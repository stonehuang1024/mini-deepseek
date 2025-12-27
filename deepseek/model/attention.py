"""
DeepSeek V3 Attention Module
============================

Implements Multi-head Latent Attention (MLA) - the key innovation in DeepSeek V3.

MLA uses low-rank compression for Keys and Values to reduce KV cache memory
during inference while maintaining model quality.

Key Features:
1. Low-rank KV compression: Projects KV to smaller dimension d_c
2. Decoupled RoPE: Separate dimensions for position-aware and position-free features
3. Efficient inference: Compressed KV cache reduces memory ~O(d_c) instead of O(H*d_h)

References:
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- DeepSeek-V3 Technical Report
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm, commonly used in modern LLMs.
    
    Shape:
        Input: (B, L, D) - batch, sequence length, hidden dimension
        Output: (B, L, D) - same shape as input
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Args:
            hidden_size: D - dimension to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # Shape: (D,)
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, L, D)
        Returns:
            Normalized tensor, shape (B, L, D)
        """
        # Compute RMS: sqrt(mean(x^2))
        variance = x.pow(2).mean(-1, keepdim=True)  # Shape: (B, L, 1)
        x = x * torch.rsqrt(variance + self.eps)    # Shape: (B, L, D)
        return self.weight * x  # Element-wise scaling


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes position information through rotation in 2D subspaces.
    Used for the position-aware portion of MLA queries and keys.
    
    Shape:
        Input: (B, H, L, d_h) - batch, heads, seq_len, head_dim
        Output: (B, H, L, d_h) - rotated embeddings
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            dim: d_h^rope - dimension of RoPE portion (must be even)
            max_position_embeddings: Maximum sequence length to cache
            base: Theta base frequency
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
        # Shape: (d/2,)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache cos/sin values
        self._set_cos_sin_cache(max_position_embeddings, device)
    
    def _set_cos_sin_cache(
        self, 
        seq_len: int, 
        device: Optional[torch.device] = None
    ):
        """Pre-compute cos and sin for all positions up to seq_len."""
        self.max_seq_len_cached = seq_len
        
        # Position indices: shape (L,)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        # Outer product: shape (L, d/2)
        freqs = torch.outer(t, self.inv_freq)
        
        # Create full rotation: shape (L, d) by repeating each freq twice
        # This matches the rotation pattern: [cos, cos, ...], [sin, sin, ...]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin: shape (1, 1, L, d)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin for given positions.
        
        Args:
            x: Input tensor, shape (B, H, L, d) - used only to get seq_len
            position_ids: Optional position indices, shape (B, L)
            
        Returns:
            cos: Cosine values, shape (1, 1, L, d) or (B, 1, L, d)
            sin: Sine values, shape (1, 1, L, d) or (B, 1, L, d)
        """
        seq_len = x.shape[2]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)
        
        if position_ids is None:
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        else:
            # Gather cos/sin for specific positions
            # position_ids shape: (B, L)
            # cos_cached shape: (1, 1, max_L, d)
            # Output shape should be (B, 1, L, d) for proper broadcasting
            B = position_ids.shape[0]
            # Index with position_ids: (1, 1, max_L, d) -> select positions -> (B, L, d)
            cos = self.cos_cached[0, 0, position_ids, :]  # (B, L, d)
            sin = self.sin_cached[0, 0, position_ids, :]  # (B, L, d)
            # Add head dimension for broadcasting: (B, L, d) -> (B, 1, L, d)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    For RoPE, we need to rotate pairs: [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]
    
    Args:
        x: Input tensor, shape (..., d)
    Returns:
        Rotated tensor, shape (..., d)
    """
    x1 = x[..., : x.shape[-1] // 2]  # First half
    x2 = x[..., x.shape[-1] // 2 :]  # Second half
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    The rotation formula: x' = x * cos + rotate_half(x) * sin
    
    Args:
        q: Query tensor, shape (B, H, L, d_h^rope)
        k: Key tensor, shape (B, H, L, d_h^rope)
        cos: Cosine values, shape (1, 1, L, d_h^rope) or (B, 1, L, d_h^rope)
        sin: Sine values, shape (1, 1, L, d_h^rope) or (B, 1, L, d_h^rope)
        
    Returns:
        q_embed: Rotated queries, shape (B, H, L, d_h^rope)
        k_embed: Rotated keys, shape (B, H, L, d_h^rope)
    """
    # Ensure cos/sin can broadcast to (B, H, L, d) - expand the head dimension
    # cos/sin shape: (1, 1, L, d) -> broadcasts to (B, H, L, d)
    # or if shape is (B, 1, L, d), it also broadcasts correctly
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) - DeepSeek V3's efficient attention.
    
    MLA compresses Keys and Values through low-rank projections to reduce
    memory usage during inference. It decouples position encoding into
    separate dimensions.
    
    Architecture:
    1. Compress input to latent dimension d_c for KV
    2. Expand latent to produce K and V
    3. Q uses separate low-rank compression
    4. Apply RoPE only to the "rope" portion of Q and K
    5. Concatenate rope and nope portions for attention
    
    Tensor shapes:
    - Input: (B, L, D)
    - c_kv (compressed KV): (B, L, d_c)
    - K after expansion: (B, L, H, d_h^nope + d_h^rope)
    - V after expansion: (B, L, H, d_h^v)
    - Q: (B, L, H, d_h^nope + d_h^rope)
    - Output: (B, L, D)
    """
    
    def __init__(
        self,
        hidden_size: int,          # D: model dimension
        num_attention_heads: int,  # H: number of heads
        kv_lora_rank: int,         # d_c: compressed KV dimension
        q_lora_rank: int,          # d_c': compressed Q dimension
        qk_nope_head_dim: int,     # d_h^nope: non-position head dim
        qk_rope_head_dim: int,     # d_h^rope: position head dim
        v_head_dim: int,           # d_h^v: value head dim
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.layer_idx = layer_idx
        
        # Total query/key head dimension
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        
        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)
        
        # ============ Q projections ============
        # Down projection: D -> d_c' (low-rank compression)
        # Shape: (D, d_c')
        self.q_down_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        
        # Up projection: d_c' -> H * (d_h^nope + d_h^rope)
        # Shape: (d_c', H * qk_head_dim)
        self.q_up_proj = nn.Linear(
            q_lora_rank, 
            num_attention_heads * self.qk_head_dim, 
            bias=False
        )
        
        # ============ KV projections ============
        # Down projection: D -> d_c (compress KV together)
        # Shape: (D, d_c)
        self.kv_down_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        
        # K up projection: d_c -> H * (d_h^nope + d_h^rope)
        # We also add a separate rope key dimension for each head
        # Shape: (d_c, H * qk_head_dim)
        self.k_up_proj = nn.Linear(
            kv_lora_rank, 
            num_attention_heads * self.qk_head_dim, 
            bias=False
        )
        
        # V up projection: d_c -> H * d_h^v
        # Shape: (d_c, H * v_head_dim)
        self.v_up_proj = nn.Linear(
            kv_lora_rank, 
            num_attention_heads * v_head_dim, 
            bias=False
        )
        
        # ============ Output projection ============
        # Shape: (H * d_h^v, D)
        self.o_proj = nn.Linear(
            num_attention_heads * v_head_dim, 
            hidden_size, 
            bias=False
        )
        
        # ============ RoPE ============
        self.rotary_emb = RotaryEmbedding(
            qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        
        # Dropout
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        # Storage for visualization
        self.last_attention_weights = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,       # Shape: (B, L, D)
        attention_mask: Optional[torch.Tensor] = None,  # Shape: (B, 1, L, L) or (B, 1, 1, L)
        position_ids: Optional[torch.Tensor] = None,    # Shape: (B, L)
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for MLA.
        
        Args:
            hidden_states: Input, shape (B, L, D)
            attention_mask: Attention mask, shape (B, 1, L, L) or (B, 1, 1, L)
            position_ids: Position indices, shape (B, L)
            past_key_value: Cached KV for incremental decoding
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights
            
        Returns:
            output: Attention output, shape (B, L, D)
            attention_weights: Optional attention weights, shape (B, H, L, L)
            past_key_value: Optional updated KV cache
        """
        B, L, D = hidden_states.shape
        
        # ============ Q computation ============
        # Down project: (B, L, D) -> (B, L, d_c')
        q_compressed = self.q_down_proj(hidden_states)
        
        # Up project: (B, L, d_c') -> (B, L, H * qk_head_dim)
        q = self.q_up_proj(q_compressed)
        
        # Reshape: (B, L, H * qk_head_dim) -> (B, L, H, qk_head_dim)
        q = q.view(B, L, self.num_heads, self.qk_head_dim)
        
        # Transpose: (B, L, H, qk_head_dim) -> (B, H, L, qk_head_dim)
        q = q.transpose(1, 2)
        
        # ============ KV computation ============
        # Down project: (B, L, D) -> (B, L, d_c)
        kv_compressed = self.kv_down_proj(hidden_states)
        
        # Up project K: (B, L, d_c) -> (B, L, H * qk_head_dim)
        k = self.k_up_proj(kv_compressed)
        k = k.view(B, L, self.num_heads, self.qk_head_dim)
        k = k.transpose(1, 2)  # (B, H, L, qk_head_dim)
        
        # Up project V: (B, L, d_c) -> (B, L, H * v_head_dim)
        v = self.v_up_proj(kv_compressed)
        v = v.view(B, L, self.num_heads, self.v_head_dim)
        v = v.transpose(1, 2)  # (B, H, L, v_head_dim)
        
        # ============ Apply RoPE to rope portion ============
        # Split Q and K into nope and rope portions
        # q_nope: (B, H, L, d_h^nope), q_rope: (B, H, L, d_h^rope)
        q_nope, q_rope = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        k_nope, k_rope = k.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        
        # Get rotary embeddings
        cos, sin = self.rotary_emb(q_rope, position_ids)
        
        # Apply RoPE to rope portions only
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        
        # Concatenate back: (B, H, L, qk_head_dim)
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        
        # ============ Handle KV cache ============
        if past_key_value is not None:
            # Concatenate with cached KV
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)  # (B, H, L_past + L, qk_head_dim)
            v = torch.cat([past_v, v], dim=2)  # (B, H, L_past + L, v_head_dim)
        
        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None
        
        # ============ Attention computation ============
        # Q @ K^T: (B, H, L, qk_head_dim) @ (B, H, qk_head_dim, L_kv) -> (B, H, L, L_kv)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            # attention_mask is typically (B, 1, L, L_kv) with -inf for masked positions
            attention_scores = attention_scores + attention_mask
        
        # Softmax: (B, H, L, L_kv)
        attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_weights = attention_weights.to(q.dtype)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Store for visualization
        self.last_attention_weights = attention_weights.detach()
        
        # Attention @ V: (B, H, L, L_kv) @ (B, H, L_kv, v_head_dim) -> (B, H, L, v_head_dim)
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape: (B, H, L, v_head_dim) -> (B, L, H, v_head_dim) -> (B, L, H * v_head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(B, L, -1)
        
        # Output projection: (B, L, H * v_head_dim) -> (B, L, D)
        output = self.o_proj(attention_output)
        
        if output_attentions:
            return output, attention_weights, present_key_value
        return output, None, present_key_value


class StandardAttention(nn.Module):
    """
    Standard Multi-Head Attention for comparison.
    
    Uses traditional Q, K, V projections without low-rank compression.
    
    Shape:
        Input: (B, L, D)
        Output: (B, L, D)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        
        self.scale = 1.0 / math.sqrt(head_dim)
        
        # QKV projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.last_attention_weights = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Standard attention forward pass."""
        B, L, D = hidden_states.shape
        
        # Project Q, K, V: (B, L, D) -> (B, L, H * d_h)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape and transpose: (B, L, H * d_h) -> (B, H, L, d_h)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present_key_value = (k, v) if use_cache else None
        
        # Attention: (B, H, L, d_h) @ (B, H, d_h, L_kv) -> (B, H, L, L_kv)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_weights = attention_weights.to(q.dtype)
        attention_weights = self.attention_dropout(attention_weights)
        
        self.last_attention_weights = attention_weights.detach()
        
        # Output: (B, H, L, L_kv) @ (B, H, L_kv, d_h) -> (B, H, L, d_h)
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape: (B, H, L, d_h) -> (B, L, H * d_h)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(B, L, -1)
        
        # Output projection: (B, L, H * d_h) -> (B, L, D)
        output = self.o_proj(attention_output)
        
        if output_attentions:
            return output, attention_weights, present_key_value
        return output, None, present_key_value
