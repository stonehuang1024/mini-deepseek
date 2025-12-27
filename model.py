"""
DeepSeek V3 Model Implementation
================================

Complete implementation of DeepSeek V3 architecture with:
1. Multi-head Latent Attention (MLA) - efficient KV compression
2. DeepSeekMoE - Mixture of Experts with shared experts
3. Multi-Token Prediction (MTP) - auxiliary training objective

Tensor shapes are annotated throughout:
- B: batch size
- L: sequence length
- D: hidden size (d_model)
- H: number of attention heads
- N: number of experts
- K: experts per token
- V: vocabulary size
"""

import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import (
    MultiHeadLatentAttention,
    StandardAttention,
    RMSNorm,
)
from config import ModelConfig, MoEConfig
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class SwiGLU(nn.Module):
    """
    SwiGLU activation: Swish-Gated Linear Unit.
    
    SwiGLU(x) = Swish(xW_gate) * (xW_up)
    where Swish(x) = x * sigmoid(x)
    
    Shape:
        Input: (B, L, D)
        Output: (B, L, D)
    """
    
    def __init__(
        self,
        hidden_size: int,      # D: input dimension
        intermediate_size: int, # I: FFN hidden dimension
        bias: bool = False,
    ):
        super().__init__()
        # Gate projection: D -> I
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        # Up projection: D -> I
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        # Down projection: I -> D
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, L, D)
        Returns:
            Output tensor, shape (B, L, D)
        """
        # gate: (B, L, I), up: (B, L, I)
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        # (B, L, I) -> (B, L, D)
        return self.down_proj(gate * up)


class Expert(nn.Module):
    """
    Single Expert FFN module.
    
    Each expert is an independent FFN that processes tokens routed to it.
    Uses SwiGLU activation following LLaMA/DeepSeek design.
    
    Shape:
        Input: (num_tokens, D) - flattened tokens assigned to this expert
        Output: (num_tokens, D)
    """
    
    def __init__(
        self,
        hidden_size: int,       # D: model dimension
        expert_hidden_size: int, # Expert FFN hidden dimension
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ffn = SwiGLU(hidden_size, expert_hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (num_tokens, D)
        Returns:
            Output tensor, shape (num_tokens, D)
        """
        return self.dropout(self.ffn(x))


class MoEGate(nn.Module):
    """
    MoE Routing Gate with auxiliary load balancing.
    
    Routes each token to top-K experts using softmax probabilities.
    Implements auxiliary loss for load balancing across experts.
    
    Shape:
        Input: (B, L, D)
        Output:
            - router_probs: (B, L, N) - routing probabilities
            - selected_experts: (B, L, K) - indices of selected experts
            - expert_weights: (B, L, K) - normalized weights for selected experts
    """
    
    def __init__(
        self,
        hidden_size: int,       # D: input dimension
        num_experts: int,       # N: total number of experts
        num_experts_per_tok: int,  # K: experts selected per token
        aux_loss_alpha: float = 0.001,
        router_bias: bool = False,
        router_jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.aux_loss_alpha = aux_loss_alpha
        self.router_jitter_noise = router_jitter_noise
        
        # Router: projects hidden states to expert scores
        # Shape: (D, N)
        self.gate = nn.Linear(hidden_size, num_experts, bias=router_bias)
        
        # Storage for auxiliary loss
        self.aux_loss = None
        self.expert_usage = None
    
    def forward(
        self, 
        hidden_states: torch.Tensor,  # Shape: (B, L, D)
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing for MoE layer.
        
        Args:
            hidden_states: Input tensor, shape (B, L, D)
            training: Whether in training mode
            
        Returns:
            router_probs: Full probability distribution, shape (B, L, N)
            selected_experts: Indices of top-K experts, shape (B, L, K)
            expert_weights: Normalized weights for selected, shape (B, L, K)
        """
        B, L, D = hidden_states.shape
        
        # Add jitter noise during training for load balancing
        if training and self.router_jitter_noise > 0:
            hidden_states = hidden_states + torch.randn_like(hidden_states) * self.router_jitter_noise
        
        # Compute router logits: (B, L, D) -> (B, L, N)
        router_logits = self.gate(hidden_states)
        
        # Softmax to get probabilities: (B, L, N)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Select top-K experts: (B, L, K)
        expert_weights, selected_experts = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        
        # Normalize selected weights to sum to 1: (B, L, K)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights.to(hidden_states.dtype)
        
        # Compute auxiliary load balancing loss
        if training:
            self._compute_aux_loss(router_probs, selected_experts)
        
        return router_probs, selected_experts, expert_weights
    
    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,   # (B, L, N)
        selected_experts: torch.Tensor,  # (B, L, K)
    ):
        """
        Compute auxiliary loss for load balancing.
        
        The loss encourages balanced expert utilization:
        L_aux = alpha * sum_i(f_i * P_i)
        where f_i is frequency of expert i, P_i is average probability
        """
        B, L, N = router_probs.shape
        num_tokens = B * L
        
        # Count how many tokens each expert receives: (N,)
        expert_mask = F.one_hot(selected_experts, num_classes=N).sum(dim=2)  # (B, L, N)
        expert_counts = expert_mask.sum(dim=(0, 1)).float()  # (N,)
        
        # Fraction of tokens routed to each expert: (N,)
        f_i = expert_counts / (num_tokens * self.num_experts_per_tok)
        
        # Average probability for each expert: (N,)
        P_i = router_probs.mean(dim=(0, 1))
        
        # Auxiliary loss: encourage uniform f_i and P_i
        self.aux_loss = self.aux_loss_alpha * N * (f_i * P_i).sum()
        
        # Store for visualization
        self.expert_usage = expert_counts / expert_counts.sum()


class DeepSeekMoE(nn.Module):
    """
    DeepSeek Mixture of Experts layer.
    
    Combines:
    1. Shared experts (always activated) - provide common knowledge
    2. Routed experts (top-K selected) - provide specialized knowledge
    
    DeepSeek V3 uses auxiliary-loss-free load balancing, but we include
    auxiliary loss for educational purposes.
    
    Shape:
        Input: (B, L, D)
        Output: (B, L, D)
    """
    
    def __init__(
        self,
        hidden_size: int,
        moe_config: MoEConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = moe_config.num_experts
        self.num_experts_per_tok = moe_config.num_experts_per_tok
        self.num_shared_experts = moe_config.num_shared_experts
        self.routed_scaling_factor = moe_config.routed_scaling_factor
        self.layer_idx = layer_idx
        
        # Shared experts (always active)
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                Expert(
                    hidden_size=hidden_size,
                    expert_hidden_size=moe_config.expert_hidden_size,
                    dropout=moe_config.expert_dropout if hasattr(moe_config, 'expert_dropout') else 0.0,
                )
                for _ in range(self.num_shared_experts)
            ])
        else:
            self.shared_experts = None
        
        # Routed experts
        self.experts = nn.ModuleList([
            Expert(
                hidden_size=hidden_size,
                expert_hidden_size=moe_config.expert_hidden_size,
                dropout=moe_config.expert_dropout if hasattr(moe_config, 'expert_dropout') else 0.0,
            )
            for _ in range(self.num_experts)
        ])
        
        # Router gate
        self.gate = MoEGate(
            hidden_size=hidden_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            aux_loss_alpha=moe_config.aux_loss_alpha,
            router_bias=moe_config.router_bias,
            router_jitter_noise=moe_config.router_jitter_noise,
        )
        
        # Storage for visualization
        self.last_router_probs = None
        self.last_expert_weights = None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: Input tensor, shape (B, L, D)
            
        Returns:
            Output tensor, shape (B, L, D)
        """
        B, L, D = hidden_states.shape
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # 1. Shared expert computation (always active)
        if self.shared_experts is not None:
            shared_output = torch.zeros_like(hidden_states)
            for expert in self.shared_experts:
                # Each shared expert processes all tokens
                shared_output = shared_output + expert(hidden_states)
            # Average shared expert outputs
            shared_output = shared_output / self.num_shared_experts
            output = output + shared_output
        
        # 2. Routed expert computation
        # Get routing: (B, L, N), (B, L, K), (B, L, K)
        router_probs, selected_experts, expert_weights = self.gate(
            hidden_states, training=self.training
        )
        
        # Store for visualization
        self.last_router_probs = router_probs.detach()
        self.last_expert_weights = expert_weights.detach()
        
        # Flatten batch and sequence: (B*L, D)
        hidden_flat = hidden_states.view(-1, D)
        selected_flat = selected_experts.view(-1, self.num_experts_per_tok)  # (B*L, K)
        weights_flat = expert_weights.view(-1, self.num_experts_per_tok)     # (B*L, K)
        
        # Process each expert
        routed_output = torch.zeros(B * L, D, device=hidden_states.device, dtype=hidden_states.dtype)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert: (B*L, K) -> boolean mask
            # For each position (token), check if any of the K slots selected this expert
            expert_mask = (selected_flat == expert_idx)  # (B*L, K)
            
            # Get indices of tokens that selected this expert in any slot
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]  # (num_tokens,)
            
            if len(token_indices) == 0:
                continue
            
            # Get the tokens for this expert: (num_tokens, D)
            expert_input = hidden_flat[token_indices]
            
            # Get weights for these tokens (sum across slots where expert was selected)
            # expert_mask[token_indices]: (num_tokens, K)
            # weights_flat[token_indices]: (num_tokens, K)
            token_weights = (expert_mask[token_indices].float() * weights_flat[token_indices]).sum(dim=-1, keepdim=True)  # (num_tokens, 1)
            
            # Process through expert: (num_tokens, D)
            expert_output = self.experts[expert_idx](expert_input)
            
            # Weighted output
            weighted_output = expert_output * token_weights
            
            # Accumulate to output
            routed_output.index_add_(0, token_indices, weighted_output)
        
        # Reshape and add scaled routed output
        routed_output = routed_output.view(B, L, D)
        output = output + self.routed_scaling_factor * routed_output
        
        return output
    
    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """Get auxiliary loss from router."""
        return self.gate.aux_loss


class TransformerBlock(nn.Module):
    """
    Single Transformer block with MLA attention and optional MoE FFN.
    
    Architecture:
        x -> LayerNorm -> Attention -> Residual
          -> LayerNorm -> FFN/MoE -> Residual
    
    Shape:
        Input: (B, L, D)
        Output: (B, L, D)
    """
    
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Pre-attention LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Multi-head Latent Attention
        self.attention = MultiHeadLatentAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            layer_idx=layer_idx,
        )
        
        # Pre-FFN LayerNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # FFN: either MoE or standard
        self.use_moe = config.is_moe_layer(layer_idx)
        
        if self.use_moe:
            self.ffn = DeepSeekMoE(
                hidden_size=config.hidden_size,
                moe_config=config.moe,
                layer_idx=layer_idx,
            )
        else:
            self.ffn = SwiGLU(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: Input, shape (B, L, D)
            attention_mask: Attention mask, shape (B, 1, L, L)
            position_ids: Position indices, shape (B, L)
            past_key_value: KV cache from previous forward
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights
            
        Returns:
            hidden_states: Output, shape (B, L, D)
            attention_weights: Optional attention weights
            present_key_value: Optional KV cache
            aux_loss: Optional MoE auxiliary loss
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, attention_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self.dropout(attn_output)
        
        # FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + self.dropout(ffn_output)
        
        # Get MoE auxiliary loss if applicable
        aux_loss = None
        if self.use_moe and hasattr(self.ffn, 'aux_loss'):
            aux_loss = self.ffn.aux_loss
        
        return hidden_states, attention_weights, present_key_value, aux_loss


class MTPHead(nn.Module):
    """
    Multi-Token Prediction Head.
    
    Predicts multiple future tokens simultaneously as an auxiliary objective.
    Each prediction head shares the embedding but has separate output projections.
    
    Shape:
        Input: (B, L, D)
        Output: List of (B, L, V) for each prediction depth
    """
    
    def __init__(
        self,
        hidden_size: int,      # D: hidden dimension
        vocab_size: int,       # V: vocabulary size
        num_predict_tokens: int,  # Number of additional tokens to predict
        tie_embeddings: bool = True,
        embedding_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_predict_tokens = num_predict_tokens
        
        # Separate projection for each prediction depth
        # Each transforms hidden state to predict token at depth d
        self.projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_predict_tokens)
        ])
        
        # Output heads (share with embedding if tie_embeddings)
        if tie_embeddings and embedding_weight is not None:
            self.output_heads = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(num_predict_tokens)
            ])
            # Tie weights
            for head in self.output_heads:
                head.weight = embedding_weight
        else:
            self.output_heads = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(num_predict_tokens)
            ])
        
        self.layer_norms = nn.ModuleList([
            RMSNorm(hidden_size)
            for _ in range(num_predict_tokens)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute multi-token predictions.
        
        Args:
            hidden_states: Final hidden states, shape (B, L, D)
            
        Returns:
            List of logits for each prediction depth, each shape (B, L, V)
        """
        predictions = []
        
        for i in range(self.num_predict_tokens):
            # Project hidden states for this depth
            h = self.projections[i](hidden_states)
            h = self.layer_norms[i](h)
            # Compute logits: (B, L, D) -> (B, L, V)
            logits = self.output_heads[i](h)
            predictions.append(logits)
        
        return predictions


class DeepSeekV3Model(nn.Module):
    """
    Complete DeepSeek V3 Language Model.
    
    Combines:
    - Token embeddings
    - Transformer blocks with MLA and MoE
    - Multi-Token Prediction heads
    - Language modeling head
    
    Shape:
        Input: (B, L) - token indices
        Output:
            - logits: (B, L, V)
            - mtp_logits: List of (B, L, V) if MTP enabled
            - aux_loss: Total MoE auxiliary loss
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        
        # Token embedding: V -> D
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings with LM head
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Multi-Token Prediction head
        self.mtp_enabled = config.mtp.enabled
        if self.mtp_enabled:
            self.mtp_head = MTPHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                num_predict_tokens=config.mtp.num_predict_tokens,
                tie_embeddings=config.tie_word_embeddings,
                embedding_weight=self.embed_tokens.weight if config.tie_word_embeddings else None,
            )
            self.mtp_loss_weight = config.mtp.mtp_loss_weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights using normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get token embedding layer."""
        return self.embed_tokens
    
    def set_input_embeddings(self, embeddings: nn.Embedding):
        """Set token embedding layer."""
        self.embed_tokens = embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,                    # Shape: (B, L)
        attention_mask: Optional[torch.Tensor] = None,  # Shape: (B, L)
        position_ids: Optional[torch.Tensor] = None,    # Shape: (B, L)
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        labels: Optional[torch.Tensor] = None,      # Shape: (B, L)
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token indices, shape (B, L)
            attention_mask: 1 for real tokens, 0 for padding, shape (B, L)
            position_ids: Position indices, shape (B, L)
            past_key_values: List of KV caches for each layer
            use_cache: Return KV cache for incremental decoding
            output_attentions: Return attention weights
            output_hidden_states: Return all hidden states
            labels: Target token ids for loss computation, shape (B, L)
            return_dict: Return dict instead of tuple
            
        Returns:
            Dict containing:
                - logits: (B, L, V)
                - loss: scalar if labels provided
                - mtp_logits: List[(B, L, V)] if MTP enabled
                - aux_loss: MoE load balancing loss
                - attentions: List of attention weights if requested
                - hidden_states: List of hidden states if requested
                - past_key_values: KV cache if use_cache
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Get embeddings: (B, L) -> (B, L, D)
        hidden_states = self.embed_tokens(input_ids)
        
        # Create position ids if not provided
        if position_ids is None:
            if past_key_values is not None and len(past_key_values) > 0:
                past_length = past_key_values[0][0].shape[2]
                position_ids = torch.arange(
                    past_length, past_length + L, device=device
                ).unsqueeze(0).expand(B, -1)
            else:
                position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        
        # Create causal attention mask
        # Shape: (B, 1, L, L_total) where L_total includes past length
        causal_mask = self._create_causal_mask(
            B, L, 
            past_length=past_key_values[0][0].shape[2] if past_key_values else 0,
            device=device,
            dtype=hidden_states.dtype,
        )
        
        # Apply padding mask if provided
        if attention_mask is not None:
            # Expand attention_mask: (B, L) -> (B, 1, 1, L)
            # Then broadcast with causal mask
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask.masked_fill(padding_mask == 0, float('-inf'))
        
        # Process through layers
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        all_present_key_values = [] if use_cache else None
        total_aux_loss = 0.0
        
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            past_key_value = past_key_values[idx] if past_key_values else None
            
            hidden_states, attention_weights, present_key_value, aux_loss = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            if use_cache:
                all_present_key_values.append(present_key_value)
            if output_attentions:
                all_attentions.append(attention_weights)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)  # (B, L, D)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Language modeling head: (B, L, D) -> (B, L, V)
        logits = self.lm_head(hidden_states)
        
        # MTP predictions
        mtp_logits = None
        mtp_loss = 0.0
        if self.mtp_enabled:
            mtp_logits = self.mtp_head(hidden_states)  # List of (B, L, V)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            # logits: (B, L-1, V), labels: (B, L-1)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Cross entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
            
            # Add MTP loss
            if self.mtp_enabled and mtp_logits is not None:
                for depth, mtp_pred in enumerate(mtp_logits, start=1):
                    # Predict token at position i+depth from position i
                    if L > depth + 1:
                        mtp_shift_logits = mtp_pred[:, :-depth-1, :].contiguous()
                        mtp_shift_labels = labels[:, depth+1:].contiguous()
                        mtp_loss = mtp_loss + loss_fct(
                            mtp_shift_logits.view(-1, self.vocab_size),
                            mtp_shift_labels.view(-1)
                        )
                
                mtp_loss = self.mtp_loss_weight * mtp_loss / len(mtp_logits)
                loss = loss + mtp_loss
            
            # Add auxiliary loss
            if total_aux_loss > 0:
                loss = loss + total_aux_loss
        
        return {
            'logits': logits,
            'loss': loss,
            'mtp_logits': mtp_logits,
            'aux_loss': total_aux_loss if isinstance(total_aux_loss, torch.Tensor) else None,
            'attentions': all_attentions,
            'hidden_states': all_hidden_states,
            'past_key_values': all_present_key_values,
        }
    
    def _create_causal_mask(
        self,
        batch_size: int,
        seq_length: int,
        past_length: int = 0,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Create causal attention mask.
        
        Returns mask with 0 for allowed positions and -inf for masked positions.
        
        Shape: (B, 1, L, L_total) where L_total = past_length + L
        """
        total_length = past_length + seq_length
        
        # Create causal mask: lower triangular
        # For position i, can attend to positions 0..i
        mask = torch.ones(seq_length, total_length, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=past_length + 1)  # Upper triangular (excluding diagonal)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        # Expand for batch and head dimensions: (1, 1, L, L_total)
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        return mask
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,                # (B, L)
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Initial token ids, shape (B, L)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-K sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy)
            pad_token_id: Padding token id
            eos_token_id: End of sequence token id
            use_cache: Use KV cache for efficiency
            
        Returns:
            Generated token ids, shape (B, L + max_new_tokens)
        """
        B = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize
        generated = input_ids
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Get input for this step
            if use_cache and past_key_values is not None:
                # Only need the last token
                curr_input = generated[:, -1:]
            else:
                curr_input = generated
            
            # Forward pass
            outputs = self.forward(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            
            # Get logits for last position: (B, V)
            logits = outputs['logits'][:, -1, :]
            
            # Update cache
            if use_cache:
                past_key_values = outputs['past_key_values']
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Get next token
            if do_sample:
                # Top-K filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-P filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        
        return generated


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: DeepSeekV3Model, config: ModelConfig):
    """Print model architecture summary."""
    logger.info("=" * 70)
    logger.info("DeepSeek V3 Model Summary")
    logger.info("=" * 70)
    logger.info(f"Hidden size: {config.hidden_size}")
    logger.info(f"Num layers: {config.num_hidden_layers}")
    logger.info(f"Num attention heads: {config.num_attention_heads}")
    logger.info(f"Vocab size: {config.vocab_size}")
    logger.info(f"Max position embeddings: {config.max_position_embeddings}")
    logger.info("-" * 70)
    logger.info("MLA Configuration:")
    logger.info(f"  KV LoRA rank: {config.kv_lora_rank}")
    logger.info(f"  Q LoRA rank: {config.q_lora_rank}")
    logger.info(f"  QK nope head dim: {config.qk_nope_head_dim}")
    logger.info(f"  QK rope head dim: {config.qk_rope_head_dim}")
    logger.info(f"  V head dim: {config.v_head_dim}")
    logger.info("-" * 70)
    logger.info("MoE Configuration:")
    logger.info(f"  Enabled: {config.moe.enabled}")
    if config.moe.enabled:
        logger.info(f"  Num experts: {config.moe.num_experts}")
        logger.info(f"  Experts per token: {config.moe.num_experts_per_tok}")
        logger.info(f"  Shared experts: {config.moe.num_shared_experts}")
        logger.info(f"  Expert hidden size: {config.moe.expert_hidden_size}")
    logger.info("-" * 70)
    logger.info("MTP Configuration:")
    logger.info(f"  Enabled: {config.mtp.enabled}")
    if config.mtp.enabled:
        logger.info(f"  Predict tokens: {config.mtp.num_predict_tokens}")
        logger.info(f"  Loss weight: {config.mtp.mtp_loss_weight}")
    logger.info("-" * 70)
    logger.info(f"Total parameters: {count_parameters(model):,}")
    logger.info("=" * 70)


if __name__ == "__main__":
    # Test model instantiation
    from config import load_config
    
    config = load_config()
    model = DeepSeekV3Model(config.model)
    
    print_model_summary(model, config.model)
    
    # Test forward pass
    B, L = 2, 64
    input_ids = torch.randint(0, config.model.vocab_size, (B, L))
    labels = torch.randint(0, config.model.vocab_size, (B, L))
    
    logger.info("Test forward pass...")
    outputs = model(input_ids=input_ids, labels=labels, output_attentions=True)
    
    logger.info(f"Logits shape: {outputs['logits'].shape}")  # (B, L, V)
    logger.info(f"Loss: {outputs['loss'].item():.4f}")
    if outputs['mtp_logits']:
        logger.info(f"MTP logits: {len(outputs['mtp_logits'])} predictions")
    if outputs['aux_loss'] is not None:
        logger.info(f"Aux loss: {outputs['aux_loss'].item():.6f}")
    
    # Test generation
    logger.info("Test generation...")
    generated = model.generate(input_ids[:1, :10], max_new_tokens=20, do_sample=True)
    logger.info(f"Generated shape: {generated.shape}")
