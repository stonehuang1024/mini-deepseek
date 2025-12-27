"""
DeepSeek V3 Model Module
========================

Contains the model architecture implementations:
- Multi-head Latent Attention (MLA)
- DeepSeek V3 Model with MoE and MTP
"""

from .model import (
    DeepSeekV3Model,
    TransformerBlock,
    DeepSeekMoE,
    Expert,
    MTPHead,
    SwiGLU,
    count_parameters,
    print_model_summary,
)

from .attention import (
    MultiHeadLatentAttention,
    StandardAttention,
    RotaryEmbedding,
    RMSNorm,
    apply_rotary_pos_emb,
)

__all__ = [
    # Model
    "DeepSeekV3Model",
    "TransformerBlock",
    "DeepSeekMoE",
    "Expert",
    "MTPHead",
    "SwiGLU",
    "count_parameters",
    "print_model_summary",
    # Attention
    "MultiHeadLatentAttention",
    "StandardAttention",
    "RotaryEmbedding",
    "RMSNorm",
    "apply_rotary_pos_emb",
]
