"""
DeepSeek V3 Learning Implementation
===================================

A complete, educational implementation of DeepSeek V3 architecture.

Key Features:
- Multi-head Latent Attention (MLA)
- DeepSeekMoE (Mixture of Experts)
- Multi-Token Prediction (MTP)
- GRPO (Group Relative Policy Optimization)

Usage:
    from deepseek_v3 import DeepSeekV3Model, load_config
    
    config = load_config("config_default.yaml")
    model = DeepSeekV3Model(config.model)
"""

from .config import (
    DeepSeekV3Config,
    ModelConfig,
    TrainingConfig,
    SFTConfig,
    RLConfig,
    DataConfig,
    VisualizationConfig,
    InferenceConfig,
    MoEConfig,
    MTPConfig,
    load_config,
    get_device,
)

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

from .dataset import (
    get_tokenizer,
    create_dataloaders,
    PretrainDataset,
    SFTDataset,
    RLDataset,
)

from .trainer import (
    BaseTrainer,
    PretrainTrainer,
    SFTTrainer,
    GRPOTrainer,
    create_trainer,
    Visualizer,
)

from .inference import (
    DeepSeekInference,
    load_model_for_inference,
)

from .logger import (
    get_logger,
    set_log_level,
    setup_file_logging,
    get_training_logger,
    get_model_logger,
    get_data_logger,
    get_inference_logger,
    DeepSeekLogger,
    ColoredFormatter,
)

__version__ = "0.1.0"
__author__ = "DeepSeek V3 Learning Project"

__all__ = [
    # Config
    "DeepSeekV3Config",
    "ModelConfig",
    "TrainingConfig",
    "SFTConfig",
    "RLConfig",
    "DataConfig",
    "VisualizationConfig",
    "InferenceConfig",
    "MoEConfig",
    "MTPConfig",
    "load_config",
    "get_device",
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
    # Dataset
    "get_tokenizer",
    "create_dataloaders",
    "PretrainDataset",
    "SFTDataset",
    "RLDataset",
    # Trainer
    "BaseTrainer",
    "PretrainTrainer",
    "SFTTrainer",
    "GRPOTrainer",
    "create_trainer",
    "Visualizer",
    # Inference
    "DeepSeekInference",
    "load_model_for_inference",
    # Logger
    "get_logger",
    "set_log_level",
    "setup_file_logging",
    "get_training_logger",
    "get_model_logger",
    "get_data_logger",
    "get_inference_logger",
    "DeepSeekLogger",
    "ColoredFormatter",
]
