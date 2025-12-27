"""
DeepSeek V3 Configuration Module
================================
Provides configuration management for model architecture, training, and inference.
Supports loading from YAML files with validation and defaults.

Key Components:
- ModelConfig: DeepSeek V3 model architecture settings (MLA, MoE, MTP)
- TrainingConfig: Training hyperparameters for pretrain/SFT/RL
- DataConfig: Dataset and tokenizer settings
- VisualizationConfig: TensorBoard visualization settings
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from logger import get_logger

# Initialize logger
logger = get_logger(__name__)


@dataclass
class MoEConfig:
    """
    Mixture of Experts Configuration
    
    DeepSeek V3 uses auxiliary-loss-free load balancing for MoE training.
    Experts are routed using top-k selection with shared experts always activated.
    """
    enabled: bool = True
    num_experts: int = 16              # N: total number of routed experts
    num_experts_per_tok: int = 2       # K_r: experts activated per token
    num_shared_experts: int = 2        # K_s: shared experts (always active)
    expert_hidden_size: int = 768      # Hidden dim in each expert FFN
    aux_loss_alpha: float = 0.001      # Load balancing loss coefficient
    seq_aux_loss: bool = True          # Sequence-level auxiliary loss
    routed_scaling_factor: float = 1.0 # Scaling factor for routed outputs
    router_bias: bool = False          # Use bias in gating network
    router_jitter_noise: float = 0.0   # Noise for training stability


@dataclass
class MTPConfig:
    """
    Multi-Token Prediction Configuration
    
    DeepSeek V3 uses MTP as an auxiliary training objective and for
    speculative decoding during inference.
    """
    enabled: bool = True
    num_predict_tokens: int = 2        # Additional tokens to predict (D in paper)
    mtp_loss_weight: float = 0.3       # Weight for MTP auxiliary loss


@dataclass
class ModelConfig:
    """
    DeepSeek V3 Model Architecture Configuration
    
    Key innovations:
    1. Multi-head Latent Attention (MLA): Low-rank KV compression
    2. DeepSeekMoE: Mixture of Experts with shared experts
    3. Multi-Token Prediction: Auxiliary objective for training
    
    Tensor shapes (in comments) use:
    - B: batch size
    - L: sequence length
    - D: hidden size (d_model)
    - H: number of attention heads
    - d_h: head dimension
    - d_c: KV compression rank
    - N: number of experts
    - K: experts per token
    """
    # Basic dimensions
    vocab_size: int = 50257            # V: vocabulary size
    hidden_size: int = 512             # D: model hidden dimension
    max_position_embeddings: int = 1024  # Maximum sequence length
    num_hidden_layers: int = 6         # L_layers: transformer blocks
    
    # MLA (Multi-head Latent Attention) parameters
    num_attention_heads: int = 8       # H: number of query heads
    head_dim: int = 64                 # d_h: dimension per head
    kv_lora_rank: int = 64             # d_c: KV compression dimension
    q_lora_rank: int = 96              # d_c': Q compression dimension
    qk_nope_head_dim: int = 32         # d_h^nope: non-RoPE head dim
    qk_rope_head_dim: int = 32         # d_h^rope: RoPE head dim
    v_head_dim: int = 64               # d_h^v: value head dim
    
    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # MoE (Mixture of Experts)
    moe: MoEConfig = field(default_factory=MoEConfig)
    moe_layer_freq: int = 2            # Apply MoE every N layers
    first_moe_layer: int = 1           # First MoE layer index
    
    # Standard FFN
    intermediate_size: int = 1408      # I: FFN hidden dimension
    hidden_act: str = "silu"           # Activation function
    
    # MTP (Multi-Token Prediction)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    
    # Regularization
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    expert_dropout: float = 0.0
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Embeddings
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    
    def __post_init__(self):
        """Validate and convert nested configs."""
        if isinstance(self.moe, dict):
            self.moe = MoEConfig(**self.moe)
        if isinstance(self.mtp, dict):
            self.mtp = MTPConfig(**self.mtp)
    
    @property
    def total_head_dim(self) -> int:
        """Total query head dimension: d_h^nope + d_h^rope."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim
    
    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if layer uses MoE."""
        if not self.moe.enabled:
            return False
        if layer_idx < self.first_moe_layer:
            return False
        return (layer_idx - self.first_moe_layer) % self.moe_layer_freq == 0


@dataclass
class TrainingConfig:
    """
    Training Configuration
    
    Supports pretrain, SFT, and RL training modes.
    """
    # Basic
    batch_size: int = 16
    num_epochs: int = 5
    max_steps: int = 5000
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 200
    warmup_ratio: float = 0.0
    
    # Gradient
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # LR scheduler
    lr_scheduler: str = "cosine"
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    logging_steps: int = 20
    tensorboard_dir: str = "runs"
    
    # Evaluation
    eval_steps: int = 200
    eval_samples: int = 500
    
    # Device
    device: str = "auto"
    mixed_precision: str = "no"
    
    # Reproducibility
    seed: int = 42


@dataclass
class SFTConfig(TrainingConfig):
    """SFT-specific training configuration."""
    batch_size: int = 8
    num_epochs: int = 3
    max_steps: int = 2000
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    adam_beta2: float = 0.999
    checkpoint_dir: str = "checkpoints/sft"
    tensorboard_dir: str = "runs/sft"
    
    # LoRA
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class RLConfig(TrainingConfig):
    """
    RL Training Configuration
    
    Supports multiple RL algorithms:
    - DPO: Direct Preference Optimization (offline)
    - GRPO: Group Relative Policy Optimization (DeepSeek style)
    - PPO: Proximal Policy Optimization (online)
    """
    # Algorithm selection: "dpo", "grpo", "ppo"
    algorithm: str = "grpo"
    
    # Basic training
    batch_size: int = 4
    num_epochs: int = 1
    max_steps: int = 500
    learning_rate: float = 5e-7
    weight_decay: float = 0.0
    
    # GRPO specific parameters
    group_size: int = 4            # G: responses per prompt for group scoring
    
    # DPO specific parameters
    dpo_beta: float = 0.1          # Temperature for DPO (higher = more aggressive)
    dpo_label_smoothing: float = 0.0  # Label smoothing for robustness
    
    # PPO specific parameters
    ppo_epochs: int = 4            # PPO update epochs per batch
    value_coef: float = 0.5        # Value function loss coefficient
    entropy_coef: float = 0.01     # Entropy bonus coefficient
    gae_lambda: float = 0.95       # GAE lambda parameter
    
    # Common RL parameters
    kl_coef: float = 0.1           # KL divergence penalty
    clip_range: float = 0.2        # PPO-style clipping
    gamma: float = 1.0             # Discount factor
    
    # Reward configuration
    reward_model: str = "rule_based"  # "rule_based" or path to reward model
    
    # Gradient settings
    gradient_accumulation_steps: int = 4
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/rl"
    tensorboard_dir: str = "runs/rl"
    
    # Generation parameters for online RL
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # Early stopping
    target_kl: float = 0.02        # Stop PPO update if KL exceeds this
    
    # Reference model
    use_reference_model: bool = True  # Whether to use reference for KL


@dataclass
class DataConfig:
    """
    Data Configuration
    
    Supports different datasets for pretrain, SFT, and RL.
    """
    # Pretrain
    pretrain_dataset_name: str = "openwebtext"
    pretrain_dataset_config: Optional[str] = None
    pretrain_data_dir: str = "data/pretrain"
    pretrain_max_seq_length: int = 512
    pretrain_max_samples: Optional[int] = 2000000  # Limit samples (~10GB)
    pretrain_streaming: bool = False  # Streaming mode for very large datasets
    
    # SFT
    sft_dataset_name: str = "tatsu-lab/alpaca"
    sft_data_dir: str = "data/sft"
    sft_max_seq_length: int = 512
    sft_max_samples: int = 10000
    
    # RL
    rl_dataset_name: str = "Anthropic/hh-rlhf"
    rl_dataset_config: str = "default"
    rl_data_dir: str = "data/rl"
    rl_max_seq_length: int = 256
    rl_max_samples: int = 1000
    
    # Tokenizer
    tokenizer_name: str = "gpt2"
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True


@dataclass
class VisualizationConfig:
    """TensorBoard Visualization Settings."""
    # Attention
    visualize_attention: bool = True
    attention_log_steps: int = 200
    num_attention_heads_to_show: int = 4
    num_attention_layers_to_show: int = 2
    
    # MoE
    visualize_moe: bool = True
    moe_log_steps: int = 200
    
    # MTP
    visualize_mtp: bool = True
    mtp_log_steps: int = 200
    
    # RoPE
    visualize_rope: bool = True
    rope_log_steps: int = 1000
    
    # Mask
    visualize_masks: bool = True
    mask_log_steps: int = 1000
    
    # Embeddings
    visualize_embeddings: bool = True
    embedding_log_steps: int = 2000
    num_embedding_samples: int = 500
    
    # Loss
    visualize_loss: bool = True
    
    # Weights
    visualize_weights: bool = True
    weights_log_steps: int = 1000
    
    # Gradients
    visualize_gradients: bool = True
    gradient_log_steps: int = 200
    
    # Generation
    visualize_generation: bool = True
    generation_log_steps: int = 500
    generation_prompts: List[str] = field(default_factory=lambda: [
        "The meaning of life is",
        "In the beginning",
        "Once upon a time",
        "Artificial intelligence will",
        "The future of computing"
    ])
    generation_max_length: int = 100
    generation_temperature: float = 0.7


@dataclass
class InferenceConfig:
    """Inference Configuration."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    use_mtp_decoding: bool = True
    batch_size: int = 4
    device: str = "auto"


@dataclass
class DeepSeekV3Config:
    """
    Complete DeepSeek V3 Configuration
    
    Combines all sub-configurations into a single config object.
    """
    experiment_name: str = "deepseek_v3"
    model: ModelConfig = field(default_factory=ModelConfig)
    pretraining: TrainingConfig = field(default_factory=TrainingConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    def __post_init__(self):
        """Convert nested dicts to config objects."""
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.pretraining, dict):
            self.pretraining = TrainingConfig(**self.pretraining)
        if isinstance(self.sft, dict):
            self.sft = SFTConfig(**self.sft)
        if isinstance(self.rl, dict):
            self.rl = RLConfig(**self.rl)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.visualization, dict):
            self.visualization = VisualizationConfig(**self.visualization)
        if isinstance(self.inference, dict):
            self.inference = InferenceConfig(**self.inference)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DeepSeekV3Config":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            DeepSeekV3Config instance
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DeepSeekV3Config":
        """
        Create config from dictionary.
        
        Handles nested configuration structures.
        """
        # Extract top-level settings
        experiment_name = config_dict.get("experiment_name", "deepseek_v3")
        
        # Parse model config
        model_dict = config_dict.get("model", {})
        model_config = ModelConfig(**{
            k: v for k, v in model_dict.items()
            if k in ModelConfig.__dataclass_fields__
        })
        
        # Parse training configs
        pretrain_dict = config_dict.get("pretraining", {})
        pretrain_config = TrainingConfig(**{
            k: v for k, v in pretrain_dict.items()
            if k in TrainingConfig.__dataclass_fields__
        })
        
        sft_dict = config_dict.get("sft", {})
        sft_config = SFTConfig(**{
            k: v for k, v in sft_dict.items()
            if k in SFTConfig.__dataclass_fields__
        })
        
        rl_dict = config_dict.get("rl", {})
        rl_config = RLConfig(**{
            k: v for k, v in rl_dict.items()
            if k in RLConfig.__dataclass_fields__
        })
        
        # Parse data config
        data_dict = config_dict.get("data", {})
        data_config = cls._parse_data_config(data_dict)
        
        # Parse visualization config
        vis_dict = config_dict.get("visualization", {})
        vis_config = VisualizationConfig(**{
            k: v for k, v in vis_dict.items()
            if k in VisualizationConfig.__dataclass_fields__
        })
        
        # Parse inference config
        inf_dict = config_dict.get("inference", {})
        inf_config = InferenceConfig(**{
            k: v for k, v in inf_dict.items()
            if k in InferenceConfig.__dataclass_fields__
        })
        
        return cls(
            experiment_name=experiment_name,
            model=model_config,
            pretraining=pretrain_config,
            sft=sft_config,
            rl=rl_config,
            data=data_config,
            visualization=vis_config,
            inference=inf_config
        )
    
    @staticmethod
    def _parse_data_config(data_dict: Dict[str, Any]) -> DataConfig:
        """Parse nested data configuration."""
        flat_dict = {}
        
        # Flatten pretrain config
        pretrain = data_dict.get("pretrain", {})
        flat_dict["pretrain_dataset_name"] = pretrain.get("dataset_name", "wikitext")
        flat_dict["pretrain_dataset_config"] = pretrain.get("dataset_config", "wikitext-2-raw-v1")
        flat_dict["pretrain_data_dir"] = pretrain.get("data_dir", "data/pretrain")
        flat_dict["pretrain_max_seq_length"] = pretrain.get("max_seq_length", 512)
        flat_dict["pretrain_max_samples"] = pretrain.get("max_samples", 2000000)
        flat_dict["pretrain_streaming"] = pretrain.get("streaming", False)
        
        # Flatten SFT config
        sft = data_dict.get("sft", {})
        flat_dict["sft_dataset_name"] = sft.get("dataset_name", "tatsu-lab/alpaca")
        flat_dict["sft_data_dir"] = sft.get("data_dir", "data/sft")
        flat_dict["sft_max_seq_length"] = sft.get("max_seq_length", 512)
        flat_dict["sft_max_samples"] = sft.get("max_samples", 10000)
        
        # Flatten RL config
        rl = data_dict.get("rl", {})
        flat_dict["rl_dataset_name"] = rl.get("dataset_name", "Anthropic/hh-rlhf")
        flat_dict["rl_dataset_config"] = rl.get("dataset_config", "default")
        flat_dict["rl_data_dir"] = rl.get("data_dir", "data/rl")
        flat_dict["rl_max_seq_length"] = rl.get("max_seq_length", 256)
        flat_dict["rl_max_samples"] = rl.get("max_samples", 1000)
        
        # Common settings
        flat_dict["tokenizer_name"] = data_dict.get("tokenizer_name", "gpt2")
        flat_dict["num_workers"] = data_dict.get("num_workers", 4)
        flat_dict["pin_memory"] = data_dict.get("pin_memory", True)
        flat_dict["shuffle"] = data_dict.get("shuffle", True)
        
        return DataConfig(**flat_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def print_config(self):
        """Print configuration summary."""
        logger.info("=" * 60)
        logger.info(f"DeepSeek V3 Configuration: {self.experiment_name}")
        logger.info("=" * 60)
        logger.info(f"Model Architecture:")
        logger.info(f"  Hidden size: {self.model.hidden_size}")
        logger.info(f"  Layers: {self.model.num_hidden_layers}")
        logger.info(f"  Attention heads: {self.model.num_attention_heads}")
        logger.info(f"  KV LoRA rank: {self.model.kv_lora_rank}")
        logger.info(f"  MoE enabled: {self.model.moe.enabled}")
        if self.model.moe.enabled:
            logger.info(f"    Experts: {self.model.moe.num_experts}")
            logger.info(f"    Experts per token: {self.model.moe.num_experts_per_tok}")
            logger.info(f"    Shared experts: {self.model.moe.num_shared_experts}")
        logger.info(f"  MTP enabled: {self.model.mtp.enabled}")
        if self.model.mtp.enabled:
            logger.info(f"    Predict tokens: {self.model.mtp.num_predict_tokens}")
        logger.info("=" * 60)


def load_config(config_path: Optional[str] = None) -> DeepSeekV3Config:
    """
    Load configuration from file or return defaults.
    
    Args:
        config_path: Path to YAML config file, or None for defaults
        
    Returns:
        DeepSeekV3Config instance
    """
    if config_path is None:
        # Try to load default config
        default_path = Path(__file__).parent / "config_default.yaml"
        if default_path.exists():
            return DeepSeekV3Config.from_yaml(str(default_path))
        return DeepSeekV3Config()
    
    return DeepSeekV3Config.from_yaml(config_path)


def get_device(device_str: str = "auto") -> str:
    """
    Determine the best available device.
    
    Args:
        device_str: "auto", "cuda", "mps", or "cpu"
        
    Returns:
        Device string for PyTorch
    """
    import torch
    
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    config.print_config()
    
    # Test model config
    logger.info(f"Is layer 0 MoE? {config.model.is_moe_layer(0)}")
    logger.info(f"Is layer 1 MoE? {config.model.is_moe_layer(1)}")
    logger.info(f"Is layer 2 MoE? {config.model.is_moe_layer(2)}")
    
    # Test device detection
    device = get_device(config.pretraining.device)
    logger.info(f"Selected device: {device}")
