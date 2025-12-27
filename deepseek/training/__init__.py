"""
DeepSeek V3 Training Module
===========================

Contains trainer implementations for all training stages:
- PretrainTrainer: Language model pretraining
- SFTTrainer: Supervised fine-tuning
- RL trainers: GRPO, PPO, DPO
"""

from .trainer import (
    BaseTrainer,
    PretrainTrainer,
    SFTTrainer,
    GRPOTrainer as GRPOTrainerBase,
    create_trainer,
    Visualizer,
)

from .rl_trainer_base import (
    DPOTrainer,
    RuleBasedReward,
    CompositeReward,
    LengthReward,
)

from .rl_trainer_algorithms import (
    GRPOTrainer,
    PPOTrainer,
)

__all__ = [
    # Base trainers
    "BaseTrainer",
    "PretrainTrainer",
    "SFTTrainer",
    "GRPOTrainerBase",
    "create_trainer",
    "Visualizer",
    # RL trainers
    "DPOTrainer",
    "GRPOTrainer",
    "PPOTrainer",
    # Rewards
    "RuleBasedReward",
    "CompositeReward",
    "LengthReward",
]
