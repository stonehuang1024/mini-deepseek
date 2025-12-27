"""
DeepSeek V3 Data Module
=======================

Contains dataset implementations for all training stages:
- Pretrain datasets (WikiText-2, OpenWebText)
- SFT datasets (Alpaca)
- RL datasets (HH-RLHF)
"""

from .dataset import (
    get_tokenizer,
    create_dataloaders,
    PretrainDataset,
    SFTDataset,
    RLDataset,
)

from .rl_dataset import (
    create_rl_dataloaders,
    DPODataset,
    GRPODataset,
    PPODataset,
)

__all__ = [
    # Common
    "get_tokenizer",
    "create_dataloaders",
    # Pretrain/SFT
    "PretrainDataset",
    "SFTDataset",
    "RLDataset",
    # RL specific
    "create_rl_dataloaders",
    "DPODataset",
    "GRPODataset",
    "PPODataset",
]
