#!/usr/bin/env python3
"""
DeepSeek V3 Training Script
===========================

Main entry point for training DeepSeek V3 model.

Usage:
    # Pretraining
    python train.py --mode pretrain --config config_default.yaml
    
    # SFT (from pretrained)
    python train.py --mode sft --checkpoint checkpoints/pretrain/best.pt
    
    # RL (from SFT)
    python train.py --mode rl --checkpoint checkpoints/sft/best.pt

Features:
- Automatic dataset download
- Configurable via YAML
- TensorBoard visualization
- Checkpoint resume
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, DeepSeekV3Config, get_device
from model import DeepSeekV3Model, print_model_summary, count_parameters
from dataset import get_tokenizer, create_dataloaders
from trainer import create_trainer
from logger import get_logger, set_log_level

# Initialize logger
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DeepSeek V3 Training")
    
    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        choices=["pretrain", "sft", "rl"],
        help="Training mode",
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="config_default.yaml",
        help="Path to config YAML file",
    )
    
    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for fine-tuning or resuming",
    )
    
    # Override some common settings
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    
    # Quick test mode
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test with minimal steps",
    )
    
    # Dataset scale selection
    parser.add_argument(
        "--dataset_scale",
        type=str,
        default="small",
        choices=["small", "large"],
        help="Dataset scale: 'small' for WikiText-2 (~13MB), 'large' for OpenWebText (~10GB)",
    )
    
    return parser.parse_args()


def apply_overrides(config: DeepSeekV3Config, args):
    """Apply command line overrides to config."""
    if args.batch_size is not None:
        if args.mode == "pretrain":
            config.pretraining.batch_size = args.batch_size
        elif args.mode == "sft":
            config.sft.batch_size = args.batch_size
        elif args.mode == "rl":
            config.rl.batch_size = args.batch_size
    
    if args.learning_rate is not None:
        if args.mode == "pretrain":
            config.pretraining.learning_rate = args.learning_rate
        elif args.mode == "sft":
            config.sft.learning_rate = args.learning_rate
        elif args.mode == "rl":
            config.rl.learning_rate = args.learning_rate
    
    if args.max_steps is not None:
        if args.mode == "pretrain":
            config.pretraining.max_steps = args.max_steps
        elif args.mode == "sft":
            config.sft.max_steps = args.max_steps
        elif args.mode == "rl":
            config.rl.max_steps = args.max_steps
    
    if args.device is not None:
        config.pretraining.device = args.device
        config.sft.device = args.device
        config.rl.device = args.device
    
    # Test mode overrides
    if args.test:
        config.pretraining.max_steps = 50
        config.pretraining.eval_steps = 20
        config.pretraining.save_steps = 25
        config.pretraining.logging_steps = 5
        config.sft.max_steps = 30
        config.sft.eval_steps = 15
        config.rl.max_steps = 20
        config.data.sft_max_samples = 100
        config.data.rl_max_samples = 50
    
    # Dataset scale selection for pretrain mode
    if hasattr(args, 'dataset_scale') and args.mode == "pretrain":
        if args.dataset_scale == "small":
            # WikiText-2: ~13MB, good for quick experiments
            config.data.pretrain_dataset_name = "wikitext"
            config.data.pretrain_dataset_config = "wikitext-2-raw-v1"
            config.data.pretrain_max_samples = None
            config.data.pretrain_streaming = False
            logger.info("[Dataset] Using SMALL dataset: WikiText-2 (~13MB)")
        elif args.dataset_scale == "large":
            # OpenWebText: ~40GB, using ~10GB subset
            config.data.pretrain_dataset_name = "openwebtext"
            config.data.pretrain_dataset_config = None
            config.data.pretrain_max_samples = 2000000  # ~10GB of text
            config.data.pretrain_streaming = False
            # Also increase training steps for larger dataset
            if config.pretraining.max_steps < 50000:
                config.pretraining.max_steps = 50000
            logger.info(f"[Dataset] Using LARGE dataset: OpenWebText (~10GB, {config.data.pretrain_max_samples:,} samples)")
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Set log level based on test mode
    if args.test:
        set_log_level(logging.DEBUG)
    
    logger.info("=" * 70)
    logger.info("DeepSeek V3 Training")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    if args.checkpoint:
        logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("=" * 70)
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        logger.warning(f"Config file not found: {config_path}")
        logger.warning("Using default configuration")
        config = DeepSeekV3Config()
    
    # Apply command line overrides
    config = apply_overrides(config, args)
    
    # Print config summary
    config.print_config()
    
    # Initialize tokenizer
    logger.info("-" * 70)
    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer(config.data)
    logger.info(f"Tokenizer: {config.data.tokenizer_name}")
    logger.info(f"Vocab size: {len(tokenizer)}")
    
    # Update model vocab size to match tokenizer
    config.model.vocab_size = len(tokenizer)
    
    # Create model
    logger.info("-" * 70)
    logger.info("Creating model...")
    model = DeepSeekV3Model(config.model)
    print_model_summary(model, config.model)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Checkpoint loaded successfully")
    
    # Create reference model for RL
    ref_model = None
    if args.mode == "rl":
        logger.info("Creating reference model for RL...")
        ref_model = DeepSeekV3Model(config.model)
        ref_model.load_state_dict(model.state_dict())
    
    # Create data loaders
    logger.info("-" * 70)
    logger.info("Loading datasets...")
    
    # Get batch size for current mode
    if args.mode == "pretrain":
        batch_size = config.pretraining.batch_size
    elif args.mode == "sft":
        batch_size = config.sft.batch_size
    else:
        batch_size = config.rl.batch_size
    
    train_loader, val_loader = create_dataloaders(
        config=config.data,
        tokenizer=tokenizer,
        mode=args.mode,
        batch_size=batch_size,
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create trainer
    logger.info("-" * 70)
    logger.info("Creating trainer...")
    trainer = create_trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        mode=args.mode,
        ref_model=ref_model,
    )
    
    # Train
    logger.info("-" * 70)
    metrics = trainer.train()
    
    # Report final metrics
    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return metrics


if __name__ == "__main__":
    main()
