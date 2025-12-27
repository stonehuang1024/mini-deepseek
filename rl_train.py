"""
DeepSeek V3 RL Training Module
==============================

Unified interface for RL training with multiple algorithms:
- DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- PPO (Proximal Policy Optimization)

Usage:
    python rl_train.py --algorithm dpo --config config.yaml
    python rl_train.py --algorithm grpo --max_steps 200
    python rl_train.py --algorithm ppo --batch_size 2
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, DeepSeekV3Config, RLConfig, get_device
from deepseek.model import DeepSeekV3Model
from deepseek.data import get_tokenizer, create_rl_dataloaders, DPODataset, GRPODataset, PPODataset
from deepseek.training import DPOTrainer, RuleBasedReward, CompositeReward, LengthReward
from deepseek.training.rl_trainer_algorithms import GRPOTrainer, PPOTrainer
from deepseek.utils import get_logger

# Initialize logger
logger = get_logger(__name__)


# =============================================================================
# Factory Functions
# =============================================================================

def create_rl_trainer(
    algorithm: str,
    model: DeepSeekV3Model,
    ref_model: Optional[DeepSeekV3Model],
    config: DeepSeekV3Config,
    train_loader,
    val_loader,
    tokenizer,
    **kwargs,
):
    """
    Create RL trainer for specified algorithm.
    
    Args:
        algorithm: "dpo", "grpo", or "ppo"
        model: Policy model
        ref_model: Reference model (None = copy of model)
        config: Full configuration
        train_loader: Training data
        val_loader: Validation data
        tokenizer: Tokenizer
        **kwargs: Additional algorithm-specific arguments
        
    Returns:
        Trainer instance
    """
    # Create reward function for GRPO/PPO
    reward_fn = CompositeReward([
        (RuleBasedReward(), 0.7),
        (LengthReward(target_length=50), 0.3),
    ])
    
    if algorithm == "dpo":
        return DPOTrainer(
            model=model,
            ref_model=ref_model,
            config=config.rl,
            vis_config=config.visualization,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            beta=kwargs.get('beta', 0.1),
            label_smoothing=kwargs.get('label_smoothing', 0.0),
        )
    
    elif algorithm == "grpo":
        return GRPOTrainer(
            model=model,
            ref_model=ref_model,
            config=config.rl,
            vis_config=config.visualization,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
        )
    
    elif algorithm == "ppo":
        return PPOTrainer(
            model=model,
            ref_model=ref_model,
            config=config.rl,
            vis_config=config.visualization,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            value_coef=kwargs.get('value_coef', 0.5),
            entropy_coef=kwargs.get('entropy_coef', 0.01),
        )
    
    else:
        raise ValueError(f"Unknown RL algorithm: {algorithm}")


def run_rl_training(
    algorithm: str = "grpo",
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Run RL training with specified algorithm.
    
    Args:
        algorithm: RL algorithm ("dpo", "grpo", "ppo")
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint to load
        max_steps: Override max training steps
        batch_size: Override batch size
        max_samples: Max samples from dataset
        **kwargs: Additional arguments
        
    Returns:
        Training results
    """
    logger.info(f"{'='*70}")
    logger.info(f"DeepSeek V3 RL Training - {algorithm.upper()}")
    logger.info(f"{'='*70}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Override settings if provided
    if max_steps is not None:
        config.rl.max_steps = max_steps
    if batch_size is not None:
        config.rl.batch_size = batch_size
    
    # Update algorithm
    config.rl.algorithm = algorithm
    config.rl.tensorboard_dir = f"runs/rl_{algorithm}"
    config.rl.checkpoint_dir = f"checkpoints/rl_{algorithm}"
    
    # Get device
    device = get_device(config.rl.device)
    logger.info(f"Device: {device}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer(config.data)
    
    # Create model
    logger.info("Creating model...")
    model = DeepSeekV3Model(config.model)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create data loaders
    logger.info("Loading RL dataset...")
    train_loader, val_loader = create_rl_dataloaders(
        config.data,
        tokenizer,
        algorithm=algorithm,
        batch_size=config.rl.batch_size,
        max_samples=max_samples,
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create trainer
    logger.info(f"Initializing {algorithm.upper()} trainer...")
    trainer = create_rl_trainer(
        algorithm=algorithm,
        model=model,
        ref_model=None,  # Will create copy
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        **kwargs,
    )
    
    # Run training
    logger.info("Starting training...")
    results = trainer.train()
    
    logger.info(f"{'='*70}")
    logger.info(f"Training complete!")
    logger.info(f"Results: {results}")
    logger.info(f"{'='*70}")
    
    return results


# =============================================================================
# Test Functions
# =============================================================================

def test_dpo(max_steps: int = 10, max_samples: int = 20):
    """Test DPO training."""
    logger.info("="*70)
    logger.info("Testing DPO Training")
    logger.info("="*70)
    
    return run_rl_training(
        algorithm="dpo",
        max_steps=max_steps,
        batch_size=2,
        max_samples=max_samples,
    )


def test_grpo(max_steps: int = 5, max_samples: int = 10):
    """Test GRPO training."""
    logger.info("="*70)
    logger.info("Testing GRPO Training")
    logger.info("="*70)
    
    config = load_config()
    config.rl.group_size = 2  # Small group for testing
    
    return run_rl_training(
        algorithm="grpo",
        max_steps=max_steps,
        batch_size=1,
        max_samples=max_samples,
    )


def test_ppo(max_steps: int = 5, max_samples: int = 10):
    """Test PPO training."""
    logger.info("="*70)
    logger.info("Testing PPO Training")
    logger.info("="*70)
    
    return run_rl_training(
        algorithm="ppo",
        max_steps=max_steps,
        batch_size=1,
        max_samples=max_samples,
    )


def test_all_rl():
    """Test all RL algorithms."""
    logger.info("="*70)
    logger.info("Testing All RL Algorithms")
    logger.info("="*70)
    
    results = {}
    
    # Test DPO
    try:
        results['dpo'] = test_dpo(max_steps=5, max_samples=10)
        logger.info("✓ DPO test passed")
    except Exception as e:
        logger.error(f"✗ DPO test failed: {e}")
        results['dpo'] = {'error': str(e)}
    
    # Test GRPO
    try:
        results['grpo'] = test_grpo(max_steps=3, max_samples=5)
        logger.info("✓ GRPO test passed")
    except Exception as e:
        logger.error(f"✗ GRPO test failed: {e}")
        results['grpo'] = {'error': str(e)}
    
    # Test PPO
    try:
        results['ppo'] = test_ppo(max_steps=3, max_samples=5)
        logger.info("✓ PPO test passed")
    except Exception as e:
        logger.error(f"✗ PPO test failed: {e}")
        results['ppo'] = {'error': str(e)}
    
    logger.info("="*70)
    logger.info("RL Testing Summary")
    logger.info("="*70)
    for algo, result in results.items():
        status = "PASS" if 'error' not in result else "FAIL"
        logger.info(f"  {algo.upper()}: {status}")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepSeek V3 RL Training")
    
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        choices=["dpo", "grpo", "ppo", "all"],
        default="grpo",
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to use"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test"
    )
    
    args = parser.parse_args()
    
    if args.test or args.algorithm == "all":
        test_all_rl()
    else:
        run_rl_training(
            algorithm=args.algorithm,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
