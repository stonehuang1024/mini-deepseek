"""
DeepSeek V3 RL Test Suite
=========================

Comprehensive tests for all RL algorithms:
- DPO: Direct Preference Optimization
- GRPO: Group Relative Policy Optimization
- PPO: Proximal Policy Optimization

Usage:
    python test_rl.py           # Run all tests
    python test_rl.py --dpo     # Test only DPO
    python test_rl.py --grpo    # Test only GRPO
    python test_rl.py --ppo     # Test only PPO
"""

import os
import sys
import argparse
import time
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import load_config, DeepSeekV3Config
from deepseek.model import DeepSeekV3Model
from deepseek.data import get_tokenizer
from deepseek.utils import get_logger

# Initialize logger
logger = get_logger(__name__)


def print_header(title: str):
    """Print formatted header."""
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def print_results(results: Dict[str, Any], algorithm: str):
    """Print test results."""
    logger.info(f"{algorithm} Results:")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


# =============================================================================
# Dataset Tests
# =============================================================================

def test_rl_datasets() -> bool:
    """Test all RL dataset classes."""
    print_header("Testing RL Datasets")
    
    from deepseek.data import (
        DPODataset, GRPODataset, PPODataset,
        create_rl_dataloaders
    )
    
    config = load_config()
    tokenizer = get_tokenizer(config.data)
    
    tests_passed = True
    
    # Test DPO Dataset
    logger.info("1. Testing DPODataset...")
    try:
        train_loader, val_loader = create_rl_dataloaders(
            config.data, tokenizer, algorithm="dpo",
            batch_size=2, max_samples=10
        )
        batch = next(iter(train_loader))
        assert 'chosen_input_ids' in batch
        assert 'rejected_input_ids' in batch
        logger.info(f"   ✓ DPO dataset: {len(train_loader.dataset)} samples")
        logger.info(f"   ✓ chosen_input_ids shape: {batch['chosen_input_ids'].shape}")
    except Exception as e:
        logger.error(f"   ✗ DPO dataset failed: {e}")
        tests_passed = False
    
    # Test GRPO Dataset
    logger.info("2. Testing GRPODataset...")
    try:
        train_loader, val_loader = create_rl_dataloaders(
            config.data, tokenizer, algorithm="grpo",
            batch_size=2, max_samples=10
        )
        batch = next(iter(train_loader))
        assert 'input_ids' in batch
        assert 'prompt_text' in batch
        logger.info(f"   ✓ GRPO dataset: {len(train_loader.dataset)} samples")
        logger.info(f"   ✓ input_ids shape: {batch['input_ids'].shape}")
    except Exception as e:
        logger.error(f"   ✗ GRPO dataset failed: {e}")
        tests_passed = False
    
    # Test PPO Dataset
    logger.info("3. Testing PPODataset...")
    try:
        train_loader, val_loader = create_rl_dataloaders(
            config.data, tokenizer, algorithm="ppo",
            batch_size=2, max_samples=10
        )
        batch = next(iter(train_loader))
        assert 'input_ids' in batch
        assert 'prompt_text' in batch
        logger.info(f"   ✓ PPO dataset: {len(train_loader.dataset)} samples")
        logger.info(f"   ✓ input_ids shape: {batch['input_ids'].shape}")
    except Exception as e:
        logger.error(f"   ✗ PPO dataset failed: {e}")
        tests_passed = False
    
    return tests_passed


# =============================================================================
# Reward Function Tests
# =============================================================================

def test_reward_functions() -> bool:
    """Test reward functions."""
    print_header("Testing Reward Functions")
    
    from deepseek.training import (
        RuleBasedReward, LengthReward, CompositeReward
    )
    
    tests_passed = True
    
    # Test RuleBasedReward
    logger.info("1. Testing RuleBasedReward...")
    try:
        reward_fn = RuleBasedReward()
        
        # Good response
        good = "This is a well-structured sentence. It has multiple parts and proper punctuation."
        r1 = reward_fn(good)
        
        # Poor response
        poor = "bad"
        r2 = reward_fn(poor)
        
        assert r1 > r2, "Good response should score higher"
        logger.info(f"   ✓ Good response reward: {r1:.4f}")
        logger.info(f"   ✓ Poor response reward: {r2:.4f}")
    except Exception as e:
        logger.error(f"   ✗ RuleBasedReward failed: {e}")
        tests_passed = False
    
    # Test LengthReward
    logger.info("2. Testing LengthReward...")
    try:
        reward_fn = LengthReward(target_length=10)
        
        # Exactly target length
        exact = " ".join(["word"] * 10)
        r1 = reward_fn(exact)
        
        # Much longer
        long = " ".join(["word"] * 100)
        r2 = reward_fn(long)
        
        assert r1 > r2, "Target length should score higher"
        logger.info(f"   ✓ Target length reward: {r1:.4f}")
        logger.info(f"   ✓ Long text reward: {r2:.4f}")
    except Exception as e:
        logger.error(f"   ✗ LengthReward failed: {e}")
        tests_passed = False
    
    # Test CompositeReward
    logger.info("3. Testing CompositeReward...")
    try:
        reward_fn = CompositeReward([
            (RuleBasedReward(), 0.7),
            (LengthReward(target_length=20), 0.3),
        ])
        r = reward_fn("This is a test response with moderate length.")
        logger.info(f"   ✓ Composite reward: {r:.4f}")
    except Exception as e:
        logger.error(f"   ✗ CompositeReward failed: {e}")
        tests_passed = False
    
    return tests_passed


# =============================================================================
# DPO Training Test
# =============================================================================

def test_dpo_training(max_steps: int = 10) -> Dict[str, Any]:
    """Test DPO training pipeline."""
    print_header("Testing DPO Training")
    
    from deepseek.data import create_rl_dataloaders
    from deepseek.training import DPOTrainer
    
    config = load_config()
    config.rl.max_steps = max_steps
    config.rl.batch_size = 2
    config.rl.gradient_accumulation_steps = 1
    config.rl.logging_steps = max(1, max_steps // 5)
    config.rl.save_steps = max_steps + 1  # Don't save
    config.rl.tensorboard_dir = 'runs/test_dpo'
    config.rl.checkpoint_dir = 'checkpoints/test_dpo'
    
    tokenizer = get_tokenizer(config.data)
    model = DeepSeekV3Model(config.model)
    
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader, val_loader = create_rl_dataloaders(
        config.data, tokenizer, algorithm="dpo",
        batch_size=config.rl.batch_size, max_samples=20
    )
    
    logger.info(f"  Training samples: {len(train_loader.dataset)}")
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        config=config.rl,
        vis_config=config.visualization,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        beta=0.1,
    )
    
    start_time = time.time()
    results = trainer.train()
    elapsed = time.time() - start_time
    
    results['elapsed_time'] = elapsed
    print_results(results, "DPO")
    
    return results


# =============================================================================
# GRPO Training Test
# =============================================================================

def test_grpo_training(max_steps: int = 5) -> Dict[str, Any]:
    """Test GRPO training pipeline."""
    print_header("Testing GRPO Training")
    
    from deepseek.data import create_rl_dataloaders
    from deepseek.training.rl_trainer_algorithms import GRPOTrainer
    from deepseek.training import RuleBasedReward
    
    config = load_config()
    config.rl.max_steps = max_steps
    config.rl.batch_size = 1
    config.rl.group_size = 2
    config.rl.gradient_accumulation_steps = 1
    config.rl.logging_steps = 1
    config.rl.save_steps = max_steps + 1
    config.rl.max_new_tokens = 32
    config.rl.tensorboard_dir = 'runs/test_grpo'
    config.rl.checkpoint_dir = 'checkpoints/test_grpo'
    
    tokenizer = get_tokenizer(config.data)
    model = DeepSeekV3Model(config.model)
    
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader, val_loader = create_rl_dataloaders(
        config.data, tokenizer, algorithm="grpo",
        batch_size=config.rl.batch_size, max_samples=10
    )
    
    logger.info(f"  Training samples: {len(train_loader.dataset)}")
    
    trainer = GRPOTrainer(
        model=model,
        ref_model=None,
        config=config.rl,
        vis_config=config.visualization,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        reward_fn=RuleBasedReward(),
    )
    
    start_time = time.time()
    results = trainer.train()
    elapsed = time.time() - start_time
    
    results['elapsed_time'] = elapsed
    print_results(results, "GRPO")
    
    return results


# =============================================================================
# PPO Training Test
# =============================================================================

def test_ppo_training(max_steps: int = 5) -> Dict[str, Any]:
    """Test PPO training pipeline."""
    print_header("Testing PPO Training")
    
    from deepseek.data import create_rl_dataloaders
    from deepseek.training.rl_trainer_algorithms import PPOTrainer
    from deepseek.training import RuleBasedReward
    
    config = load_config()
    config.rl.max_steps = max_steps
    config.rl.batch_size = 2
    config.rl.gradient_accumulation_steps = 1
    config.rl.logging_steps = 1
    config.rl.save_steps = max_steps + 1
    config.rl.max_new_tokens = 32
    config.rl.tensorboard_dir = 'runs/test_ppo'
    config.rl.checkpoint_dir = 'checkpoints/test_ppo'
    
    tokenizer = get_tokenizer(config.data)
    model = DeepSeekV3Model(config.model)
    
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader, val_loader = create_rl_dataloaders(
        config.data, tokenizer, algorithm="ppo",
        batch_size=config.rl.batch_size, max_samples=10
    )
    
    logger.info(f"  Training samples: {len(train_loader.dataset)}")
    
    trainer = PPOTrainer(
        model=model,
        ref_model=None,
        config=config.rl,
        vis_config=config.visualization,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        reward_fn=RuleBasedReward(),
        ppo_epochs=2,
    )
    
    start_time = time.time()
    results = trainer.train()
    elapsed = time.time() - start_time
    
    results['elapsed_time'] = elapsed
    print_results(results, "PPO")
    
    return results


# =============================================================================
# Full Integration Test
# =============================================================================

def test_full_rl_pipeline() -> Dict[str, Any]:
    """Test complete RL pipeline: Pretrain -> SFT -> RL."""
    print_header("Testing Full RL Pipeline")
    
    from deepseek.data import create_rl_dataloaders, create_dataloaders
    from deepseek.training import DPOTrainer, PretrainTrainer, SFTTrainer
    
    results = {}
    
    # 1. Quick pretrain
    logger.info("1. Quick Pretrain Phase...")
    config = load_config()
    config.pretraining.max_steps = 10
    config.pretraining.batch_size = 4
    config.pretraining.gradient_accumulation_steps = 1
    config.pretraining.logging_steps = 5
    config.pretraining.save_steps = 100
    config.pretraining.tensorboard_dir = 'runs/test_pipeline_pretrain'
    config.pretraining.checkpoint_dir = 'checkpoints/test_pipeline_pretrain'
    
    tokenizer = get_tokenizer(config.data)
    model = DeepSeekV3Model(config.model)
    
    train_loader, val_loader = create_dataloaders(
        config.data, tokenizer, mode="pretrain",
        batch_size=config.pretraining.batch_size, max_samples=50
    )
    
    pretrain_trainer = PretrainTrainer(
        model=model, config=config.pretraining,
        vis_config=config.visualization,
        train_loader=train_loader, val_loader=val_loader,
        tokenizer=tokenizer
    )
    pretrain_results = pretrain_trainer.train()
    results['pretrain'] = pretrain_results
    logger.info(f"   ✓ Pretrain complete: loss={pretrain_results.get('avg_train_loss', 0):.4f}")
    
    # 2. Quick SFT
    logger.info("2. Quick SFT Phase...")
    config.sft.max_steps = 10
    config.sft.batch_size = 2
    config.sft.gradient_accumulation_steps = 1
    config.sft.logging_steps = 5
    config.sft.save_steps = 100
    config.sft.tensorboard_dir = 'runs/test_pipeline_sft'
    config.sft.checkpoint_dir = 'checkpoints/test_pipeline_sft'
    
    sft_train_loader, sft_val_loader = create_dataloaders(
        config.data, tokenizer, mode="sft",
        batch_size=config.sft.batch_size, max_samples=30
    )
    
    sft_trainer = SFTTrainer(
        model=model, config=config.sft,
        vis_config=config.visualization,
        train_loader=sft_train_loader, val_loader=sft_val_loader,
        tokenizer=tokenizer
    )
    sft_results = sft_trainer.train()
    results['sft'] = sft_results
    logger.info(f"   ✓ SFT complete: loss={sft_results.get('avg_train_loss', 0):.4f}")
    
    # 3. Quick DPO
    logger.info("3. Quick DPO Phase...")
    config.rl.max_steps = 5
    config.rl.batch_size = 2
    config.rl.gradient_accumulation_steps = 1
    config.rl.logging_steps = 2
    config.rl.save_steps = 100
    config.rl.tensorboard_dir = 'runs/test_pipeline_dpo'
    config.rl.checkpoint_dir = 'checkpoints/test_pipeline_dpo'
    
    dpo_train_loader, dpo_val_loader = create_rl_dataloaders(
        config.data, tokenizer, algorithm="dpo",
        batch_size=config.rl.batch_size, max_samples=20
    )
    
    dpo_trainer = DPOTrainer(
        model=model, ref_model=None, config=config.rl,
        vis_config=config.visualization,
        train_loader=dpo_train_loader, val_loader=dpo_val_loader,
        tokenizer=tokenizer
    )
    dpo_results = dpo_trainer.train()
    results['dpo'] = dpo_results
    logger.info(f"   ✓ DPO complete: accuracy={dpo_results.get('avg_accuracy', 0):.4f}")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepSeek V3 RL Test Suite")
    
    parser.add_argument("--dpo", action="store_true", help="Test DPO only")
    parser.add_argument("--grpo", action="store_true", help="Test GRPO only")
    parser.add_argument("--ppo", action="store_true", help="Test PPO only")
    parser.add_argument("--datasets", action="store_true", help="Test datasets only")
    parser.add_argument("--rewards", action="store_true", help="Test rewards only")
    parser.add_argument("--pipeline", action="store_true", help="Test full pipeline")
    parser.add_argument("--max_steps", type=int, default=10, help="Max training steps")
    
    args = parser.parse_args()
    
    # Determine what to test
    run_specific = args.dpo or args.grpo or args.ppo or args.datasets or args.rewards or args.pipeline
    
    results = {}
    all_passed = True
    
    print_header("DeepSeek V3 RL Test Suite")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')}")
    
    # Run tests
    if args.datasets or not run_specific:
        if not test_rl_datasets():
            all_passed = False
    
    if args.rewards or not run_specific:
        if not test_reward_functions():
            all_passed = False
    
    if args.dpo or not run_specific:
        try:
            results['dpo'] = test_dpo_training(args.max_steps)
        except Exception as e:
            logger.error(f"✗ DPO test failed: {e}")
            all_passed = False
    
    if args.grpo or not run_specific:
        try:
            results['grpo'] = test_grpo_training(max(3, args.max_steps // 2))
        except Exception as e:
            logger.error(f"✗ GRPO test failed: {e}")
            all_passed = False
    
    if args.ppo or not run_specific:
        try:
            results['ppo'] = test_ppo_training(max(3, args.max_steps // 2))
        except Exception as e:
            logger.error(f"✗ PPO test failed: {e}")
            all_passed = False
    
    if args.pipeline:
        try:
            results['pipeline'] = test_full_rl_pipeline()
        except Exception as e:
            logger.error(f"✗ Pipeline test failed: {e}")
            all_passed = False
    
    # Summary
    print_header("Test Summary")
    
    if all_passed:
        logger.info("✓ All tests passed!")
    else:
        logger.error("✗ Some tests failed!")
    
    for name, result in results.items():
        if isinstance(result, dict) and 'error' not in result:
            logger.info(f"{name.upper()}:")
            for k, v in result.items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.4f}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
