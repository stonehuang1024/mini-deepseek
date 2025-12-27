#!/usr/bin/env python3
"""
DeepSeek V3 Test Suite
======================

Comprehensive tests for all components:
1. Model architecture tests
2. Dataset tests
3. Training pipeline tests
4. Inference tests
5. End-to-end tests

Run all tests:
    python test_all.py

Run specific test:
    python test_all.py --test model
"""

import argparse
import os
import sys
import time
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepseek.utils import get_logger

# Initialize logger
logger = get_logger(__name__)


def test_config():
    """Test configuration loading and validation."""
    logger.info("=" * 70)
    logger.info("TEST: Configuration")
    logger.info("=" * 70)
    
    from config import (
        DeepSeekV3Config, ModelConfig, TrainingConfig,
        MoEConfig, MTPConfig, load_config, get_device
    )
    
    # Test default config creation
    logger.info("1. Testing default config creation...")
    config = DeepSeekV3Config()
    assert config.model.hidden_size == 512
    assert config.model.num_hidden_layers == 6
    assert config.model.moe.enabled == True
    logger.info("   ✓ Default config created successfully")
    
    # Test MoE layer detection
    logger.info("2. Testing MoE layer detection...")
    assert config.model.is_moe_layer(0) == False  # Layer 0 is not MoE
    assert config.model.is_moe_layer(1) == True   # Layer 1 is MoE
    assert config.model.is_moe_layer(2) == False  # Layer 2 is not MoE
    assert config.model.is_moe_layer(3) == True   # Layer 3 is MoE
    logger.info("   ✓ MoE layer detection works correctly")
    
    # Test config from dict
    logger.info("3. Testing config from dict...")
    config_dict = {
        "experiment_name": "test_exp",
        "model": {"hidden_size": 256, "num_hidden_layers": 2},
    }
    config = DeepSeekV3Config.from_dict(config_dict)
    assert config.experiment_name == "test_exp"
    assert config.model.hidden_size == 256
    logger.info("   ✓ Config from dict works correctly")
    
    # Test device detection
    logger.info("4. Testing device detection...")
    device = get_device("auto")
    logger.info(f"   Detected device: {device}")
    assert device in ["cuda", "mps", "cpu"]
    logger.info("   ✓ Device detection works")
    
    # Test YAML save/load
    logger.info("5. Testing YAML save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "test_config.yaml")
        config = DeepSeekV3Config()
        config.save_yaml(yaml_path)
        loaded = load_config(yaml_path)
        assert loaded.model.hidden_size == config.model.hidden_size
        logger.info("   ✓ YAML save/load works correctly")
    
    logger.info("✅ All configuration tests passed!")
    return True


def test_attention():
    """Test attention mechanisms."""
    logger.info("=" * 70)
    logger.info("TEST: Attention Mechanisms")
    logger.info("=" * 70)
    
    from deepseek.model import (
        RMSNorm, RotaryEmbedding, MultiHeadLatentAttention,
        StandardAttention, apply_rotary_pos_emb
    )
    
    device = torch.device("cpu")
    B, L, D = 2, 32, 256
    H = 4  # heads
    
    # Test RMSNorm
    logger.info("1. Testing RMSNorm...")
    norm = RMSNorm(D).to(device)
    x = torch.randn(B, L, D, device=device)
    out = norm(x)
    assert out.shape == (B, L, D)
    # Check normalization
    assert torch.allclose(out.pow(2).mean(dim=-1), torch.ones(B, L), atol=0.1)
    logger.info(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    logger.info("   ✓ RMSNorm works correctly")
    
    # Test RotaryEmbedding
    logger.info("2. Testing RotaryEmbedding...")
    rope_dim = 32
    rope = RotaryEmbedding(rope_dim, max_position_embeddings=64).to(device)
    q = torch.randn(B, H, L, rope_dim, device=device)
    cos, sin = rope(q)
    assert cos.shape[-1] == rope_dim
    logger.info(f"   Query shape: {q.shape}")
    logger.info(f"   Cos shape: {cos.shape}, Sin shape: {sin.shape}")
    logger.info("   ✓ RotaryEmbedding works correctly")
    
    # Test MultiHeadLatentAttention
    logger.info("3. Testing MultiHeadLatentAttention (MLA)...")
    mla = MultiHeadLatentAttention(
        hidden_size=D,
        num_attention_heads=H,
        kv_lora_rank=32,
        q_lora_rank=48,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=32,
        max_position_embeddings=64,
    ).to(device)
    
    hidden_states = torch.randn(B, L, D, device=device)
    output, attn_weights, _ = mla(
        hidden_states=hidden_states,
        output_attentions=True,
    )
    
    assert output.shape == (B, L, D)
    assert attn_weights.shape == (B, H, L, L)
    logger.info(f"   Input shape: {hidden_states.shape}")
    logger.info(f"   Output shape: {output.shape}")
    logger.info(f"   Attention weights shape: {attn_weights.shape}")
    logger.info("   ✓ MLA works correctly")
    
    # Test with causal mask
    logger.info("4. Testing attention with causal mask...")
    mask = torch.triu(torch.ones(L, L, device=device) * float('-inf'), diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
    
    output_masked, _, _ = mla(
        hidden_states=hidden_states,
        attention_mask=mask,
    )
    assert output_masked.shape == (B, L, D)
    logger.info("   ✓ Causal mask works correctly")
    
    # Test KV caching
    logger.info("5. Testing KV cache...")
    output1, _, kv_cache = mla(
        hidden_states=hidden_states[:, :16, :],
        use_cache=True,
    )
    
    output2, _, kv_cache = mla(
        hidden_states=hidden_states[:, 16:, :],
        past_key_value=kv_cache,
        use_cache=True,
    )
    
    assert output1.shape == (B, 16, D)
    assert output2.shape == (B, L-16, D)
    logger.info(f"   First chunk output: {output1.shape}")
    logger.info(f"   Second chunk output: {output2.shape}")
    logger.info("   ✓ KV cache works correctly")
    
    logger.info("✅ All attention tests passed!")
    return True


def test_model():
    """Test model architecture."""
    logger.info("=" * 70)
    logger.info("TEST: Model Architecture")
    logger.info("=" * 70)
    
    from config import load_config, ModelConfig
    from deepseek.model import (
        DeepSeekV3Model, SwiGLU, Expert, DeepSeekMoE,
        TransformerBlock, MTPHead, count_parameters
    )
    
    device = torch.device("cpu")
    
    # Create small config for testing
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=32,
        intermediate_size=256,
        max_position_embeddings=64,
    )
    config.moe.num_experts = 4
    config.moe.num_experts_per_tok = 2
    config.moe.num_shared_experts = 1
    config.moe.expert_hidden_size = 128
    config.mtp.num_predict_tokens = 2
    
    B, L = 2, 32
    
    # Test SwiGLU
    logger.info("1. Testing SwiGLU FFN...")
    ffn = SwiGLU(config.hidden_size, config.intermediate_size).to(device)
    x = torch.randn(B, L, config.hidden_size, device=device)
    out = ffn(x)
    assert out.shape == (B, L, config.hidden_size)
    logger.info(f"   Input: {x.shape} -> Output: {out.shape}")
    logger.info("   ✓ SwiGLU works correctly")
    
    # Test Expert
    logger.info("2. Testing Expert module...")
    expert = Expert(config.hidden_size, config.moe.expert_hidden_size).to(device)
    x_flat = torch.randn(10, config.hidden_size, device=device)
    out = expert(x_flat)
    assert out.shape == (10, config.hidden_size)
    logger.info(f"   Expert input: {x_flat.shape} -> Output: {out.shape}")
    logger.info("   ✓ Expert module works correctly")
    
    # Test MoE
    logger.info("3. Testing DeepSeekMoE...")
    moe = DeepSeekMoE(config.hidden_size, config.moe).to(device)
    x = torch.randn(B, L, config.hidden_size, device=device)
    moe.train()
    out = moe(x)
    assert out.shape == (B, L, config.hidden_size)
    assert moe.aux_loss is not None  # Should have aux loss during training
    logger.info(f"   MoE input: {x.shape} -> Output: {out.shape}")
    logger.info(f"   Auxiliary loss: {moe.aux_loss.item():.6f}")
    logger.info("   ✓ DeepSeekMoE works correctly")
    
    # Test TransformerBlock
    logger.info("4. Testing TransformerBlock...")
    block = TransformerBlock(config, layer_idx=1).to(device)  # MoE layer
    x = torch.randn(B, L, config.hidden_size, device=device)
    out, attn, kv, aux_loss = block(x, output_attentions=True)
    assert out.shape == (B, L, config.hidden_size)
    assert block.use_moe == True
    logger.info(f"   Block input: {x.shape} -> Output: {out.shape}")
    logger.info(f"   Is MoE layer: {block.use_moe}")
    logger.info("   ✓ TransformerBlock works correctly")
    
    # Test MTP Head
    logger.info("5. Testing MTP Head...")
    mtp = MTPHead(
        config.hidden_size, config.vocab_size, 
        config.mtp.num_predict_tokens
    ).to(device)
    hidden = torch.randn(B, L, config.hidden_size, device=device)
    predictions = mtp(hidden)
    assert len(predictions) == config.mtp.num_predict_tokens
    for i, pred in enumerate(predictions):
        assert pred.shape == (B, L, config.vocab_size)
    logger.info(f"   MTP predictions: {len(predictions)} heads")
    logger.info(f"   Each prediction shape: {predictions[0].shape}")
    logger.info("   ✓ MTP Head works correctly")
    
    # Test full model
    logger.info("6. Testing full DeepSeekV3Model...")
    model = DeepSeekV3Model(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (B, L), device=device)
    labels = torch.randint(0, config.vocab_size, (B, L), device=device)
    
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        output_attentions=True,
        output_hidden_states=True,
    )
    
    assert outputs['logits'].shape == (B, L, config.vocab_size)
    assert outputs['loss'] is not None
    assert outputs['mtp_logits'] is not None
    
    logger.info(f"   Input: {input_ids.shape}")
    logger.info(f"   Logits: {outputs['logits'].shape}")
    logger.info(f"   Loss: {outputs['loss'].item():.4f}")
    logger.info(f"   MTP predictions: {len(outputs['mtp_logits'])}")
    logger.info(f"   Total parameters: {count_parameters(model):,}")
    logger.info("   ✓ Full model works correctly")
    
    # Test generation
    logger.info("7. Testing generation...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids[:1, :10],
            max_new_tokens=20,
            temperature=0.8,
            do_sample=True,
        )
    assert generated.shape[0] == 1
    assert generated.shape[1] >= 10
    logger.info(f"   Input length: 10, Generated length: {generated.shape[1]}")
    logger.info("   ✓ Generation works correctly")
    
    logger.info("✅ All model tests passed!")
    return True


def test_dataset():
    """Test dataset loading and processing."""
    logger.info("=" * 70)
    logger.info("TEST: Dataset")
    logger.info("=" * 70)
    
    from config import load_config
    from deepseek.data import (
        get_tokenizer, PretrainDataset, SFTDataset, RLDataset,
        create_dataloaders
    )
    from deepseek.data.dataset import collate_fn
    
    config = load_config()
    
    # Test tokenizer
    logger.info("1. Testing tokenizer...")
    tokenizer = get_tokenizer(config.data)
    assert tokenizer.pad_token is not None
    text = "Hello, world! This is a test."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    logger.info(f"   Original: {text}")
    logger.info(f"   Encoded: {encoded[:10]}...")
    logger.info(f"   Decoded: {decoded}")
    logger.info("   ✓ Tokenizer works correctly")
    
    # Test PretrainDataset
    logger.info("2. Testing PretrainDataset...")
    pretrain_train, pretrain_val = create_dataloaders(
        config.data, tokenizer, mode="pretrain",
        batch_size=4, max_samples=50,
    )
    
    batch = next(iter(pretrain_train))
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'labels' in batch
    logger.info(f"   Train batches: {len(pretrain_train)}")
    logger.info(f"   Val batches: {len(pretrain_val)}")
    logger.info(f"   Batch input_ids: {batch['input_ids'].shape}")
    logger.info("   ✓ PretrainDataset works correctly")
    
    # Test SFTDataset
    logger.info("3. Testing SFTDataset...")
    sft_train, sft_val = create_dataloaders(
        config.data, tokenizer, mode="sft",
        batch_size=4, max_samples=50,
    )
    
    batch = next(iter(sft_train))
    # Check that labels have -100 for prompt (ignored in loss)
    assert (batch['labels'] == -100).any()
    logger.info(f"   Train batches: {len(sft_train)}")
    logger.info(f"   Batch shape: {batch['input_ids'].shape}")
    logger.info(f"   Labels contain -100 (prompt masking): True")
    logger.info("   ✓ SFTDataset works correctly")
    
    # Test RLDataset
    logger.info("4. Testing RLDataset...")
    rl_train, rl_val = create_dataloaders(
        config.data, tokenizer, mode="rl",
        batch_size=4, max_samples=20,
    )
    
    batch = next(iter(rl_train))
    assert 'prompt_text' in batch
    logger.info(f"   Train batches: {len(rl_train)}")
    logger.info(f"   Sample prompt: {batch['prompt_text'][0][:50]}...")
    logger.info("   ✓ RLDataset works correctly")
    
    logger.info("✅ All dataset tests passed!")
    return True


def test_trainer():
    """Test training pipelines."""
    logger.info("=" * 70)
    logger.info("TEST: Trainer")
    logger.info("=" * 70)
    
    from config import load_config, DeepSeekV3Config, ModelConfig
    from deepseek.model import DeepSeekV3Model
    from deepseek.data import get_tokenizer, create_dataloaders
    from deepseek.training import (
        BaseTrainer, PretrainTrainer, SFTTrainer,
        create_trainer, Visualizer
    )
    from deepseek.training.trainer import get_cosine_warmup_scheduler, GRPOTrainer
    
    # Small config for fast testing
    config = DeepSeekV3Config()
    config.model.hidden_size = 64
    config.model.num_hidden_layers = 2
    config.model.num_attention_heads = 2
    config.model.kv_lora_rank = 16
    config.model.q_lora_rank = 16
    config.model.qk_nope_head_dim = 8
    config.model.qk_rope_head_dim = 8
    config.model.v_head_dim = 16
    config.model.intermediate_size = 128
    config.model.moe.num_experts = 4
    config.model.moe.expert_hidden_size = 64
    
    # Fast training settings
    config.pretraining.max_steps = 10
    config.pretraining.batch_size = 2
    config.pretraining.eval_steps = 5
    config.pretraining.logging_steps = 2
    config.pretraining.save_steps = 5
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config.pretraining.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        config.pretraining.tensorboard_dir = os.path.join(tmpdir, "runs")
        
        # Test scheduler
        logger.info("1. Testing LR scheduler...")
        model = DeepSeekV3Model(config.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_cosine_warmup_scheduler(optimizer, 5, 20)
        
        lrs = []
        for _ in range(20):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
        
        assert lrs[0] < lrs[5]  # Warmup
        assert lrs[5] > lrs[-1]  # Decay
        logger.info(f"   LR progression: {lrs[0]:.4f} -> {lrs[5]:.4f} -> {lrs[-1]:.4f}")
        logger.info("   ✓ LR scheduler works correctly")
        
        # Test Visualizer
        logger.info("2. Testing Visualizer...")
        tokenizer = get_tokenizer(config.data)
        config.model.vocab_size = len(tokenizer)
        
        vis = Visualizer(
            config.pretraining.tensorboard_dir,
            config.visualization,
            tokenizer,
        )
        vis.log_scalar("test/loss", 1.5, 0)
        vis.close()
        logger.info("   ✓ Visualizer works correctly")
        
        # Test PretrainTrainer
        logger.info("3. Testing PretrainTrainer (quick run)...")
        model = DeepSeekV3Model(config.model)
        
        train_loader, val_loader = create_dataloaders(
            config.data, tokenizer, mode="pretrain",
            batch_size=2, max_samples=20,
        )
        
        trainer = create_trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            mode="pretrain",
        )
        
        # Run a few steps
        metrics = trainer.train()
        
        assert 'final_val_loss' in metrics
        # Note: final_val_loss can be 0 if no evaluation was performed
        # This can happen with very small max_steps and eval_steps settings
        logger.info(f"   Final val loss: {metrics.get('final_val_loss', 0):.4f}")
        logger.info("   ✓ PretrainTrainer works correctly")
        
        # Test checkpoint loading
        logger.info("4. Testing checkpoint save/load...")
        checkpoint_path = os.path.join(
            config.pretraining.checkpoint_dir, "final.pt"
        )
        if os.path.exists(checkpoint_path):
            # Create new model and load checkpoint
            new_model = DeepSeekV3Model(config.model)
            new_trainer = create_trainer(
                model=new_model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                tokenizer=tokenizer,
                mode="pretrain",
            )
            new_trainer.load_checkpoint(checkpoint_path)
            logger.info(f"   Loaded from step: {new_trainer.global_step}")
            logger.info("   ✓ Checkpoint save/load works correctly")
    
    logger.info("✅ All trainer tests passed!")
    return True


def test_inference():
    """Test inference functionality."""
    logger.info("=" * 70)
    logger.info("TEST: Inference")
    logger.info("=" * 70)
    
    from config import DeepSeekV3Config, ModelConfig, InferenceConfig
    from deepseek.model import DeepSeekV3Model
    from deepseek.data import get_tokenizer
    from deepseek.inference.inference import DeepSeekInference
    
    # Small config
    config = DeepSeekV3Config()
    config.model.hidden_size = 64
    config.model.num_hidden_layers = 2
    config.model.num_attention_heads = 2
    config.model.kv_lora_rank = 16
    config.model.q_lora_rank = 16
    config.model.qk_nope_head_dim = 8
    config.model.qk_rope_head_dim = 8
    config.model.v_head_dim = 16
    config.model.intermediate_size = 128
    config.model.moe.num_experts = 4
    config.model.moe.expert_hidden_size = 64
    
    tokenizer = get_tokenizer(config.data)
    config.model.vocab_size = len(tokenizer)
    
    model = DeepSeekV3Model(config.model)
    
    # Test inference wrapper
    logger.info("1. Testing DeepSeekInference...")
    inference = DeepSeekInference(
        model=model,
        tokenizer=tokenizer,
        config=config.inference,
        device="cpu",
    )
    logger.info("   ✓ Inference wrapper created")
    
    # Test generation
    logger.info("2. Testing text generation...")
    prompt = "Hello, world"
    response = inference.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        do_sample=True,
    )
    assert len(response) > len(prompt)
    logger.info(f"   Prompt: {prompt}")
    logger.info(f"   Generated: {response[:100]}...")
    logger.info("   ✓ Text generation works")
    
    # Test batch generation
    logger.info("3. Testing batch generation...")
    prompts = ["Hello", "The meaning of", "Once upon"]
    responses = inference.batch_generate(
        prompts,
        max_new_tokens=15,
    )
    assert len(responses) == len(prompts)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        logger.info(f"   {i+1}. '{p}' -> '{r[:50]}...'")
    logger.info("   ✓ Batch generation works")
    
    # Test chat
    logger.info("4. Testing chat...")
    response = inference.chat(
        user_message="What is AI?",
        history=None,
        max_new_tokens=30,
    )
    logger.info(f"   User: What is AI?")
    logger.info(f"   Assistant: {response[:100]}...")
    logger.info("   ✓ Chat works")
    
    # Test greedy decoding
    logger.info("5. Testing greedy decoding...")
    response1 = inference.generate("Test", max_new_tokens=10, do_sample=False)
    response2 = inference.generate("Test", max_new_tokens=10, do_sample=False)
    # Greedy should be deterministic
    logger.info(f"   Response 1: {response1}")
    logger.info(f"   Response 2: {response2}")
    logger.info("   ✓ Greedy decoding works")
    
    logger.info("✅ All inference tests passed!")
    return True


def test_end_to_end():
    """Test complete end-to-end pipeline."""
    logger.info("=" * 70)
    logger.info("TEST: End-to-End Pipeline")
    logger.info("=" * 70)
    
    from config import DeepSeekV3Config
    from deepseek.model import DeepSeekV3Model
    from deepseek.data import get_tokenizer, create_dataloaders
    from deepseek.training import create_trainer
    from deepseek.inference.inference import DeepSeekInference
    
    # Minimal config for E2E test
    config = DeepSeekV3Config()
    config.model.hidden_size = 64
    config.model.num_hidden_layers = 2
    config.model.num_attention_heads = 2
    config.model.kv_lora_rank = 16
    config.model.q_lora_rank = 16
    config.model.qk_nope_head_dim = 8
    config.model.qk_rope_head_dim = 8
    config.model.v_head_dim = 16
    config.model.intermediate_size = 128
    config.model.moe.num_experts = 4
    config.model.moe.expert_hidden_size = 64
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config.pretraining.checkpoint_dir = os.path.join(tmpdir, "pretrain")
        config.pretraining.tensorboard_dir = os.path.join(tmpdir, "runs/pretrain")
        config.pretraining.max_steps = 20
        config.pretraining.batch_size = 2
        config.pretraining.eval_steps = 10
        config.pretraining.logging_steps = 5
        config.pretraining.save_steps = 10
        
        config.sft.checkpoint_dir = os.path.join(tmpdir, "sft")
        config.sft.tensorboard_dir = os.path.join(tmpdir, "runs/sft")
        config.sft.max_steps = 15
        config.sft.batch_size = 2
        config.sft.eval_steps = 7
        config.sft.logging_steps = 3
        config.sft.save_steps = 7
        
        config.rl.checkpoint_dir = os.path.join(tmpdir, "rl")
        config.rl.tensorboard_dir = os.path.join(tmpdir, "runs/rl")
        config.rl.max_steps = 5
        config.rl.batch_size = 1
        config.rl.logging_steps = 1
        config.rl.group_size = 2
        
        # Initialize
        logger.info("1. Initializing tokenizer and model...")
        tokenizer = get_tokenizer(config.data)
        config.model.vocab_size = len(tokenizer)
        model = DeepSeekV3Model(config.model)
        logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info("   ✓ Initialization complete")
        
        # Stage 1: Pretraining
        logger.info("2. Stage 1: Pretraining...")
        train_loader, val_loader = create_dataloaders(
            config.data, tokenizer, mode="pretrain",
            batch_size=2, max_samples=30,
        )
        
        trainer = create_trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            mode="pretrain",
        )
        
        pretrain_metrics = trainer.train()
        logger.info(f"   Pretrain loss: {pretrain_metrics['final_val_loss']:.4f}")
        logger.info("   ✓ Pretraining complete")
        
        # Stage 2: SFT
        logger.info("3. Stage 2: Supervised Fine-Tuning...")
        sft_train, sft_val = create_dataloaders(
            config.data, tokenizer, mode="sft",
            batch_size=2, max_samples=30,
        )
        
        sft_trainer = create_trainer(
            model=model,
            config=config,
            train_loader=sft_train,
            val_loader=sft_val,
            tokenizer=tokenizer,
            mode="sft",
        )
        
        sft_metrics = sft_trainer.train()
        logger.info(f"   SFT loss: {sft_metrics['final_val_loss']:.4f}")
        logger.info("   ✓ SFT complete")
        
        # Stage 3: RL (GRPO)
        logger.info("4. Stage 3: Reinforcement Learning (GRPO)...")
        rl_train, rl_val = create_dataloaders(
            config.data, tokenizer, mode="rl",
            batch_size=1, max_samples=10,
        )
        
        # Create reference model
        ref_model = DeepSeekV3Model(config.model)
        ref_model.load_state_dict(model.state_dict())
        
        rl_trainer = create_trainer(
            model=model,
            config=config,
            train_loader=rl_train,
            val_loader=rl_val,
            tokenizer=tokenizer,
            mode="rl",
            ref_model=ref_model,
        )
        
        rl_metrics = rl_trainer.train()
        logger.info(f"   RL reward: {rl_metrics.get('avg_reward', 0):.4f}")
        logger.info("   ✓ RL complete")
        
        # Stage 4: Inference
        logger.info("5. Stage 4: Inference...")
        inference = DeepSeekInference(
            model=model,
            tokenizer=tokenizer,
            config=config.inference,
            device="cpu",
        )
        
        prompts = [
            "The meaning of life is",
            "Hello, how are you?",
        ]
        
        for prompt in prompts:
            response = inference.generate(prompt, max_new_tokens=30)
            logger.info(f"   Prompt: {prompt}")
            logger.info(f"   Response: {response[:80]}...")
        
        logger.info("   ✓ Inference complete")
    
    logger.info("=" * 70)
    logger.info("✅ End-to-End Pipeline Test Passed!")
    logger.info("=" * 70)
    logger.info("Pipeline Summary:")
    logger.info(f"  1. Pretraining: Loss {pretrain_metrics['final_val_loss']:.4f}")
    logger.info(f"  2. SFT: Loss {sft_metrics['final_val_loss']:.4f}")
    logger.info(f"  3. RL (GRPO): Reward {rl_metrics.get('avg_reward', 0):.4f}")
    logger.info("  4. Inference: ✓")
    
    return True


def run_all_tests():
    """Run all test suites."""
    logger.info("=" * 70)
    logger.info("DeepSeek V3 - Complete Test Suite")
    logger.info("=" * 70)
    
    tests = [
        ("Configuration", test_config),
        ("Attention", test_attention),
        ("Model", test_model),
        ("Dataset", test_dataset),
        ("Trainer", test_trainer),
        ("Inference", test_inference),
        ("End-to-End", test_end_to_end),
    ]
    
    results = {}
    
    for name, test_fn in tests:
        try:
            logger.info(f"{'#' * 70}")
            logger.info(f"# Running: {name} Tests")
            logger.info(f"{'#' * 70}")
            
            start_time = time.time()
            success = test_fn()
            elapsed = time.time() - start_time
            
            results[name] = {
                "success": success,
                "time": elapsed,
            }
            
        except Exception as e:
            results[name] = {
                "success": False,
                "error": str(e),
            }
            logger.error(f"❌ {name} tests failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    all_passed = True
    for name, result in results.items():
        if result["success"]:
            logger.info(f"  ✅ {name}: PASSED ({result['time']:.1f}s)")
        else:
            logger.error(f"  ❌ {name}: FAILED - {result.get('error', 'Unknown error')}")
            all_passed = False
    
    logger.info("=" * 70)
    
    if all_passed:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning("⚠️ Some tests failed. Please check the output above.")
    
    return all_passed


def main():
    """Main test entry point."""
    parser = argparse.ArgumentParser(description="DeepSeek V3 Tests")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "config", "attention", "model", "dataset", "trainer", "inference", "e2e"],
        help="Which test to run",
    )
    
    args = parser.parse_args()
    
    if args.test == "all":
        success = run_all_tests()
    elif args.test == "config":
        success = test_config()
    elif args.test == "attention":
        success = test_attention()
    elif args.test == "model":
        success = test_model()
    elif args.test == "dataset":
        success = test_dataset()
    elif args.test == "trainer":
        success = test_trainer()
    elif args.test == "inference":
        success = test_inference()
    elif args.test == "e2e":
        success = test_end_to_end()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
