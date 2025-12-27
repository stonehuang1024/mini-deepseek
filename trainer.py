"""
DeepSeek V3 Trainer Module
==========================

Training infrastructure for:
1. Pretraining: Next-token prediction on raw text
2. SFT: Supervised fine-tuning on instruction data
3. RL (GRPO): Group Relative Policy Optimization

Features:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- TensorBoard visualization
"""

import os
import time
import math
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter

from config import (
    DeepSeekV3Config, TrainingConfig, SFTConfig, RLConfig,
    VisualizationConfig, get_device,
)
from model import DeepSeekV3Model
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

def get_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Linear warmup then linear decay scheduler.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Steps for warmup
        num_training_steps: Total training steps
        
    Returns:
        LR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


def get_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Linear warmup then cosine decay scheduler.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Steps for warmup
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as fraction of initial
        
    Returns:
        LR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Visualizer
# =============================================================================

class Visualizer:
    """
    TensorBoard visualization for training.
    
    Visualizes:
    - Loss curves
    - Attention patterns
    - MoE routing statistics
    - Gradient distributions
    - Generated text samples
    """
    
    def __init__(
        self,
        log_dir: str,
        config: VisualizationConfig,
        tokenizer: Any = None,
    ):
        """
        Args:
            log_dir: TensorBoard log directory
            config: Visualization configuration
            tokenizer: Tokenizer for decoding
        """
        self.writer = SummaryWriter(log_dir)
        self.config = config
        self.tokenizer = tokenizer
        
        logger.info(f"TensorBoard logging to: {log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of values."""
        self.writer.add_histogram(tag, values.cpu().detach().numpy(), step)
    
    def log_attention(
        self,
        model: DeepSeekV3Model,
        step: int,
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Log attention patterns.
        
        Visualizes attention weights as heatmaps.
        """
        if not self.config.visualize_attention:
            return
        
        if layer_indices is None:
            layer_indices = list(range(min(
                self.config.num_attention_layers_to_show,
                len(model.layers)
            )))
        
        for layer_idx in layer_indices:
            layer = model.layers[layer_idx]
            if hasattr(layer.attention, 'last_attention_weights'):
                attn_weights = layer.attention.last_attention_weights
                if attn_weights is not None:
                    # Shape: (B, H, L, L) -> take first batch, subset of heads
                    attn = attn_weights[0]  # (H, L, L)
                    num_heads = min(self.config.num_attention_heads_to_show, attn.shape[0])
                    
                    for head_idx in range(num_heads):
                        # Create heatmap image
                        attn_map = attn[head_idx].cpu()  # (L, L)
                        # Normalize for visualization
                        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                        
                        self.writer.add_image(
                            f"attention/layer{layer_idx}_head{head_idx}",
                            attn_map.unsqueeze(0),  # (1, L, L)
                            step,
                        )
    
    def log_moe(self, model: DeepSeekV3Model, step: int):
        """
        Log MoE routing statistics.
        
        Visualizes expert utilization and load balancing.
        """
        if not self.config.visualize_moe:
            return
        
        for layer_idx, layer in enumerate(model.layers):
            if hasattr(layer, 'use_moe') and layer.use_moe:
                ffn = layer.ffn
                if hasattr(ffn, 'gate') and hasattr(ffn.gate, 'expert_usage'):
                    expert_usage = ffn.gate.expert_usage
                    if expert_usage is not None:
                        # Log expert usage distribution
                        self.writer.add_histogram(
                            f"moe/layer{layer_idx}_expert_usage",
                            expert_usage.cpu(),
                            step,
                        )
                        
                        # Log load balance metrics
                        if ffn.last_router_probs is not None:
                            router_probs = ffn.last_router_probs[0]  # (L, N)
                            # Entropy of routing (higher = more balanced)
                            entropy = -(router_probs * (router_probs + 1e-10).log()).sum(dim=-1).mean()
                            self.writer.add_scalar(
                                f"moe/layer{layer_idx}_routing_entropy",
                                entropy.item(),
                                step,
                            )
    
    def log_gradients(self, model: DeepSeekV3Model, step: int):
        """Log gradient statistics."""
        if not self.config.visualize_gradients:
            return
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                self.writer.add_scalar(f"gradients/{name}_norm", grad.norm().item(), step)
                if step % (self.config.gradient_log_steps * 5) == 0:
                    self.writer.add_histogram(f"gradients/{name}", grad.cpu(), step)
    
    def log_weights(self, model: DeepSeekV3Model, step: int):
        """Log weight distributions."""
        if not self.config.visualize_weights:
            return
        
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"weights/{name}", param.cpu().detach(), step)
    
    def log_generation(
        self,
        model: DeepSeekV3Model,
        step: int,
        device: torch.device,
    ):
        """Log generated text samples."""
        if not self.config.visualize_generation or self.tokenizer is None:
            return
        
        model.eval()
        
        for prompt in self.config.generation_prompts[:3]:  # Limit prompts
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=self.config.generation_max_length,
                    temperature=self.config.generation_temperature,
                    do_sample=True,
                )
            
            generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            
            self.writer.add_text(
                f"generation/{prompt[:30]}...",
                generated_text,
                step,
            )
        
        model.train()
    
    def log_mask(self, mask: torch.Tensor, name: str, step: int):
        """Log attention mask pattern."""
        if not self.config.visualize_masks:
            return
        
        # Normalize mask for visualization
        if mask.dim() == 4:  # (B, 1, L, L)
            mask_vis = mask[0, 0].float().cpu()
        elif mask.dim() == 2:  # (L, L)
            mask_vis = mask.float().cpu()
        else:
            return
        
        # Convert -inf to 0 for visualization
        mask_vis = torch.where(
            mask_vis == float('-inf'),
            torch.zeros_like(mask_vis),
            torch.ones_like(mask_vis),
        )
        
        self.writer.add_image(
            f"masks/{name}",
            mask_vis.unsqueeze(0),
            step,
        )
    
    def close(self):
        """Close the writer."""
        self.writer.close()


# =============================================================================
# Base Trainer
# =============================================================================

class BaseTrainer:
    """
    Base trainer class with common functionality.
    
    Handles:
    - Optimizer and scheduler setup
    - Training loop
    - Evaluation
    - Checkpointing
    - Visualization
    """
    
    def __init__(
        self,
        model: DeepSeekV3Model,
        config: TrainingConfig,
        vis_config: VisualizationConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: Any = None,
    ):
        """
        Args:
            model: The model to train
            config: Training configuration
            vis_config: Visualization configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            tokenizer: Tokenizer for text decoding
        """
        self.model = model
        self.config = config
        self.vis_config = vis_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        
        # Setup device
        self.device = torch.device(get_device(config.device))
        self.model.to(self.device)
        
        logger.info(f"Training on device: {self.device}")
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Calculate total steps
        steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
        self.total_steps = min(
            config.max_steps,
            steps_per_epoch * config.num_epochs
        )
        
        # Setup scheduler
        warmup_steps = config.warmup_steps
        if config.warmup_ratio > 0:
            warmup_steps = int(self.total_steps * config.warmup_ratio)
        
        self.scheduler = get_cosine_warmup_scheduler(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
            min_lr_ratio=config.min_learning_rate / config.learning_rate,
        )
        
        # Setup visualization
        os.makedirs(config.tensorboard_dir, exist_ok=True)
        self.visualizer = Visualizer(
            config.tensorboard_dir,
            vis_config,
            tokenizer,
        )
        
        # Setup checkpointing
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        return AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
    
    def train(self) -> Dict[str, float]:
        """
        Main training loop with comprehensive monitoring.
        
        Returns:
            Training metrics
        """
        logger.info(f"{'='*70}")
        logger.info("Starting training...")
        logger.info(f"Total steps: {self.total_steps}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"Logging steps: {self.config.logging_steps}")
        logger.info(f"Eval steps: {self.config.eval_steps}")
        logger.info(f"Save steps: {self.config.save_steps}")
        logger.info(f"{'='*70}")
        
        self.model.train()
        
        # Enhanced tracking metrics
        total_loss = 0.0
        accumulation_loss = 0.0
        num_batches = 0
        start_time = time.time()
        epoch_start_time = time.time()
        
        # Loss tracking for detailed monitoring
        self.loss_history = []  # Recent loss values for smoothing
        self.tokens_processed = 0
        self.samples_processed = 0
        self.step_times = []  # Track step times for speed estimation
        last_log_time = time.time()
        last_step_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_steps = 0
            
            logger.info(f"{'='*70}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*70}")
            
            for batch_idx, batch in enumerate(self.train_loader):
                batch_start_time = time.time()
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                loss, extra_metrics = self._training_step_with_metrics(batch)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                accumulation_loss += loss.item()
                num_batches += 1
                
                # Track tokens and samples
                batch_size = batch['input_ids'].shape[0]
                seq_len = batch['input_ids'].shape[1]
                self.tokens_processed += batch_size * seq_len
                self.samples_processed += batch_size
                
                # Optimizer step
                if num_batches % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    grad_norm = 0.0
                    if self.config.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        ).item()
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    total_loss += accumulation_loss
                    epoch_loss += accumulation_loss
                    epoch_steps += 1
                    
                    # Track loss history for smoothing
                    self.loss_history.append(accumulation_loss)
                    if len(self.loss_history) > 100:  # Keep last 100 losses
                        self.loss_history.pop(0)
                    
                    # Track step time
                    step_time = time.time() - last_step_time
                    self.step_times.append(step_time)
                    if len(self.step_times) > 50:  # Keep last 50 step times
                        self.step_times.pop(0)
                    last_step_time = time.time()
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        self._log_training_enhanced(
                            accumulation_loss, 
                            start_time, 
                            epoch, 
                            grad_norm,
                            extra_metrics
                        )
                    
                    accumulation_loss = 0.0
                    
                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        val_loss, val_metrics = self.evaluate_with_metrics()
                        self.model.train()
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint("best")
                            logger.info(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
                    
                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
                    
                    # Visualization
                    self._visualize()
                    
                    # Check if done
                    if self.global_step >= self.total_steps:
                        break
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(1, epoch_steps)
            logger.info(f"{'─'*70}")
            logger.info(f"Epoch {epoch + 1} Summary:")
            logger.info(f"  Loss: {avg_epoch_loss:.4f}")
            logger.info(f"  Steps: {epoch_steps}")
            logger.info(f"  Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
            logger.info(f"  Samples/sec: {self.samples_processed / epoch_time:.1f}")
            logger.info(f"{'─'*70}")
            
            if self.global_step >= self.total_steps:
                break
        
        # Final evaluation and save
        final_val_loss, final_metrics = self.evaluate_with_metrics()
        self.save_checkpoint("final")
        
        self.visualizer.close()
        
        total_time = time.time() - start_time
        avg_loss = total_loss / max(1, self.global_step)
        
        logger.info(f"{'='*70}")
        logger.info("Training Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Average training loss: {avg_loss:.4f}")
        logger.info(f"Total training time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
        logger.info(f"Total steps: {self.global_step}")
        logger.info(f"Total tokens processed: {self.tokens_processed:,}")
        logger.info(f"Average tokens/sec: {self.tokens_processed / total_time:.1f}")
        logger.info(f"{'='*70}")
        
        return {
            'final_val_loss': final_val_loss,
            'best_val_loss': self.best_val_loss,
            'avg_train_loss': avg_loss,
            'total_time': total_time,
            'tokens_processed': self.tokens_processed,
        }
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Single training step. Override in subclasses for custom behavior.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss tensor
        """
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch['labels'],
        )
        return outputs['loss']
    
    def _training_step_with_metrics(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step with detailed metrics.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss tensor and extra metrics dict
        """
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch['labels'],
        )
        
        extra_metrics = {}
        
        # Extract MTP loss if available
        if 'mtp_loss' in outputs and outputs['mtp_loss'] is not None:
            extra_metrics['mtp_loss'] = outputs['mtp_loss'].item()
        
        # Extract aux loss (MoE load balancing) if available
        if 'aux_loss' in outputs and outputs['aux_loss'] is not None:
            extra_metrics['aux_loss'] = outputs['aux_loss'].item()
        
        # Main language modeling loss
        if 'lm_loss' in outputs:
            extra_metrics['lm_loss'] = outputs['lm_loss'].item()
        
        return outputs['loss'], extra_metrics
    
    def _log_training_enhanced(
        self, 
        loss: float, 
        start_time: float, 
        epoch: int,
        grad_norm: float,
        extra_metrics: Dict[str, float],
    ):
        """Enhanced training logging with progress bar and detailed metrics."""
        elapsed = time.time() - start_time
        lr = self.scheduler.get_last_lr()[0]
        
        # Calculate progress
        progress = self.global_step / self.total_steps * 100
        
        # Calculate ETA
        if self.global_step > 0:
            avg_step_time = elapsed / self.global_step
            remaining_steps = self.total_steps - self.global_step
            eta_seconds = remaining_steps * avg_step_time
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "N/A"
        
        # Calculate smoothed loss (moving average)
        smoothed_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else loss
        
        # Calculate tokens per second
        tokens_per_sec = self.tokens_processed / elapsed if elapsed > 0 else 0
        
        # Calculate samples per second  
        samples_per_sec = self.samples_processed / elapsed if elapsed > 0 else 0
        
        # Calculate steps per second
        if self.step_times:
            recent_step_time = sum(self.step_times) / len(self.step_times)
            steps_per_sec = 1.0 / recent_step_time if recent_step_time > 0 else 0
        else:
            steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * self.global_step / self.total_steps)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Print detailed progress
        logger.info(f"┌{'─'*68}┐")
        logger.info(f"│ Step: {self.global_step:>6}/{self.total_steps} [{bar}] {progress:>5.1f}%       │")
        logger.info(f"├{'─'*68}┤")
        logger.info(f"│ Loss: {loss:>8.4f}  (smoothed: {smoothed_loss:>8.4f})                       │")
        logger.info(f"│ LR: {lr:>10.2e}  Grad norm: {grad_norm:>8.4f}                       │")
        logger.info(f"│ Epoch: {epoch + 1}                                                      │")
        logger.info(f"├{'─'*68}┤")
        logger.info(f"│ Speed: {tokens_per_sec:>8.0f} tok/s  {samples_per_sec:>6.1f} samples/s  {steps_per_sec:>5.2f} steps/s │")
        logger.info(f"│ Time: {self._format_time(elapsed):>10}  ETA: {eta_str:>10}                        │")
        logger.info(f"│ Tokens: {self.tokens_processed:>12,}                                     │")
        logger.info(f"└{'─'*68}┘")
        
        # Print extra metrics if available
        if extra_metrics:
            metrics_str = "  ".join([f"{k}: {v:.4f}" for k, v in extra_metrics.items()])
            logger.info(f"  Extra metrics: {metrics_str}")
        
        # Log to TensorBoard
        self.visualizer.log_scalar("train/loss", loss, self.global_step)
        self.visualizer.log_scalar("train/loss_smoothed", smoothed_loss, self.global_step)
        self.visualizer.log_scalar("train/lr", lr, self.global_step)
        self.visualizer.log_scalar("train/grad_norm", grad_norm, self.global_step)
        self.visualizer.log_scalar("train/tokens_per_sec", tokens_per_sec, self.global_step)
        self.visualizer.log_scalar("train/samples_per_sec", samples_per_sec, self.global_step)
        self.visualizer.log_scalar("train/progress_percent", progress, self.global_step)
        
        # Log extra metrics to TensorBoard
        for key, value in extra_metrics.items():
            self.visualizer.log_scalar(f"train/{key}", value, self.global_step)
        
        # Log perplexity (exp of loss)
        perplexity = math.exp(min(loss, 20))  # Cap to avoid overflow
        self.visualizer.log_scalar("train/perplexity", perplexity, self.global_step)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def _visualize(self):
        """Run visualization logging."""
        step = self.global_step
        
        # Attention patterns
        if step % self.vis_config.attention_log_steps == 0:
            self.visualizer.log_attention(self.model, step)
        
        # MoE routing
        if step % self.vis_config.moe_log_steps == 0:
            self.visualizer.log_moe(self.model, step)
        
        # Gradients
        if step % self.vis_config.gradient_log_steps == 0:
            self.visualizer.log_gradients(self.model, step)
        
        # Weights
        if step % self.vis_config.weights_log_steps == 0:
            self.visualizer.log_weights(self.model, step)
        
        # Generation samples
        if step % self.vis_config.generation_log_steps == 0:
            self.visualizer.log_generation(self.model, step, self.device)
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate on validation set.
        
        Returns:
            Average validation loss
        """
        val_loss, _ = self.evaluate_with_metrics()
        return val_loss
    
    @torch.no_grad()
    def evaluate_with_metrics(self) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate on validation set with detailed metrics.
        
        Returns:
            Average validation loss and metrics dict
        """
        self.model.eval()
        
        total_loss = 0.0
        total_lm_loss = 0.0
        total_mtp_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0
        total_tokens = 0
        
        eval_start_time = time.time()
        
        logger.info(f"  Evaluating...")
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels=batch['labels'],
            )
            
            total_loss += outputs['loss'].item()
            
            # Track sub-losses if available
            if 'lm_loss' in outputs and outputs['lm_loss'] is not None:
                total_lm_loss += outputs['lm_loss'].item()
            if 'mtp_loss' in outputs and outputs['mtp_loss'] is not None:
                total_mtp_loss += outputs['mtp_loss'].item()
            if 'aux_loss' in outputs and outputs['aux_loss'] is not None:
                total_aux_loss += outputs['aux_loss'].item()
            
            num_batches += 1
            total_tokens += batch['input_ids'].numel()
            
            if num_batches >= self.config.eval_samples // self.config.batch_size:
                break
        
        avg_loss = total_loss / max(1, num_batches)
        eval_time = time.time() - eval_start_time
        
        # Calculate perplexity
        perplexity = math.exp(min(avg_loss, 20))
        
        # Build metrics dict
        metrics = {
            'val_loss': avg_loss,
            'perplexity': perplexity,
            'eval_time': eval_time,
            'eval_tokens': total_tokens,
        }
        
        if total_lm_loss > 0:
            metrics['lm_loss'] = total_lm_loss / num_batches
        if total_mtp_loss > 0:
            metrics['mtp_loss'] = total_mtp_loss / num_batches
        if total_aux_loss > 0:
            metrics['aux_loss'] = total_aux_loss / num_batches
        
        # Print evaluation results
        logger.info(f"  ┌{'─'*50}┐")
        logger.info(f"  │ Validation Results                              │")
        logger.info(f"  ├{'─'*50}┤")
        logger.info(f"  │ Loss: {avg_loss:>10.4f}  Perplexity: {perplexity:>10.2f}    │")
        logger.info(f"  │ Batches: {num_batches:>5}  Tokens: {total_tokens:>10,}       │")
        logger.info(f"  │ Time: {eval_time:>6.1f}s                                  │")
        logger.info(f"  └{'─'*50}┘")
        
        # Log to TensorBoard
        self.visualizer.log_scalar("val/loss", avg_loss, self.global_step)
        self.visualizer.log_scalar("val/perplexity", perplexity, self.global_step)
        for key, value in metrics.items():
            if key not in ['val_loss', 'perplexity']:
                self.visualizer.log_scalar(f"val/{key}", value, self.global_step)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"  Checkpoint saved: {checkpoint_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = []
        for f in os.listdir(self.config.checkpoint_dir):
            if f.startswith("step_") and f.endswith(".pt"):
                step = int(f.replace("step_", "").replace(".pt", ""))
                checkpoints.append((step, f))
        
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        for _, filename in checkpoints[self.config.save_total_limit:]:
            path = os.path.join(self.config.checkpoint_dir, filename)
            os.remove(path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Resuming from step {self.global_step}")


# =============================================================================
# Pretrain Trainer
# =============================================================================

class PretrainTrainer(BaseTrainer):
    """Trainer for pretraining on raw text."""
    pass  # Uses base trainer implementation


# =============================================================================
# SFT Trainer
# =============================================================================

class SFTTrainer(BaseTrainer):
    """Trainer for supervised fine-tuning."""
    
    def __init__(
        self,
        model: DeepSeekV3Model,
        config: SFTConfig,
        vis_config: VisualizationConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: Any = None,
    ):
        super().__init__(model, config, vis_config, train_loader, val_loader, tokenizer)
        self.sft_config = config
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """SFT training step with instruction-following loss."""
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch['labels'],  # Labels already have -100 for prompt
        )
        return outputs['loss']


# =============================================================================
# GRPO Trainer (RL)
# =============================================================================

class GRPOTrainer(BaseTrainer):
    """
    Group Relative Policy Optimization Trainer.
    
    GRPO generates multiple responses per prompt, then uses relative
    rewards within each group to update the policy.
    
    Key steps:
    1. Generate G responses for each prompt
    2. Score responses with reward model
    3. Compute relative advantages within groups
    4. Update policy using clipped objective
    """
    
    def __init__(
        self,
        model: DeepSeekV3Model,
        ref_model: Optional[DeepSeekV3Model],
        config: RLConfig,
        vis_config: VisualizationConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: Any,
        reward_fn: Optional[Callable] = None,
    ):
        """
        Args:
            model: Policy model to train
            ref_model: Reference model for KL penalty (frozen)
            config: RL configuration
            vis_config: Visualization config
            train_loader: Prompt data loader
            val_loader: Validation loader
            tokenizer: Tokenizer for generation
            reward_fn: Reward function (response -> scalar)
        """
        super().__init__(model, config, vis_config, train_loader, val_loader, tokenizer)
        
        self.rl_config = config
        self.group_size = config.group_size
        self.kl_coef = config.kl_coef
        self.clip_range = config.clip_range
        
        # Reference model (frozen)
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # Reward function
        self.reward_fn = reward_fn or self._default_reward
        
        # RL-specific metrics
        self.reward_history = []
    
    def _default_reward(self, response: str) -> float:
        """
        Default rule-based reward function.
        
        Rewards:
        - Longer responses (up to a point)
        - Coherent sentences
        - Avoiding repetition
        """
        reward = 0.0
        
        # Length reward (prefer moderate length)
        words = response.split()
        if 10 <= len(words) <= 100:
            reward += 1.0
        elif len(words) < 10:
            reward += len(words) / 10.0
        else:
            reward += max(0, 1.0 - (len(words) - 100) / 100.0)
        
        # Sentence structure reward
        sentences = response.split('.')
        if len(sentences) > 1:
            reward += 0.5
        
        # Repetition penalty
        unique_words = set(words)
        if len(words) > 0:
            uniqueness = len(unique_words) / len(words)
            reward += uniqueness * 0.5
        
        # Ending reward
        if response.strip().endswith(('.', '!', '?')):
            reward += 0.3
        
        return reward
    
    def train(self) -> Dict[str, float]:
        """GRPO training loop."""
        logger.info(f"{'='*70}")
        logger.info("Starting GRPO Training...")
        logger.info(f"Group size: {self.group_size}")
        logger.info(f"KL coefficient: {self.kl_coef}")
        logger.info(f"Clip range: {self.clip_range}")
        logger.info(f"{'='*70}")
        
        self.model.train()
        
        total_reward = 0.0
        total_loss = 0.0
        num_steps = 0
        start_time = time.time()
        
        for epoch in range(self.rl_config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.rl_config.num_epochs}")
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Get prompts
                prompts = batch['prompt_text']
                prompt_ids = batch['input_ids'].to(self.device)
                
                # Generate responses for each prompt
                all_responses = []
                all_rewards = []
                all_log_probs = []
                all_ref_log_probs = []
                
                for i, prompt in enumerate(prompts):
                    prompt_input = prompt_ids[i:i+1]
                    
                    # Generate G responses
                    responses, log_probs, ref_log_probs = self._generate_group(
                        prompt_input,
                    )
                    
                    # Score responses
                    rewards = [self.reward_fn(r) for r in responses]
                    
                    all_responses.append(responses)
                    all_rewards.append(rewards)
                    all_log_probs.append(log_probs)
                    all_ref_log_probs.append(ref_log_probs)
                
                # Compute GRPO loss
                loss, metrics = self._grpo_step(
                    all_rewards, all_log_probs, all_ref_log_probs
                )
                
                # Backward
                loss.backward()
                
                # Optimizer step
                if (batch_idx + 1) % self.rl_config.gradient_accumulation_steps == 0:
                    if self.rl_config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.rl_config.max_grad_norm,
                        )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    num_steps += 1
                    
                    total_loss += loss.item()
                    total_reward += metrics['mean_reward']
                    self.reward_history.append(metrics['mean_reward'])
                    
                    # Logging
                    if self.global_step % self.rl_config.logging_steps == 0:
                        self._log_rl(loss.item(), metrics, start_time)
                    
                    # Save
                    if self.global_step % self.rl_config.save_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
                    
                    if self.global_step >= self.rl_config.max_steps:
                        break
            
            if self.global_step >= self.rl_config.max_steps:
                break
        
        self.save_checkpoint("final")
        self.visualizer.close()
        
        avg_reward = total_reward / max(1, num_steps)
        avg_loss = total_loss / max(1, num_steps)
        
        logger.info(f"{'='*70}")
        logger.info("GRPO Training complete!")
        logger.info(f"Average reward: {avg_reward:.4f}")
        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info(f"{'='*70}")
        
        return {
            'avg_reward': avg_reward,
            'avg_loss': avg_loss,
        }
    
    def _generate_group(
        self,
        prompt_ids: torch.Tensor,
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate G responses for a prompt.
        
        Returns:
            responses: List of G response strings
            log_probs: List of log probability tensors
            ref_log_probs: List of reference model log probs
        """
        responses = []
        log_probs = []
        ref_log_probs = []
        generated_outputs = []
        
        # Step 1: Generate responses (no gradient needed)
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.group_size):
                # Generate
                output_ids = self.model.generate(
                    prompt_ids,
                    max_new_tokens=self.rl_config.max_new_tokens,
                    temperature=self.rl_config.temperature,
                    top_p=self.rl_config.top_p,
                    do_sample=self.rl_config.do_sample,
                )
                
                # Decode response
                response_ids = output_ids[0, prompt_ids.shape[1]:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                responses.append(response_text)
                generated_outputs.append(output_ids)
        
        # Step 2: Compute log probs WITH gradient for policy model
        self.model.train()
        for output_ids in generated_outputs:
            # Compute log probs for response (WITH gradient)
            full_output = self.model(
                input_ids=output_ids,
                labels=output_ids,
            )
            logits = full_output['logits']
            
            # Log probs of generated tokens
            shift_logits = logits[:, prompt_ids.shape[1]-1:-1, :]
            shift_labels = output_ids[:, prompt_ids.shape[1]:]
            
            log_prob = F.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction='none',
            ).sum(dim=-1)
            log_probs.append(-log_prob)  # Negative because CE gives loss
            
            # Reference model log probs (no gradient needed)
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_output = self.ref_model(
                        input_ids=output_ids,
                        labels=output_ids,
                    )
                    ref_logits = ref_output['logits']
                    shift_ref_logits = ref_logits[:, prompt_ids.shape[1]-1:-1, :]
                    
                    ref_log_prob = F.cross_entropy(
                        shift_ref_logits.transpose(1, 2),
                        shift_labels,
                        reduction='none',
                    ).sum(dim=-1)
                    ref_log_probs.append(-ref_log_prob)
                else:
                    ref_log_probs.append(log_probs[-1].detach())
        
        return responses, log_probs, ref_log_probs
    
    def _grpo_step(
        self,
        all_rewards: List[List[float]],
        all_log_probs: List[List[torch.Tensor]],
        all_ref_log_probs: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss.
        
        Uses group-relative advantages for policy gradient.
        """
        total_loss = 0.0
        mean_reward = 0.0
        num_groups = len(all_rewards)
        
        for rewards, log_probs, ref_log_probs in zip(
            all_rewards, all_log_probs, all_ref_log_probs
        ):
            # Convert to tensors
            rewards_t = torch.tensor(rewards, device=self.device)
            
            # Group-relative advantages (normalize within group)
            mean_r = rewards_t.mean()
            std_r = rewards_t.std() + 1e-8
            advantages = (rewards_t - mean_r) / std_r
            
            # Compute loss for each response in group
            for adv, log_p, ref_log_p in zip(advantages, log_probs, ref_log_probs):
                # KL penalty
                kl = (log_p - ref_log_p).mean()
                
                # Policy gradient loss with advantage
                pg_loss = -adv * log_p.mean()
                
                # Total loss
                loss = pg_loss + self.kl_coef * kl
                total_loss = total_loss + loss
            
            mean_reward += rewards_t.mean().item()
        
        # Average across groups
        total_loss = total_loss / (num_groups * self.group_size)
        mean_reward = mean_reward / num_groups
        
        return total_loss, {
            'mean_reward': mean_reward,
            'loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        }
    
    def _log_rl(self, loss: float, metrics: Dict[str, float], start_time: float):
        """Log RL training metrics."""
        elapsed = time.time() - start_time
        lr = self.scheduler.get_last_lr()[0]
        
        logger.info(f"Step {self.global_step}/{self.rl_config.max_steps} | "
              f"Loss: {loss:.4f} | Reward: {metrics['mean_reward']:.4f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s")
        
        self.visualizer.log_scalar("rl/loss", loss, self.global_step)
        self.visualizer.log_scalar("rl/reward", metrics['mean_reward'], self.global_step)
        self.visualizer.log_scalar("rl/lr", lr, self.global_step)


# =============================================================================
# Factory Functions
# =============================================================================

def create_trainer(
    model: DeepSeekV3Model,
    config: DeepSeekV3Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: Any,
    mode: str = "pretrain",
    ref_model: Optional[DeepSeekV3Model] = None,
) -> BaseTrainer:
    """
    Create appropriate trainer for training mode.
    
    Args:
        model: Model to train
        config: Full configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        tokenizer: Tokenizer
        mode: Training mode ("pretrain", "sft", "rl")
        ref_model: Reference model for RL
        
    Returns:
        Trainer instance
    """
    if mode == "pretrain":
        return PretrainTrainer(
            model=model,
            config=config.pretraining,
            vis_config=config.visualization,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
        )
    elif mode == "sft":
        return SFTTrainer(
            model=model,
            config=config.sft,
            vis_config=config.visualization,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
        )
    elif mode == "rl":
        return GRPOTrainer(
            model=model,
            ref_model=ref_model,
            config=config.rl,
            vis_config=config.visualization,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown training mode: {mode}")
