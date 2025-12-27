"""
DeepSeek V3 RL Trainer Module - Part 1
======================================

Base RL trainer and DPO implementation.
"""

import os
import time
import math
import copy
from typing import Optional, Dict, Any, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from deepseek.model import DeepSeekV3Model
from deepseek.utils import get_logger

# Import config - support both package and standalone usage
try:
    from config import RLConfig, VisualizationConfig, get_device
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import RLConfig, VisualizationConfig, get_device

# Initialize logger
logger = get_logger(__name__)


# =============================================================================
# Reward Functions
# =============================================================================

class RewardFunction:
    """
    Base class for reward functions.
    
    Can be rule-based or model-based.
    """
    
    def __call__(self, response: str, prompt: str = "", context: str = "") -> float:
        """
        Compute reward for a response.
        
        Args:
            response: Generated response text
            prompt: Original prompt
            context: Additional context
            
        Returns:
            Scalar reward value
        """
        raise NotImplementedError


class RuleBasedReward(RewardFunction):
    """
    Rule-based reward function.
    
    Computes reward based on:
    - Response length (moderate length preferred)
    - Sentence structure
    - Word diversity (avoid repetition)
    - Proper ending punctuation
    """
    
    def __init__(
        self,
        length_weight: float = 1.0,
        structure_weight: float = 0.5,
        diversity_weight: float = 0.5,
        ending_weight: float = 0.3,
        min_length: int = 10,
        max_length: int = 150,
    ):
        self.length_weight = length_weight
        self.structure_weight = structure_weight
        self.diversity_weight = diversity_weight
        self.ending_weight = ending_weight
        self.min_length = min_length
        self.max_length = max_length
    
    def __call__(self, response: str, prompt: str = "", context: str = "") -> float:
        reward = 0.0
        words = response.split()
        num_words = len(words)
        
        # Length reward: prefer moderate length
        if self.min_length <= num_words <= self.max_length:
            # Peak reward at middle of range
            optimal = (self.min_length + self.max_length) / 2
            deviation = abs(num_words - optimal) / optimal
            reward += self.length_weight * max(0, 1.0 - deviation)
        elif num_words < self.min_length:
            reward += self.length_weight * (num_words / self.min_length) * 0.5
        else:
            reward += self.length_weight * max(0, 1.0 - (num_words - self.max_length) / self.max_length) * 0.5
        
        # Structure reward: has multiple sentences
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) > 1:
            reward += self.structure_weight
        elif len(sentences) == 1 and len(words) > 5:
            reward += self.structure_weight * 0.5
        
        # Diversity reward: unique words ratio
        if num_words > 0:
            unique_words = set(w.lower() for w in words)
            diversity = len(unique_words) / num_words
            reward += self.diversity_weight * diversity
        
        # Ending reward: proper punctuation
        stripped = response.strip()
        if stripped and stripped[-1] in '.!?':
            reward += self.ending_weight
        
        return reward


class LengthReward(RewardFunction):
    """Simple length-based reward."""
    
    def __init__(self, target_length: int = 50, penalty_scale: float = 0.01):
        self.target_length = target_length
        self.penalty_scale = penalty_scale
    
    def __call__(self, response: str, prompt: str = "", context: str = "") -> float:
        words = response.split()
        deviation = abs(len(words) - self.target_length)
        return max(0, 1.0 - self.penalty_scale * deviation)


class CompositeReward(RewardFunction):
    """Combines multiple reward functions."""
    
    def __init__(self, rewards: List[Tuple[RewardFunction, float]]):
        """
        Args:
            rewards: List of (reward_fn, weight) tuples
        """
        self.rewards = rewards
    
    def __call__(self, response: str, prompt: str = "", context: str = "") -> float:
        total = 0.0
        for reward_fn, weight in self.rewards:
            total += weight * reward_fn(response, prompt, context)
        return total


# =============================================================================
# Base RL Trainer
# =============================================================================

class BaseRLTrainer:
    """
    Base class for RL trainers.
    
    Provides common functionality:
    - Model and optimizer setup
    - Reference model handling
    - Checkpointing
    - Visualization
    - KL divergence computation
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
        reward_fn: Optional[RewardFunction] = None,
    ):
        """
        Args:
            model: Policy model to train
            ref_model: Reference model for KL (frozen) - if None, copies model
            config: RL configuration
            vis_config: Visualization config
            train_loader: Training data
            val_loader: Validation data
            tokenizer: Tokenizer for text processing
            reward_fn: Reward function
        """
        self.config = config
        self.vis_config = vis_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        
        # Device setup
        self.device = torch.device(get_device(config.device))
        
        # Model setup
        self.model = model.to(self.device)
        
        # Reference model (frozen copy for KL)
        if ref_model is not None:
            self.ref_model = ref_model.to(self.device)
        else:
            # Create frozen copy using state_dict to avoid deepcopy issues
            try:
                self.ref_model = copy.deepcopy(model).to(self.device)
            except Exception:
                # Fallback: create new model and load weights
                from model import DeepSeekV3Model
                self.ref_model = DeepSeekV3Model(model.config).to(self.device)
                self.ref_model.load_state_dict(model.state_dict())
        
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Reward function
        self.reward_fn = reward_fn or RuleBasedReward()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.total_steps = config.max_steps
        warmup_steps = config.warmup_steps
        if config.warmup_ratio > 0:
            warmup_steps = int(self.total_steps * config.warmup_ratio)
        
        self.scheduler = self._create_scheduler(warmup_steps)
        
        # Visualization
        os.makedirs(config.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(config.tensorboard_dir)
        
        # Checkpointing
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_reward = float('-inf')
        
        # Set seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        logger.info(f"RL Trainer initialized on {self.device}")
        logger.info(f"  Algorithm: {config.algorithm}")
        logger.info(f"  Max steps: {config.max_steps}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with appropriate weight decay."""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        return AdamW(
            [
                {'params': decay_params, 'weight_decay': self.config.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
    
    def _create_scheduler(self, warmup_steps: int) -> LambdaLR:
        """Create cosine schedule with warmup."""
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def compute_log_probs(
        self,
        model: DeepSeekV3Model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities of tokens.
        
        Args:
            model: Model to use
            input_ids: Input token IDs, shape (B, L)
            attention_mask: Attention mask, shape (B, L)
            labels: Labels for computing log probs, shape (B, L)
            
        Returns:
            Log probabilities per sequence, shape (B,)
        """
        with torch.set_grad_enabled(model.training):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs['logits']  # (B, L, V)
        
        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()  # (B, L-1, V)
        shift_labels = labels[:, 1:].contiguous()  # (B, L-1)
        
        # Compute per-token log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, L-1, V)
        
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, L-1)
        
        # Mask out padding and prompt tokens
        mask = (shift_labels != -100) & (shift_labels != self.tokenizer.pad_token_id)
        mask = mask.float()  # (B, L-1)
        
        # Sum log probs for each sequence
        seq_log_probs = (token_log_probs * mask).sum(dim=-1)  # (B,)
        
        return seq_log_probs
    
    def compute_kl_divergence(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference.
        
        KL(π || π_ref) ≈ log(π) - log(π_ref)
        
        Args:
            policy_log_probs: Log probs from policy, shape (B,)
            ref_log_probs: Log probs from reference, shape (B,)
            
        Returns:
            KL divergence per sequence, shape (B,)
        """
        return policy_log_probs - ref_log_probs
    
    def train(self) -> Dict[str, float]:
        """Main training loop - to be implemented by subclasses."""
        raise NotImplementedError
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'config': self.config,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"  Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        
        logger.info(f"Loaded checkpoint from {path}, step {self.global_step}")
    
    def close(self):
        """Cleanup resources."""
        self.writer.close()


# =============================================================================
# DPO Trainer
# =============================================================================

class DPOTrainer(BaseRLTrainer):
    """
    Direct Preference Optimization Trainer.
    
    DPO directly optimizes the policy using preference pairs without
    explicit reward modeling or online generation.
    
    Loss = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
    
    where y_w is chosen (winner) and y_l is rejected (loser).
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
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            model: Policy model
            ref_model: Reference model
            config: RL config
            vis_config: Visualization config
            train_loader: DPO data loader (preference pairs)
            val_loader: Validation loader
            tokenizer: Tokenizer
            beta: Temperature parameter for DPO (higher = more aggressive)
            label_smoothing: Label smoothing for robustness
        """
        # DPO doesn't use reward function
        super().__init__(
            model, ref_model, config, vis_config,
            train_loader, val_loader, tokenizer, reward_fn=None
        )
        
        self.beta = beta
        self.label_smoothing = label_smoothing
        
        logger.info(f"DPO Trainer initialized")
        logger.info(f"  Beta: {self.beta}")
        logger.info(f"  Label smoothing: {self.label_smoothing}")
    
    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Policy log probs for chosen, shape (B,)
            policy_rejected_logps: Policy log probs for rejected, shape (B,)
            ref_chosen_logps: Reference log probs for chosen, shape (B,)
            ref_rejected_logps: Reference log probs for rejected, shape (B,)
            
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        # Compute log ratios
        # π(y_w|x) / π_ref(y_w|x)
        chosen_logratios = policy_chosen_logps - ref_chosen_logps  # (B,)
        # π(y_l|x) / π_ref(y_l|x)
        rejected_logratios = policy_rejected_logps - ref_rejected_logps  # (B,)
        
        # DPO implicit reward difference
        # r(y_w) - r(y_l) ∝ log(π(y_w)/π_ref(y_w)) - log(π(y_l)/π_ref(y_l))
        logits = self.beta * (chosen_logratios - rejected_logratios)  # (B,)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Soft labels: (1 - ε, ε) instead of (1, 0)
            losses = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-logits) * self.label_smoothing
            )
        else:
            # Standard DPO: maximize probability that chosen > rejected
            losses = -F.logsigmoid(logits)
        
        loss = losses.mean()
        
        # Compute metrics
        with torch.no_grad():
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            
            # Accuracy: how often do we correctly prefer chosen?
            accuracy = (logits > 0).float().mean()
            
            # Reward margins
            reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_reward': chosen_rewards.mean().item(),
            'rejected_reward': rejected_rewards.mean().item(),
            'reward_margin': reward_margin.item(),
            'chosen_logratio': chosen_logratios.mean().item(),
            'rejected_logratio': rejected_logratios.mean().item(),
        }
        
        return loss, metrics
    
    def train(self) -> Dict[str, float]:
        """DPO training loop."""
        logger.info(f"{'='*70}")
        logger.info("Starting DPO Training...")
        logger.info(f"{'='*70}")
        
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_steps = 0
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Move to device
                chosen_ids = batch['chosen_input_ids'].to(self.device)
                rejected_ids = batch['rejected_input_ids'].to(self.device)
                chosen_labels = batch['chosen_labels'].to(self.device)
                rejected_labels = batch['rejected_labels'].to(self.device)
                chosen_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_mask = batch['rejected_attention_mask'].to(self.device)
                
                # Compute policy log probs
                policy_chosen_logps = self.compute_log_probs(
                    self.model, chosen_ids, chosen_mask, chosen_labels
                )
                policy_rejected_logps = self.compute_log_probs(
                    self.model, rejected_ids, rejected_mask, rejected_labels
                )
                
                # Compute reference log probs
                with torch.no_grad():
                    ref_chosen_logps = self.compute_log_probs(
                        self.ref_model, chosen_ids, chosen_mask, chosen_labels
                    )
                    ref_rejected_logps = self.compute_log_probs(
                        self.ref_model, rejected_ids, rejected_mask, rejected_labels
                    )
                
                # Compute DPO loss
                loss, metrics = self.compute_dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                
                # Backward
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Optimizer step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    num_steps += 1
                    
                    total_loss += metrics['loss']
                    total_accuracy += metrics['accuracy']
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        self._log_metrics(metrics, start_time)
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
                    
                    if self.global_step >= self.config.max_steps:
                        break
            
            if self.global_step >= self.config.max_steps:
                break
        
        # Final save
        self.save_checkpoint("final")
        self.close()
        
        avg_loss = total_loss / max(1, num_steps)
        avg_accuracy = total_accuracy / max(1, num_steps)
        
        logger.info(f"{'='*70}")
        logger.info("DPO Training complete!")
        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info(f"Average accuracy: {avg_accuracy:.4f}")
        logger.info(f"{'='*70}")
        
        return {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
        }
    
    def _log_metrics(self, metrics: Dict[str, float], start_time: float):
        """Log training metrics."""
        elapsed = time.time() - start_time
        lr = self.scheduler.get_last_lr()[0]
        
        logger.info(f"Step {self.global_step}/{self.config.max_steps} | "
              f"Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f} | "
              f"Margin: {metrics['reward_margin']:.4f} | LR: {lr:.2e} | "
              f"Time: {elapsed:.1f}s")
        
        # TensorBoard
        self.writer.add_scalar("dpo/loss", metrics['loss'], self.global_step)
        self.writer.add_scalar("dpo/accuracy", metrics['accuracy'], self.global_step)
        self.writer.add_scalar("dpo/reward_margin", metrics['reward_margin'], self.global_step)
        self.writer.add_scalar("dpo/chosen_reward", metrics['chosen_reward'], self.global_step)
        self.writer.add_scalar("dpo/rejected_reward", metrics['rejected_reward'], self.global_step)
        self.writer.add_scalar("dpo/lr", lr, self.global_step)
