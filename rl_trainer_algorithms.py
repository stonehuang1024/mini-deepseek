"""
DeepSeek V3 RL Trainer Module - Part 2
======================================

GRPO and PPO implementations.
"""

import time
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import RLConfig, VisualizationConfig
from model import DeepSeekV3Model
from rl_trainer_base import BaseRLTrainer, RewardFunction, RuleBasedReward
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)


# =============================================================================
# GRPO Trainer (Group Relative Policy Optimization)
# =============================================================================

class GRPOTrainer(BaseRLTrainer):
    """
    Group Relative Policy Optimization Trainer.
    
    GRPO generates G responses per prompt and uses group-relative
    advantages for more stable training without a learned reward model.
    
    Key features:
    1. Generate group of responses per prompt
    2. Compute rewards using reward function
    3. Normalize rewards within group (relative advantage)
    4. Policy gradient with KL penalty
    
    Loss = -E[A(y) * log π(y|x)] + β * KL(π || π_ref)
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
        super().__init__(
            model, ref_model, config, vis_config,
            train_loader, val_loader, tokenizer, reward_fn
        )
        
        self.group_size = config.group_size
        self.kl_coef = config.kl_coef
        self.clip_range = config.clip_range
        
        # Metrics tracking
        self.reward_history = []
        self.kl_history = []
        
        logger.info(f"GRPO Trainer initialized")
        logger.info(f"  Group size: {self.group_size}")
        logger.info(f"  KL coefficient: {self.kl_coef}")
    
    @torch.no_grad()
    def generate_responses(
        self,
        prompt_ids: torch.Tensor,
        num_responses: int,
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Generate multiple responses for a prompt.
        
        Args:
            prompt_ids: Tokenized prompt, shape (1, L_prompt)
            num_responses: Number of responses to generate
            
        Returns:
            response_ids: List of response tensors
            response_texts: List of decoded response texts
        """
        self.model.eval()
        
        response_ids = []
        response_texts = []
        
        for _ in range(num_responses):
            # Generate with sampling
            output_ids = self.model.generate(
                prompt_ids,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
            )  # (1, L_prompt + L_response)
            
            # Extract response (remove prompt)
            response = output_ids[:, prompt_ids.shape[1]:]  # (1, L_response)
            response_ids.append(output_ids)
            
            # Decode
            text = self.tokenizer.decode(response[0], skip_special_tokens=True)
            response_texts.append(text)
        
        self.model.train()
        return response_ids, response_texts
    
    def compute_grpo_loss(
        self,
        prompt_ids: torch.Tensor,
        response_ids_list: List[torch.Tensor],
        rewards: List[float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss for a group of responses.
        
        Args:
            prompt_ids: Prompt tokens, shape (1, L_prompt)
            response_ids_list: List of full sequences (prompt + response)
            rewards: List of rewards for each response
            
        Returns:
            loss: Scalar loss
            metrics: Training metrics
        """
        G = len(rewards)
        
        # Convert rewards to tensor and normalize within group
        rewards_t = torch.tensor(rewards, device=self.device)  # (G,)
        
        # Group-relative advantage: (r - mean) / std
        mean_reward = rewards_t.mean()
        std_reward = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean_reward) / std_reward  # (G,)
        
        total_pg_loss = 0.0
        total_kl = 0.0
        
        for i, (full_ids, adv) in enumerate(zip(response_ids_list, advantages)):
            # Create labels: -100 for prompt, actual ids for response
            labels = full_ids.clone()
            labels[:, :prompt_ids.shape[1]] = -100
            
            # Attention mask
            attention_mask = (full_ids != self.tokenizer.pad_token_id).long()
            
            # Policy log probs
            policy_logps = self.compute_log_probs(
                self.model, full_ids, attention_mask, labels
            )  # (1,)
            
            # Reference log probs
            with torch.no_grad():
                ref_logps = self.compute_log_probs(
                    self.ref_model, full_ids, attention_mask, labels
                )  # (1,)
            
            # KL divergence
            kl = policy_logps - ref_logps  # (1,)
            
            # Policy gradient loss: -advantage * log_prob
            pg_loss = -adv * policy_logps.mean()
            
            total_pg_loss = total_pg_loss + pg_loss
            total_kl = total_kl + kl.mean()
        
        # Average over group
        pg_loss = total_pg_loss / G
        kl_loss = total_kl / G
        
        # Total loss
        loss = pg_loss + self.kl_coef * kl_loss
        
        metrics = {
            'loss': loss.item(),
            'pg_loss': pg_loss.item() if isinstance(pg_loss, torch.Tensor) else pg_loss,
            'kl': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            'mean_reward': mean_reward.item(),
            'std_reward': std_reward.item(),
        }
        
        return loss, metrics
    
    def train(self) -> Dict[str, float]:
        """GRPO training loop."""
        logger.info(f"{'='*70}")
        logger.info("Starting GRPO Training...")
        logger.info(f"{'='*70}")
        
        total_reward = 0.0
        total_loss = 0.0
        num_steps = 0
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(self.train_loader):
                prompt_texts = batch['prompt_text']
                prompt_ids = batch['input_ids'].to(self.device)  # (B, L)
                
                batch_loss = 0.0
                batch_reward = 0.0
                batch_size = len(prompt_texts)
                
                # Process each prompt in batch
                for i in range(batch_size):
                    prompt = prompt_ids[i:i+1]  # (1, L)
                    prompt_text = prompt_texts[i]
                    
                    # Generate group of responses
                    response_ids, response_texts = self.generate_responses(
                        prompt, self.group_size
                    )
                    
                    # Compute rewards
                    rewards = [
                        self.reward_fn(resp, prompt_text)
                        for resp in response_texts
                    ]
                    
                    # Compute loss
                    loss, metrics = self.compute_grpo_loss(
                        prompt, response_ids, rewards
                    )
                    
                    batch_loss += loss
                    batch_reward += metrics['mean_reward']
                
                # Average over batch
                batch_loss = batch_loss / batch_size
                batch_reward = batch_reward / batch_size
                
                # Backward
                batch_loss = batch_loss / self.config.gradient_accumulation_steps
                batch_loss.backward()
                
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
                    
                    total_loss += batch_loss.item() * self.config.gradient_accumulation_steps
                    total_reward += batch_reward
                    self.reward_history.append(batch_reward)
                    
                    # Update best reward (check every step)
                    if batch_reward > self.best_reward:
                        self.best_reward = batch_reward
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        self._log_grpo_metrics(batch_reward, start_time)
                    
                    # Save
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
                        
                        # Save best model
                        if batch_reward >= self.best_reward:
                            self.save_checkpoint("best")
                    
                    if self.global_step >= self.config.max_steps:
                        break
            
            if self.global_step >= self.config.max_steps:
                break
        
        # Final save
        self.save_checkpoint("final")
        self.close()
        
        avg_reward = total_reward / max(1, num_steps)
        avg_loss = total_loss / max(1, num_steps)
        
        logger.info(f"{'='*70}")
        logger.info("GRPO Training complete!")
        logger.info(f"Average reward: {avg_reward:.4f}")
        logger.info(f"Best reward: {self.best_reward:.4f}")
        logger.info(f"{'='*70}")
        
        return {'avg_reward': avg_reward, 'avg_loss': avg_loss, 'best_reward': self.best_reward}
    
    def _log_grpo_metrics(self, reward: float, start_time: float):
        """Log GRPO metrics."""
        elapsed = time.time() - start_time
        lr = self.scheduler.get_last_lr()[0]
        
        # Compute running average
        window = min(10, len(self.reward_history))
        avg_reward = sum(self.reward_history[-window:]) / window if window > 0 else reward
        
        logger.info(f"Step {self.global_step}/{self.config.max_steps} | "
              f"Reward: {reward:.4f} (avg: {avg_reward:.4f}) | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s")
        
        self.writer.add_scalar("grpo/reward", reward, self.global_step)
        self.writer.add_scalar("grpo/reward_avg", avg_reward, self.global_step)
        self.writer.add_scalar("grpo/lr", lr, self.global_step)


# =============================================================================
# PPO Trainer
# =============================================================================

class PPOTrainer(BaseRLTrainer):
    """
    Proximal Policy Optimization Trainer.
    
    Full PPO implementation with:
    1. Value function for advantage estimation
    2. GAE (Generalized Advantage Estimation)
    3. Clipped surrogate objective
    4. Value function loss
    5. Entropy bonus
    
    Loss = L_clip + c1 * L_vf - c2 * H(π)
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
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 4,
    ):
        """
        Args:
            value_coef: Coefficient for value function loss
            entropy_coef: Coefficient for entropy bonus
            gae_lambda: GAE lambda parameter
            ppo_epochs: Number of PPO update epochs per batch
        """
        super().__init__(
            model, ref_model, config, vis_config,
            train_loader, val_loader, tokenizer, reward_fn
        )
        
        self.clip_range = config.clip_range
        self.gamma = config.gamma
        self.kl_coef = config.kl_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        
        # Simple value head (linear on top of model hidden states)
        self.value_head = nn.Linear(
            model.config.hidden_size, 1
        ).to(self.device)
        
        # Add value head to optimizer
        self.optimizer.add_param_group({'params': self.value_head.parameters()})
        
        logger.info(f"PPO Trainer initialized")
        logger.info(f"  Clip range: {self.clip_range}")
        logger.info(f"  GAE lambda: {self.gae_lambda}")
        logger.info(f"  PPO epochs: {self.ppo_epochs}")
    
    def compute_value(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value estimates.
        
        Args:
            input_ids: Input tokens, shape (B, L)
            attention_mask: Attention mask, shape (B, L)
            
        Returns:
            values: Value estimates, shape (B,)
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.get('hidden_states')
            if hidden_states is not None:
                # Use last hidden state at last position
                last_hidden = hidden_states[-1]  # (B, L, D)
                # Get hidden state at last non-pad position
                seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
                batch_indices = torch.arange(input_ids.shape[0], device=self.device)
                final_hidden = last_hidden[batch_indices, seq_lens]  # (B, D)
            else:
                # Fallback: use logits mean
                final_hidden = outputs['logits'].mean(dim=(1, 2)).unsqueeze(-1)
                final_hidden = final_hidden.expand(-1, self.model.config.hidden_size)
        
        values = self.value_head(final_hidden).squeeze(-1)  # (B,)
        return values
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Rewards, shape (T,)
            values: Value estimates, shape (T+1,)
            dones: Episode done flags, shape (T,)
            
        Returns:
            advantages: GAE advantages, shape (T,)
            returns: TD(λ) returns, shape (T,)
        """
        T = rewards.shape[0]
        advantages = torch.zeros(T, device=self.device)
        
        lastgaelam = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = values[t + 1] if values.shape[0] > T else 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * lastgaelam
        
        returns = advantages + values[:T]
        return advantages, returns
    
    def compute_ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO clipped objective.
        
        Args:
            old_log_probs: Log probs from old policy, shape (B,)
            new_log_probs: Log probs from current policy, shape (B,)
            advantages: Normalized advantages, shape (B,)
            values: Value estimates, shape (B,)
            returns: Target returns, shape (B,)
            entropy: Policy entropy, shape (B,)
            
        Returns:
            loss: Total PPO loss
            metrics: Training metrics
        """
        # Probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)  # (B,)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss (also clipped)
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus (encourage exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Metrics
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean()
            clip_frac = ((ratio - 1).abs() > self.clip_range).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'approx_kl': approx_kl.item(),
            'clip_frac': clip_frac.item(),
        }
        
        return loss, metrics
    
    def train(self) -> Dict[str, float]:
        """PPO training loop."""
        logger.info(f"{'='*70}")
        logger.info("Starting PPO Training...")
        logger.info(f"{'='*70}")
        
        total_reward = 0.0
        total_loss = 0.0
        num_steps = 0
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(self.train_loader):
                prompt_texts = batch['prompt_text']
                prompt_ids = batch['input_ids'].to(self.device)
                prompt_mask = batch['attention_mask'].to(self.device)
                batch_size = len(prompt_texts)
                
                # Collect experience (generate responses, compute rewards)
                experience = self._collect_experience(
                    prompt_ids, prompt_mask, prompt_texts
                )
                
                if experience is None:
                    continue
                
                # PPO update epochs
                for ppo_epoch in range(self.ppo_epochs):
                    loss, metrics = self._ppo_update(experience)
                    
                    # Early stopping on high KL
                    if metrics['approx_kl'] > 0.02:
                        break
                
                self.global_step += 1
                num_steps += 1
                
                total_loss += metrics['loss']
                total_reward += experience['mean_reward']
                
                # Update best reward (check every step)
                if experience['mean_reward'] > self.best_reward:
                    self.best_reward = experience['mean_reward']
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_ppo_metrics(metrics, experience['mean_reward'], start_time)
                
                # Save
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
                    
                    # Save best model
                    if experience['mean_reward'] >= self.best_reward:
                        self.save_checkpoint("best")
                
                if self.global_step >= self.config.max_steps:
                    break
            
            if self.global_step >= self.config.max_steps:
                break
        
        # Final save
        self.save_checkpoint("final")
        self.close()
        
        avg_reward = total_reward / max(1, num_steps)
        avg_loss = total_loss / max(1, num_steps)
        
        logger.info(f"{'='*70}")
        logger.info("PPO Training complete!")
        logger.info(f"Average reward: {avg_reward:.4f}")
        logger.info(f"Best reward: {self.best_reward:.4f}")
        logger.info(f"{'='*70}")
        
        return {'avg_reward': avg_reward, 'avg_loss': avg_loss, 'best_reward': self.best_reward}
    
    def _collect_experience(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        prompt_texts: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Collect rollout experience."""
        self.model.eval()
        
        all_response_ids = []
        all_rewards = []
        all_old_log_probs = []
        all_values = []
        
        with torch.no_grad():
            for i in range(prompt_ids.shape[0]):
                prompt = prompt_ids[i:i+1]
                prompt_text = prompt_texts[i]
                
                # Generate response
                output_ids = self.model.generate(
                    prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                )
                
                # Get response text
                response = output_ids[:, prompt.shape[1]:]
                response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
                
                # Compute reward
                reward = self.reward_fn(response_text, prompt_text)
                
                # Create labels
                labels = output_ids.clone()
                labels[:, :prompt.shape[1]] = -100
                
                # Attention mask
                attention_mask = (output_ids != self.tokenizer.pad_token_id).long()
                
                # Compute log probs and value
                log_prob = self.compute_log_probs(
                    self.model, output_ids, attention_mask, labels
                )
                value = self.compute_value(output_ids, attention_mask)
                
                all_response_ids.append(output_ids)
                all_rewards.append(reward)
                all_old_log_probs.append(log_prob)
                all_values.append(value)
        
        self.model.train()
        
        if not all_rewards:
            return None
        
        # Compute advantages
        rewards_t = torch.tensor(all_rewards, device=self.device)
        values_t = torch.cat(all_values)
        
        # Simple advantage: r - V (no GAE for simplicity in this demo)
        advantages = rewards_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'response_ids': all_response_ids,
            'rewards': rewards_t,
            'old_log_probs': torch.cat(all_old_log_probs),
            'values': values_t,
            'advantages': advantages,
            'mean_reward': rewards_t.mean().item(),
        }
    
    def _ppo_update(self, experience: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform PPO update."""
        total_loss = 0.0
        total_metrics = {}
        
        for i, output_ids in enumerate(experience['response_ids']):
            # Create labels and mask
            prompt_len = (output_ids[0] == self.tokenizer.pad_token_id).long().argmax().item()
            if prompt_len == 0:
                prompt_len = output_ids.shape[1] // 2
            
            labels = output_ids.clone()
            labels[:, :prompt_len] = -100
            attention_mask = (output_ids != self.tokenizer.pad_token_id).long()
            
            # New log probs
            new_log_prob = self.compute_log_probs(
                self.model, output_ids, attention_mask, labels
            )
            
            # New value
            new_value = self.compute_value(output_ids, attention_mask)
            
            # Entropy (approximate)
            outputs = self.model(input_ids=output_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
            
            # Compute loss
            loss, metrics = self.compute_ppo_loss(
                experience['old_log_probs'][i:i+1],
                new_log_prob,
                experience['advantages'][i:i+1],
                new_value,
                experience['values'][i:i+1] + experience['advantages'][i:i+1],
                entropy.unsqueeze(0),
            )
            
            loss.backward()
            total_loss += loss.item()
            
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
        
        # Optimizer step
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.value_head.parameters()),
                self.config.max_grad_norm
            )
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        n = len(experience['response_ids'])
        return total_loss / n, {k: v / n for k, v in total_metrics.items()}
    
    def _log_ppo_metrics(self, metrics: Dict[str, float], reward: float, start_time: float):
        """Log PPO metrics."""
        elapsed = time.time() - start_time
        lr = self.scheduler.get_last_lr()[0]
        
        logger.info(f"Step {self.global_step}/{self.config.max_steps} | "
              f"Reward: {reward:.4f} | Loss: {metrics['loss']:.4f} | "
              f"KL: {metrics['approx_kl']:.4f} | LR: {lr:.2e} | "
              f"Time: {elapsed:.1f}s")
        
        self.writer.add_scalar("ppo/reward", reward, self.global_step)
        self.writer.add_scalar("ppo/loss", metrics['loss'], self.global_step)
        self.writer.add_scalar("ppo/policy_loss", metrics['policy_loss'], self.global_step)
        self.writer.add_scalar("ppo/value_loss", metrics['value_loss'], self.global_step)
        self.writer.add_scalar("ppo/entropy", metrics['entropy'], self.global_step)
        self.writer.add_scalar("ppo/approx_kl", metrics['approx_kl'], self.global_step)
        self.writer.add_scalar("ppo/lr", lr, self.global_step)
