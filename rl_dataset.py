"""
DeepSeek V3 RL Dataset Module
=============================

Provides dataset classes for different RL algorithms:
1. DPODataset: Preference pairs (chosen, rejected) for DPO training
2. GRPODataset: Prompts for group relative policy optimization
3. PPODataset: Prompts with reward model for PPO training

All datasets support automatic downloading and caching.
"""

import os
import json
import random
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from config import DataConfig
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)


# =============================================================================
# DPO Dataset (Preference Pairs)
# =============================================================================

class DPODataset(Dataset):
    """
    Dataset for Direct Preference Optimization (DPO).
    
    Contains preference pairs: (prompt, chosen_response, rejected_response)
    Used to train model to prefer chosen over rejected responses.
    
    Shape:
        - prompt_input_ids: (L_prompt,) - tokenized prompt
        - chosen_input_ids: (L_full,) - prompt + chosen response
        - rejected_input_ids: (L_full,) - prompt + rejected response
        - chosen_labels: (L_full,) - labels for chosen (prompt=-100)
        - rejected_labels: (L_full,) - labels for rejected (prompt=-100)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: Tokenizer for encoding
            config: Data configuration
            split: Dataset split
            max_samples: Maximum samples to load
        """
        self.tokenizer = tokenizer
        self.max_seq_length = config.rl_max_seq_length
        self.split = split
        max_samples = max_samples or config.rl_max_samples
        
        # Load data
        self.examples = self._load_data(config, split, max_samples)
        
        logger.info(f"DPODataset ({split}): {len(self.examples)} preference pairs")
    
    def _load_data(
        self,
        config: DataConfig,
        split: str,
        max_samples: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Load DPO preference pairs from HH-RLHF dataset."""
        try:
            from datasets import load_dataset
            
            # Load HH-RLHF dataset (contains chosen/rejected pairs)
            # Use 'default' config if specified config fails
            try:
                dataset = load_dataset(
                    config.rl_dataset_name,
                    config.rl_dataset_config,
                    cache_dir=config.rl_data_dir,
                )
            except Exception:
                # Fallback to default config
                dataset = load_dataset(
                    config.rl_dataset_name,
                    "default",
                    cache_dir=config.rl_data_dir,
                )
            
            # Get appropriate split
            if split in dataset:
                data = dataset[split]
            else:
                data = dataset["train"]
            
            # Process examples
            examples = []
            for i, item in enumerate(data):
                if i >= max_samples:
                    break
                
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")
                
                if not chosen or not rejected:
                    continue
                
                # Extract prompt and responses
                prompt, chosen_response, rejected_response = self._parse_conversation(
                    chosen, rejected
                )
                
                if prompt and chosen_response and rejected_response:
                    example = self._process_example(
                        prompt, chosen_response, rejected_response
                    )
                    if example is not None:
                        examples.append(example)
            
            # Split for validation
            if split == "validation":
                examples = examples[:int(len(examples) * 0.1)]
            elif split == "train":
                examples = examples[int(len(examples) * 0.1):]
            
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load DPO dataset: {e}")
            logger.warning("Generating synthetic preference data...")
            return self._generate_synthetic_data(split, max_samples)
    
    def _parse_conversation(
        self,
        chosen: str,
        rejected: str,
    ) -> Tuple[str, str, str]:
        """
        Parse HH-RLHF conversation format.
        
        Format: "Human: ... Assistant: ..."
        """
        # Extract prompt (human turn)
        if "Human:" not in chosen:
            return "", "", ""
        
        # Get prompt from chosen (should be same in both)
        parts = chosen.split("Assistant:")
        if len(parts) < 2:
            return "", "", ""
        
        prompt = parts[0].replace("Human:", "").strip()
        chosen_response = parts[1].strip() if len(parts) > 1 else ""
        
        # Get rejected response
        rejected_parts = rejected.split("Assistant:")
        rejected_response = rejected_parts[1].strip() if len(rejected_parts) > 1 else ""
        
        # Clean up responses (remove trailing human turns)
        if "Human:" in chosen_response:
            chosen_response = chosen_response.split("Human:")[0].strip()
        if "Human:" in rejected_response:
            rejected_response = rejected_response.split("Human:")[0].strip()
        
        return prompt, chosen_response, rejected_response
    
    def _process_example(
        self,
        prompt: str,
        chosen_response: str,
        rejected_response: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Process single preference pair into tokenized format."""
        # Format full texts
        prompt_text = f"Human: {prompt}\n\nAssistant:"
        chosen_full = f"{prompt_text} {chosen_response}"
        rejected_full = f"{prompt_text} {rejected_response}"
        
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        
        # Tokenize chosen
        chosen_ids = self.tokenizer.encode(
            chosen_full,
            max_length=self.max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )
        
        # Tokenize rejected
        rejected_ids = self.tokenizer.encode(
            rejected_full,
            max_length=self.max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )
        
        # Create labels: -100 for prompt tokens
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids[len(prompt_ids):]
        rejected_labels = [-100] * len(prompt_ids) + rejected_ids[len(prompt_ids):]
        
        # Pad to same length
        max_len = max(len(chosen_ids), len(rejected_ids))
        max_len = min(max_len, self.max_seq_length)
        
        def pad_sequence(ids, labels, target_len):
            pad_len = target_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
            else:
                ids = ids[:target_len]
                labels = labels[:target_len]
            return ids, labels
        
        chosen_ids, chosen_labels = pad_sequence(chosen_ids, chosen_labels, max_len)
        rejected_ids, rejected_labels = pad_sequence(rejected_ids, rejected_labels, max_len)
        
        return {
            'prompt_input_ids': torch.tensor(prompt_ids, dtype=torch.long),
            'chosen_input_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_input_ids': torch.tensor(rejected_ids, dtype=torch.long),
            'chosen_labels': torch.tensor(chosen_labels, dtype=torch.long),
            'rejected_labels': torch.tensor(rejected_labels, dtype=torch.long),
            'chosen_attention_mask': torch.tensor(
                [1 if t != self.tokenizer.pad_token_id else 0 for t in chosen_ids],
                dtype=torch.long
            ),
            'rejected_attention_mask': torch.tensor(
                [1 if t != self.tokenizer.pad_token_id else 0 for t in rejected_ids],
                dtype=torch.long
            ),
        }
    
    def _generate_synthetic_data(
        self,
        split: str,
        max_samples: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate synthetic preference data for testing."""
        preference_pairs = [
            ("What is AI?", 
             "Artificial intelligence is a field of computer science that aims to create intelligent machines.",
             "AI is when computers do stuff."),
            ("Explain photosynthesis.",
             "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
             "Plants eat sun."),
            ("What causes rain?",
             "Rain forms when water evaporates, rises into the atmosphere, condenses into clouds, and falls back to Earth.",
             "Clouds cry."),
            ("How do airplanes fly?",
             "Airplanes fly due to the principles of aerodynamics. Wings create lift by having air move faster over the top surface.",
             "Magic."),
            ("Why is the sky blue?",
             "The sky appears blue due to Rayleigh scattering, where shorter blue wavelengths of sunlight scatter more in our atmosphere.",
             "It just is."),
        ]
        
        num_samples = min(max_samples, 100 if split == "train" else 10)
        examples = []
        
        for i in range(num_samples):
            prompt, chosen, rejected = random.choice(preference_pairs)
            example = self._process_example(prompt, chosen, rejected)
            if example is not None:
                examples.append(example)
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# =============================================================================
# GRPO Dataset (Group Prompts)
# =============================================================================

class GRPODataset(Dataset):
    """
    Dataset for Group Relative Policy Optimization.
    
    Contains prompts for generating multiple responses per prompt.
    Responses are generated during training, not stored.
    
    Shape:
        - input_ids: (L,) - tokenized prompt
        - attention_mask: (L,) - 1 for real tokens
        - prompt_text: str - original prompt text
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = config.rl_max_seq_length
        self.split = split
        max_samples = max_samples or config.rl_max_samples
        
        self.prompts, self.tokenized = self._load_data(config, split, max_samples)
        logger.info(f"GRPODataset ({split}): {len(self.prompts)} prompts")
    
    def _load_data(
        self,
        config: DataConfig,
        split: str,
        max_samples: int,
    ) -> Tuple[List[str], List[Dict[str, torch.Tensor]]]:
        """Load prompts from dataset."""
        try:
            from datasets import load_dataset
            
            # Try primary config first, fallback to 'default'
            try:
                dataset = load_dataset(
                    config.rl_dataset_name,
                    config.rl_dataset_config,
                    cache_dir=config.rl_data_dir,
                )
            except Exception:
                dataset = load_dataset(
                    config.rl_dataset_name,
                    "default",
                    cache_dir=config.rl_data_dir,
                )
            
            if split in dataset:
                data = dataset[split]
            else:
                data = dataset["train"]
            
            prompts = []
            for item in data:
                chosen = item.get("chosen", "")
                if "Human:" in chosen:
                    human_turn = chosen.split("Human:")[1].split("Assistant:")[0].strip()
                    if human_turn:
                        prompts.append(f"Human: {human_turn}\n\nAssistant:")
                
                if len(prompts) >= max_samples:
                    break
            
            # Split for validation
            if split == "validation":
                prompts = prompts[:int(len(prompts) * 0.1)]
            elif split == "train":
                prompts = prompts[int(len(prompts) * 0.1):]
            
            tokenized = self._tokenize_prompts(prompts)
            return prompts, tokenized
            
        except Exception as e:
            logger.error(f"Failed to load GRPO dataset: {e}")
            return self._generate_synthetic_prompts(split, max_samples)
    
    def _tokenize_prompts(
        self,
        prompts: List[str],
    ) -> List[Dict[str, torch.Tensor]]:
        """Tokenize prompts."""
        tokenized = []
        for prompt in prompts:
            encoding = self.tokenizer(
                prompt,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            tokenized.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
            })
        return tokenized
    
    def _generate_synthetic_prompts(
        self,
        split: str,
        max_samples: int,
    ) -> Tuple[List[str], List[Dict[str, torch.Tensor]]]:
        """Generate synthetic prompts."""
        templates = [
            "Human: Explain quantum computing in simple terms.\n\nAssistant:",
            "Human: Write a short poem about nature.\n\nAssistant:",
            "Human: What are the benefits of exercise?\n\nAssistant:",
            "Human: How does machine learning work?\n\nAssistant:",
            "Human: Describe the water cycle.\n\nAssistant:",
            "Human: What makes a good leader?\n\nAssistant:",
            "Human: Explain how vaccines work.\n\nAssistant:",
            "Human: What is climate change?\n\nAssistant:",
        ]
        
        num_samples = min(max_samples, 100 if split == "train" else 10)
        prompts = [random.choice(templates) for _ in range(num_samples)]
        tokenized = self._tokenize_prompts(prompts)
        
        return prompts, tokenized
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            **self.tokenized[idx],
            'prompt_text': self.prompts[idx],
        }


# =============================================================================
# PPO Dataset (Prompts + Rewards)
# =============================================================================

class PPODataset(Dataset):
    """
    Dataset for Proximal Policy Optimization.
    
    Similar to GRPO but may include additional context for reward computation.
    
    Shape:
        - input_ids: (L,) - tokenized prompt
        - attention_mask: (L,) - 1 for real tokens
        - prompt_text: str - original prompt
        - context: str - additional context (optional)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = config.rl_max_seq_length
        self.split = split
        max_samples = max_samples or config.rl_max_samples
        
        self.data = self._load_data(config, split, max_samples)
        logger.info(f"PPODataset ({split}): {len(self.data)} samples")
    
    def _load_data(
        self,
        config: DataConfig,
        split: str,
        max_samples: int,
    ) -> List[Dict[str, Any]]:
        """Load PPO training data."""
        try:
            from datasets import load_dataset
            
            # Try primary config first, fallback to 'default'
            try:
                dataset = load_dataset(
                    config.rl_dataset_name,
                    config.rl_dataset_config,
                    cache_dir=config.rl_data_dir,
                )
            except Exception:
                dataset = load_dataset(
                    config.rl_dataset_name,
                    "default",
                    cache_dir=config.rl_data_dir,
                )
            
            if split in dataset:
                data = dataset[split]
            else:
                data = dataset["train"]
            
            samples = []
            for item in data:
                chosen = item.get("chosen", "")
                if "Human:" in chosen:
                    human_turn = chosen.split("Human:")[1].split("Assistant:")[0].strip()
                    if human_turn:
                        prompt = f"Human: {human_turn}\n\nAssistant:"
                        encoding = self.tokenizer(
                            prompt,
                            max_length=self.max_seq_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt',
                        )
                        samples.append({
                            'input_ids': encoding['input_ids'].squeeze(0),
                            'attention_mask': encoding['attention_mask'].squeeze(0),
                            'prompt_text': prompt,
                            'context': human_turn,
                        })
                
                if len(samples) >= max_samples:
                    break
            
            # Split for validation
            if split == "validation":
                samples = samples[:int(len(samples) * 0.1)]
            elif split == "train":
                samples = samples[int(len(samples) * 0.1):]
            
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load PPO dataset: {e}")
            return self._generate_synthetic_data(split, max_samples)
    
    def _generate_synthetic_data(
        self,
        split: str,
        max_samples: int,
    ) -> List[Dict[str, Any]]:
        """Generate synthetic PPO data."""
        contexts = [
            "quantum computing", "machine learning", "climate change",
            "space exploration", "renewable energy", "artificial intelligence",
        ]
        
        num_samples = min(max_samples, 100 if split == "train" else 10)
        samples = []
        
        for _ in range(num_samples):
            context = random.choice(contexts)
            prompt = f"Human: Tell me about {context}.\n\nAssistant:"
            
            encoding = self.tokenizer(
                prompt,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            
            samples.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'prompt_text': prompt,
                'context': context,
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


# =============================================================================
# Collate Functions
# =============================================================================

def dpo_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate DPO batch with variable length handling."""
    # Find max lengths
    max_chosen_len = max(ex['chosen_input_ids'].shape[0] for ex in batch)
    max_rejected_len = max(ex['rejected_input_ids'].shape[0] for ex in batch)
    max_len = max(max_chosen_len, max_rejected_len)
    
    result = {
        'chosen_input_ids': [],
        'rejected_input_ids': [],
        'chosen_labels': [],
        'rejected_labels': [],
        'chosen_attention_mask': [],
        'rejected_attention_mask': [],
    }
    
    pad_token_id = batch[0]['chosen_input_ids'][0].item()  # Fallback
    
    for ex in batch:
        for key in result.keys():
            tensor = ex[key]
            pad_len = max_len - tensor.shape[0]
            if pad_len > 0:
                if 'labels' in key:
                    pad_value = -100
                elif 'mask' in key:
                    pad_value = 0
                else:
                    pad_value = pad_token_id
                tensor = torch.cat([
                    tensor,
                    torch.full((pad_len,), pad_value, dtype=tensor.dtype)
                ])
            result[key].append(tensor)
    
    return {k: torch.stack(v) for k, v in result.items()}


def rl_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate GRPO/PPO batch."""
    result = {}
    
    tensor_keys = [k for k in batch[0].keys() if isinstance(batch[0][k], torch.Tensor)]
    
    for key in tensor_keys:
        result[key] = torch.stack([ex[key] for ex in batch])
    
    for key in batch[0].keys():
        if key not in tensor_keys:
            result[key] = [ex[key] for ex in batch]
    
    return result


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_rl_dataloaders(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    algorithm: str = "grpo",  # "dpo", "grpo", "ppo"
    batch_size: int = 4,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create RL DataLoaders for specific algorithm.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer
        algorithm: RL algorithm type
        batch_size: Batch size
        max_samples: Max samples
        
    Returns:
        train_loader, val_loader
    """
    # Select dataset class
    if algorithm == "dpo":
        DatasetClass = DPODataset
        collate = dpo_collate_fn
    elif algorithm == "grpo":
        DatasetClass = GRPODataset
        collate = rl_collate_fn
    elif algorithm == "ppo":
        DatasetClass = PPODataset
        collate = rl_collate_fn
    else:
        raise ValueError(f"Unknown RL algorithm: {algorithm}")
    
    # Create datasets
    train_dataset = DatasetClass(
        tokenizer=tokenizer,
        config=config,
        split="train",
        max_samples=max_samples,
    )
    
    val_dataset = DatasetClass(
        tokenizer=tokenizer,
        config=config,
        split="validation",
        max_samples=max_samples // 10 if max_samples else None,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(config.num_workers, 2),
        pin_memory=config.pin_memory,
        collate_fn=collate,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(config.num_workers, 2),
        pin_memory=config.pin_memory,
        collate_fn=collate,
    )
    
    return train_loader, val_loader


# =============================================================================
# Test
# =============================================================================

def test_rl_datasets():
    """Test all RL dataset classes."""
    from config import load_config
    from dataset import get_tokenizer
    
    logger.info("=" * 70)
    logger.info("Testing RL Datasets")
    logger.info("=" * 70)
    
    config = load_config()
    tokenizer = get_tokenizer(config.data)
    
    # Test DPO
    logger.info("-" * 70)
    logger.info("Testing DPODataset...")
    dpo_train, dpo_val = create_rl_dataloaders(
        config.data, tokenizer, algorithm="dpo",
        batch_size=2, max_samples=20,
    )
    batch = next(iter(dpo_train))
    logger.info(f"  chosen_input_ids shape: {batch['chosen_input_ids'].shape}")
    logger.info(f"  rejected_input_ids shape: {batch['rejected_input_ids'].shape}")
    
    # Test GRPO
    logger.info("-" * 70)
    logger.info("Testing GRPODataset...")
    grpo_train, grpo_val = create_rl_dataloaders(
        config.data, tokenizer, algorithm="grpo",
        batch_size=2, max_samples=20,
    )
    batch = next(iter(grpo_train))
    logger.info(f"  input_ids shape: {batch['input_ids'].shape}")
    logger.info(f"  prompt_text: {batch['prompt_text'][0][:50]}...")
    
    # Test PPO
    logger.info("-" * 70)
    logger.info("Testing PPODataset...")
    ppo_train, ppo_val = create_rl_dataloaders(
        config.data, tokenizer, algorithm="ppo",
        batch_size=2, max_samples=20,
    )
    batch = next(iter(ppo_train))
    logger.info(f"  input_ids shape: {batch['input_ids'].shape}")
    
    logger.info("=" * 70)
    logger.info("All RL dataset tests passed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    test_rl_datasets()
