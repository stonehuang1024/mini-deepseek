"""
DeepSeek V3 Dataset Module
==========================

Handles data loading and preprocessing for:
1. Pretraining: WikiText-2 dataset
2. SFT: Alpaca instruction-following dataset
3. RL: HH-RLHF prompts dataset

All datasets are automatically downloaded and cached.
"""

import os
import json
import random
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from deepseek.utils import get_logger

# Use local cache if exists, avoid downloading every time
# REUSE_DATASET_IF_EXISTS: Skip download and processing if cached dataset exists
DOWNLOAD_MODE = "reuse_dataset_if_exists"

# Import config - support both package and standalone usage
try:
    from config import DataConfig
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import DataConfig

# Initialize logger
logger = get_logger(__name__)


# =============================================================================
# Tokenizer Management
# =============================================================================

def get_tokenizer(config: DataConfig) -> PreTrainedTokenizer:
    """
    Load and configure tokenizer.
    
    Args:
        config: DataConfig with tokenizer settings
        
    Returns:
        Configured tokenizer
    """
    import os
    
    # Try to load from HuggingFace with local caching
    try:
        # Set local cache directory
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "tokenizer")
        os.makedirs(cache_dir, exist_ok=True)
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            trust_remote_code=True,
            local_files_only=False,
            cache_dir=cache_dir,
        )
    except Exception as e:
        # If network fails, try local_only mode
        logger.warning(f"Failed to load tokenizer from HuggingFace: {e}")
        logger.info("Attempting to load from local cache...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e2:
            logger.error(f"Failed to load tokenizer from local cache: {e2}")
            raise RuntimeError(
                f"Failed to load tokenizer '{config.tokenizer_name}'. "
                f"Please ensure you have internet connection to download it, "
                f"or use a pre-downloaded tokenizer."
            ) from e
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


# =============================================================================
# Pretrain Dataset
# =============================================================================

class PretrainDataset(Dataset):
    """
    Dataset for pretraining on raw text.
    
    Uses OpenWebText or WikiText for language modeling.
    Chunks text into sequences of max_seq_length.
    
    Supports:
    - OpenWebText (~40GB) - Large scale pretraining
    - WikiText-2 (~13MB) - Quick testing
    
    Shape:
        - input_ids: (L,) - token indices
        - attention_mask: (L,) - 1 for real tokens
        - labels: (L,) - same as input_ids for LM
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
            tokenizer: Tokenizer for encoding text
            config: Data configuration
            split: Dataset split ("train", "validation", "test")
            max_samples: Maximum number of samples (for debugging)
        """
        self.tokenizer = tokenizer
        self.max_seq_length = config.pretrain_max_seq_length
        self.split = split
        self.max_samples = max_samples or config.pretrain_max_samples
        
        # Load dataset
        logger.info(f"Loading {config.pretrain_dataset_name} dataset (split={split})...")
        logger.info(f"This may take a while for large datasets...")
        self.data = self._load_data(config, split)
        
        # Tokenize and chunk
        self.examples = self._prepare_examples()
        
        logger.info(f"PretrainDataset ({split}): {len(self.examples)} examples")
    
    def _load_data(self, config: DataConfig, split: str) -> List[str]:
        """Load raw text data from various sources."""
        try:
            from datasets import load_dataset
            import time
            
            start_time = time.time()
            dataset_name = config.pretrain_dataset_name
            
            # Different loading strategies for different datasets
            if dataset_name == "openwebtext":
                # OpenWebText is ~40GB, load with streaming or limit samples
                logger.info(f"Loading OpenWebText dataset...")
                logger.info(f"Max samples: {self.max_samples}")
                
                if config.pretrain_streaming:
                    # Streaming mode for very large datasets
                    dataset = load_dataset(
                        "openwebtext",
                        split=split if split != "validation" else "train",
                        streaming=True,
                        cache_dir=config.pretrain_data_dir,
                        download_mode=DOWNLOAD_MODE,
                    )
                    texts = []
                    for i, item in enumerate(dataset):
                        if self.max_samples and i >= self.max_samples:
                            break
                        texts.append(item["text"].strip())
                        if (i + 1) % 100000 == 0:
                            logger.info(f"  Loaded {i + 1} samples...")
                else:
                    # Non-streaming mode
                    dataset = load_dataset(
                        "openwebtext",
                        split="train",  # OpenWebText only has train split
                        cache_dir=config.pretrain_data_dir,
                        download_mode=DOWNLOAD_MODE,
                    )
                    # For validation, use last 10% of data
                    total_size = len(dataset)
                    if split == "validation":
                        start_idx = int(total_size * 0.9)
                        indices = range(start_idx, min(start_idx + (self.max_samples or total_size) // 10, total_size))
                    else:
                        # Train uses first 90%
                        end_idx = min(self.max_samples or total_size, int(total_size * 0.9))
                        indices = range(0, end_idx)
                    
                    texts = []
                    for i in indices:
                        texts.append(dataset[i]["text"].strip())
                        if (len(texts)) % 100000 == 0:
                            logger.info(f"  Loaded {len(texts)} samples...")
                            
            elif dataset_name == "wikitext":
                # WikiText is small (~13MB)
                dataset = load_dataset(
                    config.pretrain_dataset_name,
                    config.pretrain_dataset_config,
                    cache_dir=config.pretrain_data_dir,
                    download_mode=DOWNLOAD_MODE,
                )
                
                if split == "train":
                    texts = dataset["train"]["text"]
                elif split == "validation":
                    texts = dataset["validation"]["text"]
                else:
                    texts = dataset["test"]["text"]
                
                texts = [t.strip() for t in texts if t.strip()]
                
            else:
                # Generic loading for other datasets
                dataset = load_dataset(
                    dataset_name,
                    config.pretrain_dataset_config if config.pretrain_dataset_config else None,
                    cache_dir=config.pretrain_data_dir,
                    download_mode=DOWNLOAD_MODE,
                )
                
                if split in dataset:
                    data_split = dataset[split]
                else:
                    data_split = dataset["train"]
                    
                # Try common text field names
                text_field = None
                for field in ["text", "content", "document", "sentence"]:
                    if field in data_split.features:
                        text_field = field
                        break
                
                if text_field is None:
                    text_field = list(data_split.features.keys())[0]
                    
                texts = []
                for i, item in enumerate(data_split):
                    if self.max_samples and i >= self.max_samples:
                        break
                    text = str(item[text_field]).strip()
                    if text:
                        texts.append(text)
            
            elapsed = time.time() - start_time
            logger.info(f"  Dataset loaded in {elapsed:.1f}s, {len(texts)} texts")
            
            # Filter empty lines
            texts = [t for t in texts if t]
            return texts
            
        except Exception as e:
            logger.error(f"Failed to load dataset from HuggingFace: {e}")
            logger.warning("Generating synthetic data for demonstration...")
            return self._generate_synthetic_data(split)
    
    def _generate_synthetic_data(self, split: str) -> List[str]:
        """Generate synthetic data for testing."""
        templates = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was darkness and then there was light.",
            "Machine learning is transforming the way we process information.",
            "The history of computing dates back to the early calculators.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models have achieved remarkable results in various tasks.",
            "The transformer architecture revolutionized natural language processing.",
            "Attention mechanisms allow models to focus on relevant parts of input.",
            "Large language models are trained on vast amounts of text data.",
            "The development of AI has accelerated in recent years.",
        ]
        
        num_samples = 5000 if split == "train" else 500
        data = []
        for _ in range(num_samples):
            # Combine random templates
            num_sentences = random.randint(3, 10)
            text = " ".join(random.choices(templates, k=num_sentences))
            data.append(text)
        
        return data
    
    def _prepare_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Tokenize and chunk text into examples with progress tracking."""
        import time
        
        logger.info(f"Tokenizing and chunking data...")
        start_time = time.time()
        
        # Process in batches to show progress
        batch_size = 1000
        all_tokens = []
        
        for i in range(0, len(self.data), batch_size):
            batch = self.data[i:i + batch_size]
            batch_text = " ".join(batch)
            tokens = self.tokenizer.encode(batch_text, add_special_tokens=False)
            all_tokens.extend(tokens)
            
            if (i + batch_size) % 50000 == 0 or i + batch_size >= len(self.data):
                elapsed = time.time() - start_time
                logger.info(f"  Processed {min(i + batch_size, len(self.data))}/{len(self.data)} texts, "
                      f"{len(all_tokens)} tokens, {elapsed:.1f}s")
        
        # Chunk into sequences
        examples = []
        max_examples = self.max_samples if self.max_samples else float('inf')
        
        for i in range(0, len(all_tokens) - self.max_seq_length, self.max_seq_length):
            chunk = all_tokens[i:i + self.max_seq_length]
            
            examples.append({
                'input_ids': torch.tensor(chunk, dtype=torch.long),
                'attention_mask': torch.ones(len(chunk), dtype=torch.long),
                'labels': torch.tensor(chunk, dtype=torch.long),
            })
            
            if len(examples) >= max_examples:
                break
        
        elapsed = time.time() - start_time
        total_tokens = len(examples) * self.max_seq_length
        logger.info(f"  Created {len(examples)} examples ({total_tokens:,} tokens) in {elapsed:.1f}s")
        logger.info(f"  Estimated data size: {total_tokens * 2 / 1024 / 1024 / 1024:.2f} GB (tokens * 2 bytes)")
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# =============================================================================
# SFT Dataset
# =============================================================================

@dataclass
class SFTExample:
    """Single SFT example with instruction and response."""
    instruction: str
    input: str
    output: str


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.
    
    Uses Alpaca-format instruction data with:
    - instruction: Task description
    - input: Optional context
    - output: Expected response
    
    Formats as: "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    
    Shape:
        - input_ids: (L,) - token indices
        - attention_mask: (L,) - 1 for real tokens
        - labels: (L,) - -100 for prompt tokens, token ids for response
    """
    
    # Prompt template
    PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
    
    PROMPT_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""
    
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
            max_samples: Maximum samples to use
        """
        self.tokenizer = tokenizer
        self.max_seq_length = config.sft_max_seq_length
        self.split = split
        max_samples = max_samples or config.sft_max_samples
        
        # Load dataset
        self.examples = self._load_data(config, split, max_samples)
        
        logger.info(f"SFTDataset ({split}): {len(self.examples)} examples")
    
    def _load_data(
        self, 
        config: DataConfig, 
        split: str,
        max_samples: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Load and process SFT data."""
        try:
            from datasets import load_dataset
            
            # Load Alpaca dataset
            dataset = load_dataset(
                config.sft_dataset_name,
                cache_dir=config.sft_data_dir,
                download_mode=DOWNLOAD_MODE,
            )
            
            # Get data
            if "train" in dataset:
                data = dataset["train"]
            else:
                data = list(dataset.values())[0]
            
            # Process examples
            examples = []
            for i, item in enumerate(data):
                if i >= max_samples:
                    break
                
                example = self._process_example(
                    instruction=item.get("instruction", ""),
                    input_text=item.get("input", ""),
                    output=item.get("output", ""),
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
            logger.error(f"Failed to load SFT dataset: {e}")
            logger.warning("Generating synthetic SFT data...")
            return self._generate_synthetic_data(split, max_samples)
    
    def _process_example(
        self,
        instruction: str,
        input_text: str,
        output: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Process single example into tokenized format."""
        # Format prompt
        if input_text.strip():
            prompt = self.PROMPT_TEMPLATE.format(
                instruction=instruction,
                input=input_text,
                output="",
            )
            full_text = self.PROMPT_TEMPLATE.format(
                instruction=instruction,
                input=input_text,
                output=output,
            )
        else:
            prompt = self.PROMPT_TEMPLATE_NO_INPUT.format(
                instruction=instruction,
                output="",
            )
            full_text = self.PROMPT_TEMPLATE_NO_INPUT.format(
                instruction=instruction,
                output=output,
            )
        
        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
        
        # Truncate if necessary
        if len(full_ids) > self.max_seq_length:
            full_ids = full_ids[:self.max_seq_length]
            # Adjust prompt length if it exceeds full length
            prompt_ids = prompt_ids[:min(len(prompt_ids), len(full_ids))]
        
        # Create labels: -100 for prompt (ignored in loss), token ids for response
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]
        
        # Pad to max length
        padding_length = self.max_seq_length - len(full_ids)
        if padding_length > 0:
            full_ids = full_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
            attention_mask = [1] * (self.max_seq_length - padding_length) + [0] * padding_length
        else:
            attention_mask = [1] * len(full_ids)
        
        return {
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }
    
    def _generate_synthetic_data(
        self, 
        split: str, 
        max_samples: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate synthetic SFT data."""
        instructions = [
            ("Write a short story about a robot.", "", "Once upon a time, there was a robot named R2 who dreamed of exploring the stars."),
            ("Summarize the following text.", "Machine learning is a subset of AI.", "ML is part of artificial intelligence."),
            ("Translate to French.", "Hello, how are you?", "Bonjour, comment allez-vous?"),
            ("Explain quantum computing.", "", "Quantum computing uses quantum bits or qubits that can exist in multiple states."),
            ("Write a poem about nature.", "", "The trees sway gently in the breeze,\nWhile birds sing songs with ease."),
            ("Calculate the result.", "What is 15 + 27?", "The result is 42."),
            ("Correct the grammar.", "She don't like apples.", "She doesn't like apples."),
            ("List three benefits of exercise.", "", "1. Improved cardiovascular health\n2. Better mental well-being\n3. Increased energy levels"),
        ]
        
        num_samples = min(max_samples, 1000 if split == "train" else 100)
        examples = []
        
        for i in range(num_samples):
            instr, inp, out = random.choice(instructions)
            example = self._process_example(instr, inp, out)
            if example is not None:
                examples.append(example)
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


# =============================================================================
# RL Dataset
# =============================================================================

class RLDataset(Dataset):
    """
    Dataset for Reinforcement Learning (GRPO).
    
    Uses HH-RLHF prompts for generating responses.
    Only contains prompts, responses are generated during RL training.
    
    Shape:
        - input_ids: (L,) - tokenized prompt
        - attention_mask: (L,) - 1 for real tokens
        - prompt_text: str - original prompt text (for generation)
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
            max_samples: Maximum samples
        """
        self.tokenizer = tokenizer
        self.max_seq_length = config.rl_max_seq_length
        self.split = split
        max_samples = max_samples or config.rl_max_samples
        
        # Load dataset
        self.prompts, self.tokenized = self._load_data(config, split, max_samples)
        
        logger.info(f"RLDataset ({split}): {len(self.prompts)} prompts")
    
    def _load_data(
        self,
        config: DataConfig,
        split: str,
        max_samples: int,
    ) -> Tuple[List[str], List[Dict[str, torch.Tensor]]]:
        """Load RL prompts."""
        try:
            from datasets import load_dataset
            
            # Load HH-RLHF dataset
            dataset = load_dataset(
                config.rl_dataset_name,
                config.rl_dataset_config,
                cache_dir=config.rl_data_dir,
                download_mode=DOWNLOAD_MODE,
            )
            
            # Get split
            if split in dataset:
                data = dataset[split]
            else:
                data = dataset["train"]
            
            # Extract prompts (human turns from conversations)
            prompts = []
            for item in data:
                # HH-RLHF format: "Human: ... Assistant: ..."
                chosen = item.get("chosen", "")
                if "Human:" in chosen:
                    # Extract first human turn
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
            
            # Tokenize
            tokenized = self._tokenize_prompts(prompts)
            
            return prompts, tokenized
            
        except Exception as e:
            logger.error(f"Failed to load RL dataset: {e}")
            logger.warning("Generating synthetic RL prompts...")
            return self._generate_synthetic_prompts(split, max_samples)
    
    def _tokenize_prompts(
        self, 
        prompts: List[str],
    ) -> List[Dict[str, torch.Tensor]]:
        """Tokenize prompts for model input."""
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
        """Generate synthetic RL prompts."""
        prompt_templates = [
            "Human: Can you explain what machine learning is?\n\nAssistant:",
            "Human: Write a short poem about nature.\n\nAssistant:",
            "Human: What are the benefits of exercise?\n\nAssistant:",
            "Human: How do computers work?\n\nAssistant:",
            "Human: Tell me about space exploration.\n\nAssistant:",
            "Human: What is the meaning of life?\n\nAssistant:",
            "Human: Explain quantum physics simply.\n\nAssistant:",
            "Human: What makes a good leader?\n\nAssistant:",
            "Human: How can I learn to code?\n\nAssistant:",
            "Human: Describe a perfect day.\n\nAssistant:",
        ]
        
        num_samples = min(max_samples, 200 if split == "train" else 20)
        prompts = [random.choice(prompt_templates) for _ in range(num_samples)]
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
# Data Collator
# =============================================================================

def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collate batch of examples.
    
    Args:
        batch: List of example dicts
        
    Returns:
        Batched tensors
    """
    # Stack tensors
    result = {}
    
    # Get all keys except non-tensor fields
    tensor_keys = [k for k in batch[0].keys() if isinstance(batch[0][k], torch.Tensor)]
    
    for key in tensor_keys:
        result[key] = torch.stack([ex[key] for ex in batch])
    
    # Handle non-tensor fields (like prompt_text)
    for key in batch[0].keys():
        if key not in tensor_keys:
            result[key] = [ex[key] for ex in batch]
    
    return result


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_dataloaders(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    mode: str = "pretrain",  # "pretrain", "sft", or "rl"
    batch_size: int = 16,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer
        mode: Training mode
        batch_size: Batch size
        max_samples: Max samples per split
        
    Returns:
        train_loader, val_loader
    """
    # Select dataset class
    if mode == "pretrain":
        DatasetClass = PretrainDataset
    elif mode == "sft":
        DatasetClass = SFTDataset
    elif mode == "rl":
        DatasetClass = RLDataset
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
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
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader


# =============================================================================
# Test Functions
# =============================================================================

def test_datasets():
    """Test all dataset classes."""
    from config import load_config
    
    logger.info("=" * 70)
    logger.info("Testing DeepSeek V3 Datasets")
    logger.info("=" * 70)
    
    # Load config and tokenizer
    config = load_config()
    tokenizer = get_tokenizer(config.data)
    
    logger.info(f"Tokenizer: {config.data.tokenizer_name}")
    logger.info(f"Vocab size: {len(tokenizer)}")
    logger.info(f"Pad token: {tokenizer.pad_token}")
    
    # Test pretrain dataset
    logger.info("-" * 70)
    logger.info("Testing PretrainDataset...")
    pretrain_train, pretrain_val = create_dataloaders(
        config.data, tokenizer, mode="pretrain",
        batch_size=4, max_samples=100,
    )
    
    batch = next(iter(pretrain_train))
    logger.info(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    logger.info(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")
    logger.info(f"  Batch labels shape: {batch['labels'].shape}")
    logger.info(f"  Sample text: {tokenizer.decode(batch['input_ids'][0][:50])}...")
    
    # Test SFT dataset
    logger.info("-" * 70)
    logger.info("Testing SFTDataset...")
    sft_train, sft_val = create_dataloaders(
        config.data, tokenizer, mode="sft",
        batch_size=4, max_samples=100,
    )
    
    batch = next(iter(sft_train))
    logger.info(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    logger.info(f"  Sample instruction: {tokenizer.decode(batch['input_ids'][0][:100])}...")
    
    # Test RL dataset
    logger.info("-" * 70)
    logger.info("Testing RLDataset...")
    rl_train, rl_val = create_dataloaders(
        config.data, tokenizer, mode="rl",
        batch_size=4, max_samples=50,
    )
    
    batch = next(iter(rl_train))
    logger.info(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    logger.info(f"  Sample prompt: {batch['prompt_text'][0][:100]}...")
    
    logger.info("=" * 70)
    logger.info("All dataset tests passed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    test_datasets()
