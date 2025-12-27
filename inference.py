#!/usr/bin/env python3
"""
DeepSeek V3 Inference Module
============================

Provides inference capabilities including:
1. Text generation (greedy, sampling, beam search)
2. MTP (Multi-Token Prediction) speculative decoding
3. Batch inference
4. Interactive chat mode

Usage:
    python inference.py --checkpoint checkpoints/sft/best.pt --prompt "Hello, world!"
    python inference.py --checkpoint checkpoints/sft/best.pt --interactive
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, InferenceConfig, get_device
from model import DeepSeekV3Model
from dataset import get_tokenizer
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class DeepSeekInference:
    """
    Inference wrapper for DeepSeek V3 model.
    
    Supports:
    - Standard autoregressive generation
    - MTP speculative decoding for faster inference
    - Various sampling strategies
    """
    
    def __init__(
        self,
        model: DeepSeekV3Model,
        tokenizer: Any,
        config: InferenceConfig,
        device: str = "auto",
    ):
        """
        Args:
            model: Trained DeepSeek V3 model
            tokenizer: Tokenizer
            config: Inference configuration
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.device = torch.device(get_device(device))
        self.model.to(self.device)
        self.model.eval()
        
        # Get special token ids
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        use_mtp: Optional[bool] = None,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-K sampling
            do_sample: Whether to sample (vs greedy)
            repetition_penalty: Penalty for repeating tokens
            use_mtp: Use MTP speculative decoding
            
        Returns:
            Generated text
        """
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        use_mtp = use_mtp if use_mtp is not None else self.config.use_mtp_decoding
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        if use_mtp and self.model.mtp_enabled:
            output_ids = self._generate_with_mtp(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            )
        else:
            output_ids = self._generate_standard(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return generated_text
    
    def _generate_standard(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Standard autoregressive generation."""
        generated = input_ids
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated
            
            outputs = self.model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs['logits'][:, -1, :]  # (B, V)
            past_key_values = outputs['past_key_values']
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Sample next token
            next_token = self._sample_token(
                logits, temperature, top_p, top_k, do_sample
            )
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if next_token[0, 0].item() == self.eos_token_id:
                break
        
        return generated
    
    def _generate_with_mtp(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """
        Generation with MTP speculative decoding.
        
        Uses MTP heads to predict multiple tokens at once,
        then verify with the main model for faster generation.
        """
        generated = input_ids
        past_key_values = None
        num_generated = 0
        
        # Number of tokens to speculate per step
        num_speculative = self.model.config.mtp.num_predict_tokens + 1
        
        while num_generated < max_new_tokens:
            # Forward pass to get main and MTP predictions
            outputs = self.model(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs['logits'][:, -1, :]  # Main prediction
            mtp_logits = outputs.get('mtp_logits', None)  # MTP predictions
            past_key_values = outputs['past_key_values']
            
            # Get candidate tokens
            candidates = []
            
            # Main token
            main_token = self._sample_token(logits, temperature, top_p, top_k, do_sample)
            candidates.append(main_token)
            
            # MTP tokens (speculative)
            if mtp_logits:
                for mtp_pred in mtp_logits:
                    mtp_token = self._sample_token(
                        mtp_pred[:, -1, :], temperature, top_p, top_k, do_sample
                    )
                    candidates.append(mtp_token)
            
            # For simplicity, accept all candidates
            # In full implementation, would verify with model
            for token in candidates:
                generated = torch.cat([generated, token], dim=1)
                num_generated += 1
                
                if token[0, 0].item() == self.eos_token_id:
                    return generated
                
                if num_generated >= max_new_tokens:
                    break
            
            # Reset cache after speculative tokens (simplified)
            past_key_values = None
        
        return generated
    
    def _sample_token(
        self,
        logits: torch.Tensor,  # (B, V)
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> torch.Tensor:
        """Sample a single token from logits."""
        # Temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        
        if not do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)
        
        # Top-K filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def chat(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Chat-style generation with conversation history.
        
        Args:
            user_message: User's input message
            history: List of {"role": "user/assistant", "content": "..."} dicts
            system_prompt: Optional system prompt
            **kwargs: Generation parameters
            
        Returns:
            Assistant's response
        """
        # Build conversation prompt
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n\n")
        
        if history:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    prompt_parts.append(f"Human: {content}\n\n")
                else:
                    prompt_parts.append(f"Assistant: {content}\n\n")
        
        prompt_parts.append(f"Human: {user_message}\n\nAssistant:")
        
        full_prompt = "".join(prompt_parts)
        
        # Generate
        response = self.generate(full_prompt, **kwargs)
        
        # Extract assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
    def interactive_chat(self, system_prompt: Optional[str] = None):
        """Run interactive chat session."""
        print("\n" + "=" * 70)
        print("DeepSeek V3 Interactive Chat")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'clear' to reset conversation history")
        print("=" * 70 + "\n")
        
        history = []
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                history = []
                print("Conversation history cleared.\n")
                continue
            
            # Generate response
            response = self.chat(
                user_message=user_input,
                history=history,
                system_prompt=system_prompt,
            )
            
            print(f"Assistant: {response}\n")
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})


def load_model_for_inference(
    checkpoint_path: str,
    config_path: Optional[str] = None,
) -> tuple:
    """
    Load model and tokenizer for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file
        
    Returns:
        model, tokenizer, config
    """
    # Load config
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config_path = Path(checkpoint_path).parent.parent / "config_default.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            from config import DeepSeekV3Config
            config = DeepSeekV3Config()
    
    # Load tokenizer
    tokenizer = get_tokenizer(config.data)
    config.model.vocab_size = len(tokenizer)
    
    # Create model
    model = DeepSeekV3Model(config.model)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Using randomly initialized model")
    
    return model, tokenizer, config


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="DeepSeek V3 Inference")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pretrain/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info("Loading model...")
    model, tokenizer, config = load_model_for_inference(
        args.checkpoint,
        args.config,
    )
    
    # Update inference config
    config.inference.max_new_tokens = args.max_new_tokens
    config.inference.temperature = args.temperature
    config.inference.top_p = args.top_p
    config.inference.device = args.device
    
    # Create inference wrapper
    inference = DeepSeekInference(
        model=model,
        tokenizer=tokenizer,
        config=config.inference,
        device=args.device,
    )
    
    # Run inference
    if args.interactive:
        inference.interactive_chat()
    elif args.prompt:
        logger.info(f"Prompt: {args.prompt}")
        logger.info("-" * 50)
        response = inference.generate(args.prompt)
        logger.info(f"Generated:\n{response}")
    else:
        # Demo generation
        prompts = [
            "The meaning of life is",
            "In a galaxy far, far away",
            "def fibonacci(n):",
            "The future of artificial intelligence",
        ]
        
        logger.info("Demo Generation:")
        logger.info("=" * 70)
        
        for prompt in prompts:
            logger.info(f"Prompt: {prompt}")
            logger.info("-" * 50)
            response = inference.generate(prompt, max_new_tokens=100)
            logger.info(f"Generated:\n{response}")


if __name__ == "__main__":
    main()
