#!/usr/bin/env python3
"""
DeepSeek V3 Web Chat Interface
==============================

A ChatGPT-style web interface for the DeepSeek V3 model.

Features:
- Model selection from checkpoints directory
- Streaming token generation (SSE)
- Cancel generation support
- Dark theme UI

Usage:
    python chat/app.py
    # or
    ./scripts/run.sh web-chat
"""

import os
import sys
import json
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, Response, jsonify, stream_with_context

import torch
import torch.nn.functional as F

from config import load_config, get_device
from deepseek.model import DeepSeekV3Model
from deepseek.data import get_tokenizer
from deepseek.utils import get_logger

logger = get_logger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global state
_model_cache: Dict[str, Any] = {}
_model_lock = threading.Lock()
_current_model_name: Optional[str] = None
_cancel_generation = threading.Event()
_generation_in_progress = threading.Event()


@dataclass
class CheckpointInfo:
    """Information about a checkpoint file."""
    name: str
    path: str
    size_mb: float
    category: str
    modified_time: float


def list_checkpoints(checkpoints_dir: str = "checkpoints") -> List[CheckpointInfo]:
    """
    Discover all checkpoint files in the checkpoints directory.
    
    Returns:
        List of CheckpointInfo objects sorted by category and name
    """
    checkpoints = []
    base_path = PROJECT_ROOT / checkpoints_dir
    
    if not base_path.exists():
        logger.warning(f"Checkpoints directory not found: {base_path}")
        return checkpoints
    
    for pt_file in base_path.rglob("*.pt"):
        rel_path = pt_file.relative_to(base_path)
        parts = rel_path.parts
        
        # Determine category from directory structure
        if len(parts) > 1:
            category = parts[0].upper()
        else:
            category = "OTHER"
        
        # Map category names
        category_map = {
            "PRETRAIN": "Pretrained",
            "SFT": "SFT",
            "RL": "RL",
            "GRPO": "RL-GRPO",
            "PPO": "RL-PPO",
            "DPO": "RL-DPO",
        }
        category = category_map.get(category, category)
        
        size_mb = pt_file.stat().st_size / (1024 * 1024)
        modified_time = pt_file.stat().st_mtime
        
        checkpoints.append(CheckpointInfo(
            name=str(rel_path),
            path=str(pt_file),
            size_mb=size_mb,
            category=category,
            modified_time=modified_time,
        ))
    
    # Sort by category, then by name
    checkpoints.sort(key=lambda x: (x.category, x.name))
    return checkpoints


def get_default_checkpoint() -> Optional[str]:
    """Get the default checkpoint path (prefer pretrain/final.pt)."""
    checkpoints = list_checkpoints()
    
    if not checkpoints:
        return None
    
    # Prefer pretrain checkpoints
    for ckpt in checkpoints:
        if "pretrain" in ckpt.name.lower() and "final" in ckpt.name.lower():
            return ckpt.path
    
    for ckpt in checkpoints:
        if "pretrain" in ckpt.name.lower() and "best" in ckpt.name.lower():
            return ckpt.path
    
    # Return first available
    return checkpoints[0].path


def load_model(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a model from checkpoint with caching.
    
    Returns:
        Dict with 'model', 'tokenizer', 'config'
    """
    global _current_model_name
    
    with _model_lock:
        if checkpoint_path in _model_cache:
            _current_model_name = checkpoint_path
            logger.info(f"Using cached model: {checkpoint_path}")
            return _model_cache[checkpoint_path]
        
        logger.info(f"Loading model from: {checkpoint_path}")
        
        # Load config
        config = load_config(str(PROJECT_ROOT / "configs" / "config_default.yaml"))
        
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
            logger.warning(f"Checkpoint not found: {checkpoint_path}, using random weights")
        
        # Move to device - use CPU for web server to avoid MPS threading issues
        # MPS has issues with Flask's threaded mode on macOS
        device = "cpu"
        model.to(device)
        model.eval()
        
        result = {
            'model': model,
            'tokenizer': tokenizer,
            'config': config,
            'device': device,
        }
        
        _model_cache[checkpoint_path] = result
        _current_model_name = checkpoint_path
        
        return result


def generate_stream(
    model: DeepSeekV3Model,
    tokenizer: Any,
    device: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> Generator[str, None, None]:
    """
    Stream token generation.

    Yields:
        Generated tokens one at a time
    """
    _generation_in_progress.set()
    _cancel_generation.clear()
    
    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated = input_ids
        past_key_values = None
        
        eos_token_id = tokenizer.eos_token_id
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Check for cancellation
                if _cancel_generation.is_set():
                    logger.info("Generation cancelled by user")
                    yield "[CANCELLED]"
                    break
                
                # Forward pass
                if past_key_values is not None:
                    curr_input = generated[:, -1:]
                else:
                    curr_input = generated
                
                outputs = model(
                    input_ids=curr_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs['logits'][:, -1, :]
                past_key_values = outputs['past_key_values']
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    vocab_size = logits.size(-1)
                    for token_id in set(generated[0].tolist()):
                        if 0 <= token_id < vocab_size:
                            logits[0, token_id] /= repetition_penalty
                
                # Temperature scaling
                if temperature > 0 and temperature != 1.0:
                    logits = logits / temperature
                
                # Top-K filtering
                if top_k > 0:
                    top_k_val = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k_val)[0][:, -1, None]
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
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if next_token[0, 0].item() == eos_token_id:
                    break
                
                # Decode and yield token
                token_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
                if token_text:
                    yield token_text
    
    finally:
        _generation_in_progress.clear()


# Flask Routes

@app.route('/')
def index():
    """Serve the main chat page."""
    return render_template('index.html')


@app.route('/api/checkpoints', methods=['GET'])
def api_list_checkpoints():
    """List available checkpoints."""
    checkpoints = list_checkpoints()
    default_path = get_default_checkpoint()
    
    return jsonify({
        'checkpoints': [
            {
                'name': ckpt.name,
                'path': ckpt.path,
                'size_mb': round(ckpt.size_mb, 2),
                'category': ckpt.category,
            }
            for ckpt in checkpoints
        ],
        'default': default_path,
        'current': _current_model_name,
    })


@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    """Load a specific model."""
    data = request.json
    checkpoint_path = data.get('checkpoint_path')
    
    if not checkpoint_path:
        return jsonify({'error': 'No checkpoint path provided'}), 400
    
    # Cancel any ongoing generation
    if _generation_in_progress.is_set():
        _cancel_generation.set()
        time.sleep(0.5)
    
    try:
        load_model(checkpoint_path)
        return jsonify({
            'success': True,
            'message': f'Model loaded: {checkpoint_path}',
            'current': _current_model_name,
        })
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_status', methods=['GET'])
def api_model_status():
    """Get current model status."""
    return jsonify({
        'current_model': _current_model_name,
        'is_generating': _generation_in_progress.is_set(),
        'cached_models': list(_model_cache.keys()),
    })


@app.route('/api/cancel', methods=['POST'])
def api_cancel():
    """Cancel ongoing generation."""
    if _generation_in_progress.is_set():
        _cancel_generation.set()
        return jsonify({'success': True, 'message': 'Cancellation requested'})
    else:
        return jsonify({'success': False, 'message': 'No generation in progress'})


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Generate text with streaming response."""
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Get generation parameters
    max_new_tokens = data.get('max_new_tokens', 256)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 50)
    repetition_penalty = data.get('repetition_penalty', 1.1)
    
    # Load model if not loaded
    checkpoint_path = data.get('checkpoint_path') or _current_model_name or get_default_checkpoint()
    
    if not checkpoint_path:
        return jsonify({'error': 'No checkpoint available'}), 400
    
    try:
        model_data = load_model(checkpoint_path)
    except Exception as e:
        return jsonify({'error': f'Failed to load model: {e}'}), 500
    
    def generate():
        """Generator for SSE streaming."""
        try:
            for token in generate_stream(
                model=model_data['model'],
                tokenizer=model_data['tokenizer'],
                device=model_data['device'],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            ):
                if token == "[CANCELLED]":
                    yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


def create_app():
    """Create and configure the Flask app."""
    # Create templates and static directories
    templates_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)
    
    return app


if __name__ == '__main__':
    logger.info("Starting DeepSeek V3 Web Chat Interface")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    # List available checkpoints
    checkpoints = list_checkpoints()
    if checkpoints:
        logger.info(f"Found {len(checkpoints)} checkpoint(s):")
        for ckpt in checkpoints:
            logger.info(f"  - {ckpt.name} ({ckpt.size_mb:.2f} MB) [{ckpt.category}]")
    else:
        logger.warning("No checkpoints found in checkpoints/ directory")
    
    # Pre-load default model
    default_ckpt = get_default_checkpoint()
    if default_ckpt:
        logger.info(f"Pre-loading default model: {default_ckpt}")
        try:
            load_model(default_ckpt)
        except Exception as e:
            logger.error(f"Failed to pre-load model: {e}")
    
    # Run server
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
