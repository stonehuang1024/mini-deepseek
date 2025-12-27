"""
DeepSeek V3 Web Chat Interface
==============================

Flask-based web chat interface with:
- Model switching from checkpoints directory
- Streaming token output
- ChatGPT-style UI
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, Generator, Optional
from queue import Queue

from flask import Flask, render_template, request, jsonify, Response, stream_with_context

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config, get_device
from deepseek.inference.inference import DeepSeekInference, load_model_for_inference

app = Flask(__name__)

# Global state
models_cache: Dict[str, DeepSeekInference] = {}
models_lock = threading.Lock()
model_loading_status: Dict[str, str] = {}

# Generation task management
generation_tasks: Dict[str, bool] = {}  # task_id -> should_cancel
generation_tasks_lock = threading.Lock()


def list_checkpoints(checkpoints_dir: str = "checkpoints") -> Dict[str, dict]:
    """
    List all available model checkpoints.
    
    Returns:
        Dict mapping model names to metadata
    """
    checkpoints_path = PROJECT_ROOT / checkpoints_dir
    if not checkpoints_path.exists():
        return {}
    
    models = {}
    
    for root, dirs, files in os.walk(checkpoints_path):
        for file in files:
            if file.endswith('.pt'):
                # Create relative path from checkpoints directory
                full_path = Path(root) / file
                model_name = str(full_path.relative_to(checkpoints_path)).replace('\\', '/')
                
                # Get file size and modification time
                stat = full_path.stat()
                
                # Determine model type from path
                model_type = "unknown"
                if "pretrain" in model_name.lower():
                    model_type = "Pretrained"
                elif "sft" in model_name.lower():
                    model_type = "SFT"
                elif "rl" in model_name.lower() or "grpo" in model_name.lower() or "ppo" in model_name.lower() or "dpo" in model_name.lower():
                    model_type = "RL"
                
                models[model_name] = {
                    "path": str(full_path),
                    "name": model_name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                    "type": model_type
                }
    
    return models


def load_model(model_name: str) -> DeepSeekInference:
    """
    Load a model with caching.
    
    Args:
        model_name: Path to model checkpoint (relative to checkpoints)
        
    Returns:
        DeepSeekInference instance
    """
    global models_cache, model_loading_status
    
    with models_lock:
        # Check if already loading
        if model_name in model_loading_status and model_loading_status[model_name] == "loading":
            raise ValueError(f"Model {model_name} is already loading. Please wait.")
        
        # Check if already loaded
        if model_name in models_cache:
            return models_cache[model_name]
        
        # Set loading status
        model_loading_status[model_name] = "loading"
    
    try:
        # Build full path - model_name is relative to checkpoints directory
        full_path = PROJECT_ROOT / "checkpoints" / model_name
        
        if not full_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {full_path}")
        
        print(f"Loading model from: {full_path}")
        
        # Load model
        model, tokenizer, config = load_model_for_inference(str(full_path))
        
        print(f"Model loaded successfully. Creating inference wrapper...")
        
        # Create inference wrapper
        inference = DeepSeekInference(
            model=model,
            tokenizer=tokenizer,
            config=config.inference,
            device="auto"
        )
        
        print(f"Inference wrapper created. Caching model...")
        
        # Cache the model
        with models_lock:
            models_cache[model_name] = inference
            model_loading_status[model_name] = "loaded"
        
        print(f"Model {model_name} loaded and cached successfully")
        
        return inference
    
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        with models_lock:
            model_loading_status[model_name] = "error"
        raise e


def unload_model(model_name: str):
    """Unload a model from cache to free memory."""
    global models_cache, model_loading_status
    
    with models_lock:
        if model_name in models_cache:
            del models_cache[model_name]
        if model_name in model_loading_status:
            del model_loading_status[model_name]


def generate_stream(
    inference: DeepSeekInference,
    prompt: str,
    task_id: str,
    **kwargs
) -> Generator[str, None, None]:
    """
    Stream text generation.
    
    Args:
        inference: DeepSeekInference instance
        prompt: Input prompt
        task_id: Unique identifier for this generation task
        **kwargs: Generation parameters
        
    Yields:
        Generated tokens
    """
    import torch
    import torch.nn.functional as F
    
    device = inference.device
    tokenizer = inference.tokenizer
    
    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    except Exception as e:
        yield json.dumps({"error": f"Tokenization failed: {str(e)}", "done": True}) + "\n"
        return
    
    # Generation parameters
    max_new_tokens = kwargs.get('max_new_tokens', 512)
    temperature = kwargs.get('temperature', 0.7)
    top_p = kwargs.get('top_p', 0.9)
    top_k = kwargs.get('top_k', 50)
    do_sample = kwargs.get('do_sample', True)
    repetition_penalty = kwargs.get('repetition_penalty', 1.1)
    
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
    
    generated = input_ids
    past_key_values = None
    response_tokens = []
    
    def sample_token(logits, temperature, top_p, top_k, do_sample):
        """Sample a single token from logits."""
        # Temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        
        if not do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)
        
        # Top-K filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, indices, values)
        
        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep first token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    try:
        for step in range(max_new_tokens):
            # Check for cancellation
            with generation_tasks_lock:
                if task_id in generation_tasks and generation_tasks[task_id]:
                    yield json.dumps({"token": "", "done": True, "cancelled": True}) + "\n"
                    return
            
            # Forward pass
            if past_key_values is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated
            
            outputs = inference.model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs['logits'][:, -1, :]
            past_key_values = outputs['past_key_values']
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Sample next token
            next_token = sample_token(
                logits, temperature, top_p, top_k, do_sample
            )
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            response_tokens.append(next_token[0, 0].item())
            
            # Check for EOS
            if eos_token_id is not None and next_token[0, 0].item() == eos_token_id:
                break
            
            # Decode and yield tokens
            full_response = tokenizer.decode(
                torch.tensor(response_tokens), 
                skip_special_tokens=True
            )
            
            # Extract assistant's response
            if "Assistant:" in full_response:
                assistant_response = full_response.split("Assistant:")[-1]
            else:
                assistant_response = full_response
            
            # Stream only new text
            if not hasattr(inference, '_last_streamed_response'):
                inference._last_streamed_response = ""
            
            new_text = assistant_response[len(inference._last_streamed_response):]
            if new_text:
                yield json.dumps({"token": new_text, "done": False}) + "\n"
            
            inference._last_streamed_response = assistant_response
        
        # Reset state
        if hasattr(inference, '_last_streamed_response'):
            del inference._last_streamed_response
        
        # Signal completion
        yield json.dumps({"token": "", "done": True}) + "\n"
    
    except Exception as e:
        import traceback
        error_msg = f"Generation error: {str(e)}"
        print(f"Error in generate_stream: {error_msg}")
        traceback.print_exc()
        yield json.dumps({"error": error_msg, "done": True}) + "\n"


@app.route('/')
def index():
    """Serve the main chat page."""
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    models = list_checkpoints()
    return jsonify({
        "models": models,
        "count": len(models)
    })


@app.route('/api/models/status', methods=['GET'])
def get_model_status():
    """Get loading status of a model."""
    model_name = request.args.get('model')
    
    if not model_name:
        return jsonify({"error": "Model name is required"}), 400
    
    with models_lock:
        status = model_loading_status.get(model_name, "not_loaded")
        loaded = model_name in models_cache
    return jsonify({
        "name": model_name,
        "status": status,
        "loaded": loaded
    })


@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate text with streaming."""
    data = request.json
    
    model_name = data.get('model')
    message = data.get('message', '')
    history = data.get('history', [])
    
    # Generation parameters
    max_new_tokens = data.get('max_new_tokens', 512)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    top_k = data.get('top_k', 50)
    do_sample = data.get('do_sample', True)
    repetition_penalty = data.get('repetition_penalty', 1.1)
    
    if not model_name:
        return jsonify({"error": "Model name is required"}), 400
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        # Load model
        inference = load_model(model_name)
        
        # Build conversation prompt
        prompt_parts = []
        
        if history:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    prompt_parts.append(f"Human: {content}\n\n")
                else:
                    prompt_parts.append(f"Assistant: {content}\n\n")
        
        prompt_parts.append(f"Human: {message}\n\nAssistant:")
        full_prompt = "".join(prompt_parts)
        
        # Generate unique task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Register task
        with generation_tasks_lock:
            generation_tasks[task_id] = False
        
        # Stream generation
        def generate():
            try:
                for token in generate_stream(
                    inference, full_prompt, task_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty
                ):
                    yield token
            except GeneratorExit:
                # Generator was closed
                pass
            except Exception as e:
                # Stream the error to the client
                error_msg = str(e)
                import traceback
                traceback.print_exc()
                yield json.dumps({"error": error_msg, "done": True}) + "\n"
            finally:
                # Clean up task
                with generation_tasks_lock:
                    if task_id in generation_tasks:
                        del generation_tasks[task_id]
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'X-Task-ID': task_id
            }
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear chat history (no-op for now, handled client-side)."""
    return jsonify({"success": True})


@app.route('/api/generate/cancel', methods=['POST'])
def cancel_generation():
    """Cancel an ongoing generation."""
    data = request.json
    task_id = data.get('task_id')
    
    if not task_id:
        return jsonify({"error": "Task ID is required"}), 400
    
    with generation_tasks_lock:
        if task_id in generation_tasks:
            generation_tasks[task_id] = True
            return jsonify({"success": True, "message": "Generation cancelled"})
        else:
            return jsonify({"error": "Task not found"}), 404


@app.route('/api/models/unload', methods=['POST'])
def unload_model_endpoint():
    """Unload a model from memory."""
    model_name = request.json.get('model') if request.json else request.args.get('model')
    
    if not model_name:
        return jsonify({"error": "Model name is required"}), 400
    
    unload_model(model_name)
    return jsonify({"success": True, "message": f"Model {model_name} unloaded"})


if __name__ == '__main__':
    print("=" * 60)
    print("DeepSeek V3 Web Chat Interface")
    print("=" * 60)
    print(f"Server starting at http://localhost:5001")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)
    
    # List available models
    models = list_checkpoints()
    print(f"\nAvailable models: {len(models)}")
    for name, info in list(models.items())[:5]:
        print(f"  - {name} ({info.get('size_mb', 0)} MB)")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")
    
    # Create templates directory if not exists
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Run app
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
