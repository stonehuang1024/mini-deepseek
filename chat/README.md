# DeepSeek V3 Web Chat Interface

A ChatGPT-style web interface for DeepSeek V3 model inference.

## Features

- 🎨 **Modern UI**: Clean, responsive ChatGPT-style interface
- 🤖 **Model Switching**: Switch between different checkpoint models dynamically
- 💬 **Streaming Output**: Real-time token-by-token text generation
- 💬 **Chat History**: Maintain conversation context across multiple turns
- 🎛️ **Parameter Control**: Adjustable temperature, top-p, and other generation parameters

## Quick Start

### Start the Web Server

```bash
# Option 1: Using run.sh script
./scripts/run.sh web-chat

# Option 2: Direct python execution
python3 chat/app.py
```

The web interface will be available at: `http://localhost:5001`

### Using the Web Interface

1. **Select a Model**: Use the dropdown in the header to select a model from your `checkpoints/` directory
2. **Send a Message**: Type your message in the input box and press Enter or click the send button
3. **View Streaming Response**: Watch the response generate token by token
4. **Clear Chat**: Click the "Clear Chat" button to start a fresh conversation

## Model Management

### Available Models

The interface automatically discovers all `.pt` checkpoint files in the `checkpoints/` directory and its subdirectories.

Models are organized by type:
- **Pretrained**: Base pre-trained models
- **SFT**: Supervised fine-tuned models  
- **RL**: RL-aligned models (GRPO, PPO, DPO)

### Loading Models

When you select a model:
1. A loading spinner will appear while the model loads
2. The model is cached in memory after the first load
3. Subsequent switches to the same model are instant
4. To free memory, you can unload models (feature available via API)

## API Endpoints

### List Models

```bash
GET /api/models
```

Returns a list of all available models.

**Response:**
```json
{
  "models": {
    "pretrain/best.pt": {
      "path": "/path/to/checkpoints/pretrain/best.pt",
      "name": "pretrain/best.pt",
      "size_mb": 125.5,
      "modified": "2025-12-27 14:30:00",
      "type": "Pretrained"
    }
  },
  "count": 1
}
```

### Check Model Status

```bash
GET /api/models/<model_name>/status
```

Check if a model is currently loaded or loading.

**Response:**
```json
{
  "name": "pretrain/best.pt",
  "status": "loaded",
  "loaded": true
}
```

### Generate Text (Streaming)

```bash
POST /api/generate
```

Generate text with streaming output.

**Request Body:**
```json
{
  "model": "pretrain/best.pt",
  "message": "Hello, how are you?",
  "history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ],
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "do_sample": true,
  "repetition_penalty": 1.1
}
```

**Response (Streaming):**
```
data: {"token": "Hello", "done": false}
data: {"token": "!", "done": false}
data: {"token": " ", "done": false}
...
data: {"token": "", "done": true}
```

### Clear Chat

```bash
POST /api/chat/clear
```

Clear chat history (handled client-side).

### Unload Model

```bash
POST /api/models/<model_name>/unload
```

Unload a model from memory to free resources.

**Response:**
```json
{
  "success": true,
  "message": "Model pretrain/best.pt unloaded"
}
```

## Architecture

### Backend (Flask)

- **app.py**: Main Flask application
  - Model caching and loading management
  - Streaming generation support
  - RESTful API endpoints
  - Thread-safe model handling

### Frontend (HTML/CSS/JS)

- **templates/index.html**: Single-page application
  - Responsive chat interface
  - Real-time streaming via Server-Sent Events (SSE)
  - Model selection dropdown
  - Auto-resizing input textarea

### Integration with Existing Code

The web chat interface reuses:
- `deepseek/inference/inference.py`: Core inference logic
- `config.py`: Configuration management
- `deepseek/model/model.py`: Model architecture

## Configuration

### Server Configuration

Edit `chat/app.py` to change:

```python
# Server settings
app.run(
    host='0.0.0.0',      # Listen on all interfaces
    port=5001,           # Port number
    debug=True,          # Enable debug mode
    threaded=True        # Handle multiple requests
)
```

### Default Generation Parameters

Default parameters are set in the frontend JavaScript:

```javascript
{
    max_new_tokens: 512,
    temperature: 0.7,
    top_p: 0.9,
    do_sample: true
}
```

## Troubleshooting

### Models Not Showing Up

1. Ensure checkpoint files have `.pt` extension
2. Check that `checkpoints/` directory exists
3. Verify file permissions

### Model Loading Errors

1. Check that the checkpoint file is valid
2. Ensure all required dependencies are installed
3. Check server logs for detailed error messages

### Streaming Not Working

1. Ensure your browser supports Server-Sent Events (most modern browsers do)
2. Check that no proxy or firewall is blocking the streaming connection
3. Verify debug mode is not interfering

### Memory Issues

- Use the unload endpoint to free memory
- Consider loading only one model at a time
- Use smaller checkpoints if available

## Advanced Usage

### Programmatic Access

Use curl to interact with the API:

```bash
# List models
curl http://localhost:5001/api/models

# Generate with streaming
curl -X POST http://localhost:5001/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pretrain/best.pt",
    "message": "Tell me a joke"
  }'
```

### Custom Frontend

The API is fully documented above, allowing you to build custom clients:
- Mobile apps
- Command-line tools
- Desktop applications
- Integration with other systems

## Requirements

- Python 3.7+
- Flask 3.0+
- PyTorch 2.0+
- transformers 4.30+
- Existing DeepSeek V3 dependencies

## License

Same as the parent DeepSeek V3 project.
