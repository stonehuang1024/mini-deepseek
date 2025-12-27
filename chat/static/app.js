/**
 * DeepSeek V3 Chat Interface - Frontend JavaScript
 */

class ChatApp {
    constructor() {
        this.chatMessages = document.getElementById('chat-messages');
        this.userInput = document.getElementById('user-input');
        this.sendBtn = document.getElementById('send-btn');
        this.cancelBtn = document.getElementById('cancel-btn');
        this.modelSelect = document.getElementById('model-select');
        this.modelLoading = document.getElementById('model-loading');
        this.modelInfo = document.getElementById('model-info');
        this.clearChatBtn = document.getElementById('clear-chat');
        this.errorToast = document.getElementById('error-toast');
        
        // Settings
        this.temperatureInput = document.getElementById('temperature');
        this.maxTokensInput = document.getElementById('max-tokens');
        this.topPInput = document.getElementById('top-p');
        this.topKInput = document.getElementById('top-k');
        
        // State
        this.isGenerating = false;
        this.currentModel = null;
        this.conversationHistory = [];
        this.abortController = null;
        
        this.init();
    }
    
    async init() {
        this.setupEventListeners();
        this.setupSettingsListeners();
        await this.loadCheckpoints();
    }
    
    setupEventListeners() {
        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Enter to send, Shift+Enter for new line
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.userInput.addEventListener('input', () => {
            this.userInput.style.height = 'auto';
            this.userInput.style.height = Math.min(this.userInput.scrollHeight, 200) + 'px';
        });
        
        // Cancel generation
        this.cancelBtn.addEventListener('click', () => this.cancelGeneration());
        
        // Model selection
        this.modelSelect.addEventListener('change', () => this.loadSelectedModel());
        
        // Clear chat
        this.clearChatBtn.addEventListener('click', () => this.clearChat());
    }
    
    setupSettingsListeners() {
        // Temperature
        this.temperatureInput.addEventListener('input', (e) => {
            document.getElementById('temp-value').textContent = e.target.value;
        });
        
        // Max tokens
        this.maxTokensInput.addEventListener('input', (e) => {
            document.getElementById('tokens-value').textContent = e.target.value;
        });
        
        // Top-P
        this.topPInput.addEventListener('input', (e) => {
            document.getElementById('topp-value').textContent = e.target.value;
        });
        
        // Top-K
        this.topKInput.addEventListener('input', (e) => {
            document.getElementById('topk-value').textContent = e.target.value;
        });
    }
    
    async loadCheckpoints() {
        try {
            const response = await fetch('/api/checkpoints');
            const data = await response.json();
            
            this.modelSelect.innerHTML = '';
            
            if (data.checkpoints.length === 0) {
                this.modelSelect.innerHTML = '<option value="">No models found</option>';
                this.showError('No checkpoint files found in checkpoints/ directory');
                return;
            }
            
            // Group by category
            const grouped = {};
            data.checkpoints.forEach(ckpt => {
                if (!grouped[ckpt.category]) {
                    grouped[ckpt.category] = [];
                }
                grouped[ckpt.category].push(ckpt);
            });
            
            // Create optgroups
            for (const [category, checkpoints] of Object.entries(grouped)) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = category;
                
                checkpoints.forEach(ckpt => {
                    const option = document.createElement('option');
                    option.value = ckpt.path;
                    option.textContent = `${ckpt.name} (${ckpt.size_mb} MB)`;
                    if (ckpt.path === data.default || ckpt.path === data.current) {
                        option.selected = true;
                    }
                    optgroup.appendChild(option);
                });
                
                this.modelSelect.appendChild(optgroup);
            }
            
            this.modelSelect.disabled = false;
            this.currentModel = data.current || data.default;
            
            // Update model info
            if (this.currentModel) {
                const ckpt = data.checkpoints.find(c => c.path === this.currentModel);
                if (ckpt) {
                    this.modelInfo.textContent = `Loaded: ${ckpt.name}`;
                }
            }
            
            // Enable input
            this.userInput.disabled = false;
            this.sendBtn.disabled = false;
            
        } catch (error) {
            console.error('Failed to load checkpoints:', error);
            this.showError('Failed to load model list');
        }
    }
    
    async loadSelectedModel() {
        const checkpointPath = this.modelSelect.value;
        if (!checkpointPath || checkpointPath === this.currentModel) {
            return;
        }
        
        // Cancel any ongoing generation first
        if (this.isGenerating) {
            await this.cancelGeneration();
        }
        
        this.modelSelect.disabled = true;
        this.modelLoading.style.display = 'block';
        this.modelInfo.textContent = 'Loading model...';
        
        try {
            const response = await fetch('/api/load_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ checkpoint_path: checkpointPath })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentModel = checkpointPath;
                const name = checkpointPath.split('/').slice(-2).join('/');
                this.modelInfo.textContent = `Loaded: ${name}`;
            } else {
                this.showError(data.error || 'Failed to load model');
                this.modelInfo.textContent = 'Load failed';
            }
        } catch (error) {
            console.error('Failed to load model:', error);
            this.showError('Failed to load model');
            this.modelInfo.textContent = 'Load failed';
        } finally {
            this.modelSelect.disabled = false;
            this.modelLoading.style.display = 'none';
        }
    }
    
    async sendMessage() {
        const message = this.userInput.value.trim();
        if (!message || this.isGenerating) {
            return;
        }
        
        // Clear welcome message
        const welcome = this.chatMessages.querySelector('.welcome-message');
        if (welcome) {
            welcome.remove();
        }
        
        // Add user message
        this.addMessage('user', message);
        this.conversationHistory.push({ role: 'user', content: message });
        
        // Clear input
        this.userInput.value = '';
        this.userInput.style.height = 'auto';
        
        // Build prompt with history
        const prompt = this.buildPrompt(message);
        
        // Start generation
        await this.generateResponse(prompt);
    }
    
    buildPrompt(userMessage) {
        let prompt = '';
        
        // Add conversation history
        for (const msg of this.conversationHistory.slice(0, -1)) {
            if (msg.role === 'user') {
                prompt += `Human: ${msg.content}\n\n`;
            } else {
                prompt += `Assistant: ${msg.content}\n\n`;
            }
        }
        
        // Add current message
        prompt += `Human: ${userMessage}\n\nAssistant:`;
        
        return prompt;
    }
    
    async generateResponse(prompt) {
        this.isGenerating = true;
        this.sendBtn.style.display = 'none';
        this.cancelBtn.style.display = 'flex';
        this.userInput.disabled = true;
        
        // Create assistant message element
        const messageDiv = this.addMessage('assistant', '', true);
        const contentDiv = messageDiv.querySelector('.message-content');
        
        let fullResponse = '';
        
        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    checkpoint_path: this.currentModel,
                    max_new_tokens: parseInt(this.maxTokensInput.value),
                    temperature: parseFloat(this.temperatureInput.value),
                    top_p: parseFloat(this.topPInput.value),
                    top_k: parseInt(this.topKInput.value),
                })
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.type === 'token') {
                                fullResponse += data.content;
                                contentDiv.textContent = fullResponse;
                                this.scrollToBottom();
                            } else if (data.type === 'cancelled') {
                                contentDiv.classList.remove('streaming');
                                const cancelledNote = document.createElement('div');
                                cancelledNote.className = 'message-cancelled';
                                cancelledNote.textContent = '[Generation cancelled]';
                                contentDiv.appendChild(cancelledNote);
                            } else if (data.type === 'error') {
                                this.showError(data.message);
                            } else if (data.type === 'done') {
                                contentDiv.classList.remove('streaming');
                            }
                        } catch (e) {
                            // Ignore parse errors for incomplete chunks
                        }
                    }
                }
            }
            
            // Save to history
            if (fullResponse) {
                this.conversationHistory.push({ role: 'assistant', content: fullResponse });
            }
            
        } catch (error) {
            console.error('Generation error:', error);
            if (error.name !== 'AbortError') {
                this.showError('Generation failed: ' + error.message);
            }
            contentDiv.classList.remove('streaming');
        } finally {
            this.isGenerating = false;
            this.sendBtn.style.display = 'flex';
            this.cancelBtn.style.display = 'none';
            this.userInput.disabled = false;
            this.userInput.focus();
        }
    }
    
    async cancelGeneration() {
        try {
            await fetch('/api/cancel', { method: 'POST' });
        } catch (error) {
            console.error('Cancel error:', error);
        }
    }
    
    addMessage(role, content, streaming = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'U' : 'AI';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        if (streaming) {
            contentDiv.classList.add('streaming');
        }
        contentDiv.textContent = content;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    clearChat() {
        this.chatMessages.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to DeepSeek V3 Chat</h2>
                <p>Select a model and start chatting. Your messages will be processed locally.</p>
            </div>
        `;
        this.conversationHistory = [];
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    showError(message) {
        this.errorToast.textContent = message;
        this.errorToast.style.display = 'block';
        
        setTimeout(() => {
            this.errorToast.style.display = 'none';
        }, 5000);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});
