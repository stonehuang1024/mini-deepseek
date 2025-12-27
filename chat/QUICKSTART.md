# Web Chat Quick Start Guide

## 启动Web聊天界面

### 方法1: 使用run.sh脚本（推荐）
```bash
./scripts/run.sh web-chat
```

### 方法2: 直接运行Python
```bash
python3 chat/app.py
```

## 访问界面

启动后，在浏览器中访问：
```
http://localhost:5001
```

## 功能说明

### 1. 模型切换
- 页面右上角的下拉菜单可以切换不同的模型
- 模型来自 `checkpoints/` 目录下的所有 `.pt` 文件
- 切换时会有转圈圈加载效果
- 模型按类型分组显示（Pretrained、SFT、RL）

### 2. 流式输出
- 每个token会实时显示在聊天框中
- 使用Server-Sent Events (SSE)技术实现

### 3. 聊天历史
- 支持多轮对话
- 对话历史会在发送消息时传递给模型
- 点击"Clear Chat"按钮可以清空对话

### 4. 自动发现模型
系统会自动发现以下checkpoint：
- ✓ test_grpo/final.pt (RL模型, 1131.82 MB)
- ✓ test_dpo/final.pt (RL模型, 1131.82 MB)
- ✓ pretrain/final.pt (预训练模型, 79.35 MB)
- ✓ test_ppo/final.pt (RL模型, 1131.82 MB)

## 测试

在启动服务前，可以先运行测试：
```bash
python3 chat/test_api.py
```

## API端点

### 获取模型列表
```bash
curl http://localhost:5001/api/models
```

### 生成文本（流式）
```bash
curl -X POST http://localhost:5001/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pretrain/final.pt",
    "message": "Hello"
  }'
```

## 注意事项

1. 首次加载模型需要几秒钟时间
2. 大模型（1GB+）首次加载可能需要更长时间
3. 模型会缓存在内存中，再次切换会很快
4. 建议使用GPU加速（CUDA或MPS）

## 故障排查

### 如果模型列表为空
- 检查 `checkpoints/` 目录下是否有 `.pt` 文件
- 确认文件权限正确

### 如果加载失败
- 查看服务器日志输出的错误信息
- 确认PyTorch和transformers依赖已安装
- 检查checkpoint文件是否完整

### 如果流式输出不工作
- 确认浏览器支持Server-Sent Events
- 检查防火墙或代理是否阻止了流式连接
