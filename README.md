# DeepSeek V3 完整训练指南

> 本项目是一个教育性的 DeepSeek V3 实现，涵盖 **Pretrain → SFT → RL** 完整训练流程。

---

## 目录

1. [项目概述](#1-项目概述)
2. [架构设计](#2-架构设计)
   - [Multi-head Latent Attention (MLA)](#21-multi-head-latent-attention-mla)
   - [DeepSeekMoE (混合专家)](#22-deepseekmoemixtrue-of-experts)
   - [Multi-Token Prediction (MTP)](#23-multi-token-prediction-mtp)
3. [项目结构](#3-项目结构)
4. [数据集说明](#4-数据集说明)
5. [Tokenizer 处理](#5-tokenizer-处理)
6. [训练流程](#6-训练流程)
   - [Pretrain 预训练](#61-pretrain-预训练)
   - [SFT 有监督微调](#62-sft-有监督微调)
   - [RL 强化学习](#63-rl-强化学习)
7. [RL 强化学习详解](#7-rl-强化学习详解)
   - [GRPO 算法原理](#71-grpo-group-relative-policy-optimization)
   - [PPO 算法原理](#72-ppo-proximal-policy-optimization)
   - [DPO 算法原理](#73-dpo-direct-preference-optimization)
   - [Loss 函数与注意事项](#74-loss-函数与注意事项)
8. [快速开始](#8-快速开始)
9. [配置说明](#9-配置说明)
10. [监控与可视化](#10-监控与可视化)

---

## 1. 项目概述

本项目实现了 DeepSeek V3 的核心架构和完整训练流程，包括：

| 阶段 | 描述 | 数据集 | 目标 |
|------|------|--------|------|
| **Pretrain** | 语言模型预训练 | WikiText-2 / OpenWebText | 学习语言知识 |
| **SFT** | 有监督微调 | Alpaca | 学习指令跟随 |
| **RL** | 强化学习对齐 | HH-RLHF | 与人类偏好对齐 |

### 核心创新

1. **MLA (Multi-head Latent Attention)**: 低秩 KV 压缩，减少推理时内存占用
2. **DeepSeekMoE**: 共享专家 + 路由专家的混合专家架构
3. **MTP (Multi-Token Prediction)**: 多 token 预测作为辅助训练目标

---

## 2. 架构设计

### 2.1 Multi-head Latent Attention (MLA)

MLA 是 DeepSeek V3 的核心注意力机制，通过低秩压缩 KV 来减少内存占用。

#### 原理

传统 Attention 的 KV 缓存大小为 `O(H × d_h)`，MLA 将其压缩到 `O(d_c)`。

```
Input: x ∈ R^(B × L × D)

# KV 压缩 (核心创新)
c_kv = W_down(x)           # (B, L, d_c)     - 压缩到低维
K = W_k_up(c_kv)           # (B, L, H, d_h)  - 扩展为 Key
V = W_v_up(c_kv)           # (B, L, H, d_h^v)- 扩展为 Value

# Query 使用独立压缩
c_q = W_q_down(x)          # (B, L, d_c')
Q = W_q_up(c_q)            # (B, L, H, d_h)

# Decoupled RoPE (解耦位置编码)
Q_nope, Q_rope = split(Q)  # 分离位置相关和位置无关部分
K_nope, K_rope = split(K)

# 仅对 rope 部分应用旋转位置编码
Q_rope, K_rope = apply_rope(Q_rope, K_rope)

# 重新组合
Q = concat(Q_nope, Q_rope)
K = concat(K_nope, K_rope)

# 标准 Attention
Output = softmax(QK^T / √d_h) · V
```

#### 关键参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `kv_lora_rank` | KV 压缩维度 d_c | 64 |
| `q_lora_rank` | Q 压缩维度 d_c' | 96 |
| `qk_nope_head_dim` | 非 RoPE 头维度 | 32 |
| `qk_rope_head_dim` | RoPE 头维度 | 32 |
| `v_head_dim` | Value 头维度 | 64 |

#### 代码位置

- 实现: [attention.py](attention.py) - `MultiHeadLatentAttention` 类

---

### 2.2 DeepSeekMoE（Mixture of Experts）

DeepSeekMoE 结合了共享专家和路由专家，既保证通用知识又提供专业能力。

#### 架构

```
Input: x ∈ R^(B × L × D)

# 1. 共享专家 (Shared Experts) - 始终激活
shared_out = Σ expert_s(x) / n_shared

# 2. 路由专家 (Routed Experts) - Top-K 选择
router_probs = softmax(gate(x))           # (B, L, N) 路由概率
top_k_probs, top_k_idx = topk(router_probs, K)  # 选择 Top-K 专家
routed_out = Σ (prob_i × expert_i(x))     # 加权输出

# 3. 最终输出
output = shared_out + routed_scaling_factor × routed_out
```

#### 负载均衡损失

为了防止专家使用不均衡，引入辅助损失：

```
L_aux = α × N × Σ(f_i × P_i)

其中:
- f_i: 专家 i 接收的 token 比例
- P_i: 专家 i 的平均路由概率
- α: 损失系数 (默认 0.001)
- N: 专家总数
```

#### 关键参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `num_experts` | 路由专家总数 N | 16 |
| `num_experts_per_tok` | 每 token 激活专家数 K | 2 |
| `num_shared_experts` | 共享专家数 | 2 |
| `expert_hidden_size` | 专家 FFN 隐藏维度 | 768 |
| `aux_loss_alpha` | 辅助损失系数 | 0.001 |

#### 代码位置

- 实现: [model.py](model.py) - `DeepSeekMoE`, `MoEGate`, `Expert` 类

---

### 2.3 Multi-Token Prediction (MTP)

MTP 同时预测多个未来 token，作为辅助训练目标，同时支持推理时的投机解码。

#### 原理

```
Input: hidden_states ∈ R^(B × L × D)

# 对每个预测深度 d ∈ [1, D_predict]
for d in range(1, num_predict_tokens + 1):
    # 独立的投影层
    h_d = projection_d(hidden_states)
    h_d = layer_norm_d(h_d)
    logits_d = output_head_d(h_d)  # 预测位置 i+d 处的 token

# 训练时计算 MTP Loss
mtp_loss = Σ CE(logits_d[:, :-d-1], labels[:, d+1:])
total_loss = lm_loss + mtp_weight × mtp_loss
```

#### 关键参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `num_predict_tokens` | 额外预测的 token 数 | 2 |
| `mtp_loss_weight` | MTP 损失权重 | 0.3 |

#### 代码位置

- 实现: [model.py](model.py) - `MTPHead` 类

---

## 3. 项目结构

```
deepseek_v3/
├── config.py               # 配置管理 (所有配置类定义)
├── config_default.yaml     # 默认配置 (小数据集)
├── config_large.yaml       # 大规模训练配置
├── attention.py            # MLA 注意力实现
├── model.py                # DeepSeek V3 模型主体
├── dataset.py              # 数据集处理 (Pretrain/SFT/RL)
├── rl_dataset.py           # RL 专用数据集
├── trainer.py              # 训练器 (Pretrain/SFT/GRPO)
├── rl_trainer_base.py      # RL 训练基类
├── rl_trainer_algorithms.py # RL 算法实现 (GRPO/PPO)
├── train.py                # 训练入口脚本
├── rl_train.py             # RL 训练入口
├── inference.py            # 推理和生成
├── logger.py               # 日志模块 (彩色输出、多级别日志)
├── run.sh                  # 便捷运行脚本
├── run_pretrain.sh         # 预训练专用脚本
├── test_all.py             # 测试套件
├── test_rl.py              # RL 测试
├── requirements.txt        # Python 依赖
└── README.md               # 项目说明
```

### 3.1 日志模块 (logger.py)

项目提供统一的日志管理功能，支持彩色输出和不同日志级别。

#### 日志级别颜色

| 级别 | 颜色 | 符号 |
|------|------|------|
| DEBUG | 青色 (Cyan) | 🔍 |
| INFO | 绿色 (Green) | ℹ️ |
| WARNING | 黄色 (Yellow) | ⚠️ |
| ERROR | 红色 (Red) | ❌ |
| CRITICAL | 粗体红色 | 🔥 |

#### 日志格式

```
[时间] [PID:TID] [符号] [级别] [文件:行号] 消息
```

示例:
```
2025-12-27 10:30:00 [12345:67890] ℹ️  [  INFO  ] [train.py:100] Training started
```

#### 使用方法

```python
from logger import get_logger, set_log_level
import logging

# 获取 logger
logger = get_logger(__name__)

# 使用不同级别的日志
logger.debug("Detailed debugging info")
logger.info("General information")
logger.warning("Potential issue")
logger.error("Error occurred")

# 设置全局日志级别
set_log_level(logging.DEBUG)

# 添加文件日志
from logger import setup_file_logging
setup_file_logging("logs/training.log")
```

---

## 4. 数据集说明

### 4.1 Pretrain 数据集

| 数据集 | 规模 | 参数 | 用途 |
|--------|------|------|------|
| **WikiText-2** | ~13MB | `--dataset_scale small` | 快速测试/实验 |
| **OpenWebText** | ~40GB | `--dataset_scale large` | 正式训练 |

#### WikiText-2 格式
原始文本，每行是一段文章：
```text
= Valkyria Chronicles III =
Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場の...
```

#### OpenWebText 格式
Reddit 外链文章的文本内容：
```python
{
    "text": "The full article content..."
}
```

### 4.2 SFT 数据集

| 数据集 | 规模 | 格式 |
|--------|------|------|
| **Alpaca** | ~52K 样本 | instruction-input-output |

#### Alpaca 格式
```json
{
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1. Eat a balanced diet...\n2. Exercise regularly...\n3. Get enough sleep..."
}
```

#### 格式化模板
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

### 4.3 RL 数据集

| 数据集 | 规模 | 格式 |
|--------|------|------|
| **HH-RLHF** | ~170K | chosen/rejected pairs |

#### HH-RLHF 格式
```json
{
    "chosen": "Human: What is...\n\nAssistant: The answer is...",
    "rejected": "Human: What is...\n\nAssistant: I don't know..."
}
```

---

## 5. Tokenizer 处理

本项目使用 GPT-2 Tokenizer（可配置其他 HuggingFace tokenizer）。

### 5.1 加载 Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 设置 padding token (GPT-2 默认没有)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
```

### 5.2 Pretrain 数据处理

```python
# 1. 将所有文本拼接
all_text = " ".join(texts)

# 2. Tokenize
tokens = tokenizer.encode(all_text, add_special_tokens=False)

# 3. 切分成固定长度的序列
for i in range(0, len(tokens) - max_seq_length, max_seq_length):
    chunk = tokens[i:i + max_seq_length]
    examples.append({
        'input_ids': torch.tensor(chunk),
        'attention_mask': torch.ones(len(chunk)),
        'labels': torch.tensor(chunk),  # 自回归，labels = input_ids
    })
```

### 5.3 SFT 数据处理

```python
# 1. 格式化 prompt 和完整文本
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
full_text = prompt + output

# 2. Tokenize
prompt_ids = tokenizer.encode(prompt)
full_ids = tokenizer.encode(full_text)

# 3. 创建 labels (prompt 部分为 -100，不计算 loss)
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

### 5.4 关键配置

```yaml
data:
  tokenizer_name: "gpt2"           # 可改为其他 tokenizer
  pretrain_max_seq_length: 512     # 预训练序列长度
  sft_max_seq_length: 512          # SFT 序列长度
  rl_max_seq_length: 256           # RL 序列长度
```

---

## 6. 训练流程

### 6.1 Pretrain 预训练

#### 目标
学习语言模型的基础能力：语法、知识、推理。

#### Loss 函数
```python
# Next-Token Prediction Loss
loss = CrossEntropyLoss(logits[:, :-1], labels[:, 1:])

# + MTP Loss (if enabled)
for d in range(1, num_predict_tokens + 1):
    mtp_loss += CrossEntropyLoss(mtp_logits_d[:, :-d-1], labels[:, d+1:])
loss += mtp_weight * mtp_loss / num_predict_tokens

# + MoE Auxiliary Loss (if enabled)
loss += aux_loss
```

#### 运行命令

```bash
# 小数据集快速测试 (WikiText-2, ~13MB)
python train.py --mode pretrain --dataset_scale small --test

# 小数据集完整训练
python train.py --mode pretrain --dataset_scale small

# 大数据集训练 (OpenWebText, ~10GB)
python train.py --mode pretrain --dataset_scale large

# 使用 run.sh 脚本
./run.sh pretrain          # 小数据集
./run.sh pretrain-large    # 大数据集
./run.sh pretrain-test     # 快速测试
```

#### 关键参数

```yaml
pretraining:
  batch_size: 16
  learning_rate: 3e-4
  max_steps: 5000
  warmup_steps: 200
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
```

---

### 6.2 SFT 有监督微调

#### 目标
学习遵循指令、生成有帮助的回复。

#### Loss 函数
```python
# 只在 response 部分计算 loss
# labels 中 prompt 部分设为 -100
loss = CrossEntropyLoss(logits, labels, ignore_index=-100)
```

#### 运行命令

```bash
# 从预训练检查点开始 SFT
python train.py --mode sft --checkpoint checkpoints/pretrain/best.pt

# 使用 run.sh
./run.sh sft checkpoints/pretrain/best.pt
./run.sh sft-test  # 快速测试
```

#### 关键参数

```yaml
sft:
  batch_size: 8
  learning_rate: 2e-5      # 比预训练小
  max_steps: 2000
  warmup_ratio: 0.03
  weight_decay: 0.0        # SFT 通常不用 weight decay
```

---

### 6.3 RL 强化学习

#### 目标
将模型与人类偏好对齐，生成更有帮助、更安全的回复。

#### 支持的算法

| 算法 | 类型 | 特点 |
|------|------|------|
| **GRPO** | Online | DeepSeek 风格，组内相对优势 |
| **PPO** | Online | 经典 RLHF，需要 value function |
| **DPO** | Offline | 直接优化偏好，无需 reward model |

#### 运行命令

```bash
# GRPO (默认)
python train.py --mode rl --checkpoint checkpoints/sft/best.pt

# 指定算法
python rl_train.py --algorithm grpo --checkpoint checkpoints/sft/best.pt
python rl_train.py --algorithm ppo --checkpoint checkpoints/sft/best.pt
python rl_train.py --algorithm dpo --checkpoint checkpoints/sft/best.pt

# 使用 run.sh
./run.sh rl checkpoints/sft/best.pt
./run.sh rl-test
```

---

## 7. RL 强化学习详解

### 7.1 GRPO (Group Relative Policy Optimization)

GRPO 是 DeepSeek 提出的 RL 算法，无需学习 reward model，使用组内相对优势。

#### 算法流程

```
For each prompt x:
    1. 生成 G 个响应 {y_1, y_2, ..., y_G}
    2. 计算每个响应的 reward r_i = R(x, y_i)
    3. 计算组内相对优势:
       A_i = (r_i - mean(r)) / (std(r) + ε)
    4. 计算 policy gradient loss:
       L_PG = -E[A_i × log π(y_i|x)]
    5. 计算 KL 惩罚:
       L_KL = β × KL(π || π_ref)
    6. 总损失:
       L = L_PG + L_KL
```

#### 核心代码

```python
# 组内相对优势归一化
rewards_t = torch.tensor(rewards)
mean_r = rewards_t.mean()
std_r = rewards_t.std() + 1e-8
advantages = (rewards_t - mean_r) / std_r

# Policy Gradient Loss
for adv, log_prob in zip(advantages, log_probs):
    pg_loss += -adv * log_prob.mean()

# KL Penalty
kl = (policy_logps - ref_logps).mean()
loss = pg_loss + kl_coef * kl
```

#### 关键参数

| 参数 | 含义 | 默认值 | 说明 |
|------|------|--------|------|
| `group_size` | 每个 prompt 生成的响应数 | 4 | 越大方差估计越准，但计算量大 |
| `kl_coef` | KL 惩罚系数 β | 0.1 | 防止偏离参考模型太远 |
| `temperature` | 采样温度 | 0.7 | 控制生成多样性 |

---

### 7.2 PPO (Proximal Policy Optimization)

PPO 是经典的 RLHF 算法，使用 value function 估计优势。

#### 算法流程

```
1. Rollout: 生成响应，计算 reward
2. 计算 GAE (Generalized Advantage Estimation):
   δ_t = r_t + γ V(s_{t+1}) - V(s_t)
   A_t = Σ (γλ)^k δ_{t+k}
3. PPO Update (多个 epoch):
   a. 计算 probability ratio: ρ = π(a|s) / π_old(a|s)
   b. Clipped surrogate objective:
      L_clip = min(ρ A, clip(ρ, 1-ε, 1+ε) A)
   c. Value function loss:
      L_VF = MSE(V(s), R_t)
   d. Entropy bonus:
      H = -Σ π log π
   e. Total loss:
      L = -L_clip + c1 × L_VF - c2 × H
```

#### 关键参数

| 参数 | 含义 | 默认值 | 说明 |
|------|------|--------|------|
| `clip_range` | PPO 裁剪范围 ε | 0.2 | 限制策略更新幅度 |
| `value_coef` | Value loss 系数 c1 | 0.5 | |
| `entropy_coef` | Entropy bonus 系数 c2 | 0.01 | 鼓励探索 |
| `gae_lambda` | GAE λ 参数 | 0.95 | 方差-偏差权衡 |
| `ppo_epochs` | 每批数据的 PPO 更新次数 | 4 | |
| `target_kl` | KL 早停阈值 | 0.02 | 超过则停止更新 |

---

### 7.3 DPO (Direct Preference Optimization)

DPO 直接从偏好数据学习，无需显式 reward model。

#### 算法原理

```
给定偏好数据 (x, y_w, y_l)，其中 y_w 是人类偏好的响应，y_l 是不偏好的响应

DPO Loss:
L_DPO = -E[log σ(β × (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

简化为:
L_DPO = -E[log σ(β × (r_w - r_l))]

其中:
r = log π(y|x) - log π_ref(y|x)  # 隐式 reward
```

#### 核心代码

```python
# 计算 policy 和 reference 的 log probabilities
policy_logps_w = compute_log_probs(model, y_w)
policy_logps_l = compute_log_probs(model, y_l)
ref_logps_w = compute_log_probs(ref_model, y_w)
ref_logps_l = compute_log_probs(ref_model, y_l)

# 计算 reward margin
logits_w = policy_logps_w - ref_logps_w
logits_l = policy_logps_l - ref_logps_l
logits_diff = logits_w - logits_l

# DPO Loss
loss = -F.logsigmoid(beta * logits_diff).mean()
```

#### 关键参数

| 参数 | 含义 | 默认值 | 说明 |
|------|------|--------|------|
| `dpo_beta` | 温度参数 β | 0.1 | 越大策略变化越激进 |
| `dpo_label_smoothing` | 标签平滑 | 0.0 | 增加鲁棒性 |

---

### 7.4 Loss 函数与注意事项

#### RL 训练的主要 Loss 组件

| Loss | 公式 | 作用 |
|------|------|------|
| **Policy Gradient** | `-A × log π(y|x)` | 提高高 reward 响应的概率 |
| **KL Penalty** | `β × KL(π || π_ref)` | 防止偏离参考模型太远 |
| **Value Loss** | `MSE(V, R)` | 准确估计状态价值 |
| **Entropy Bonus** | `-H(π)` | 鼓励探索 |

#### 训练注意事项

1. **学习率**：RL 阶段使用非常小的学习率（~5e-7），防止模型能力退化

2. **KL 控制**：
   - 监控 KL 散度，过大说明策略变化太快
   - 使用 target_kl 早停机制
   - 适当调整 kl_coef

3. **Reward 设计**：
   - 本项目使用规则 reward（长度、连贯性、重复惩罚等）
   - 生产环境应使用 learned reward model

4. **Reward Hacking**：
   - 模型可能找到 reward 的漏洞
   - 使用多样化的 reward 信号
   - 保持 KL 约束

5. **训练稳定性**：
   - 使用梯度裁剪 `max_grad_norm: 1.0`
   - 使用梯度累积平滑更新
   - 监控 reward 和 loss 曲线

6. **Reference Model**：
   - 保持冻结，不更新参数
   - 用于计算 KL 散度
   - 防止模型退化

---

## 8. 快速开始

### 8.1 安装依赖

```bash
cd learn/deepseek_v3
pip install -r requirements.txt
chmod +x run.sh
```

### 8.2 运行测试

```bash
# 测试模型和训练流程
./run.sh test-quick

# 完整测试
./run.sh test
```

### 8.3 完整训练流程

```bash
# 1. Pretrain (小数据集快速验证)
./run.sh pretrain-test

# 或完整预训练
./run.sh pretrain

# 2. SFT
./run.sh sft checkpoints/pretrain/best.pt

# 3. RL (GRPO)
./run.sh rl checkpoints/sft/best.pt

# 4. 推理
./run.sh inference checkpoints/rl/best.pt

# 5. 交互式对话
./run.sh chat checkpoints/rl/best.pt
```

### 8.4 一键完整流程

```bash
# 快速测试整个流程
./run.sh full-test

# 完整训练流程
./run.sh full
```

---

## 9. 配置说明

### 9.1 配置文件

| 文件 | 用途 |
|------|------|
| `config_default.yaml` | 默认配置（小数据集） |
| `config_large.yaml` | 大规模训练配置 |

### 9.2 命令行参数

```bash
python train.py \
    --mode pretrain|sft|rl \     # 训练模式
    --dataset_scale small|large \ # 数据集规模
    --config config.yaml \        # 配置文件
    --checkpoint path/to/ckpt \   # 加载检查点
    --device auto|cuda|mps|cpu \  # 设备
    --test                        # 快速测试模式
```

### 9.3 关键配置项

```yaml
model:
  hidden_size: 512           # 模型维度
  num_hidden_layers: 6       # Transformer 层数
  num_attention_heads: 8     # 注意力头数
  
  moe:
    enabled: true
    num_experts: 16          # 专家数量
    num_experts_per_tok: 2   # 每 token 激活专家数
    
  mtp:
    enabled: true
    num_predict_tokens: 2    # 额外预测 token 数

pretraining:
  batch_size: 16
  learning_rate: 3e-4
  max_steps: 5000

sft:
  batch_size: 8
  learning_rate: 2e-5

rl:
  algorithm: "grpo"          # grpo, ppo, dpo
  group_size: 4
  kl_coef: 0.1
```

---

## 10. 监控与可视化

### 10.1 TensorBoard

```bash
# 启动 TensorBoard
./run.sh tensorboard

# 或手动启动
tensorboard --logdir=runs --port=6006
```

### 10.2 可视化内容

| 类别 | 内容 |
|------|------|
| **Loss** | train_loss, val_loss, perplexity |
| **Learning** | learning_rate, grad_norm |
| **Speed** | tokens/sec, samples/sec, steps/sec |
| **Attention** | 注意力权重热力图 |
| **MoE** | 专家使用分布, routing entropy |
| **Generation** | 生成文本样本 |
| **RL** | reward, kl_divergence, policy_loss |

### 10.3 训练日志示例

```
┌──────────────────────────────────────────────────────────────────────┐
│ Step:    100/5000 [██████░░░░░░░░░░░░░░░░░░░░░░░░]  2.0%             │
├──────────────────────────────────────────────────────────────────────┤
│ Loss:   6.2345  (smoothed:   6.3012)                                 │
│ LR: 2.85e-04  Grad norm:   0.8721                                    │
│ Epoch: 1                                                             │
├──────────────────────────────────────────────────────────────────────┤
│ Speed:    12543 tok/s   98.2 samples/s   6.14 steps/s                │
│ Time:      16.3s  ETA:    13.1m                                      │
│ Tokens:      203,776                                                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 参考资料

- [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
