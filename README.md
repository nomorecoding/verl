# Ming-omni-tts MoE 模型 Flow-GRPO 训练方案

基于 [verl](https://github.com/verl-project/verl) 框架，对 [Ming-omni-tts](https://github.com/inclusionAI/Ming-omni-tts) 的 MoE 模型实现 Flow-GRPO（Group Relative Policy Optimization）强化学习训练。

## 目录

- [架构概览](#架构概览)
- [两种训练模式](#两种训练模式)
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [详细步骤](#详细步骤)
- [代码结构](#代码结构)
- [关键设计决策](#关键设计决策)
- [常见问题](#常见问题)

---

## 架构概览

Ming-omni-tts 的生成流程不同于标准的 LLM 文本生成，它是 **自回归LLM + Flow Matching** 的混合架构：

```
文本输入 → MoE-LLM 编码 → hidden state → Flow Matching 生成音频 latent → 音频解码
                ↑                                    │
                └── latent 投影回 embedding ←─────────┘
```

每一步生成过程中：
1. MoE-LLM（BailingMoeForCausalLM，16个专家，top-2路由）处理输入，产生 hidden state
2. Flow Matching 头（CFM + DiT）以 hidden state 为条件，从噪声采样得到音频 latent
3. 音频 latent 经 Aggregator 投影回 LLM 的 embedding 空间，作为下一步的输入
4. 重复直至 stop_head 预测生成结束

**GRPO 强化学习目标**：通过 reward（语音质量、可懂度、说话人相似度等）来优化生成策略。

---

## 两种训练模式

### MODE A: verl 原生 GRPO（推荐）

只训练 MoE-LLM backbone，flow matching 头冻结。

**优点**：
- 直接复用 verl 的 FSDP/Megatron 分布式训练 + vLLM 推理引擎
- 基础设施成熟，分布式扩展性好
- MoE-LLM 的条件表示更好 → 音频质量自然提升

**适用场景**：
- 大规模多机多卡训练
- 需要 verl 的完整功能（rollout correction、checkpointing、wandb 集成等）

### MODE B: 完整管线 Flow-GRPO

同时训练 MoE-LLM 和 flow matching 头。

**优点**：
- 端到端优化，flow matching 也能适应 reward 信号
- 使用自定义的 surrogate log-prob（基于 CFM loss）进行策略更新

**适用场景**：
- 单机少卡实验
- 需要优化 flow matching 头本身
- 研究/探索目的

---

## 环境安装

### 1. 基础依赖

```bash
pip install torch>=2.1 transformers>=4.40 accelerate
pip install pandas pyarrow
pip install x-transformers   # Ming-omni-tts DiT 模块需要
```

### 2. 安装 verl（MODE A 需要）

```bash
# 推荐从源码安装
git clone https://github.com/verl-project/verl.git
cd verl
pip install -e ".[all]"

# 安装 vLLM（用于高效 rollout）
pip install vllm>=0.6
```

### 3. 克隆 Ming-omni-tts 模型代码

```bash
git clone https://github.com/inclusionAI/Ming-omni-tts.git
export MING_MODEL_DIR=$(pwd)/Ming-omni-tts
export PYTHONPATH="${MING_MODEL_DIR}:${PYTHONPATH}"
```

### 4. 下载模型权重

根据 Ming-omni-tts 仓库的说明下载预训练权重。

---

## 数据准备

### verl 数据格式

verl 期望 Parquet 格式的训练数据，包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `prompt` | list[dict] | 对话格式 `[{"role": "user", "content": "..."}]` |
| `data_source` | str | 数据来源标识 |
| `reward_model` | dict | 奖励计算所需信息 |
| `extra_info` | dict | 辅助元数据 |

### 从 TTS 语料制作训练数据

```bash
# 方式1: 从 JSONL manifest 转换
python -m ming_moe_verl.data.preprocess \
    --manifest /path/to/tts_manifest.jsonl \
    --output_dir ./data/tts_grpo \
    --split train

# 方式2: 创建示例数据（测试用）
python -m ming_moe_verl.data.preprocess --create_demo --output_dir ./data/tts_grpo
```

JSONL manifest 格式：
```json
{"text": "你好世界", "audio_path": "/data/audio/001.wav", "speaker_id": "spk01", "emotion": "neutral", "duration": 3.2}
```

---

## 快速开始

### MODE A: verl 原生 GRPO

```bash
# 1. 准备数据
python -m ming_moe_verl.data.preprocess --create_demo --output_dir ./data/tts_grpo

# 2. 注册模型并启动训练
export MING_MODEL_DIR=/path/to/Ming-omni-tts
export MODEL_PATH=/path/to/bailing-moe-weights

bash ming_moe_verl/scripts/run_flow_grpo_verl.sh
```

### MODE B: 完整管线 Flow-GRPO

```bash
# 1. 准备数据
python -m ming_moe_verl.data.preprocess --create_demo --output_dir ./data/tts_grpo

# 2. 启动训练
export MODEL_PATH=/path/to/ming-omni-tts-weights
export TRAIN_DATA=./data/tts_grpo/train.parquet
export VAL_DATA=./data/tts_grpo/test.parquet

bash ming_moe_verl/scripts/run_flow_grpo_standalone.sh
```

---

## 详细步骤

### 步骤 1: 理解 MoE 模型结构

BailingMoeForCausalLM 是一个 MoE Transformer：
- **专家数量**: 16（可配置）
- **Top-K 路由**: top-2
- **路由门控**: 线性层 + softmax + top-k 选择
- **共享专家**: 可选（`num_shared_experts`）
- **前 K 层稠密**: `first_k_dense_replace` 控制

关键配置参数：
```python
BailingMoeConfig(
    num_experts=16,
    num_experts_per_tok=2,
    num_shared_experts=0,
    norm_topk_prob=True,
    first_k_dense_replace=0,
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
)
```

### 步骤 2: 理解 verl GRPO 算法

GRPO 的核心思想：
1. 对每个 prompt，生成 N 个响应（group）
2. 计算每个响应的 reward
3. 在 group 内做优势归一化：`advantage = (reward - mean) / std`
4. 用 PPO-clip loss 更新策略

```
对每个 prompt p:
  生成 N 个响应: y₁, y₂, ..., yₙ ~ π_θ(·|p)
  计算 reward:   r₁, r₂, ..., rₙ
  组内归一化:    aᵢ = (rᵢ - μ) / σ
  更新策略:      L = -min(ratio·a, clip(ratio)·a)
```

### 步骤 3: 将 MoE 模型注册到 HuggingFace AutoModel

```python
from transformers import AutoConfig, AutoModelForCausalLM

# 导入 Ming-omni-tts 的模型类
from configuration_bailing_moe import BailingMoeConfig
from modeling_bailing_moe import BailingMoeForCausalLM

# 注册
AutoConfig.register("bailing_moe", BailingMoeConfig)
AutoModelForCausalLM.register(BailingMoeConfig, BailingMoeForCausalLM)
```

### 步骤 4: 配置 GRPO 训练参数

关键参数说明：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `algorithm.adv_estimator` | 设为 `grpo` | `grpo` |
| `actor_rollout_ref.rollout.n` | 每个 prompt 生成数量 | 4-8 |
| `actor_rollout_ref.actor.use_kl_loss` | KL 散度正则 | `True` |
| `actor_rollout_ref.actor.kl_loss_coef` | KL 系数 | 0.001 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | PPO mini batch | batch_size/4 |
| `actor_rollout_ref.actor.optim.lr` | 学习率 | 1e-6 |

### 步骤 5: 设计 TTS Reward

我们定义了多维度的 TTS reward：

```python
reward = w₁ × intelligibility   # ASR 可懂度 (WER/CER)
       + w₂ × mos               # 音频质量 (MOS 预测)
       + w₃ × duration          # 时长合理性
       + w₄ × speaker_sim       # 说话人相似度
```

在 `ming_moe_verl/reward/tts_reward.py` 中实现，可以插入外部 ASR / MOS / 说话人模型。

### 步骤 6: 启动训练

参见 [快速开始](#快速开始) 章节。

### 步骤 7: 监控与评估

```bash
# 使用 wandb 监控
trainer.logger='["console","wandb"]'
trainer.project_name=ming_moe_flow_grpo

# 关键指标
# - avg_reward: 平均 reward（应持续上升）
# - policy_loss: PPO 策略损失
# - approx_kl: 近似 KL 散度（不应过大）
# - clip_fraction: PPO 裁剪比例（0.1-0.3 正常）
```

---

## 代码结构

```
ming_moe_verl/
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── policy_forward.py      # BailingMoeTTSForRL - 策略模型封装
│   │                          #   - compute_log_prob_for_step(): CFM surrogate log-prob
│   │                          #   - compute_sequence_log_probs(): 序列级 log-prob
│   │                          #   - generate_audio_latents(): 音频 latent 生成
│   └── rollout_worker.py      # BailingMoeFlowRollout - 自定义 rollout 引擎
│                              #   - prepare_prompt_tokens(): 文本 → token
│                              #   - generate_batch(): 批量 rollout 生成
├── reward/
│   ├── __init__.py
│   └── tts_reward.py          # TTSRewardManager - TTS reward 管理器
│                              #   - compute_tts_reward(): 多维度 reward 计算
│                              #   - TTSRewardManager: verl 兼容的 reward 接口
├── data/
│   ├── __init__.py
│   └── preprocess.py          # 数据预处理
│                              #   - create_tts_parquet(): manifest → parquet
│                              #   - create_demo_data(): 创建示例数据
├── configs/
│   ├── flow_grpo_fsdp.yaml    # verl FSDP 模式配置
│   └── flow_grpo_standalone.yaml  # 独立训练配置
├── scripts/
│   ├── run_flow_grpo_verl.sh      # MODE A 启动脚本
│   └── run_flow_grpo_standalone.sh # MODE B 启动脚本
├── train_flow_grpo.py         # MODE B: 完整管线 Flow-GRPO 训练
└── train_flow_grpo_verl_native.py  # MODE A: verl 原生 GRPO 训练
```

---

## 关键设计决策

### 1. Surrogate Log-Prob（代理对数概率）

标准 GRPO 需要 `log π(a|s)` 来计算重要性比率。但 flow matching 生成的是连续 latent 而非离散 token。

我们使用 **CFM loss 作为 surrogate log-prob**：

```
log π(aₜ | sₜ) ≈ -½ ‖v_θ(xₜ, t, c) - (x₁ - x₀)‖²
```

这在连续动作空间的 RL（如 Diffusion Policy、DDPO）中是常见做法。

### 2. MoE 专家路由在 RL 中的处理

BailingMoe 使用 top-2 路由，路由决策在前向传播中隐式确定。verl 支持 MoE 模型的 **router replay**（在训练时复用 rollout 阶段的路由决策），可以减少 off-policy 偏差。

在 Megatron backend 下可以启用：
```yaml
actor_rollout_ref.actor.router_replay.enable: True
actor_rollout_ref.actor.router_replay.mode: "R3"
```

### 3. 奖励设计

TTS 的奖励需要平衡多个维度：
- **可懂度**权重最高（0.4），因为说清楚是基本要求
- **音质**次之（0.3），MOS 预测反映主观感受
- **说话人相似度**（0.2）保证音色一致性
- **时长**（0.1）防止生成过长/过短

### 4. 分布式策略

- **FSDP**：适合单节点多卡（8x H100/A100），MoE 参数量大需要分片
- **Megatron**：适合多节点，支持 Expert Parallel + Tensor Parallel
- **vLLM rollout**：高效推理，支持 MoE 模型的 tensor parallelism

---

## 常见问题

### Q: 模型注册失败 `BailingMoe not found`

确保设置了环境变量：
```bash
export MING_MODEL_DIR=/path/to/Ming-omni-tts
export PYTHONPATH="${MING_MODEL_DIR}:${PYTHONPATH}"
```

### Q: vLLM 不支持 BailingMoe

BailingMoe 的架构与 DeepSeek-MoE 类似。可能需要在 vLLM 中添加对应的 weight loader。
verl 已有相关 patch：参见 `verl/utils/vllm/patch.py`。

作为替代，可以使用 `naive` rollout（纯 HuggingFace generate）：
```yaml
actor_rollout_ref.rollout.name: naive
```

### Q: 显存不足

- 启用 gradient checkpointing: `enable_gradient_checkpointing=True`
- 减小 micro batch size: `ppo_micro_batch_size_per_gpu=1`
- 启用参数卸载: `fsdp_config.param_offload=True`
- 减少 group size: `rollout.n=2`

### Q: 如何接入真实的 ASR/MOS 模型

在 `TTSRewardManager` 中传入外部模型：
```python
import whisper
asr_model = whisper.load_model("base")

reward_manager = TTSRewardManager(
    tokenizer=tokenizer,
    audio_decoder=model.audio,
    asr_model=asr_model,  # Whisper for ASR
    mos_model=utmos_model,  # UTMOS for MOS prediction
)
```

---

## 参考

- [verl: Volcano Engine Reinforcement Learning for LLMs](https://github.com/verl-project/verl)
- [Ming-omni-tts](https://github.com/inclusionAI/Ming-omni-tts)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [DDPO: Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301)
