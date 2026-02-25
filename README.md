# Ming-omni-tts MoE 模型 Flow-GRPO 训练

基于 [verl](https://github.com/verl-project/verl) 的 GRPO 算法思想，对 [Ming-omni-tts](https://github.com/inclusionAI/Ming-omni-tts) 的 MoE+FlowMatching 模型实现**连续动作空间**的 Flow-GRPO 强化学习训练。

---

## 核心问题：为什么不能直接用标准 GRPO?

### 标准 GRPO (verl 默认) 的假设

```
状态: s_t = (prompt, token_{<t})
动作: a_t ∈ {1, ..., V}                    ← 离散 token
策略: π(a_t|s_t) = softmax(lm_head(h_t))   ← 过 lm_head → softmax
log-prob: log_softmax(logits)[token_id]
```

### Ming-omni-tts TTS 模式的实际情况

```
状态: s_t = (text_prompt, audio_latent_{<t})
动作: a_t ∈ ℝ^{patch_size × latent_dim}     ← 连续 latent 向量
策略: MoE-LLM hidden state → flow matching   ← 不走 lm_head
log-prob: ???  (需要重新定义!)
```

**关键差异**:
1. LLM 从头到尾**没有经过 `lm_head → softmax → 离散采样`**
2. 动作空间是连续的 `ℝ^d`，不是离散的 `{1,...,V}`
3. 每步生成涉及 K 步 ODE 积分，不是单次 softmax

如果强行用 verl 的 vLLM rollout → 模型走 lm_head 生成文本 token → GRPO 信号和 TTS 品质毫无关系。

---

## 解决方案：基于 Stochastic ODE 的连续动作 GRPO

### 关键洞察

Ming-omni-tts 的 ODE Solver 在每个积分步加了**高斯随机扰动**:

```python
# Ming-omni-tts/fm/CFM.py, Solver.integrate()
y1 = y0 + dt * f0                                          # 确定性步
noise = torch.randn_like(y0)
shift = self.sigma * (self.temperature ** 0.5) * (abs(dt) ** 0.5) * noise  # 随机扰动
y0 = y1 + shift
```

这意味着每步的转移概率是一个**显式的高斯分布**:

```
p(y_{k+1} | y_k, c; θ) = N(y_{k+1} | μ_k, σ_k² · I)

其中:
  μ_k = y_k + dt_k · v_θ(y_k, t_k, c)     ← 确定性预测（依赖模型参数 θ）
  σ_k = σ · √(T · |dt_k|)                  ← 固定方差（只依赖 solver 配置）
```

### 完整的 Flow-GRPO 公式

**Per-ODE-step log-prob:**
$$\log p(y_{k+1} | y_k, c; \theta) = -\frac{1}{2\sigma_k^2} \|y_{k+1} - \mu_k\|^2 - \frac{d}{2}\log(2\pi\sigma_k^2)$$

**Per-autoregressive-step log-prob** (K 个 ODE 步求和):
$$\log \pi(a_t | s_t; \theta) = \sum_{k=0}^{K-1} \log p(y_{k+1} | y_k, c_t; \theta)$$

**Importance ratio:**
$$r_t = \exp\left(\log \pi_{\theta_{new}}(a_t | s_t) - \log \pi_{\theta_{old}}(a_t | s_t)\right)$$

**为什么 ratio 不是 1?** 当 θ 变了:
- `v_θ` (DiT 的 velocity field) 变了 → `μ_k` 变了
- `c_t` (LLM conditioning) 也变了（因为 MoE 参数变了）
- 但 `y_{k+1}` 是旧轨迹的点，固定不变
- 所以高斯 log-prob 变了 → ratio ≠ 1

**GRPO + PPO-clip loss:**
$$\mathcal{L} = -\mathbb{E}\left[\min\left(r_t \cdot A, \text{clip}(r_t, 1\pm\epsilon) \cdot A\right)\right]$$

其中 A 是 GRPO 组内归一化的 advantage。

### 梯度回传路径

```
∂L/∂θ_DiT       ← 通过 v_θ 的预测影响 μ_k
∂L/∂θ_MoE_LLM   ← 通过 conditioning c_t 影响 μ_k
∂L/∂θ_Aggregator ← 通过 latent→embedding 投影影响 c_{t+1}
```

所有组件都参与梯度更新。

---

## 两种 log-prob 计算方法

### 1. 精确方法 (`--log_prob_method exact`)

保存完整 ODE 轨迹 `{y_0, y_1, ..., y_K}`，训练时用新参数重新计算 `v_{θ_new}` 在旧轨迹点上的值。

- **精确**: 无近似误差
- **存储**: T × K × patch_size × latent_dim floats/sample

### 2. DDPO 代理方法 (`--log_prob_method surrogate`)

只保存初始噪声 `y_0` 和最终 latent `a_t`，用随机时间步近似:

$$\log \pi_\theta(a_t | s_t) \approx -\mathbb{E}_{\tau}\left[\|v_\theta(x_\tau, \tau, c) - (a_t - y_0)\|^2\right]$$

- **高效**: 存储量最小
- **近似**: 有偏但在扩散 RL (DDPO) 中被广泛验证

---

## 环境安装

```bash
pip install torch>=2.1 transformers>=4.40 accelerate
pip install pandas pyarrow
pip install x-transformers   # Ming-omni-tts DiT 依赖

# 可选: verl 组件（用于分布式训练辅助）
pip install verl

# 克隆模型代码
git clone https://github.com/inclusionAI/Ming-omni-tts.git
export PYTHONPATH="$(pwd)/Ming-omni-tts:$PYTHONPATH"
```

---

## 数据准备

```bash
# 从 TTS manifest 转换
python -m ming_moe_verl.data.preprocess \
    --manifest /path/to/tts_manifest.jsonl \
    --output_dir ./data/tts_grpo

# 或创建示例数据
python -m ming_moe_verl.data.preprocess --create_demo --output_dir ./data/tts_grpo
```

---

## 快速开始

```bash
export MODEL_PATH=/path/to/ming-omni-tts-weights
export TRAIN_DATA=./data/tts_grpo/train.parquet

# 精确 log-prob 方法
python -m ming_moe_verl.train_flow_grpo \
    --model_path $MODEL_PATH \
    --train_data $TRAIN_DATA \
    --grpo_group_size 4 \
    --log_prob_method exact \
    --ode_steps 10 \
    --lr 1e-6 \
    --ppo_clip 0.2 \
    --kl_coef 0.001

# 或用 DDPO 代理方法（更省存储）
python -m ming_moe_verl.train_flow_grpo \
    --model_path $MODEL_PATH \
    --train_data $TRAIN_DATA \
    --log_prob_method surrogate
```

---

## 代码结构

```
ming_moe_verl/
├── model/
│   ├── policy_forward.py     # FlowGRPOPolicy
│   │   ├── _compute_ode_log_prob_exact()    # 精确 ODE 轨迹 log-prob
│   │   ├── _compute_ode_log_prob_surrogate() # DDPO 代理 log-prob
│   │   ├── rollout_step()                    # 单步 rollout + 保存轨迹
│   │   └── compute_sequence_log_probs()      # teacher-forced 序列 log-prob
│   └── rollout_worker.py     # FlowGRPORollout
│       ├── rollout_single()   # 完整自回归 flow matching 生成
│       └── generate_batch()   # 批量生成 N 个 rollout
├── reward/
│   └── tts_reward.py         # TTS 多维度 reward
├── data/
│   └── preprocess.py         # TTS 数据 → parquet
├── configs/                  # 配置文件
├── scripts/                  # 启动脚本
├── train_flow_grpo.py        # 主训练入口
└── train_flow_grpo_verl_native.py  # 说明为何不能直接用 verl 原生
```

---

## 与相关工作的对比

| 方法 | 动作空间 | log-prob 来源 | 适用模型 |
|------|---------|--------------|---------|
| verl GRPO | 离散 token | `log_softmax(logits)[token]` | 标准 LLM |
| DDPO | 连续 (扩散) | denoising loss surrogate | Diffusion model |
| **Flow-GRPO (本方案)** | **连续 (flow matching)** | **stochastic ODE 高斯转移** | **MoE-LLM + FM** |
| RLHF-Flow | 连续 | CNF exact likelihood | Flow model |

---

## 参考

- [GRPO](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization
- [DDPO](https://arxiv.org/abs/2305.13301) - Training Diffusion Models with RL
- [Flow Matching](https://arxiv.org/abs/2210.02747) - Conditional Flow Matching
- [verl](https://github.com/verl-project/verl) - RL framework for LLMs
- [Ming-omni-tts](https://github.com/inclusionAI/Ming-omni-tts)
