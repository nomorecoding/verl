"""
关于用 verl 原生 pipeline 训练 Ming-omni-tts 的说明。

=== 为什么不能直接用 verl 的标准 GRPO? ===

verl 的标准 pipeline 假设:
  1. vLLM/SGLang 做 rollout → 生成离散 tokens
  2. log π = log_softmax(logits)[token_id]
  3. 动作空间 = 词表 {1, ..., V}

但 Ming-omni-tts 的 TTS 模式:
  1. LLM hidden states → flow matching → 连续 audio latents
  2. 从未经过 lm_head / softmax / 离散采样
  3. 动作空间 = ℝ^{patch_size × latent_dim}

如果强行把 BailingMoe 塞进 verl 的 vLLM rollout:
  - 模型会走 lm_head 做文本 token 采样
  - 训出来的 GRPO 信号优化的是文本生成能力
  - 与 TTS 音频质量完全无关

=== 正确方案 ===

必须用自定义训练循环 (train_flow_grpo.py):
  - 自定义 rollout: LLM + flow matching 循环
  - 自定义 log-prob: 基于 stochastic ODE solver 的高斯转移概率
  - 自定义 reward: TTS 质量指标 (ASR/MOS/说话人相似度)

=== 唯一可以用 verl 原生 GRPO 的场景 ===

如果目标是训练 BailingMoe 的**纯文本生成能力** (而非 TTS),
那么标准 verl pipeline 可以直接使用。但这不是 TTS 训练。

=== 如何利用 verl 的分布式基础设施? ===

虽然不能直接用 verl 的端到端 pipeline, 但可以复用其组件:

1. **FSDP 包装**: 用 verl 的 FSDP engine 做模型并行
   from verl.workers.engine.fsdp.transformer_impl import FSDPTransformerEngine

2. **GRPO 算法**: 直接调用 verl 的 advantage 计算
   from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

3. **Ray 调度**: 用 verl 的 ResourcePoolManager 分配 GPU
   from verl.single_controller.ray import ResourcePoolManager

4. **Checkpoint**: 用 verl 的 checkpoint engine
   from verl.checkpoint_engine import CheckpointEngineManager

这些组件都可以在自定义训练循环中组合使用。
"""


def print_explanation():
    print(__doc__)


if __name__ == "__main__":
    print_explanation()
