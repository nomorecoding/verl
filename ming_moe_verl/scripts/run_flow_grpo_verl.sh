#!/bin/bash
# =============================================================================
# 说明: 为什么不能直接用 verl 的标准 GRPO pipeline
# =============================================================================
echo "================================================================"
echo "  注意: verl 的标准 GRPO pipeline 不适用于 Ming-omni-tts TTS 训练"
echo ""
echo "  原因:"
echo "    1. verl 的 vLLM rollout 会走 lm_head → softmax → 离散 token 采样"
echo "    2. 但 TTS 模式下 LLM 用 hidden states 条件化 flow matching"
echo "    3. 从未经过 lm_head, 动作空间是连续 latent, 不是离散 token"
echo "    4. 标准 GRPO 的 log-prob = log_softmax(logits)[token] 完全不适用"
echo ""
echo "  正确方案: 使用 run_flow_grpo_standalone.sh"
echo "    - 自定义 rollout: MoE-LLM + flow matching 循环"
echo "    - 自定义 log-prob: stochastic ODE 高斯转移概率"
echo "    - 自定义 reward: TTS 质量指标"
echo ""
echo "  运行:"
echo "    bash scripts/run_flow_grpo_standalone.sh"
echo "================================================================"

echo ""
echo "可以复用 verl 的部分组件 (GRPO advantage 计算, FSDP, checkpoint 等):"
echo "  from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage"
echo "  from verl.workers.engine.fsdp.transformer_impl import FSDPTransformerEngine"
