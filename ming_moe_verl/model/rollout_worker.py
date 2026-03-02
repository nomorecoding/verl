"""
自定义 Rollout 实现: Ming-omni-tts MoE + Flow Matching TTS

=== 为什么必须自己实现 rollout? ===

verl 提供的 rollout 方案全部基于 "离散 token 自回归" 假设:

1. vLLM rollout (verl 默认):
   - vLLM 内部有一套自己的模型实现 (不是 HuggingFace 原模型)
   - 支持的 MoE: DeepSeekV2/V3, Mixtral, Qwen2-MoE, Qwen3-MoE
   - BailingMoe 不在 vLLM 支持列表中 (model_type="bailing_moe" 未注册)
   - 即使加了 vLLM 支持, vLLM 只做 lm_head → softmax → token, 不会执行 flow matching

2. SGLang rollout:
   - 同样的问题, 只支持标准 causal LM 推理

3. HFRollout / NaiveRollout (verl 的 HuggingFace 回退):
   - 调用 model.generate() 或手动 for loop: logits → softmax → sample token
   - 即使 BailingMoeForCausalLM 有 generate() 方法, 那也是做文本生成
   - TTS 生成在 BailingMMNativeForConditionalGeneration.sample() 中, 走的是完全不同的路径

Ming-omni-tts TTS 的生成循环:
  input_ids → word_embedding → LLM forward → hidden_state
       → flow matching (DiT + ODE solver) → audio latent
       → Aggregator 投影 → 回到 LLM embedding → 重复

这个循环里:
  - 不走 lm_head (所以 vLLM 的整个 sampling 逻辑无用)
  - 需要 DiT、CFM、Aggregator、stop_head 等额外模块 (vLLM 里没有)
  - 每步输入是 continuous embedding 而非 token id (vLLM 不支持)
  - ODE solver 有多步积分 (vLLM 的 KV cache 机制不覆盖)

=== 所以: 是的, rollout 必须完全自己实现 ===

本模块实现了自定义的 TTS rollout:
  - 手动执行 LLM + flow matching 的自回归循环
  - 保存完整 ODE 轨迹 (用于训练时精确重算 log-prob)
  - 与 verl 的 DataProto 无关, 使用独立的数据结构

如果未来想接入 verl 的 Ray 分布式调度, 可以将此 rollout
封装为 Ray Actor, 但生成逻辑本身必须是自定义的。
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class FlowGRPORollout:
    """
    Ming-omni-tts 的自定义 TTS Rollout。

    与 verl 的 BaseRollout 对比:
    ┌─────────────────┬───────────────────────┬──────────────────────────┐
    │                 │ verl BaseRollout      │ FlowGRPORollout (本类)     │
    ├─────────────────┼───────────────────────┼──────────────────────────┤
    │ 推理引擎         │ vLLM / SGLang / HF   │ 原生 PyTorch (手动循环)    │
    │ 输出类型         │ token ids (离散)      │ audio latents (连续)      │
    │ log-prob        │ log_softmax(logits)   │ ODE Gaussian log-prob    │
    │ 需要额外模块      │ 只需 LLM             │ LLM + DiT + CFM + Agg    │
    │ 输入下一步       │ token embedding      │ Aggregator(latent)       │
    │ 终止条件         │ EOS token            │ stop_head 概率           │
    │ 每步计算量       │ 1次 LLM forward      │ 1次 LLM + K步 ODE 积分    │
    └─────────────────┴───────────────────────┴──────────────────────────┘
    """

    def __init__(
        self,
        policy,  # FlowGRPOPolicy
        tokenizer,
        max_decode_steps: int = 200,
        cfg: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0,
        n: int = 4,
    ):
        self.policy = policy
        self.tokenizer = tokenizer
        self.max_decode_steps = max_decode_steps
        self.cfg = cfg
        self.sigma = sigma
        self.temperature = temperature
        self.n = n

    def tokenize_prompt(self, text: str) -> Dict[str, torch.Tensor]:
        """把文本编码为 BailingMoe 格式的 input_ids。

        注意这里的 token 格式与标准 chat 模板不同:
        - MoE 模型用 <role>HUMAN</role> / <role>ASSISTANT</role>
        - Dense 模型用 <|im_start|>system / user / assistant
        """
        model = self.policy.model

        if model.model_type == "dense":
            tokens = (
                self.tokenizer.encode(
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n"
                    "Please generate speech for the following text: Text input:\n"
                )
                + self.tokenizer.encode(text)
                + self.tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n<audio>")
            )
        else:
            tokens = (
                self.tokenizer.encode("<role>HUMAN</role>")
                + self.tokenizer.encode(
                    "Please generate speech for the following text: Text input:\n"
                )
                + self.tokenizer.encode(text)
                + self.tokenizer.encode("<role>ASSISTANT</role><audio>")
            )

        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def rollout_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        对一条 prompt 执行完整的 TTS 自回归 flow matching 生成。

        这是整个 rollout 的核心, 手动实现了 BailingMMNativeForConditionalGeneration.sample()
        的逻辑, 但额外保存了 ODE 轨迹信息用于后续 GRPO 训练。

        与原始 sample() 的区别:
        1. 保存每步的完整 ODE 轨迹 {y_0, ..., y_K}
        2. 计算每步的 log-prob (基于高斯转移概率)
        3. 不做音频解码 (reward 阶段单独处理)

        Returns:
            dict:
              "step_data": list[dict]  每步的 ODE 轨迹信息
              "per_step_log_probs": [1, T]  每步的 log-prob
              "total_steps": int
        """
        model = self.policy.model
        B = input_ids.shape[0]
        latent_dim = model.latent_dim
        patch_size = model.patch_size
        history_patch_size = model.history_patch_size

        inputs_embeds = model.model.get_input_embeddings()(input_ids.to(device))

        if model.model_type != "dense":
            position_ids, rope_deltas = model.get_rope_index(
                input_ids.to(device),
                image_token_id=model.config.llm_config.image_patch_token,
                video_token_id=model.config.llm_config.image_patch_token,
                image_start_token_id=getattr(
                    model.config.llm_config, "image_start_token", 0
                ),
                video_start_token_id=getattr(
                    model.config.llm_config, "video_start_token", 0
                ),
                image_grid_thw=None,
                video_grid_thw=None,
                attention_mask=attention_mask.to(device),
            )
        else:
            position_ids = (
                attention_mask.to(device).cumsum(-1) - 1
            ).masked_fill_(attention_mask.to(device) == 0, 1)
            rope_deltas = None

        past_key_values = None
        latent_history = torch.zeros(
            B, history_patch_size, latent_dim, device=device
        )
        step_data_list = []
        per_step_log_probs = []

        for step in range(self.max_decode_steps):
            # (1) MoE-LLM forward → hidden state
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model.model(
                    attention_mask=(
                        attention_mask.to(device)
                        if attention_mask is not None
                        else None
                    ),
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    audio_mask=None,
                    image_mask=None,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            past_key_values = outputs.past_key_values
            z = outputs.hidden_states[-1][:, -1:, :]  # [B, 1, D]

            # (2) Flow matching rollout (保存完整 ODE 轨迹)
            step_result = self.policy.rollout_step(
                conditioning=z,
                latent_history=latent_history,
                cfg=self.cfg,
                sigma=self.sigma,
                temperature=self.temperature,
            )

            step_data_list.append(
                {k: v.cpu() for k, v in step_result.items()}
            )
            per_step_log_probs.append(step_result["log_prob"])

            # (3) 检查 stop_head
            stop_prob = model.stop_head(z)[0, 0].softmax(dim=-1)[1]
            if stop_prob > 0.5 and step > 3:
                break

            # (4) 准备下一步: latent → Aggregator → embedding
            inputs_embeds = model.linear_proj_audio(step_result["latent"])

            if model.model_type == "dense":
                position_ids = position_ids[:, -1:] + 1
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                if past_key_values and rope_deltas is not None:
                    delta = (
                        past_key_values[0][1].shape[2] + rope_deltas
                    )
                elif past_key_values:
                    delta = torch.tensor(
                        past_key_values[0][1].shape[2], device=device
                    )
                else:
                    delta = torch.tensor(0, device=device)
                pos = torch.arange(seq_length, device=device)
                pos = pos.view(1, -1).expand(batch_size, -1).add(delta)
                position_ids = pos.unsqueeze(0).expand(3, -1, -1)

            attention_mask = torch.ones(B, 1, device=device)

            # (5) 更新 latent history
            latent_history[:, :-patch_size, :] = (
                latent_history[:, patch_size:, :].clone()
            )
            latent_history[:, -patch_size:, :] = (
                step_result["latent"]
                .reshape(B, patch_size, latent_dim)
                .to(device)
            )

        return {
            "step_data": step_data_list,
            "per_step_log_probs": torch.stack(
                per_step_log_probs, dim=1
            ),  # [B, T]
            "total_steps": len(step_data_list),
        }

    @torch.no_grad()
    def generate_batch(
        self,
        texts: List[str],
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        对一批 prompts 各生成 n 个 rollout (GRPO group)。

        Returns:
            dict:
              "rollouts": list[dict]  每个 rollout 的完整数据
              "input_ids": list  每个 rollout 的 prompt tokens
              "attention_mask": list
              "prompt_texts": list[str]
              "group_indices": np.ndarray  GRPO 分组索引
        """
        import numpy as np

        all_rollouts = []
        all_input_ids = []
        all_attn_masks = []
        all_texts = []
        group_indices = []

        for prompt_idx, text in enumerate(texts):
            tokens = self.tokenize_prompt(text)

            for _ in range(self.n):
                rollout = self.rollout_single(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    device=device,
                )
                all_rollouts.append(rollout)
                all_input_ids.append(tokens["input_ids"])
                all_attn_masks.append(tokens["attention_mask"])
                all_texts.append(text)
                group_indices.append(prompt_idx)

        return {
            "rollouts": all_rollouts,
            "input_ids": all_input_ids,
            "attention_mask": all_attn_masks,
            "prompt_texts": all_texts,
            "group_indices": np.array(group_indices),
        }
