"""
Custom rollout worker for continuous-action Flow-GRPO.

与标准 verl rollout 的核心区别:
  - 标准 verl: vLLM 做离散 token 采样, 保存 token ids 和 per-token log-probs
  - Flow-GRPO: 执行 LLM + flow matching 循环, 保存连续 latent 和 ODE 轨迹

每一步的产出:
  - audio latent a_t (最终生成结果)
  - ODE 轨迹 {y_0, y_1, ..., y_K} (用于训练时精确重算 log-prob)
  - old log-prob (rollout 时的 log π_{θ_old})
  - LLM hidden state (用于 reward 计算)
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class FlowGRPORollout:
    """
    执行 BailingMoe + FlowMatching 的 rollout,
    返回 GRPO 训练所需的全部数据。
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
        """把文本编码为 MoE 模型需要的 input_ids。"""
        model = self.policy.model

        if model.model_type == "dense":
            tokens = (
                self.tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
                + self.tokenizer.encode("<|im_start|>user\n")
                + self.tokenizer.encode("Please generate speech for the following text: Text input:\n")
                + self.tokenizer.encode(text)
                + self.tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n<audio>")
            )
        else:
            tokens = (
                self.tokenizer.encode("<role>HUMAN</role>")
                + self.tokenizer.encode("Please generate speech for the following text: Text input:\n")
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
        对一个 prompt 执行完整的自回归 flow matching 生成。

        Returns:
            dict with:
              - "step_data": list of per-step dicts (latent, ode_trajectory, etc.)
              - "per_step_log_probs": [T]
              - "total_steps": int
        """
        model = self.policy.model
        B = input_ids.shape[0]  # 通常 = 1
        latent_dim = model.latent_dim
        patch_size = model.patch_size
        history_patch_size = model.history_patch_size

        inputs_embeds = model.model.get_input_embeddings()(input_ids.to(device))

        # Position ids
        if model.model_type != "dense":
            position_ids, rope_deltas = model.get_rope_index(
                input_ids.to(device),
                image_token_id=model.config.llm_config.image_patch_token,
                video_token_id=model.config.llm_config.image_patch_token,
                image_start_token_id=getattr(model.config.llm_config, "image_start_token", 0),
                video_start_token_id=getattr(model.config.llm_config, "video_start_token", 0),
                image_grid_thw=None, video_grid_thw=None,
                attention_mask=attention_mask.to(device),
            )
        else:
            position_ids = (attention_mask.to(device).cumsum(-1) - 1).masked_fill_(
                attention_mask.to(device) == 0, 1
            )
            rope_deltas = None

        past_key_values = None
        latent_history = torch.zeros(B, history_patch_size, latent_dim, device=device)
        step_data_list = []
        per_step_log_probs = []

        for step in range(self.max_decode_steps):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model.model(
                    attention_mask=attention_mask.to(device) if attention_mask is not None else None,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    audio_mask=None, image_mask=None,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            past_key_values = outputs.past_key_values
            z = outputs.hidden_states[-1][:, -1:, :]  # [B, 1, D]

            # Flow matching rollout step (保存完整 ODE 轨迹)
            step_result = self.policy.rollout_step(
                conditioning=z,
                latent_history=latent_history,
                cfg=self.cfg,
                sigma=self.sigma,
                temperature=self.temperature,
            )

            step_data_list.append({
                "latent": step_result["latent"].cpu(),
                "ode_trajectory": step_result["ode_trajectory"].cpu(),
                "ode_timesteps": step_result["ode_timesteps"].cpu(),
                "initial_noise": step_result["initial_noise"].cpu(),
                "log_prob": step_result["log_prob"].cpu(),
            })
            per_step_log_probs.append(step_result["log_prob"])

            # 检查是否应该停止
            stop_prob = model.stop_head(z)[0, 0].softmax(dim=-1)[1]
            if stop_prob > 0.5 and step > 3:
                break

            # 准备下一步输入
            inputs_embeds = model.linear_proj_audio(step_result["latent"])

            if model.model_type == "dense":
                position_ids = position_ids[:, -1:] + 1
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                if past_key_values and rope_deltas is not None:
                    delta = past_key_values[0][1].shape[2] + rope_deltas
                elif past_key_values:
                    delta = torch.tensor(past_key_values[0][1].shape[2], device=device)
                else:
                    delta = torch.tensor(0, device=device)
                pos = torch.arange(seq_length, device=device)
                pos = pos.view(1, -1).expand(batch_size, -1).add(delta)
                position_ids = pos.unsqueeze(0).expand(3, -1, -1)

            attention_mask = torch.ones(B, 1, device=device)
            latent_history[:, :-patch_size, :] = latent_history[:, patch_size:, :].clone()
            latent_history[:, -patch_size:, :] = step_result["latent"].reshape(
                B, patch_size, latent_dim
            ).to(device)

        return {
            "step_data": step_data_list,
            "per_step_log_probs": torch.stack(per_step_log_probs, dim=1),  # [B, T]
            "total_steps": len(step_data_list),
        }

    @torch.no_grad()
    def generate_batch(
        self,
        texts: List[str],
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        对一批 prompts 生成 n 个 rollout。

        Returns dict with:
          - "input_ids": list of [1, prompt_len]
          - "attention_mask": list of [1, prompt_len]
          - "rollouts": list of rollout dicts (each from rollout_single)
          - "prompt_texts": list of str
          - "group_indices": [B*n] numpy array for GRPO grouping
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
