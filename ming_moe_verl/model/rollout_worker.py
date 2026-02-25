"""
Custom rollout worker for the Ming-omni-tts flow-matching TTS model.

verl's standard rollout workers (vLLM, SGLang) perform autoregressive
*discrete-token* sampling.  Ming-omni-tts generates *continuous* audio
latents through an iterative flow-matching process, so we implement a
custom rollout that:

1. Takes text prompts from verl's DataProto.
2. Encodes them into input_ids via the BailingTokenizer.
3. Runs the autoregressive flow-matching generation loop.
4. Returns generated latents, surrogate log-probs, and response masks
   in the format verl expects.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BailingMoeFlowRollout:
    """
    Rollout engine for BailingMoe + FlowMatching TTS.

    This replaces the standard vLLM/SGLang rollout in the verl pipeline.
    It is called by the ActorRollout worker during the rollout phase.
    """

    def __init__(
        self,
        policy_model,  # BailingMoeTTSForRL
        tokenizer,
        max_decode_steps: int = 200,
        cfg: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0,
        n: int = 4,
    ):
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.max_decode_steps = max_decode_steps
        self.cfg = cfg
        self.sigma = sigma
        self.temperature = temperature
        self.n = n

    def prepare_prompt_tokens(self, text: str, prompt_template: str = "tts") -> Dict[str, torch.Tensor]:
        """
        Convert a text prompt to tokenized input for the MoE model.
        """
        model = self.policy.model

        if model.model_type == "dense":
            prefix = self.tokenizer.encode(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
            )
            suffix = self.tokenizer.encode(
                "<|im_end|>\n<|im_start|>assistant\n<audio>"
            )
        else:
            prefix = self.tokenizer.encode("<role>HUMAN</role>")
            suffix = self.tokenizer.encode("<role>ASSISTANT</role><audio>")

        if prompt_template == "tts":
            task_prefix = self.tokenizer.encode("Please generate speech for the following text: Text input:\n")
        elif prompt_template == "tts_emotion":
            task_prefix = self.tokenizer.encode("Please generate speech with emotion: Text input:\n")
        else:
            task_prefix = []

        token_ids = prefix + task_prefix + self.tokenizer.encode(text) + suffix
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def generate_batch(
        self,
        texts: list,
        prompt_template: str = "tts",
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Generate n rollouts per prompt text.

        Returns a dict compatible with verl's DataProto expectations:
          - input_ids:       [B*n, prompt_len]
          - attention_mask:  [B*n, prompt_len + response_len]
          - responses:       [B*n, response_len]  (flattened latent indices)
          - old_log_probs:   [B*n, response_len]
          - response_mask:   [B*n, response_len]
          - audio_latents:   [B*n, max_steps, latent_dim * patch_size] (extra)
        """
        B = len(texts)
        all_results = {
            "input_ids": [],
            "attention_mask_prompt": [],
            "audio_latents": [],
            "old_log_probs": [],
            "response_mask": [],
            "prompt_texts": [],
        }

        for text in texts:
            tokens = self.prepare_prompt_tokens(text, prompt_template)
            input_ids = tokens["input_ids"].to(device)
            attn_mask = tokens["attention_mask"].to(device)

            for _ in range(self.n):
                latents, log_probs = self.policy.generate_audio_latents(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_decode_steps=self.max_decode_steps,
                    cfg=self.cfg,
                    sigma=self.sigma,
                    temperature=self.temperature,
                )
                # latents: [1, T, C*P], log_probs: [1, T]
                T = latents.shape[1]
                resp_mask = torch.ones(1, T, device=device)

                all_results["input_ids"].append(input_ids.squeeze(0))
                all_results["attention_mask_prompt"].append(attn_mask.squeeze(0))
                all_results["audio_latents"].append(latents.squeeze(0))
                all_results["old_log_probs"].append(log_probs.squeeze(0))
                all_results["response_mask"].append(resp_mask.squeeze(0))
                all_results["prompt_texts"].append(text)

        max_prompt_len = max(ids.shape[0] for ids in all_results["input_ids"])
        max_resp_len = max(lp.shape[0] for lp in all_results["old_log_probs"])
        latent_dim = all_results["audio_latents"][0].shape[-1]

        total = B * self.n
        padded = {
            "input_ids": torch.zeros(total, max_prompt_len, dtype=torch.long, device=device),
            "attention_mask": torch.zeros(total, max_prompt_len + max_resp_len, device=device),
            "audio_latents": torch.zeros(total, max_resp_len, latent_dim, device=device),
            "old_log_probs": torch.zeros(total, max_resp_len, device=device),
            "response_mask": torch.zeros(total, max_resp_len, device=device),
        }

        for i in range(total):
            pl = all_results["input_ids"][i].shape[0]
            rl = all_results["old_log_probs"][i].shape[0]

            padded["input_ids"][i, :pl] = all_results["input_ids"][i]
            padded["attention_mask"][i, :pl] = 1.0
            padded["attention_mask"][i, max_prompt_len:max_prompt_len + rl] = 1.0
            padded["audio_latents"][i, :rl] = all_results["audio_latents"][i]
            padded["old_log_probs"][i, :rl] = all_results["old_log_probs"][i]
            padded["response_mask"][i, :rl] = 1.0

        padded["prompt_texts"] = all_results["prompt_texts"]
        return padded
