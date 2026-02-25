"""
Wraps Ming-omni-tts BailingMoe+FlowMatching pipeline into a unified
policy model that verl's actor/rollout workers can drive.

The key challenge: standard verl expects discrete-token generation with
log-probs, but Ming-omni-tts generates *continuous* audio latents via
flow-matching conditioned on the MoE-LLM's hidden states.

We bridge the gap by:
1. Treating each flow-matching step as one "policy action".
2. Computing a surrogate log-probability from the CFM loss (negative MSE)
   for each generated latent, which the GRPO advantage can weight.
3. Providing a differentiable `forward_policy` for the PPO/GRPO update.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BailingMoeTTSForRL(nn.Module):
    """
    Wraps BailingMMNativeForConditionalGeneration so that it exposes:
      - compute_log_prob(input_ids, attention_mask, audio_latents, ...)
            → per-step surrogate log-probs  (shape [B, T])
      - generate(input_ids, attention_mask, ...)
            → sampled audio latents + surrogate log-probs
    """

    def __init__(self, model, tokenizer, patch_size: int = 4, history_patch_size: int = 32):
        super().__init__()
        self.model = model  # BailingMMNativeForConditionalGeneration (already loaded)
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.history_patch_size = history_patch_size

    # ------------------------------------------------------------------ #
    #  Surrogate log-prob: we re-use the CFM's training loss (MSE between
    #  predicted flow and true flow) as a *proxy* for log π(a|s).
    #
    #  log π(a_t | s_t) ≈ -½ ‖v_θ(x_t, t, c) - (x₁ - x₀)‖²
    #
    #  This is a common trick in continuous-action RL with diffusion/flow
    #  policies (e.g. Diffusion Policy, DDPO, FlowRL).
    # ------------------------------------------------------------------ #

    def compute_log_prob_for_step(
        self,
        hidden_state: torch.Tensor,  # [B, 1, D]  last hidden state from MoE LLM
        target_latent: torch.Tensor,  # [B, C, patch_size]  the generated audio latent
        latent_history: torch.Tensor,  # [B, history_len, C]
    ) -> torch.Tensor:
        """
        Compute surrogate log-prob for a single flow-matching generation step.
        Returns shape [B].
        """
        cfm = self.model.flowloss.cfm
        dit = cfm.model

        x1 = target_latent  # [B, C, patch_size] or [B, patch_size, C] depending on convention
        if x1.dim() == 3 and x1.shape[-1] != self.model.latent_dim:
            x1 = x1.transpose(1, 2)  # → [B, patch_size, C]

        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        # uniform t for the surrogate (we average over a few t values for stability)
        num_t_samples = 4
        log_probs = []
        for _ in range(num_t_samples):
            t = torch.rand(B, device=x1.device, dtype=x1.dtype)
            t_expand = t.unsqueeze(-1).unsqueeze(-1)
            x_t = (1 - t_expand) * x0 + t_expand * x1
            flow_target = x1 - x0

            pred = dit(
                x=x_t,
                t=t,
                c=hidden_state,
                latent_history=latent_history,
            )
            pred = pred[:, -self.patch_size:, :]

            mse = F.mse_loss(pred, flow_target, reduction="none")  # [B, patch, C]
            mse = mse.mean(dim=(1, 2))  # [B]
            log_probs.append(-0.5 * mse)

        return torch.stack(log_probs, dim=0).mean(dim=0)  # [B]

    def compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_latent_sequence: torch.Tensor,  # [B, num_steps, latent_dim * patch_size]
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Re-runs the autoregressive loop teacher-forced, collecting
        per-step surrogate log-probs.

        Returns: log_probs [B, num_steps]
        """
        B, num_steps, _ = audio_latent_sequence.shape
        latent_dim = self.model.latent_dim
        patch_size = self.patch_size

        audio_latents = audio_latent_sequence.view(B, num_steps, patch_size, latent_dim)

        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if self.model.model_type != "dense" and position_ids is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_token_id=self.model.config.llm_config.image_patch_token,
                video_token_id=self.model.config.llm_config.image_patch_token,
                image_start_token_id=getattr(self.model.config.llm_config, "image_start_token", 0),
                video_start_token_id=getattr(self.model.config.llm_config, "video_start_token", 0),
                image_grid_thw=None,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
        else:
            rope_deltas = None

        past_key_values = None
        log_probs_list = []
        latent_history = torch.zeros(B, self.history_patch_size, latent_dim, device=input_ids.device)

        for step in range(num_steps):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model.model(
                    attention_mask=attention_mask,
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

            target_latent = audio_latents[:, step]  # [B, patch_size, latent_dim]
            step_log_prob = self.compute_log_prob_for_step(z, target_latent, latent_history)
            log_probs_list.append(step_log_prob)

            # prepare next step inputs (teacher forcing)
            sampled = target_latent  # [B, patch_size, latent_dim]
            inputs_embeds = self.model.linear_proj_audio(sampled)  # [B, 1?, D]

            if self.model.model_type == "dense":
                position_ids = position_ids[:, -1:] + 1
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                if past_key_values and rope_deltas is not None:
                    delta = past_key_values[0][1].shape[2] + rope_deltas
                elif past_key_values:
                    delta = torch.tensor(past_key_values[0][1].shape[2], device=inputs_embeds.device)
                else:
                    delta = torch.tensor(0, device=inputs_embeds.device)
                pos = torch.arange(seq_length, device=inputs_embeds.device)
                pos = pos.view(1, -1).expand(batch_size, -1).add(delta)
                position_ids = pos.unsqueeze(0).expand(3, -1, -1)

            attention_mask = torch.ones(B, 1, device=inputs_embeds.device)

            latent_history[:, :-patch_size, :] = latent_history[:, patch_size:, :].clone()
            latent_history[:, -patch_size:, :] = sampled.view(B, patch_size, latent_dim)

        return torch.stack(log_probs_list, dim=1)  # [B, num_steps]

    @torch.no_grad()
    def generate_audio_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_decode_steps: int = 200,
        cfg: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate audio latents autoregressively.
        Returns:
            latents: [B, num_steps, patch_size * latent_dim]
            log_probs: [B, num_steps]  surrogate log-probs
        """
        B = input_ids.shape[0]
        latent_dim = self.model.latent_dim
        patch_size = self.patch_size
        device = input_ids.device

        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if self.model.model_type != "dense" and position_ids is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_token_id=self.model.config.llm_config.image_patch_token,
                video_token_id=self.model.config.llm_config.image_patch_token,
                image_start_token_id=getattr(self.model.config.llm_config, "image_start_token", 0),
                video_start_token_id=getattr(self.model.config.llm_config, "video_start_token", 0),
                image_grid_thw=None,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
        else:
            position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(attention_mask == 0, 1)
            rope_deltas = None

        past_key_values = None
        latent_history = torch.zeros(B, self.history_patch_size, latent_dim, device=device)
        all_latents = []
        all_log_probs = []

        for step in range(max_decode_steps):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model.model(
                    attention_mask=attention_mask,
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
            z = outputs.hidden_states[-1][:, -1:, :]

            sampled, _ = self.model.flowloss.sample(
                z, latent_history, cfg, patch_size, sigma=sigma, temperature=temperature
            )
            all_latents.append(sampled.view(B, -1))

            step_lp = self.compute_log_prob_for_step(z, sampled, latent_history)
            all_log_probs.append(step_lp)

            stop_probs = self.model.stop_head(z)[:, 0].softmax(dim=-1)[:, 1]
            if (stop_probs > 0.5).all() and step > 3:
                break

            inputs_embeds = self.model.linear_proj_audio(sampled)

            if self.model.model_type == "dense":
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
            latent_history[:, -patch_size:, :] = sampled.view(B, patch_size, latent_dim)

        max_len = len(all_latents)
        latent_seq = torch.stack(all_latents, dim=1)  # [B, T, C*P]
        log_prob_seq = torch.stack(all_log_probs, dim=1)  # [B, T]

        return latent_seq, log_prob_seq
