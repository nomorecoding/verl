"""
Continuous-action Flow-GRPO policy for Ming-omni-tts MoE + Flow Matching.

=== 核心问题 ===

标准 GRPO 作用于离散 token 空间:
  - 动作:  a_t ∈ {1, ..., V}  (词表中的 token id)
  - 策略:  π(a_t | s_t) = softmax(logits)[a_t]
  - log-prob:  log_softmax(logits)[a_t]

Ming-omni-tts 的 TTS 生成是连续动作空间:
  - LLM 不走 lm_head，而是用 hidden_state 条件化 flow matching
  - 动作:  a_t ∈ ℝ^(patch_size × latent_dim)  (音频 latent 向量)
  - 策略:  由随机 ODE solver 隐式定义

=== 解决方案: 利用 Stochastic ODE Solver 的显式概率 ===

Ming-omni-tts 的 Solver 在每个积分步加了高斯噪声:
    y_{k+1} = y_k + dt · v_θ(y_k, t_k, c) + σ·√(T·|dt|) · ε_k

其中 ε_k ~ N(0, I)。这使得每步的转移概率是一个显式高斯:

    p(y_{k+1} | y_k, c; θ) = N(y_{k+1} | μ_k, σ_k² · I)

    其中:
        μ_k = y_k + dt_k · v_θ(y_k, t_k, c)
        σ_k = σ · √(T · |dt_k|)

因此 per-ODE-step log-prob:
    log p(y_{k+1} | y_k, c; θ) = -1/(2σ_k²) · ‖y_{k+1} - μ_k‖² - d/2 · log(2πσ_k²)

Per-autoregressive-step log-prob (K 个 ODE 步):
    log π(a_t | s_t; θ) = log p(y_0) + Σ_{k=0}^{K-1} log p(y_{k+1} | y_k, c_t; θ)

Importance ratio for GRPO/PPO:
    r_t = exp(log π_{θ_new}(a_t | s_t) - log π_{θ_old}(a_t | s_t))

注意: y_0 的 log-prob 在 new 和 old 之间相同 (都是标准正态), 所以 ratio 中抵消。
关键差异来自 v_θ 的变化: θ 变了 → μ_k 变了 → Gaussian log-prob 变了。

=== 两种实现 ===

1. **精确方法 (ExactODELogProb)**: 保存完整 ODE 轨迹 {y_0, ..., y_K},
   训练时重新计算 v_{θ_new} 在旧轨迹点上的值。精确但需要更多存储。

2. **DDPO 代理方法 (SurrogateLogProb)**: 只保存初始噪声 y_0 和最终 latent a_t,
   用随机时间步近似 log-prob。近似但高效，在扩散 RL 中广泛使用。
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowGRPOPolicy(nn.Module):
    """
    Wraps BailingMMNativeForConditionalGeneration 为连续动作空间的 RL 策略。

    核心接口:
      - rollout(): 生成 audio latents + 保存 ODE 轨迹 + 计算 old log-probs
      - compute_log_prob(): 在新参数下重新计算 old trajectory 的 log-probs
    """

    def __init__(
        self,
        model,  # BailingMMNativeForConditionalGeneration
        tokenizer,
        ode_steps: int = 10,
        log_prob_method: str = "exact",  # "exact" or "surrogate"
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.patch_size = model.patch_size
        self.history_patch_size = model.history_patch_size
        self.latent_dim = model.latent_dim
        self.ode_steps = ode_steps
        self.log_prob_method = log_prob_method

    # ================================================================
    #  精确 log-prob: 基于保存的 ODE 轨迹
    # ================================================================

    def _compute_ode_log_prob_exact(
        self,
        ode_trajectory: torch.Tensor,  # [K+1, B, patch_size, latent_dim]
        ode_timesteps: torch.Tensor,    # [K+1]
        conditioning: torch.Tensor,     # [B, 1, hidden_dim] from LLM
        latent_history: torch.Tensor,   # [B, history_len, latent_dim]
        solver_sigma: float,
        solver_temperature: float,
    ) -> torch.Tensor:
        """
        给定旧 ODE 轨迹, 用当前 θ 计算 log π(trajectory | c; θ_new)。

        关键: y_k 是旧轨迹的点 (固定), 但 v_θ 用新参数计算。

        Returns: [B] per-sample log-prob (对 K 步求和)
        """
        dit = self.model.flowloss.cfm.model
        K = ode_trajectory.shape[0] - 1
        B = ode_trajectory.shape[1]
        d = self.patch_size * self.latent_dim

        total_log_prob = torch.zeros(B, device=conditioning.device)

        for k in range(K):
            t_k = ode_timesteps[k]
            t_k1 = ode_timesteps[k + 1]
            dt = t_k1 - t_k

            y_k = ode_trajectory[k]      # [B, patch_size, latent_dim]
            y_k1 = ode_trajectory[k + 1]  # [B, patch_size, latent_dim]

            sigma_k = solver_sigma * math.sqrt(solver_temperature * abs(dt.item()))

            if sigma_k < 1e-10:
                continue

            t_batch = t_k.expand(B)
            v_k = dit(
                x=y_k,
                t=t_batch,
                c=conditioning,
                latent_history=latent_history,
            )
            v_k = v_k[:, -self.patch_size:, :]  # [B, patch_size, latent_dim]

            # 确定性预测: μ_k = y_k + dt · v_θ(y_k, t_k, c)
            mu_k = y_k + dt * v_k  # [B, patch_size, latent_dim]

            # 高斯 log-prob: log N(y_{k+1} | μ_k, σ_k² I)
            diff = (y_k1 - mu_k).reshape(B, -1)  # [B, d]
            log_p_k = -0.5 * (diff ** 2).sum(dim=-1) / (sigma_k ** 2)
            log_p_k = log_p_k - 0.5 * d * math.log(2 * math.pi * sigma_k ** 2)

            total_log_prob = total_log_prob + log_p_k

        return total_log_prob  # [B]

    # ================================================================
    #  DDPO 代理 log-prob: 基于随机时间步
    # ================================================================

    def _compute_ode_log_prob_surrogate(
        self,
        initial_noise: torch.Tensor,    # [B, patch_size, latent_dim]  y_0
        final_latent: torch.Tensor,      # [B, patch_size, latent_dim]  a_t
        conditioning: torch.Tensor,      # [B, 1, hidden_dim]
        latent_history: torch.Tensor,    # [B, history_len, latent_dim]
        num_t_samples: int = 4,
    ) -> torch.Tensor:
        """
        DDPO 风格的代理 log-prob。

        log π(a_t | s_t) ≈ -E_τ [‖v_θ(x_τ, τ, c) - (a_t - y_0)‖²]

        其中 x_τ = (1-τ)y_0 + τ·a_t, τ ~ U(0,1)

        Returns: [B]
        """
        dit = self.model.flowloss.cfm.model
        B = final_latent.shape[0]

        flow_target = final_latent - initial_noise  # [B, patch_size, latent_dim]

        log_probs = []
        for _ in range(num_t_samples):
            tau = torch.rand(B, device=final_latent.device, dtype=final_latent.dtype)
            tau_expand = tau.unsqueeze(-1).unsqueeze(-1)
            x_tau = (1 - tau_expand) * initial_noise + tau_expand * final_latent

            v_pred = dit(
                x=x_tau,
                t=tau,
                c=conditioning,
                latent_history=latent_history,
            )
            v_pred = v_pred[:, -self.patch_size:, :]

            mse = (v_pred - flow_target).reshape(B, -1).pow(2).sum(dim=-1)
            log_probs.append(-0.5 * mse)

        return torch.stack(log_probs).mean(dim=0)  # [B]

    # ================================================================
    #  Rollout: 生成 + 保存所需信息
    # ================================================================

    @torch.no_grad()
    def rollout_step(
        self,
        conditioning: torch.Tensor,     # [B, 1, D]
        latent_history: torch.Tensor,    # [B, history_len, latent_dim]
        cfg: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        执行一个自回归步骤的 flow matching 采样,
        同时保存 ODE 轨迹用于后续 log-prob 计算。

        Returns dict with:
            - "latent": [B, patch_size, latent_dim]  生成的 audio latent
            - "ode_trajectory": [K+1, B, patch_size, latent_dim]  完整 ODE 轨迹
            - "ode_timesteps": [K+1]  ODE 时间步
            - "initial_noise": [B, patch_size, latent_dim]  初始噪声 y_0
            - "log_prob": [B]  rollout 时的 log-prob
        """
        B = conditioning.shape[0]
        device = conditioning.device

        noise = torch.randn(B, self.latent_dim, self.patch_size, device=device)

        cfm = self.model.flowloss.cfm
        dit = cfm.model

        # 构造带 CFG 的 velocity function
        def fn(t, x):
            if cfg < 1e-5:
                return dit(x=x, time=t, latent_history=latent_history)
            pred_cfg = dit.forward_with_cfg(
                x=x, t=t, c=conditioning,
                latent_history=latent_history,
                cfg_scale=cfg,
                patch_size=self.patch_size,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg

        y0 = noise.transpose(1, 2)  # [B, patch_size, latent_dim]
        initial_noise = y0.clone()

        from fm.CFM import get_epss_timesteps
        timesteps = get_epss_timesteps(self.ode_steps, device=device, dtype=noise.dtype)

        # 手动执行 Solver 以保存完整轨迹
        trajectory = [y0.clone()]
        ode_noise_vectors = []
        y = y0

        for i in range(len(timesteps) - 1):
            t0, t1 = timesteps[i], timesteps[i + 1]
            dt = t1 - t0

            v = fn(t0, y)
            y_det = y + dt * v  # 确定性步

            eps = torch.randn_like(y)
            ode_noise_vectors.append(eps)

            shift = sigma * math.sqrt(max(temperature, 0) * abs(dt.item())) * eps
            y = y_det + shift

            trajectory.append(y.clone())

        ode_trajectory = torch.stack(trajectory, dim=0)  # [K+1, B, patch, latent]
        final_latent = trajectory[-1]

        # 计算 rollout log-prob
        log_prob = self._compute_ode_log_prob_exact(
            ode_trajectory=ode_trajectory,
            ode_timesteps=timesteps,
            conditioning=conditioning,
            latent_history=latent_history,
            solver_sigma=sigma,
            solver_temperature=max(temperature, 1e-6),
        )

        return {
            "latent": final_latent,                  # [B, patch, latent]
            "ode_trajectory": ode_trajectory,         # [K+1, B, patch, latent]
            "ode_timesteps": timesteps,               # [K+1]
            "initial_noise": initial_noise,           # [B, patch, latent]
            "log_prob": log_prob,                     # [B]
        }

    # ================================================================
    #  训练时 log-prob 重算 (当前参数下对旧轨迹评估)
    # ================================================================

    def compute_new_log_prob(
        self,
        conditioning: torch.Tensor,      # [B, 1, D]  用新参数的 LLM 产出
        latent_history: torch.Tensor,     # [B, history_len, latent_dim]
        ode_trajectory: torch.Tensor,     # [K+1, B, patch, latent]
        ode_timesteps: torch.Tensor,      # [K+1]
        initial_noise: torch.Tensor,      # [B, patch, latent]
        final_latent: torch.Tensor,       # [B, patch, latent]
        solver_sigma: float = 0.25,
        solver_temperature: float = 1e-6,
    ) -> torch.Tensor:
        """
        用当前参数 θ_new 计算旧轨迹的 log π_{θ_new}(a | s)。

        两种方法:
        - exact: 用保存的 ODE 轨迹在新 v_θ 下评估高斯 log-prob
        - surrogate: DDPO 风格代理

        Returns: [B]
        """
        if self.log_prob_method == "exact":
            return self._compute_ode_log_prob_exact(
                ode_trajectory=ode_trajectory,
                ode_timesteps=ode_timesteps,
                conditioning=conditioning,
                latent_history=latent_history,
                solver_sigma=solver_sigma,
                solver_temperature=solver_temperature,
            )
        else:
            return self._compute_ode_log_prob_surrogate(
                initial_noise=initial_noise,
                final_latent=final_latent,
                conditioning=conditioning,
                latent_history=latent_history,
            )

    # ================================================================
    #  完整序列的 log-prob 计算 (teacher-forced 自回归循环)
    # ================================================================

    def compute_sequence_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rollout_data: List[Dict[str, torch.Tensor]],
        position_ids: Optional[torch.Tensor] = None,
        solver_sigma: float = 0.25,
        solver_temperature: float = 1e-6,
    ) -> torch.Tensor:
        """
        Teacher-forced 重跑整个自回归循环,
        在每步用新参数计算 log π_{θ_new}(a_t | s_t)。

        Args:
            rollout_data: list of per-step dicts from rollout,
                          each containing ode_trajectory, ode_timesteps, etc.

        Returns: [B, T] per-step log-probs
        """
        B = input_ids.shape[0]
        T = len(rollout_data)
        device = input_ids.device

        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Position ids
        if self.model.model_type != "dense" and position_ids is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_token_id=self.model.config.llm_config.image_patch_token,
                video_token_id=self.model.config.llm_config.image_patch_token,
                image_start_token_id=getattr(self.model.config.llm_config, "image_start_token", 0),
                video_start_token_id=getattr(self.model.config.llm_config, "video_start_token", 0),
                image_grid_thw=None, video_grid_thw=None,
                attention_mask=attention_mask,
            )
        else:
            if position_ids is None:
                position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(attention_mask == 0, 1)
            rope_deltas = None

        past_key_values = None
        latent_history = torch.zeros(B, self.history_patch_size, self.latent_dim, device=device)
        log_probs_list = []

        for t in range(T):
            step_data = rollout_data[t]

            # (1) LLM forward → conditioning c_t^{new}
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model.model(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    audio_mask=None, image_mask=None,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            past_key_values = outputs.past_key_values
            c_t_new = outputs.hidden_states[-1][:, -1:, :]  # [B, 1, D]

            # (2) 用新 c_t^{new} 评估旧 ODE 轨迹的 log-prob
            step_log_prob = self.compute_new_log_prob(
                conditioning=c_t_new,
                latent_history=latent_history,
                ode_trajectory=step_data["ode_trajectory"].to(device),
                ode_timesteps=step_data["ode_timesteps"].to(device),
                initial_noise=step_data["initial_noise"].to(device),
                final_latent=step_data["latent"].to(device),
                solver_sigma=solver_sigma,
                solver_temperature=solver_temperature,
            )
            log_probs_list.append(step_log_prob)

            # (3) Teacher-forcing: 用旧的 latent 做下一步输入
            old_latent = step_data["latent"].to(device)  # [B, patch, latent]
            inputs_embeds = self.model.linear_proj_audio(old_latent)

            # (4) 更新 position_ids
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

            # (5) 更新 latent history
            latent_history[:, :-self.patch_size, :] = latent_history[:, self.patch_size:, :].clone()
            latent_history[:, -self.patch_size:, :] = old_latent.reshape(B, self.patch_size, self.latent_dim)

        return torch.stack(log_probs_list, dim=1)  # [B, T]
