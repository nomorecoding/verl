"""
连续动作空间 Flow-GRPO 训练: Ming-omni-tts MoE 模型

=== 为什么不能直接用标准 GRPO? ===

标准 GRPO (verl 默认) 的 MDP:
  状态: s_t = (prompt, token_{<t})
  动作: a_t ∈ {1, ..., V}          ← 离散 token
  策略: π(a_t|s_t) = softmax(lm_head(h_t))[a_t]
  log-prob: log_softmax(logits)[token_t]

Ming-omni-tts TTS 的 MDP:
  状态: s_t = (text_prompt, audio_latent_{<t})
  动作: a_t ∈ ℝ^{patch×latent_dim}  ← 连续 latent
  策略: 由 MoE-LLM conditioning + stochastic ODE 隐式定义
  log-prob: 需要重新推导!

关键差异:
  1. LLM 不走 lm_head, 而是 hidden_state → flow matching head
  2. 动作空间是连续的, 不是离散 token
  3. 每步生成涉及 K 步 ODE 积分, 而非单次 softmax 采样

=== 正确的 log-prob 定义 ===

Ming-omni-tts 的 ODE solver 在每步加高斯噪声:
  y_{k+1} = y_k + dt·v_θ(y_k, t_k, c) + σ√(T|dt|)·ε_k,  ε_k ~ N(0,I)

这定义了一个显式的高斯转移概率:
  p(y_{k+1} | y_k, c; θ) = N(y_{k+1} | y_k + dt·v_θ(y_k, t_k, c), σ²T|dt|·I)

Per-step log-prob (K 个 ODE 步之和):
  log π(a_t | s_t; θ) = Σ_k log N(y_{k+1} | μ_k, σ_k²I)
                       = Σ_k [ -1/(2σ_k²)·‖y_{k+1} - μ_k‖² - d/2·log(2πσ_k²) ]

Importance ratio:
  r_t = exp(log π_{θ_new}(a_t|s_t) - log π_{θ_old}(a_t|s_t))

注意: θ 变了 → v_θ 变了 → μ_k 变了 → Gaussian log-prob 变了
      但 y_{k+1} 是旧轨迹的点, 固定不变。

=== 训练流程 ===

  1. Rollout: 生成 N 个音频 + 保存 ODE 轨迹 + 算 old log-probs
  2. Reward:  评估音频质量 → scalar reward per sequence
  3. GRPO:    组内 (reward - mean) / std → advantage
  4. Update:  teacher-force 重跑, 算 new log-probs, PPO-clip loss

Usage:
    python train_flow_grpo.py \\
        --model_path /path/to/ming-omni-tts \\
        --train_data /path/to/train.parquet \\
        --grpo_group_size 4
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="moe", choices=["moe", "dense"])
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)

    # GRPO
    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--norm_adv_by_std", action="store_true", default=True)
    parser.add_argument("--log_prob_method", type=str, default="exact",
                        choices=["exact", "surrogate"],
                        help="exact: 保存 ODE 轨迹精确计算; surrogate: DDPO 风格近似")

    # Training scope
    parser.add_argument("--train_mode", type=str, default="llm_only",
                        choices=["llm_only", "full"],
                        help="llm_only: 只训练 MoE-LLM backbone (冻结 DiT/CFM/Aggregator/stop_head); "
                             "full: 训练全部参数")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.001)

    # Generation
    parser.add_argument("--max_decode_steps", type=int, default=200)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--ode_steps", type=int, default=10)

    # Reward
    parser.add_argument("--reward_weights", type=str, default=None)

    # Infrastructure
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/flow_grpo")
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="flow_grpo")

    return parser.parse_args()


def compute_grpo_advantage(
    rewards: torch.Tensor,       # [B] scalar per sample
    group_indices: np.ndarray,   # [B] group assignment
    norm_by_std: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """GRPO 组内优势归一化。"""
    id2scores = defaultdict(list)
    for i, idx in enumerate(group_indices):
        id2scores[idx].append((i, rewards[i].item()))

    advantages = torch.zeros_like(rewards)
    for idx, pairs in id2scores.items():
        scores = torch.tensor([s for _, s in pairs])
        mean = scores.mean()
        std = scores.std() if len(scores) > 1 else torch.tensor(1.0)
        for i, _ in pairs:
            if norm_by_std:
                advantages[i] = (rewards[i] - mean) / (std + epsilon)
            else:
                advantages[i] = rewards[i] - mean

    return advantages


def compute_ppo_loss(
    new_log_probs: torch.Tensor,  # [B, T] per-step (连续动作)
    old_log_probs: torch.Tensor,  # [B, T]
    advantages: torch.Tensor,     # [B] per-sequence
    response_mask: torch.Tensor,  # [B, T]
    clip_ratio: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """PPO-clip loss, 适配连续动作空间。"""
    log_ratio = new_log_probs - old_log_probs
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
    ratio = torch.exp(log_ratio)

    # GRPO: scalar advantage broadcast 到每个 step
    adv = advantages.unsqueeze(-1) * response_mask

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    loss_per_step = -torch.min(surr1, surr2)

    valid = response_mask.sum()
    loss = (loss_per_step * response_mask).sum() / (valid + 1e-8)

    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        clip_frac = ((ratio - 1.0).abs() > clip_ratio).float().mean().item()

    return loss, {
        "policy_loss": loss.item(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_frac,
        "mean_ratio": ratio.mean().item(),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    reward_weights = json.loads(args.reward_weights) if args.reward_weights else None

    use_wandb = args.wandb_project is not None
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.experiment_name, config=vars(args))
        except ImportError:
            use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load Model ----
    import sys
    model_dir = os.path.dirname(args.model_path) if os.path.isfile(args.model_path) else args.model_path
    sys.path.insert(0, model_dir)

    try:
        from modeling_bailingmm import BailingMMNativeForConditionalGeneration
        from tokenization_bailing import BailingTokenizer
    except ImportError:
        print("无法导入模型类。确保 PYTHONPATH 包含 Ming-omni-tts 目录。")
        return

    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    ).to(device)
    tokenizer = BailingTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ---- 根据 train_mode 冻结参数 ----
    if args.train_mode == "llm_only":
        frozen_modules = []
        # 冻结 flow matching head (DiT + CFM)
        for p in model.flowloss.parameters():
            p.requires_grad = False
        frozen_modules.append(f"flowloss ({sum(1 for _ in model.flowloss.parameters())} params)")

        # 冻结 audio Aggregator (latent → LLM embedding 投影)
        for p in model.linear_proj_audio.parameters():
            p.requires_grad = False
        frozen_modules.append(f"linear_proj_audio ({sum(1 for _ in model.linear_proj_audio.parameters())} params)")

        # 冻结 stop head
        for p in model.stop_head.parameters():
            p.requires_grad = False
        frozen_modules.append(f"stop_head ({sum(1 for _ in model.stop_head.parameters())} params)")

        # 冻结 speaker head
        if hasattr(model, "spk_head"):
            for p in model.spk_head.parameters():
                p.requires_grad = False
            frozen_modules.append(f"spk_head ({sum(1 for _ in model.spk_head.parameters())} params)")

        # 冻结 audio tokenizer (VAE)
        if hasattr(model, "audio"):
            for p in model.audio.parameters():
                p.requires_grad = False
            frozen_modules.append(f"audio VAE ({sum(1 for _ in model.audio.parameters())} params)")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train_mode=llm_only] 冻结模块: {frozen_modules}")
        print(f"  总参数: {total:,}  可训练: {trainable:,} ({trainable/total*100:.1f}%)")
        print(f"  只训练 MoE-LLM backbone (model.model = BailingMoeForCausalLM)")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"[train_mode=full] 训练全部参数: {total:,}")

    # 策略封装
    from ming_moe_verl.model.policy_forward import FlowGRPOPolicy
    policy = FlowGRPOPolicy(
        model=model,
        tokenizer=tokenizer,
        ode_steps=args.ode_steps,
        log_prob_method=args.log_prob_method,
    )

    # KL 参考模型
    ref_policy = None
    if args.kl_coef > 0:
        import copy
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_policy = FlowGRPOPolicy(
            model=ref_model,
            tokenizer=tokenizer,
            ode_steps=args.ode_steps,
            log_prob_method=args.log_prob_method,
        )

    # Rollout 引擎
    from ming_moe_verl.model.rollout_worker import FlowGRPORollout
    rollout_engine = FlowGRPORollout(
        policy=policy,
        tokenizer=tokenizer,
        max_decode_steps=args.max_decode_steps,
        cfg=args.cfg_scale,
        sigma=args.sigma,
        temperature=args.temperature,
        n=args.grpo_group_size,
    )

    # Reward
    from ming_moe_verl.reward.tts_reward import TTSRewardManager
    reward_manager = TTSRewardManager(
        tokenizer=tokenizer,
        audio_decoder=model.audio if hasattr(model, "audio") else None,
        reward_weights=reward_weights,
    )

    # 数据
    import pandas as pd
    train_df = pd.read_parquet(args.train_data)
    print(f"Loaded {len(train_df)} training samples")

    # 优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * (len(train_df) // args.batch_size + 1)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    print("=" * 70)
    print("Continuous-action Flow-GRPO Training")
    print(f"  train mode: {args.train_mode}")
    print(f"  log-prob method: {args.log_prob_method}")
    print(f"  ODE steps: {args.ode_steps}")
    print(f"  GRPO group size: {args.grpo_group_size}")
    print(f"  PPO clip: {args.ppo_clip}")
    print(f"  KL coef: {args.kl_coef}")
    if args.train_mode == "llm_only":
        print("  注意: 只训练 MoE-LLM, DiT/CFM/Aggregator 冻结")
        print("  梯度回传路径: L → log_π → μ_k → v_DiT(frozen) → c_t → θ_LLM")
    print("=" * 70)

    global_step = 0
    for epoch in range(args.epochs):
        shuffled = train_df.sample(frac=1.0).reset_index(drop=True)

        for batch_start in range(0, len(shuffled), args.batch_size):
            batch_df = shuffled.iloc[batch_start:batch_start + args.batch_size]
            if len(batch_df) == 0:
                continue

            # 提取文本
            texts = []
            for _, row in batch_df.iterrows():
                prompt = row.get("prompt", "")
                if isinstance(prompt, list):
                    text = prompt[0].get("content", "") if prompt else ""
                elif isinstance(prompt, str):
                    text = prompt
                else:
                    text = str(prompt)
                text = text.split("Text input:\n")[-1] if "Text input:\n" in text else text
                texts.append(text)

            t0 = time.time()

            # ========== Phase 1: Rollout ==========
            model.eval()
            batch_rollout = rollout_engine.generate_batch(texts=texts, device=str(device))

            B_total = len(batch_rollout["rollouts"])
            rollout_time = time.time() - t0

            # ========== Phase 2: Reward ==========
            t1 = time.time()
            scalar_rewards = torch.zeros(B_total)

            for i, rollout in enumerate(batch_rollout["rollouts"]):
                # 拼接所有步的 latent
                all_latents = torch.cat(
                    [s["latent"].squeeze(0) for s in rollout["step_data"]],
                    dim=0,
                )  # [T*patch, latent_dim]

                from ming_moe_verl.reward.tts_reward import compute_tts_reward
                score = compute_tts_reward(
                    audio_latents=all_latents,
                    prompt_text=batch_rollout["prompt_texts"][i],
                    audio_decoder=model.audio if hasattr(model, "audio") else None,
                    weights=reward_weights,
                )
                scalar_rewards[i] = score["score"]

                if i < 2:
                    print(f"  [sample {i}] steps={rollout['total_steps']} "
                          f"reward={score['score']:.4f}")

            reward_time = time.time() - t1

            # ========== Phase 3: GRPO Advantage ==========
            advantages = compute_grpo_advantage(
                scalar_rewards,
                batch_rollout["group_indices"],
                norm_by_std=args.norm_adv_by_std,
            )

            # ========== Phase 4: Policy Update ==========
            t2 = time.time()
            model.train()
            total_loss = 0.0
            all_metrics = defaultdict(float)

            for i in range(B_total):
                rollout = batch_rollout["rollouts"][i]
                input_ids = batch_rollout["input_ids"][i].to(device)
                attn_mask = batch_rollout["attention_mask"][i].to(device)

                T = rollout["total_steps"]
                old_log_probs = rollout["per_step_log_probs"].to(device)  # [1, T]

                # teacher-forced 用新参数计算 log-probs
                new_log_probs = policy.compute_sequence_log_probs(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    rollout_data=rollout["step_data"],
                    solver_sigma=args.sigma,
                    solver_temperature=max(args.temperature, 1e-6),
                )  # [1, T]

                resp_mask = torch.ones(1, T, device=device)
                adv_i = advantages[i:i+1].to(device)

                loss, metrics = compute_ppo_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs,
                    advantages=adv_i,
                    response_mask=resp_mask,
                    clip_ratio=args.ppo_clip,
                )

                # KL 正则
                if args.kl_coef > 0 and ref_policy is not None:
                    with torch.no_grad():
                        ref_lp = ref_policy.compute_sequence_log_probs(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            rollout_data=rollout["step_data"],
                            solver_sigma=args.sigma,
                            solver_temperature=max(args.temperature, 1e-6),
                        )
                    kl = (new_log_probs - ref_lp) * resp_mask
                    kl_loss = args.kl_coef * kl.sum() / (resp_mask.sum() + 1e-8)
                    loss = loss + kl_loss
                    metrics["kl_loss"] = kl_loss.item()

                (loss / B_total).backward()
                total_loss += loss.item()
                for k, v in metrics.items():
                    all_metrics[k] += v

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            update_time = time.time() - t2
            global_step += 1

            # ========== Logging ==========
            if global_step % args.log_freq == 0:
                avg_r = scalar_rewards.mean().item()
                avg_a = advantages.mean().item()
                avg_l = total_loss / B_total
                gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                print(
                    f"[Step {global_step}] ep={epoch} "
                    f"R={avg_r:.4f} A={avg_a:.4f} L={avg_l:.4f} "
                    f"|∇|={gn:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"t_roll={rollout_time:.1f}s t_rew={reward_time:.1f}s t_upd={update_time:.1f}s"
                )
                if use_wandb:
                    import wandb
                    wandb.log({
                        "reward": avg_r, "advantage": avg_a,
                        "loss": avg_l, "grad_norm": gn,
                        **{f"train/{k}": v / B_total for k, v in all_metrics.items()},
                    }, step=global_step)

            if args.save_freq > 0 and global_step % args.save_freq == 0:
                ckpt = os.path.join(args.output_dir, f"step_{global_step}")
                os.makedirs(ckpt, exist_ok=True)
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"Checkpoint saved: {ckpt}")

    final = os.path.join(args.output_dir, "final")
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"Training complete! Final model: {final}")


if __name__ == "__main__":
    main()
