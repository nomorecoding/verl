"""
Main training script for Flow-GRPO on Ming-omni-tts MoE model.

This script implements a standalone training loop that follows verl's
GRPO algorithm but uses our custom rollout (flow-matching generation)
and reward (TTS quality metrics) instead of verl's standard text-based
pipeline.

Architecture:
    ┌──────────────┐
    │  Text Prompt  │
    └──────┬───────┘
           ▼
    ┌──────────────┐   ×N (GRPO group size)
    │  MoE LLM     │──────────────────────────┐
    │  + Flow Head  │  autoregressive flow     │
    └──────────────┘  matching generation      │
           ▼                                    │
    ┌──────────────┐                            │
    │  Audio Latents│                           │
    │  + log π(a|s) │                           │
    └──────┬───────┘                            │
           ▼                                    │
    ┌──────────────┐                            │
    │  TTS Reward   │  (ASR, MOS, speaker sim) │
    └──────┬───────┘                            │
           ▼                                    │
    ┌──────────────┐                            │
    │  GRPO Adv.   │  group-normalise rewards  │
    └──────┬───────┘                            │
           ▼                                    │
    ┌──────────────┐                            │
    │  Policy Loss  │  -advantage × Δlog_prob  │
    │  + KL reg.    │                           │
    └──────────────┘                            │

Usage:
    python train_flow_grpo.py \\
        --model_path /path/to/ming-omni-tts \\
        --train_data /path/to/train.parquet \\
        --val_data /path/to/test.parquet \\
        --num_gpus 8 \\
        --grpo_group_size 4 \\
        --epochs 10
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def parse_args():
    parser = argparse.ArgumentParser(description="Flow-GRPO training for Ming-omni-tts MoE")

    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained Ming-omni-tts model")
    parser.add_argument("--model_type", type=str, default="moe", choices=["moe", "dense"])

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Path to training parquet")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation parquet")

    # GRPO parameters
    parser.add_argument("--grpo_group_size", type=int, default=4, help="Number of generations per prompt (n)")
    parser.add_argument("--norm_adv_by_std", action="store_true", default=True,
                        help="Normalise GRPO advantage by std (True=GRPO, False=Dr.GRPO)")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4, help="Number of unique prompts per batch")
    parser.add_argument("--mini_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ppo_clip", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--kl_coef", type=float, default=0.001, help="KL penalty coefficient")
    parser.add_argument("--entropy_coeff", type=float, default=0.0)

    # Generation
    parser.add_argument("--max_decode_steps", type=int, default=200)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0)

    # Reward
    parser.add_argument("--reward_weights", type=str, default=None,
                        help="JSON string of reward weights, e.g. '{\"intelligibility\": 0.4}'")

    # Infrastructure
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)

    # Logging & checkpointing
    parser.add_argument("--output_dir", type=str, default="./checkpoints/flow_grpo")
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="ming_moe_flow_grpo")

    return parser.parse_args()


def compute_grpo_advantage(
    rewards: torch.Tensor,
    group_indices: np.ndarray,
    norm_by_std: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute GRPO advantage: group-normalised rewards.

    Args:
        rewards: [B] scalar reward per sample
        group_indices: [B] integer group assignment (samples from same prompt share an index)
        norm_by_std: whether to divide by group std (True=GRPO, False=Dr.GRPO)

    Returns:
        advantages: [B] normalised advantages
    """
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


def compute_policy_loss(
    new_log_probs: torch.Tensor,    # [B, T]
    old_log_probs: torch.Tensor,    # [B, T]
    advantages: torch.Tensor,       # [B]
    response_mask: torch.Tensor,    # [B, T]
    clip_ratio: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    PPO-clip policy loss used for the GRPO update.
    """
    log_ratio = new_log_probs - old_log_probs  # [B, T]
    ratio = torch.exp(log_ratio)

    adv_expanded = advantages.unsqueeze(-1) * response_mask  # [B, T]

    surr1 = ratio * adv_expanded
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_expanded

    loss_per_token = -torch.min(surr1, surr2)

    valid_tokens = response_mask.sum()
    loss = (loss_per_token * response_mask).sum() / (valid_tokens + 1e-8)

    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        clip_frac = ((ratio - 1.0).abs() > clip_ratio).float().mean().item()

    metrics = {
        "policy_loss": loss.item(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_frac,
        "mean_ratio": ratio.mean().item(),
    }
    return loss, metrics


def main():
    args = parse_args()

    reward_weights = None
    if args.reward_weights:
        reward_weights = json.loads(args.reward_weights)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Logging ----
    use_wandb = args.wandb_project is not None
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.experiment_name, config=vars(args))
        except ImportError:
            print("wandb not installed, falling back to console logging")
            use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load Model ----
    print(f"Loading model from {args.model_path}...")
    import sys
    model_dir = os.path.dirname(args.model_path) if os.path.isfile(args.model_path) else args.model_path
    sys.path.insert(0, model_dir)

    try:
        from modeling_bailingmm import BailingMMNativeForConditionalGeneration
        from tokenization_bailing import BailingTokenizer
    except ImportError:
        print("Could not import model classes. Make sure the Ming-omni-tts model files are accessible.")
        print("Expected files: modeling_bailingmm.py, tokenization_bailing.py, etc.")
        print("You may need to add the model directory to your PYTHONPATH.")
        return

    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    ).to(device)

    tokenizer = BailingTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Wrap in our RL policy
    from ming_moe_verl.model.policy_forward import BailingMoeTTSForRL
    policy = BailingMoeTTSForRL(
        model=model,
        tokenizer=tokenizer,
        patch_size=model.patch_size,
        history_patch_size=model.history_patch_size,
    )

    # Keep a frozen copy for KL reference
    ref_model = None
    if args.kl_coef > 0:
        import copy
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_policy = BailingMoeTTSForRL(
            model=ref_model,
            tokenizer=tokenizer,
            patch_size=model.patch_size,
            history_patch_size=model.history_patch_size,
        )

    # ---- Setup rollout ----
    from ming_moe_verl.model.rollout_worker import BailingMoeFlowRollout
    rollout_engine = BailingMoeFlowRollout(
        policy_model=policy,
        tokenizer=tokenizer,
        max_decode_steps=args.max_decode_steps,
        cfg=args.cfg_scale,
        sigma=args.sigma,
        temperature=args.temperature,
        n=args.grpo_group_size,
    )

    # ---- Setup reward ----
    from ming_moe_verl.reward.tts_reward import TTSRewardManager
    reward_manager = TTSRewardManager(
        tokenizer=tokenizer,
        num_examine=2,
        audio_decoder=model.audio if hasattr(model, "audio") else None,
        reward_weights=reward_weights,
    )

    # ---- Load Data ----
    import pandas as pd
    train_df = pd.read_parquet(args.train_data)
    print(f"Loaded {len(train_df)} training samples")

    # ---- Optimiser ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * (len(train_df) // args.batch_size + 1)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    # ---- Training Loop ----
    print("=" * 60)
    print(f"Starting Flow-GRPO training")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  GRPO group size: {args.grpo_group_size}")
    print(f"  Effective batch: {args.batch_size * args.grpo_group_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  PPO clip ratio: {args.ppo_clip}")
    print(f"  KL coefficient: {args.kl_coef}")
    print("=" * 60)

    global_step = 0
    for epoch in range(args.epochs):
        shuffled = train_df.sample(frac=1.0).reset_index(drop=True)

        for batch_start in range(0, len(shuffled), args.batch_size):
            batch_df = shuffled.iloc[batch_start:batch_start + args.batch_size]
            if len(batch_df) == 0:
                continue

            texts = []
            for _, row in batch_df.iterrows():
                prompt = row.get("prompt", "")
                if isinstance(prompt, list):
                    text = prompt[0].get("content", "") if prompt else ""
                elif isinstance(prompt, str):
                    text = prompt
                else:
                    text = str(prompt)
                text_part = text.split("Text input:\n")[-1] if "Text input:\n" in text else text
                texts.append(text_part)

            t0 = time.time()

            # ============== Phase 1: Rollout ==============
            model.eval()
            rollout_data = rollout_engine.generate_batch(
                texts=texts,
                prompt_template="tts",
                device=str(device),
            )
            rollout_time = time.time() - t0

            B_total = rollout_data["old_log_probs"].shape[0]
            max_resp_len = rollout_data["response_mask"].shape[1]

            # ============== Phase 2: Reward ==============
            t1 = time.time()

            class SimpleDataProto:
                def __init__(self, batch_dict, non_tensor_dict):
                    self.batch = batch_dict
                    self.non_tensor_batch = non_tensor_dict

            reward_data = SimpleDataProto(
                batch_dict={
                    "audio_latents": rollout_data["audio_latents"],
                    "response_mask": rollout_data["response_mask"],
                },
                non_tensor_dict={
                    "prompt_text": rollout_data["prompt_texts"],
                },
            )
            reward_tensor = reward_manager(reward_data)  # [B_total, max_resp_len]
            reward_time = time.time() - t1

            # Scalar reward per sample
            scalar_rewards = reward_tensor.sum(dim=-1)  # [B_total]

            # ============== Phase 3: GRPO Advantage ==============
            group_indices = np.repeat(np.arange(len(texts)), args.grpo_group_size)
            advantages = compute_grpo_advantage(
                scalar_rewards,
                group_indices,
                norm_by_std=args.norm_adv_by_std,
            )

            # ============== Phase 4: Policy Update ==============
            t2 = time.time()
            model.train()

            input_ids = rollout_data["input_ids"]
            attn_mask = rollout_data["attention_mask_prompt"]
            audio_latents = rollout_data["audio_latents"]
            old_log_probs = rollout_data["old_log_probs"]
            response_mask = rollout_data["response_mask"]

            total_policy_loss = 0.0
            num_mini_batches = max(1, B_total // args.mini_batch_size)

            for mb_idx in range(num_mini_batches):
                mb_start = mb_idx * args.mini_batch_size
                mb_end = min(mb_start + args.mini_batch_size, B_total)

                mb_input_ids = input_ids[mb_start:mb_end]
                mb_attn = attn_mask[mb_start:mb_end] if attn_mask.dim() > 1 else attn_mask
                mb_latents = audio_latents[mb_start:mb_end]
                mb_old_lp = old_log_probs[mb_start:mb_end]
                mb_resp_mask = response_mask[mb_start:mb_end]
                mb_adv = advantages[mb_start:mb_end].to(device)

                new_log_probs = policy.compute_sequence_log_probs(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attn if mb_attn.dim() > 1 else None,
                    audio_latent_sequence=mb_latents,
                )

                # Align shapes if needed
                T_new = new_log_probs.shape[1]
                T_old = mb_old_lp.shape[1]
                T = min(T_new, T_old)
                new_lp = new_log_probs[:, :T]
                old_lp = mb_old_lp[:, :T].to(device)
                mask = mb_resp_mask[:, :T].to(device)

                loss, metrics = compute_policy_loss(
                    new_log_probs=new_lp,
                    old_log_probs=old_lp,
                    advantages=mb_adv,
                    response_mask=mask,
                    clip_ratio=args.ppo_clip,
                )

                # KL penalty
                if args.kl_coef > 0 and ref_model is not None:
                    with torch.no_grad():
                        ref_log_probs = ref_policy.compute_sequence_log_probs(
                            input_ids=mb_input_ids,
                            attention_mask=mb_attn if mb_attn.dim() > 1 else None,
                            audio_latent_sequence=mb_latents,
                        )[:, :T]
                    kl = (new_lp - ref_log_probs.to(device)) * mask
                    kl_loss = args.kl_coef * kl.sum() / (mask.sum() + 1e-8)
                    loss = loss + kl_loss
                    metrics["kl_loss"] = kl_loss.item()

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                total_policy_loss += loss.item()

            update_time = time.time() - t2
            global_step += 1

            # ============== Logging ==============
            if global_step % args.log_freq == 0:
                avg_reward = scalar_rewards.mean().item()
                avg_adv = advantages.mean().item()
                avg_loss = total_policy_loss / max(num_mini_batches, 1)

                log_dict = {
                    "step": global_step,
                    "epoch": epoch,
                    "avg_reward": avg_reward,
                    "avg_advantage": avg_adv,
                    "policy_loss": avg_loss,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "lr": scheduler.get_last_lr()[0],
                    "rollout_time": rollout_time,
                    "reward_time": reward_time,
                    "update_time": update_time,
                    "total_time": time.time() - t0,
                }
                log_dict.update({f"train/{k}": v for k, v in metrics.items()})

                print(
                    f"[Step {global_step}] epoch={epoch} "
                    f"reward={avg_reward:.4f} adv={avg_adv:.4f} "
                    f"loss={avg_loss:.4f} grad_norm={log_dict['grad_norm']:.4f} "
                    f"lr={log_dict['lr']:.2e} "
                    f"time={log_dict['total_time']:.1f}s"
                )

                if use_wandb:
                    wandb.log(log_dict, step=global_step)

            # ============== Checkpoint ==============
            if args.save_freq > 0 and global_step % args.save_freq == 0:
                ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint to {ckpt_dir}")

    # Final save
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training complete! Final model saved to {final_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
