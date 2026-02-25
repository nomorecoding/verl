"""
verl-native Flow-GRPO training for Ming-omni-tts MoE model.

This script integrates with verl's distributed training framework
(RayPPOTrainer + HybridFlow) by providing custom actor and reward
workers that handle the flow-matching TTS generation.

For models that fit the standard HuggingFace CausalLM interface,
verl can be used directly.  The BailingMoe model *does* implement
HuggingFace's CausalLM interface, so its LLM backbone can be driven
by verl's FSDP/Megatron engines.  The challenge is the flow-matching
head, which requires a custom generation loop.

Strategy:
  We split the training into two modes:

  MODE A  (Recommended – LLM-only GRPO)
  ─────────────────────────────────────
  Train only the MoE-LLM backbone with GRPO, treating the flow-matching
  head as frozen.  This is the simplest path and works well because:
    - The MoE-LLM generates the conditioning representation
    - Better LLM conditioning → better audio (the flow head is already good)
    - verl's standard FSDP + vLLM rollout works out-of-the-box

  MODE B  (Advanced – Full pipeline GRPO)
  ───────────────────────────────────────
  Train both LLM and flow-matching head jointly with GRPO using the
  custom training loop in train_flow_grpo.py.

This file implements MODE A.  See train_flow_grpo.py for MODE B.
"""

import os
import sys


def register_bailing_moe_with_transformers():
    """
    Register the BailingMoe model with HuggingFace AutoModel so that
    verl's model loading works seamlessly.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = os.environ.get("MING_MODEL_DIR", "")
    if model_dir and model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    try:
        from configuration_bailing_moe import BailingMoeConfig
        from modeling_bailing_moe import BailingMoeForCausalLM

        AutoConfig.register("bailing_moe", BailingMoeConfig)
        AutoModelForCausalLM.register(BailingMoeConfig, BailingMoeForCausalLM)
        print("Successfully registered BailingMoe with HuggingFace AutoModel")
    except Exception as e:
        print(f"Warning: Could not register BailingMoe: {e}")
        print("Make sure MING_MODEL_DIR environment variable points to the model directory")


def create_verl_config(
    model_path: str,
    train_data: str,
    val_data: str,
    n_gpus: int = 8,
    group_size: int = 4,
    lr: float = 1e-6,
    epochs: int = 10,
    batch_size: int = 128,
):
    """Create a verl-compatible OmegaConf configuration dict."""
    config = {
        "algorithm": {
            "adv_estimator": "grpo",
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
            "gamma": 1.0,
            "lam": 1.0,
            "kl_penalty": "kl",
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": 0.001,
            },
        },
        "data": {
            "train_files": train_data,
            "val_files": val_data,
            "train_batch_size": batch_size,
            "max_prompt_length": 512,
            "max_response_length": 1024,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "shuffle": True,
        },
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "model": {
                "path": model_path,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
            "actor": {
                "strategy": "fsdp",
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "kl_loss_type": "low_var_kl",
                "entropy_coeff": 0,
                "ppo_mini_batch_size": batch_size // 4,
                "ppo_micro_batch_size_per_gpu": max(1, batch_size // (4 * n_gpus)),
                "loss_agg_mode": "token-mean",
                "optim": {
                    "lr": lr,
                    "weight_decay": 0.01,
                    "lr_warmup_steps": 10,
                    "lr_decay_style": "cosine",
                },
                "fsdp_config": {
                    "param_offload": False,
                    "optimizer_offload": False,
                },
            },
            "rollout": {
                "name": "vllm",
                "n": group_size,
                "tensor_model_parallel_size": min(2, n_gpus),
                "gpu_memory_utilization": 0.6,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": -1,
                "log_prob_micro_batch_size_per_gpu": 4,
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 4,
                "fsdp_config": {
                    "param_offload": True,
                },
            },
        },
        "reward": {
            "reward_manager": {
                "name": "naive",
            },
        },
        "trainer": {
            "total_epochs": epochs,
            "project_name": "ming_moe_flow_grpo",
            "experiment_name": "bailing_moe_grpo",
            "logger": ["console"],
            "n_gpus_per_node": n_gpus,
            "nnodes": 1,
            "save_freq": 20,
            "test_freq": 5,
            "val_before_train": False,
            "balance_batch": True,
            "device": "cuda",
        },
    }
    return config


def main():
    """
    Launch verl's standard GRPO training with the BailingMoe model.

    This is the recommended approach: use verl's built-in distributed
    infrastructure with the MoE-LLM backbone.

    Example:
        # Set environment variable so we can find the model code
        export MING_MODEL_DIR=/path/to/Ming-omni-tts

        # Run with verl's standard entry point
        python -m verl.trainer.main_ppo \\
            algorithm.adv_estimator=grpo \\
            data.train_files=/path/to/train.parquet \\
            data.val_files=/path/to/test.parquet \\
            actor_rollout_ref.model.path=/path/to/bailing-moe-model \\
            actor_rollout_ref.actor.use_kl_loss=True \\
            actor_rollout_ref.rollout.n=4 \\
            trainer.n_gpus_per_node=8

        # Or use this script:
        python train_flow_grpo_verl_native.py \\
            --model_path /path/to/bailing-moe-model \\
            --train_data /path/to/train.parquet \\
            --val_data /path/to/test.parquet
    """
    import argparse

    parser = argparse.ArgumentParser(description="verl-native Flow-GRPO for BailingMoe")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    register_bailing_moe_with_transformers()

    config_dict = create_verl_config(
        model_path=args.model_path,
        train_data=args.train_data,
        val_data=args.val_data or args.train_data,
        n_gpus=args.n_gpus,
        group_size=args.group_size,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    from omegaconf import OmegaConf
    config = OmegaConf.create(config_dict)

    print("=" * 60)
    print("verl-native Flow-GRPO Configuration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 60)

    from verl.trainer.main_ppo import run_ppo
    run_ppo(config)


if __name__ == "__main__":
    main()
