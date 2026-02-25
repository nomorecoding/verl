#!/bin/bash
# =============================================================================
# verl-native GRPO training for Ming-omni-tts BailingMoe model
#
# This script uses verl's built-in distributed infrastructure (MODE A)
# to train the MoE-LLM backbone with GRPO, using FSDP + vLLM rollout.
#
# Usage:
#   bash scripts/run_flow_grpo_verl.sh
#
# Prerequisites:
#   1. pip install verl[all]   (or follow verl installation guide)
#   2. pip install vllm
#   3. Model weights downloaded to MODEL_PATH
#   4. Training data preprocessed to parquet format
#   5. Register BailingMoe model with HuggingFace AutoModel
# =============================================================================
set -x

# ----- Paths -----
MODEL_PATH=${MODEL_PATH:-"/path/to/bailing-moe-model"}
TRAIN_DATA=${TRAIN_DATA:-"./data/tts_grpo/train.parquet"}
VAL_DATA=${VAL_DATA:-"./data/tts_grpo/test.parquet"}

# Add model directory to Python path
export PYTHONPATH="${MODEL_PATH}:${PYTHONPATH}"
export MING_MODEL_DIR="${MODEL_PATH}"

# ----- Cluster -----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ----- GRPO Hyperparameters -----
N_RESP=${N_RESP:-4}           # Group size (responses per prompt)
TRAIN_BSZ=${TRAIN_BSZ:-128}   # Total training batch size (prompts)
MINI_BSZ=${MINI_BSZ:-32}      # Mini batch size for PPO updates
MICRO_BSZ=${MICRO_BSZ:-4}     # Micro batch per GPU
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-10}

# ----- vLLM Rollout -----
ROLLOUT_TP=${ROLLOUT_TP:-2}   # Tensor parallelism for vLLM

# Register the BailingMoe model before launching verl
python -c "
import sys, os
sys.path.insert(0, os.environ.get('MING_MODEL_DIR', ''))
from transformers import AutoConfig, AutoModelForCausalLM
try:
    from configuration_bailing_moe import BailingMoeConfig
    from modeling_bailing_moe import BailingMoeForCausalLM
    AutoConfig.register('bailing_moe', BailingMoeConfig)
    AutoModelForCausalLM.register(BailingMoeConfig, BailingMoeForCausalLM)
    print('BailingMoe registered successfully')
except Exception as e:
    print(f'Warning: {e}')
"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=${TRAIN_BSZ} \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${N_RESP} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.total_epochs=${EPOCHS} \
    trainer.project_name=ming_moe_flow_grpo \
    trainer.experiment_name=bailing_moe_grpo_verl \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    "$@"
