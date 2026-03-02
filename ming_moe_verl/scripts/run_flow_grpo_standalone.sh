#!/bin/bash
# =============================================================================
# 连续动作空间 Flow-GRPO 训练 (Ming-omni-tts MoE)
#
# 使用基于 stochastic ODE 的精确 log-prob 或 DDPO 代理
# =============================================================================
set -x

MODEL_PATH=${MODEL_PATH:-"/path/to/ming-omni-tts"}
TRAIN_DATA=${TRAIN_DATA:-"./data/tts_grpo/train.parquet"}
VAL_DATA=${VAL_DATA:-"./data/tts_grpo/test.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/flow_grpo"}

TRAIN_MODE=${TRAIN_MODE:-"llm_only"}  # llm_only or full
GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-4}
LOG_PROB_METHOD=${LOG_PROB_METHOD:-"exact"}  # exact or surrogate
ODE_STEPS=${ODE_STEPS:-10}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-1e-6}
PPO_CLIP=${PPO_CLIP:-0.2}
KL_COEF=${KL_COEF:-0.001}

MAX_DECODE_STEPS=${MAX_DECODE_STEPS:-200}
CFG_SCALE=${CFG_SCALE:-2.0}
SIGMA=${SIGMA:-0.25}

export PYTHONPATH="${MODEL_PATH}:${PYTHONPATH}"
export MING_MODEL_DIR="${MODEL_PATH}"

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Training data not found, creating demo dataset..."
    python -m ming_moe_verl.data.preprocess --create_demo --output_dir $(dirname ${TRAIN_DATA})
fi

python -m ming_moe_verl.train_flow_grpo \
    --model_path "${MODEL_PATH}" \
    --model_type moe \
    --train_data "${TRAIN_DATA}" \
    --train_mode ${TRAIN_MODE} \
    --grpo_group_size ${GRPO_GROUP_SIZE} \
    --log_prob_method ${LOG_PROB_METHOD} \
    --ode_steps ${ODE_STEPS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --ppo_clip ${PPO_CLIP} \
    --kl_coef ${KL_COEF} \
    --max_decode_steps ${MAX_DECODE_STEPS} \
    --cfg_scale ${CFG_SCALE} \
    --sigma ${SIGMA} \
    --output_dir "${OUTPUT_DIR}" \
    --gradient_checkpointing \
    --bf16 \
    "$@"
