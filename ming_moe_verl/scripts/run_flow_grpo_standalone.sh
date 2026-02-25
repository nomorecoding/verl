#!/bin/bash
# =============================================================================
# Standalone Flow-GRPO training for Ming-omni-tts MoE model
#
# This script runs the full-pipeline Flow-GRPO training (MODE B)
# where both the MoE-LLM backbone and flow-matching head are trained.
#
# Usage:
#   bash scripts/run_flow_grpo_standalone.sh
#
# Prerequisites:
#   1. pip install torch transformers accelerate pandas pyarrow
#   2. pip install x-transformers   (required by Ming-omni-tts DiT)
#   3. Model weights downloaded to MODEL_PATH
#   4. Training data preprocessed to parquet format
# =============================================================================
set -x

# ----- Paths -----
MODEL_PATH=${MODEL_PATH:-"/path/to/ming-omni-tts"}
TRAIN_DATA=${TRAIN_DATA:-"./data/tts_grpo/train.parquet"}
VAL_DATA=${VAL_DATA:-"./data/tts_grpo/test.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/flow_grpo"}

# ----- GRPO Hyperparameters -----
GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-4}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-4}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-2}
LR=${LR:-1e-6}
PPO_CLIP=${PPO_CLIP:-0.2}
KL_COEF=${KL_COEF:-0.001}

# ----- Generation -----
MAX_DECODE_STEPS=${MAX_DECODE_STEPS:-200}
CFG_SCALE=${CFG_SCALE:-2.0}
SIGMA=${SIGMA:-0.25}

# ----- Infrastructure -----
NUM_GPUS=${NUM_GPUS:-1}

# Add model directory to Python path so we can import model classes
export PYTHONPATH="${MODEL_PATH}:${PYTHONPATH}"
export MING_MODEL_DIR="${MODEL_PATH}"

# Create demo data if training data doesn't exist
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Training data not found, creating demo dataset..."
    python -m ming_moe_verl.data.preprocess --create_demo --output_dir $(dirname ${TRAIN_DATA})
fi

python -m ming_moe_verl.train_flow_grpo \
    --model_path "${MODEL_PATH}" \
    --model_type moe \
    --train_data "${TRAIN_DATA}" \
    --val_data "${VAL_DATA}" \
    --grpo_group_size ${GRPO_GROUP_SIZE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --mini_batch_size ${MINI_BATCH_SIZE} \
    --lr ${LR} \
    --ppo_clip ${PPO_CLIP} \
    --kl_coef ${KL_COEF} \
    --max_decode_steps ${MAX_DECODE_STEPS} \
    --cfg_scale ${CFG_SCALE} \
    --sigma ${SIGMA} \
    --num_gpus ${NUM_GPUS} \
    --output_dir "${OUTPUT_DIR}" \
    --gradient_checkpointing \
    --bf16 \
    "$@"
