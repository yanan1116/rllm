#!/usr/bin/env bash
# FinQA multi-table-v2 synchronous PRPO on .16 (2x RTX 6000 Ada), Qwen3-4B-Instruct-2507.
#
# Training hyperparameters are the single-table PRPO formal run's
# (finqa-prpo-run/train_16_prpo.sh + train_16_prpo_formal.sh), which worked:
# 64 tasks x 1 rollout, lr 1e-6, T=0.7, LoRA r32/a32 merged into vLLM, KL 0.001 low_var_kl,
# entropy 0.002, seq-mean-token-mean, clip 0.2/0.28, FSDP2 with offload, 50 epochs, one
# checkpoint per epoch, vLLM gpu_memory_utilization 0.42, tis off, min_trajs_per_group 1.
# Only what the multi-table benchmark forces differs (all set by train_16_multitable_grpo.sh):
# the 49,152-token envelope (prompt 40,960 / response 8,192 / packing ceiling 49,152), the
# multi-table rubric judged by gpt-5.4-nano, and fused log-prob kernels for the long rows.
# 991 // 64 = 15 steps per epoch, 750 steps in 50 epochs.
#
# Infrastructure, not hyperparameters: checkpoints go to .16's local disk (never NFS), and
# NCCL P2P is disabled as for every two-GPU run on .16 this month.
#
#   train_16_multitable_prpo.sh smoke            # 2 steps, no checkpoint
#   train_16_multitable_prpo.sh epoch [hydra overrides...]
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-smoke}"
CKPT_ROOT="${FINQA_MULTI_CKPT_ROOT:-/works/yanan/finqa-multitable-v2-checkpoints}"
[[ -d "$(dirname "$CKPT_ROOT")" && -w "$(dirname "$CKPT_ROOT")" ]] || {
  echo "checkpoint root $CKPT_ROOT is not on a writable local disk (run this on .16)" >&2; exit 1; }
case "$(realpath -m "$CKPT_ROOT")" in
  /home/yanan/agents/*) echo "checkpoint root $CKPT_ROOT is on NFS ~/agents; use the host's local disk" >&2; exit 1 ;;
esac
mkdir -p "$CKPT_ROOT"

export FINQA_MULTI_ALGORITHM=prpo
export FINQA_MULTI_TRAIN_BATCH_SIZE=64
export FINQA_MULTI_EPOCHS=50
export FINQA_MULTI_GPU_MEMORY_UTILIZATION=0.42
export FINQA_DISABLE_NCCL_P2P=1
export FINQA_MULTI_RUN_HOST_TAG=16
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-finqa-multitable-v2-prpo-b64-16}"

exec "$RUN_DIR/train_16_multitable_grpo.sh" "$MODE" \
  trainer.default_local_dir="$CKPT_ROOT/\${trainer.experiment_name}" \
  "${@:2}"
