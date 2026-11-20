#!/usr/bin/env bash
# FinQA multi-table-v2 synchronous GRPO on .16 (2x RTX 6000 Ada 48 GB), Qwen3-4B-Instruct-2507.
#
# Training hyperparameters are the single-table GRPO formal run's (finqa-grpo-run/train_16.sh epoch,
# experiment qwen3-4b-16-epoch): 32 tasks x 8 rollouts, mini-batch 32, lr 1e-6, T=0.7, LoRA r32/a32
# merged into vLLM, KL 0.001 low_var_kl, entropy 0.002, seq-mean-token-mean, clip 0.2/0.28, FSDP2 with
# offload, vLLM gpu_memory_utilization 0.42. 10 epochs; 991 // 32 = 30 steps per epoch, 300 steps; a
# checkpoint every half epoch (15 steps, user 2026-10-12). Only what the benchmark forces differs (set by
# train_16_multitable_grpo.sh): the 49,152-token envelope, the multi-table rubric judged by gpt-5.4-nano,
# and fused log-prob kernels for the long rows. Same settings as train_29_multitable_grpo.sh minus that
# script's 32 GB-card workarounds (Ulysses SP 2, per-GPU token cap 24,576, gpu util 0.55).
#
# Infrastructure, not hyperparameters: checkpoints go to .16's local disk (never NFS), and NCCL P2P is
# disabled as for every two-GPU run on .16 this month.
#
#   train_16_multitable_grpo_b32.sh smoke            # 2 steps, no checkpoint
#   train_16_multitable_grpo_b32.sh epoch [hydra overrides...]
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-smoke}"
[[ " $(hostname -I) " == *" 10.225.68.16 "* ]] || { echo "run this on 10.225.68.16" >&2; exit 1; }
CKPT_ROOT="${FINQA_MULTI_CKPT_ROOT:-/works/yanan/finqa-multitable-v2-checkpoints}"
[[ -d "$(dirname "$CKPT_ROOT")" && -w "$(dirname "$CKPT_ROOT")" ]] || {
  echo "checkpoint root $CKPT_ROOT is not on a writable local disk (run this on .16)" >&2; exit 1; }
case "$(realpath -m "$CKPT_ROOT")" in
  /home/yanan/agents/*) echo "checkpoint root $CKPT_ROOT is on NFS ~/agents; use the host's local disk" >&2; exit 1 ;;
esac
mkdir -p "$CKPT_ROOT"

export FINQA_MULTI_ALGORITHM=grpo
export FINQA_MULTI_TRAIN_BATCH_SIZE=32
export FINQA_MULTI_EPOCHS=10
export FINQA_MULTI_SAVE_FREQ=15
export FINQA_MULTI_GPU_MEMORY_UTILIZATION=0.42
export FINQA_DISABLE_NCCL_P2P=1
export FINQA_MULTI_RUN_HOST_TAG=16
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-finqa-multitable-v2-grpo-b32n8-16}"

exec "$RUN_DIR/train_16_multitable_grpo.sh" "$MODE" \
  trainer.default_local_dir="$CKPT_ROOT/\${trainer.experiment_name}" \
  "${@:2}"
