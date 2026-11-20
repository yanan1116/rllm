#!/usr/bin/env bash
# FinQA multi-table-v2 synchronous GRPO on .29 (2x RTX 5000 Ada 32 GB), Qwen3-4B-Instruct-2507.
#
# Training hyperparameters are the single-table GRPO formal run's (finqa-grpo-run/train_16.sh epoch,
# experiment qwen3-4b-16-epoch): 32 tasks x 8 rollouts, mini-batch 32, lr 1e-6, T=0.7, LoRA r32/a32
# merged into vLLM, KL 0.001 low_var_kl, entropy 0.002, seq-mean-token-mean, clip 0.2/0.28, FSDP2 with
# offload, 10 epochs. 991 // 32 = 30 steps per epoch, 300 steps; a checkpoint every half epoch
# (15 steps, user 2026-10-12). Only what the benchmark forces differs (set by train_16_multitable_grpo.sh):
# the 49,152-token envelope, the multi-table rubric judged by gpt-5.4-nano, and fused log-prob kernels.
#
# Hardware, not hyperparameters (the update is mathematically unchanged):
#   * Ulysses sequence parallel 2 for actor and ref: the first formal launch (2026-10-12 14:52) ran out
#     of memory in the step-1 backward on these 32 GB cards with a 49,152-token micro-batch per GPU.
#     verl multiplies the per-GPU token budget by the SP size, so the per-GPU cap is set to 24,576: each
#     micro-batch still holds up to 49,152 tokens (the longest trajectory fits) and each GPU computes half.
#   * vLLM gpu_memory_utilization 0.55, because at 0.42 a 32 GB card holds
# only ~36k tokens of KV cache, less than one 49,152-token context (vLLM sleeps during the update, so
# training memory is unaffected); NCCL P2P off as for every two-GPU run on .29; checkpoints on .29's
# local disk.
#
#   train_29_multitable_grpo.sh smoke            # 2 steps, no checkpoint
#   train_29_multitable_grpo.sh epoch [hydra overrides...]
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-smoke}"
[[ " $(hostname -I) " == *" 10.225.68.29 "* ]] || { echo "run this on 10.225.68.29" >&2; exit 1; }
CKPT_ROOT="${FINQA_MULTI_CKPT_ROOT:-/mnt/disk1t/finqa-multitable-v2-checkpoints}"
[[ -d "$(dirname "$CKPT_ROOT")" && -w "$(dirname "$CKPT_ROOT")" ]] || {
  echo "checkpoint root $CKPT_ROOT is not on a writable local disk" >&2; exit 1; }
case "$(realpath -m "$CKPT_ROOT")" in
  /home/yanan/agents/*) echo "checkpoint root $CKPT_ROOT is on NFS ~/agents; use the host's local disk" >&2; exit 1 ;;
esac
mkdir -p "$CKPT_ROOT"

export FINQA_MULTI_ALGORITHM=grpo
export FINQA_MULTI_TRAIN_BATCH_SIZE=32
export FINQA_MULTI_EPOCHS=10
export FINQA_MULTI_SAVE_FREQ=15
export FINQA_MULTI_GPU_MEMORY_UTILIZATION=0.55
export FINQA_MULTI_TRAIN_MAX_TOKENS_PER_GPU=24576
export FINQA_DISABLE_NCCL_P2P=1
export FINQA_MULTI_RUN_HOST_TAG=29
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-finqa-multitable-v2-grpo-b32n8-29}"

exec "$RUN_DIR/train_16_multitable_grpo.sh" "$MODE" \
  trainer.default_local_dir="$CKPT_ROOT/\${trainer.experiment_name}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  "${@:2}"
