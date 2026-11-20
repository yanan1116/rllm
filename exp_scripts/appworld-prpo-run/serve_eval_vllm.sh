#!/usr/bin/env bash
# Serve Qwen3.5-4B (eos-fixed) for AppWorld evaluation on one GPU, optionally with one LoRA
# adapter mounted unmerged (vLLM --enable-lora; the adapter is served as model "ckpt", the base
# as "base"). Thinking stays off per request (chat_template_kwargs), so no reasoning parser.
#
#   serve_eval_vllm.sh GPU PORT LOG [ADAPTER_DIR] [GPU_MEM_UTIL=0.42]
# Runs in the foreground (exec); the caller backgrounds it and owns its PID.
set -euo pipefail
GPU=${1:?gpu}; PORT=${2:?port}; LOG=${3:?log}; ADAPTER=${4:-}; UTIL=${5:-0.42}
MODEL=${EVAL_MODEL_PATH:-/works/yanan/models/qwen35-4b-851bf6e8-eosfix}
VENV=/home/yanan/agents/rllm/.venv
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$VENV/lib/python3.11/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$GPU" VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
ARGS=(
  serve "$MODEL" --served-model-name base --host 127.0.0.1 --port "$PORT"
  --max-model-len 32768 --gpu-memory-utilization "$UTIL" --max-num-seqs 16
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
)
if [[ -n "$ADAPTER" ]]; then
  [[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] \
    || { echo "expected a PEFT adapter at $ADAPTER" >&2; exit 1; }
  ARGS+=(--enable-lora --max-lora-rank 32 --max-loras 1 --lora-modules "ckpt=$ADAPTER")
fi
exec vllm "${ARGS[@]}" >> "$LOG" 2>&1
