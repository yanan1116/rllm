#!/usr/bin/env bash
# Extract the LoRA adapter of every checkpoint a running verl run writes, as soon as the checkpoint
# is complete, so another host can evaluate it (CPU only: verl model_merger under nice 10).
#
# A checkpoint is taken only once latest_checkpointed_iteration.txt has reached its step, i.e.
# after verl finished writing it. Each adapter is checked (some lora_B must be non-zero) and moved
# into place atomically. Exits when ADAPTER_DIR/STOP exists; one follower per ADAPTER_DIR (flock).
#
#   adapter_follow.sh CKPT_DIR ADAPTER_DIR [POLL_S=300]
set -u
CKPT=${1:?usage: $0 CKPT_DIR ADAPTER_DIR [POLL_S]}
ADP=${2:?usage: $0 CKPT_DIR ADAPTER_DIR [POLL_S]}
POLL=${3:-300}
VENV=/home/yanan/agents/rllm/.venv
mkdir -p "$ADP"
LOG=$ADP/adapter_follow.log
exec 9<>"$ADP/.follow.lock"
flock -n 9 || { echo "another adapter_follow.sh already follows $ADP" >&2; exit 1; }
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$LOG"; }
log "following $CKPT every ${POLL}s (pid $$)"
while [[ ! -e $ADP/STOP ]]; do
  latest=$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  for dir in $(ls -d "$CKPT"/global_step_* 2>/dev/null | sort -t_ -k3 -n); do
    step=$(basename "$dir"); n=${step#global_step_}
    (( n <= latest )) || continue
    [[ -f $ADP/$step/adapter_model.safetensors ]] && continue
    staging=$ADP/.staging-$step
    rm -rf "$staging"
    if CUDA_VISIBLE_DEVICES= nice -n 10 "$VENV/bin/python" -m verl.model_merger merge --backend fsdp \
         --local_dir "$dir/actor" --target_dir "$staging" >> "$LOG" 2>&1 \
       && "$VENV/bin/python" - "$staging/lora_adapter" >> "$LOG" 2>&1 <<'EOF'
import sys
from safetensors import safe_open
f = safe_open(sys.argv[1] + "/adapter_model.safetensors", "pt")
b = [k for k in f.keys() if "lora_B" in k]
assert b and max(float(f.get_tensor(k).abs().max()) for k in b) > 0, "all lora_B are zero"
print(f"adapter ok: {len(list(f.keys()))} tensors")
EOF
    then
      mv "$staging/lora_adapter" "$ADP/$step" && rm -rf "$staging" && log "$step extracted"
    else
      log "$step FAILED (see above); retried next poll"
    fi
  done
  sleep "$POLL"
done
log "STOP file found, exiting"
