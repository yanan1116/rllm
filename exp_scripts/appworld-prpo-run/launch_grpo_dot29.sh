#!/usr/bin/env bash
# One-command launch of the rLLM GRPO AppWorld run on .29 (2x RTX 5000 Ada 32 GB), Qwen3.5-4B, with
# the Gated DeltaNet packing fix. Run it ON .29:
#
#   /home/yanan/agents/rllm/exp_scripts/appworld-prpo-run/launch_grpo_dot29.sh [--check]
#
# Same algorithm settings as launch_grpo_dot16.sh: 9 tasks x 8 rollouts, 10 epochs = 100 steps,
# lr 1e-5, context 32768, a checkpoint every 5 steps (half an epoch), --gdn-varlen, NCCL P2P off.
# Only what the 32 GB cards force differs, as in the 2026-10-08 .29 GRPO run: vLLM
# gpu_memory_utilization 0.55 (0.35 leaves too little KV for a 32768 context), 48 concurrent
# episodes and max_num_seqs 32 (Mamba cache blocks). All training gates stay in train_appworld.sh;
# --check runs this script's checks and prints the command without starting anything.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-appworld-grpo-qwen35-4b-b9n8-c32768-lr1e-5-2gpu-dot29-gdnvarlen}"
CKPT_DIR=/mnt/disk1t/rllm-appworld-checkpoints/$EXPERIMENT_NAME
PID_FILE=$RUN_DIR/runtime/grpo-29-2gpu-gdnvarlen.pid
CONSOLE=$RUN_DIR/runtime/grpo-29-2gpu-gdnvarlen.console
MEMCAP_PID_FILE=$RUN_DIR/runtime/appworld_mem_cap-29.pid
CHECK_ONLY=0
[[ "${1:-}" == --check ]] && CHECK_ONLY=1
# GRPO_EPOCHS (optional): total epochs; unset keeps train_appworld.sh's default of 10. Raising it on a
# resumed run extends the same run (resume_mode auto; the LR schedule is constant).
EPOCH_ARGS=(); [[ -n "${GRPO_EPOCHS:-}" ]] && EPOCH_ARGS=(--epochs "$GRPO_EPOCHS")

die() { echo "[launch-grpo-29] FATAL: $*" >&2; exit 1; }
say() { echo "[launch-grpo-29] $*"; }

[[ " $(hostname -I) " == *" 10.225.68.29 "* ]] \
  || die "expected host 10.225.68.29, got $(hostname -I); this run's disk (/mnt/disk1t) and GPUs are .29's"
[[ -d /mnt/disk1t && -w /mnt/disk1t ]] || die "/mnt/disk1t not writable: checkpoints must go to .29's local disk"
if [[ -f $PID_FILE ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  die "a GRPO launcher already runs (pid $(cat "$PID_FILE"), $PID_FILE)"
fi
trainers=$(ps -u "$(id -u)" -o args= | grep -c '[t]rain_appworld.sh' || true)
(( trainers == 0 )) || die "$trainers train_appworld.sh process(es) still running on this host"
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 > 1000 {print "GPU" $1 "=" $2 "MiB"}' | xargs || true)
[[ -z $busy ]] || die "GPUs not free ($busy): expected both GPUs idle"
listening=$(ss -ltn | awk '{print $4}' | grep -oE ':(20[0-9]{3})$' | tr -d ':' | awk '$1 < 20048' | xargs || true)   # no listener is the normal case (grep exits 1)
[[ -z $listening ]] || die "ports in the AppWorld pool 20000-20047 already listen ($listening): orphan AppWorld servers; stop them by PID first"
if [[ -f $MEMCAP_PID_FILE ]] && ps -o args= -p "$(cat "$MEMCAP_PID_FILE")" 2>/dev/null | grep -q appworld_mem_cap.sh; then
  say "AppWorld memory cap watchdog running (pid $(cat "$MEMCAP_PID_FILE"))"
elif (( CHECK_ONLY )); then
  say "AppWorld memory cap watchdog NOT running; it would be started"
else
  setsid -f "$RUN_DIR/appworld_mem_cap.sh" 16 2 "$MEMCAP_PID_FILE" > /dev/null 2>&1 < /dev/null
  sleep 1; say "started the AppWorld memory cap watchdog (pid $(cat "$MEMCAP_PID_FILE"))"
fi
if [[ -f $CKPT_DIR/latest_checkpointed_iteration.txt ]]; then
  say "NOTE: $CKPT_DIR has checkpoints up to step $(cat "$CKPT_DIR/latest_checkpointed_iteration.txt"); the run RESUMES from there"
else
  say "fresh run: $CKPT_DIR has no checkpoint"
fi

CMD=("$RUN_DIR/train_appworld.sh" train --algorithm grpo --gpus 0,1 --disable-nccl-p2p --gdn-varlen
     --save-freq 5 --max-episodes 48 --max-num-seqs 32 --gpu-mem-util 0.55 "${EPOCH_ARGS[@]}" --experiment-name "$EXPERIMENT_NAME"
     -- +ray_init.address=local +ray_init.include_dashboard=false)
say "command: ${CMD[*]}"
if (( CHECK_ONLY )); then say "--check: all checks passed, nothing started"; exit 0; fi

setsid -f bash -c 'echo $$ > "$0"; exec "${@:1}"' "$PID_FILE" "${CMD[@]}" > "$CONSOLE" 2>&1 < /dev/null
sleep 3
say "started: pid $(cat "$PID_FILE"), console $CONSOLE"
