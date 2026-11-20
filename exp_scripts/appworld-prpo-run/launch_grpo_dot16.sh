#!/usr/bin/env bash
# One-command launch of the rLLM GRPO AppWorld run on .16 (2x RTX 6000 Ada 48 GB), Qwen3.5-4B, with
# the Gated DeltaNet packing fix. Run it ON .16:
#
#   ssh 10.225.68.16 /home/yanan/agents/rllm/exp_scripts/appworld-prpo-run/launch_grpo_dot16.sh [--check]
#
# Settings are the 2026-10-09 .16 GRPO run's (stopped after 12 steps because of the GDN leak) plus
# --gdn-varlen: 9 tasks x 8 rollouts, 10 epochs = 100 steps, lr 1e-5, context 32768, a checkpoint
# every 5 steps (half an epoch), 72 concurrent episodes, vLLM gpu_memory_utilization 0.45, NCCL P2P
# off. All training gates stay in train_appworld.sh; this script only checks that the host is ready
# and starts it detached. --check runs the checks and prints the command without starting anything.
#
# Re-running after a crash resumes from the run's latest checkpoint (train_appworld.sh resume_mode
# auto); a fresh run needs a new EXPERIMENT_NAME.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-appworld-grpo-qwen35-4b-b9n8-c32768-lr1e-5-2gpu-dot16-gdnvarlen}"
CKPT_DIR=/works/yanan/rllm-appworld-checkpoints/$EXPERIMENT_NAME
PID_FILE=$RUN_DIR/runtime/grpo-16-2gpu-gdnvarlen.pid
CONSOLE=$RUN_DIR/runtime/grpo-16-2gpu-gdnvarlen.console
MEMCAP_PID_FILE=$RUN_DIR/runtime/appworld_mem_cap-16.pid
CHECK_ONLY=0
[[ "${1:-}" == --check ]] && CHECK_ONLY=1

die() { echo "[launch-grpo-16] FATAL: $*" >&2; exit 1; }
say() { echo "[launch-grpo-16] $*"; }

[[ " $(hostname -I) " == *" 10.225.68.16 "* ]] \
  || die "expected host 10.225.68.16, got $(hostname -I); this run's disk (/works) and GPUs are .16's"
[[ -d /works/yanan && -w /works/yanan ]] || die "/works/yanan not writable: checkpoints must go to .16's local disk"

if [[ -f $PID_FILE ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  die "a GRPO launcher already runs (pid $(cat "$PID_FILE"), $PID_FILE)"
fi
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 > 1000 {print "GPU" $1 "=" $2 "MiB"}' | xargs || true)
[[ -z $busy ]] || die "GPUs not free ($busy): expected both GPUs idle; another job holds them. Wait for it or stop it first"
listening=$(ss -ltn | awk '{print $4}' | grep -oE ':(20[0-9]{3})$' | tr -d ':' | awk '$1 < 20072' | xargs || true)   # no listener is the normal case (grep exits 1)
[[ -z $listening ]] || die "ports in the AppWorld pool 20000-20071 already listen ($listening): orphan AppWorld servers; stop them by PID first"

if [[ -f $MEMCAP_PID_FILE ]] && ps -o args= -p "$(cat "$MEMCAP_PID_FILE")" 2>/dev/null | grep -q appworld_mem_cap.sh; then
  say "AppWorld memory cap watchdog running (pid $(cat "$MEMCAP_PID_FILE"))"
elif (( CHECK_ONLY )); then
  say "AppWorld memory cap watchdog NOT running; it would be started"
else
  setsid -f "$RUN_DIR/appworld_mem_cap.sh" 16 2 "$MEMCAP_PID_FILE" > /dev/null 2>&1 < /dev/null
  sleep 1; say "started the AppWorld memory cap watchdog (16 GB per server, pid $(cat "$MEMCAP_PID_FILE"))"
fi

if [[ -f $CKPT_DIR/latest_checkpointed_iteration.txt ]]; then
  say "NOTE: $CKPT_DIR has checkpoints up to step $(cat "$CKPT_DIR/latest_checkpointed_iteration.txt"); the run RESUMES from there"
else
  say "fresh run: $CKPT_DIR has no checkpoint"
fi

CMD=("$RUN_DIR/train_appworld.sh" train --algorithm grpo --gpus 0,1 --disable-nccl-p2p --gdn-varlen
     --save-freq 5 --max-episodes 72 --gpu-mem-util 0.45 --experiment-name "$EXPERIMENT_NAME"
     -- +ray_init.address=local +ray_init.include_dashboard=false)
say "command: ${CMD[*]}"
if (( CHECK_ONLY )); then say "--check: all checks passed, nothing started"; exit 0; fi

mkdir -p "$RUN_DIR/runtime"
setsid -f bash -c 'echo $$ > "$0"; exec "${@:1}"' "$PID_FILE" "${CMD[@]}" > "$CONSOLE" 2>&1 < /dev/null
sleep 3
say "started: pid $(cat "$PID_FILE"), console $CONSOLE"
say "first checks: both WorkerDicts print '[appworld_gdn_varlen] ...'; step 1 offpolicy/kl should be ~0.0005 (0.015 means the leak is back)"
