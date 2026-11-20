#!/usr/bin/env bash
# Follower on .29 (user, 2026-10-13: ".29 负责评估每个 checkpoint 一旦落盘，立刻评估"): evaluate every LoRA adapter that
# appears in ADP (extracted by appworld-prpo-run/adapter_follow.sh from the .16 run), in step order, under the
# FinQA multi-table v2 protocol (eval_lane.sh arm=lora, adapter unmerged), with multi_val on GPU 0 and multi_test on
# GPU 1 at the same time (eval_lane.sh FINQA_MULTITABLE_SPLITS). Claims are mkdir OUT/.claims/ITEM; a failed item
# keeps its claim and is not retried. Exits when OUT/STOP exists; one follower per OUT (flock).
#
#   mt_eval_follow_split_29.sh ADP OUT RUNTIME_TAG [POLL_S=300]
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
ADP=${1:?usage: $0 ADP OUT RUNTIME_TAG [POLL_S]}; OUT=${2:?usage}; TAG=${3:?usage}; POLL=${4:-300}
[[ " $(hostname -I) " == *" 10.225.68.29 "* ]] || { echo "run this on 10.225.68.29" >&2; exit 1; }
mkdir -p "$OUT/.claims"
exec 9<>"$OUT/.follow.lock"
flock -n 9 || { echo "another follower already evaluates into $OUT" >&2; exit 1; }
LOG=$DIR/logs/mt_eval_follow_split_29.$(basename "$OUT").log
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*" >> "$LOG"; }
export FINQA_LORA_ROOT=$ADP
export FINQA_MULTITABLE_OUTPUT_ROOT=$OUT
export FINQA_MULTITABLE_RUNTIME_TAG=$TAG
export FINQA_MULTITABLE_BASE_MODEL=/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log "following $ADP into $OUT every ${POLL}s (pid $$)"
while [[ ! -e $OUT/STOP ]]; do
  item=""
  for cand in $(ls -d "$ADP"/global_step_* 2>/dev/null | xargs -n1 basename 2>/dev/null | sort -t_ -k3 -n); do
    [[ -f $ADP/$cand/adapter_model.safetensors ]] || continue
    [[ -e $OUT/$cand/COMPLETE ]] && continue
    if mkdir "$OUT/.claims/$cand" 2>/dev/null; then item=$cand; break; fi
  done
  if [[ -z $item ]]; then sleep "$POLL"; continue; fi
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  (( busy == 0 )) || log "warning: $busy GPU(s) already in use before $item"
  log "evaluating $item: multi_val on GPU 0, multi_test on GPU 1"
  FINQA_MULTITABLE_SPLITS=multi_val bash "$DIR/eval_lane.sh" lora 0 29830 "$item" >> "$LOG.gpu0" 2>&1 &
  V=$!
  FINQA_MULTITABLE_SPLITS=multi_test bash "$DIR/eval_lane.sh" lora 1 29831 "$item" >> "$LOG.gpu1" 2>&1 &
  T=$!
  wait "$V"; rv=$?; wait "$T"; rt=$?
  if [[ -e $OUT/$item/COMPLETE ]]; then log "$item done"; else log "$item FAILED (multi_val exit $rv, multi_test exit $rt; claim kept)"; fi
done
log "STOP found, exiting"
