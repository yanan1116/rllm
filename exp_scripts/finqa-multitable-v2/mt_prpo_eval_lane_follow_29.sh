#!/usr/bin/env bash
# One GPU lane on .29 that keeps evaluating the .16 FinQA multi-table-v2 PRPO run (finqa-multitable-v2-prpo-b64-16)
# under the v2 protocol: base first, then every whole-epoch checkpoint in step order as its LoRA adapter appears
# (adapter_follow.sh extracts them to ADP). arm=lora: the adapter is served unmerged (vLLM --enable-lora), because
# a bf16 merge of a lr-1e-6 adapter erases most of its delta. Lanes share claims (mkdir), so two lanes never run
# the same item. Exits when STOP exists in OUT.
#
#   mt_prpo_eval_lane_follow_29.sh GPU PORT
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
GPU=${1:?usage: $0 GPU PORT}; PORT=${2:?usage: $0 GPU PORT}
ADP=/mnt/disk1t/finqa-multitable-v2-adapters/finqa-multitable-v2-prpo-b64-16
OUT=/mnt/disk1t/finqa-multitable-v2-eval-dot29/prpo-b64-16
mkdir -p "$OUT/.claims" "$DIR/logs"
LOG=$DIR/logs/mt_prpo_eval_lane_29_gpu$GPU.log
log() { echo "$(TZ=America/New_York date '+%F %T %Z') [lane gpu$GPU] $*" >> "$LOG"; }
export FINQA_LORA_ROOT=$ADP
export FINQA_MULTITABLE_OUTPUT_ROOT=$OUT
export FINQA_MULTITABLE_RUNTIME_TAG=prpo-b64-16-dot29
export FINQA_MULTITABLE_BASE_MODEL=/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log "following $ADP (pid $$)"
while [[ ! -e $OUT/STOP ]]; do
  item=""
  for cand in base $(ls -d "$ADP"/global_step_* 2>/dev/null | xargs -n1 basename 2>/dev/null | sort -t_ -k3 -n); do
    [[ -e $OUT/$cand/COMPLETE ]] && continue
    if mkdir "$OUT/.claims/$cand" 2>/dev/null; then item=$cand; break; fi
  done
  if [[ -z $item ]]; then sleep 300; continue; fi
  log "evaluating $item"
  if bash "$DIR/eval_lane.sh" lora "$GPU" "$PORT" "$item" >> "$LOG" 2>&1; then
    log "$item done"
  else
    log "$item FAILED (claim kept in $OUT/.claims/$item; see $OUT/$item)"
  fi
done
log "STOP found, exiting"
