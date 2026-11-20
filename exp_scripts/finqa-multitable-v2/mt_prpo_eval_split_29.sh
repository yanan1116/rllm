#!/usr/bin/env bash
# One shot on .29 (user, 2026-10-13: "epoch 8 保存之后 在本机上面 gpu-0 负责 val gpu-1 负责 test split. 这样可以双倍加速"):
# wait until the LoRA adapter of ITEM of the .16 FinQA multi-table-v2 PRPO run is extracted, then evaluate it under
# the same v2 protocol as mt_prpo_eval_lane_follow_29.sh, with multi_val on GPU 0 and multi_test on GPU 1 at the
# same time (eval_lane.sh FINQA_MULTITABLE_SPLITS). ITEM must be pre-claimed in OUT/.claims so the two follower
# lanes leave it alone.
#
#   mt_prpo_eval_split_29.sh ITEM          # e.g. global_step_120
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
ITEM=${1:?usage: $0 ITEM}
ADP=/mnt/disk1t/finqa-multitable-v2-adapters/finqa-multitable-v2-prpo-b64-16
OUT=/mnt/disk1t/finqa-multitable-v2-eval-dot29/prpo-b64-16
LOG=$DIR/logs/mt_prpo_eval_split_29.log
log() { echo "$(TZ=America/New_York date '+%F %T %Z') [split $ITEM] $*" >> "$LOG"; }

[[ " $(hostname -I) " == *" 10.225.68.29 "* ]] || { echo "run this on 10.225.68.29" >&2; exit 1; }
[[ -d $OUT/.claims/$ITEM ]] || { echo "pre-claim $OUT/.claims/$ITEM first, or a follower lane may take it" >&2; exit 1; }
export FINQA_LORA_ROOT=$ADP
export FINQA_MULTITABLE_OUTPUT_ROOT=$OUT
export FINQA_MULTITABLE_RUNTIME_TAG=prpo-b64-16-dot29
export FINQA_MULTITABLE_BASE_MODEL=/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

log "waiting for $ADP/$ITEM/adapter_model.safetensors"
until [[ -f $ADP/$ITEM/adapter_model.safetensors && -f $ADP/$ITEM/adapter_config.json ]]; do sleep 60; done
for _ in $(seq 1 120); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  (( busy == 0 )) && break; sleep 30
done
(( busy == 0 )) || { log "GPUs still busy after 1 h; NOT started"; exit 1; }

FINQA_MULTITABLE_SPLITS=multi_val bash "$DIR/eval_lane.sh" lora 0 29830 "$ITEM" > "$DIR/logs/mt_prpo_eval_split_29_gpu0.log" 2>&1 &
V=$!
FINQA_MULTITABLE_SPLITS=multi_test bash "$DIR/eval_lane.sh" lora 1 29831 "$ITEM" > "$DIR/logs/mt_prpo_eval_split_29_gpu1.log" 2>&1 &
T=$!
log "started multi_val on GPU 0 (pid $V) and multi_test on GPU 1 (pid $T)"
wait "$V"; rv=$?
wait "$T"; rt=$?
log "multi_val exit $rv, multi_test exit $rt; COMPLETE: $([[ -e $OUT/$ITEM/COMPLETE ]] && echo yes || echo no)"
