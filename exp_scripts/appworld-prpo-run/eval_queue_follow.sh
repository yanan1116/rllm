#!/usr/bin/env bash
# Keep one GPU of the .36 eval_gpu_pool.py pools busy with a prioritized list of evaluation queues.
#
# QUEUE_FILE has one queue per line (blank lines and # comments ignored):
#   SPLIT OUT ADAPTER_SOURCE SELECT
# SELECT picks the checkpoints of ADAPTER_SOURCE (global_step_N with an adapter): whole:N (steps
# divisible by N), half:N (the others), all; a "+base" suffix adds the base model (no adapter).
# Whenever this GPU's pool (PID in CTRL/gpuN.pid) has exited, the follower walks the queues in file
# order and starts the pool on the first pending model of the first queue that has one: no result.json /
# result.INVALID.json and fewer claims than the split's tasks. Every GPU runs the same follower with
# the same QUEUE_FILE, so the pool's shared claims split each model's tasks across the GPUs. The
# queue file is re-read at every poll, so lines can be added or reordered while it runs.
#
# Optional environment: EVAL_FOLLOW_AFTER_PIDFILE (start nothing while that PID lives, e.g. a pool
# started by an earlier follower). Required: EVAL_PYTHON, APPWORLD_ROOT. Exits when CTRL/STOP exists;
# one follower per GPU and CTRL (flock). Extra arguments go to eval_gpu_pool.py unchanged.
#
#   eval_queue_follow.sh GPU PORT CTRL QUEUE_FILE POLL_S -- [eval_gpu_pool.py args...]
set -u
GPU=${1:?usage}; PORT=${2:?usage}; CTRL=${3:?usage}; QUEUE_FILE=${4:?usage}; POLL=${5:?usage}
shift 5; [[ ${1:-} == -- ]] && shift
POOL_ARGS=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${EVAL_PYTHON:?set EVAL_PYTHON to the venv python}
ROOT=${APPWORLD_ROOT:?set APPWORLD_ROOT}
AFTER=${EVAL_FOLLOW_AFTER_PIDFILE:-}
mkdir -p "$CTRL"
PIDF=$CTRL/gpu$GPU.pid
LOG=$CTRL/follow-gpu$GPU.log
exec 9<>"$CTRL/.follower-gpu$GPU.lock"
flock -n 9 || { echo "a queue follower already runs for GPU $GPU in $CTRL" >&2; exit 1; }
log() { echo "$(TZ=America/New_York date '+%F %T %Z') [queue gpu$GPU] $*" >> "$LOG"; }
alive() { [[ -f $1 ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

pending() {  # OUT TOTAL MODEL -> true when MODEL still has work
  local out=$1 total=$2 model=$3 claimed
  [[ -f $out/$model/result.json || -f $out/$model/result.INVALID.json || -f $out/$model/SKIPPED ]] && return 1
  claimed=$(wc -l < "$out/$model/claims.txt" 2>/dev/null || echo 0)
  (( claimed < total ))
}

rows() { wc -l < "$1/episodes.jsonl" 2>/dev/null || echo 0; }
last=""; last_rows=0; fails=0   # a model whose pool runs keep ending without one new episode is skipped

log "following $QUEUE_FILE every ${POLL}s${AFTER:+, after $AFTER} (pid $$)"
while [[ ! -e $CTRL/STOP ]]; do
  # The sleeps close fd 9: an orphaned child must not keep the flock after the follower dies.
  if alive "$PIDF"; then sleep 60 9<&-; continue; fi
  if [[ -n $AFTER ]] && alive "$AFTER"; then sleep 60 9<&-; continue; fi
  if [[ -n $last ]]; then
    [[ $last == "${fail_model:-}" ]] || { fail_model=$last; fails=0; }   # count per model
    if (( $(rows "$last") == last_rows )); then fails=$((fails + 1)); else fails=0; fi
    if (( fails >= 3 )); then
      echo "$(TZ=America/New_York date '+%F %T %Z') skipped by eval_queue_follow.sh: 3 pool runs in a row ended without an episode (see gpu*.log: refused adapter, vLLM start failure)" > "$last/SKIPPED"
      log "SKIPPED $last after 3 runs without a new episode"
      fails=0
    fi
    last=""
  fi
  started=0
  while read -r split out src select; do
    [[ -z ${split:-} || $split == \#* ]] && continue
    total=$(wc -w < "$ROOT/data/datasets/$split.txt")
    base=0; [[ $select == *+base ]] && { base=1; select=${select%+base}; }
    models=()
    (( base )) && pending "$out" "$total" base && models+=(base)
    for dir in $(ls -d "$src"/global_step_* 2>/dev/null | sort -t_ -k3 -n); do
      model=$(basename "$dir"); n=${model#global_step_}
      [[ -f $dir/adapter_model.safetensors && -f $dir/adapter_config.json ]] || continue
      case $select in
        whole:*) (( n % ${select#whole:} == 0 )) || continue ;;
        half:*)  (( n % ${select#half:} != 0 )) || continue ;;
        all) ;;
        *) log "bad SELECT '$select' in $QUEUE_FILE"; continue 2 ;;
      esac
      pending "$out" "$total" "$model" && models+=("$model")
    done
    (( ${#models[@]} )) || continue
    # One model per pool run: after each model the queues are walked again, so a checkpoint saved
    # later in a higher-priority queue (e.g. a new whole epoch on dev) runs before the rest of a
    # lower-priority queue (a pool given a long list would only come back after all of it).
    models=("${models[0]}")
    mkdir -p "$out"
    list=$(IFS=,; echo "${models[*]}")
    log "starting the pool: $split $out on $list"
    last=$out/${models[0]}; last_rows=$(rows "$last")
    setsid bash -c 'echo $$ > "$0"; exec "$1" -u "$2" "${@:3}"' "$PIDF" "$PYTHON" "$HERE/eval_gpu_pool.py" \
      --gpu "$GPU" --port "$PORT" --split "$split" --output "$out" --models "$list" --adapter-source "$src" \
      "${POOL_ARGS[@]}" >> "$out/gpu$GPU.stdout" 2>&1 < /dev/null 9<&- &
    started=1
    break
  done < "$QUEUE_FILE"
  if (( started )); then sleep 30 9<&-; else sleep "$POLL" 9<&-; fi
done
log "STOP file found, exiting"
