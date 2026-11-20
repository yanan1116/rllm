#!/usr/bin/env bash
# Keep one GPU of an eval_gpu_pool.py pool busy with new checkpoints as their adapters appear.
#
# Whenever this GPU's pool process (the PID in OUT/gpuN.pid) has exited, the follower starts it
# again on every adapter in ADAPTER_SOURCE that has no result yet and still has unclaimed tasks,
# in step order. Run one follower per GPU with the same OUT: the pool's shared claims split each
# model's tasks across the GPUs, as for the first models. Exits when OUT/STOP exists; one follower
# per GPU (flock). Extra arguments go to eval_gpu_pool.py unchanged.
#
# Optional environment:
#   EVAL_FOLLOW_AFTER_PIDFILE  start nothing while the PID in this file is alive (another pool on
#                              this GPU, e.g. an earlier run's), so the two never share the GPU
#   EVAL_FOLLOW_ORDER          step (default): ascending step; epoch-first:N: while any whole epoch
#                              (step divisible by N) is pending, the pool gets only whole epochs, so
#                              a whole epoch saved later still runs before every half epoch
#
#   eval_follow.sh GPU PORT SPLIT OUT ADAPTER_SOURCE POLL_S -- [eval_gpu_pool.py args...]
set -u
GPU=${1:?usage}; PORT=${2:?usage}; SPLIT=${3:?usage}; OUT=${4:?usage}; SRC=${5:?usage}; POLL=${6:?usage}
shift 6; [[ ${1:-} == -- ]] && shift
POOL_ARGS=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${EVAL_PYTHON:?set EVAL_PYTHON to the venv python}
ROOT=${APPWORLD_ROOT:?set APPWORLD_ROOT}
TOTAL=$(wc -w < "$ROOT/data/datasets/$SPLIT.txt")
AFTER=${EVAL_FOLLOW_AFTER_PIDFILE:-}
ORDER=${EVAL_FOLLOW_ORDER:-step}
case "$ORDER" in step) ;; epoch-first:[1-9]*) ;; *) echo "EVAL_FOLLOW_ORDER must be step or epoch-first:N" >&2; exit 2 ;; esac
PIDF=$OUT/gpu$GPU.pid
LOG=$OUT/follow-gpu$GPU.log
# (.follower-*: pools started by the first version inherited fd 9 and still hold .follow-gpuN.lock)
exec 9<>"$OUT/.follower-gpu$GPU.lock"
flock -n 9 || { echo "a follower already runs for GPU $GPU in $OUT" >&2; exit 1; }
log() { echo "$(TZ=America/New_York date '+%F %T %Z') [follow gpu$GPU] $*" >> "$LOG"; }
log "following $SRC every ${POLL}s, order $ORDER${AFTER:+, after $AFTER} (pid $$)"
while [[ ! -e $OUT/STOP ]]; do
  # The sleeps close fd 9: an orphaned child must not keep the flock after the follower dies.
  if [[ -f $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then sleep 60 9<&-; continue; fi
  if [[ -n $AFTER && -f $AFTER ]] && kill -0 "$(cat "$AFTER")" 2>/dev/null; then sleep 60 9<&-; continue; fi
  dirs=$(ls -d "$SRC"/global_step_* 2>/dev/null | sort -t_ -k3 -n)
  if [[ $ORDER == epoch-first:* ]]; then
    every=${ORDER#epoch-first:}
    dirs="$(for d in $dirs; do n=${d##*_}; (( n % every == 0 )) && echo "$d"; done)
$(for d in $dirs; do n=${d##*_}; (( n % every != 0 )) && echo "$d"; done)"
  fi
  models=()
  for dir in $dirs; do
    model=$(basename "$dir")
    [[ -f $dir/adapter_model.safetensors && -f $dir/adapter_config.json ]] || continue
    [[ -f $OUT/$model/result.json || -f $OUT/$model/result.INVALID.json ]] && continue
    claimed=$(wc -l < "$OUT/$model/claims.txt" 2>/dev/null || echo 0)
    (( claimed < TOTAL )) || continue
    models+=("$model")
  done
  if [[ $ORDER == epoch-first:* ]] && (( ${#models[@]} > 0 )); then
    whole=(); for m in "${models[@]}"; do (( ${m##*_} % every == 0 )) && whole+=("$m"); done
    (( ${#whole[@]} > 0 )) && models=("${whole[@]}")
  fi
  if (( ${#models[@]} == 0 )); then sleep "$POLL" 9<&-; continue; fi
  list=$(IFS=,; echo "${models[*]}")
  log "starting the pool on $list"
  setsid bash -c 'echo $$ > "$0"; exec "$1" -u "$2" "${@:3}"' "$PIDF" "$PYTHON" "$HERE/eval_gpu_pool.py" \
    --gpu "$GPU" --port "$PORT" --split "$SPLIT" --output "$OUT" --models "$list" --adapter-source "$SRC" \
    "${POOL_ARGS[@]}" >> "$OUT/gpu$GPU.stdout" 2>&1 < /dev/null 9<&- &
  sleep 30 9<&-
done
log "STOP file found, exiting"
