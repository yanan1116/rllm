#!/usr/bin/env bash
# One evaluation lane: claim the next item (base, global_step_1, global_step_2, ...) from a queue
# shared by every lane of the run, serve it on one GPU with the LoRA adapter mounted unmerged
# (vLLM --enable-lora), evaluate one split greedy (eval_appworld.py), stop the server, repeat.
#
#   eval_lane.sh LANE GPU PORT [SPLIT=dev] [UTIL=0.42] [LAST_STEP=60]
#
# Items are claimed in order under an flock, so two lanes on one GPU evaluate two checkpoints at
# a time, oldest first. A checkpoint not yet saved by the still-running trainer is waited for.
# vLLM start-up is serialized across lanes (a second engine profiling while the first loads sees
# no room for its KV cache). A failed item is released for a retry, up to 3 failures.
# For each checkpoint the adapter is extracted once with verl's own model_merger (its
# lora_adapter/), checked to have non-zero lora_B, and the served adapter must move the base's
# log-probs (an adapter that changes nothing means the model is not mounted).
#
# Paths default to .16; another host overrides them (the defaults are .16's):
#   EVAL_CKPT_ROOT     verl checkpoints            (/works/yanan/rllm-appworld-checkpoints)
#   EVAL_ADAPTER_ROOT  extracted adapters, local   (/works/yanan/rllm-appworld-adapters)
#   EVAL_ADAPTER_SOURCE  optional: wait for each adapter in this directory (e.g. .16's adapters
#                      through the read-only sshfs mount) and copy it, instead of extracting it
#   EVAL_OUT_ROOT      results                     (/works/yanan/rllm-appworld-eval)
#   EVAL_MODEL_PATH    served model and tokenizer  (/works/yanan/models/qwen35-4b-851bf6e8-eosfix)
#   APPWORLD_ROOT      AppWorld data, per host (passed through to eval_appworld.py)
set -uo pipefail
LANE=${1:?lane}; GPU=${2:?gpu}; PORT=${3:?port}; SPLIT=${4:-dev}; UTIL=${5:-0.42}; LAST_STEP=${6:-60}
EXP=${EVAL_EXPERIMENT:-appworld-prpo-qwen35-4b-b15n1-c32768-lr1e-5-1gpu}
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT=${EVAL_CKPT_ROOT:-/works/yanan/rllm-appworld-checkpoints}/$EXP
ADP=${EVAL_ADAPTER_ROOT:-/works/yanan/rllm-appworld-adapters}/$EXP
OUT=${EVAL_OUT_ROOT:-/works/yanan/rllm-appworld-eval}/$EXP/$SPLIT
SOURCE=${EVAL_ADAPTER_SOURCE:+$EVAL_ADAPTER_SOURCE/$EXP}
MODEL_PATH=${EVAL_MODEL_PATH:-/works/yanan/models/qwen35-4b-851bf6e8-eosfix}
export EVAL_MODEL_PATH=$MODEL_PATH
VENV=/home/yanan/agents/rllm/.venv
mkdir -p "$OUT" "$ADP"
LOG="$OUT/lane$LANE.log"
log() { echo "$(TZ=America/New_York date '+%F %T %Z') [lane$LANE] $*" | tee -a "$LOG"; }
export LD_LIBRARY_PATH="$VENV/lib/python3.11/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ITEMS=(base); for ((s = 1; s <= LAST_STEP; s++)); do ITEMS+=("global_step_$s"); done

claim() {  # print the first unclaimed item and record the claim, under one lock
  exec 9<>"$OUT/.claims.lock"
  flock 9
  touch "$OUT/claims.txt"
  local item=""
  touch "$OUT/failures.txt"
  for candidate in "${ITEMS[@]}"; do
    [[ -f "$OUT/$candidate/result.json" ]] && continue
    (( $(grep -cx "$candidate" "$OUT/failures.txt") >= 3 )) && continue
    if ! grep -qx "$candidate" "$OUT/claims.txt"; then item=$candidate; echo "$item" >> "$OUT/claims.txt"; break; fi
  done
  flock -u 9; exec 9>&-
  echo "$item"
}

extract_adapter() {  # $1 = global_step_N; leaves $ADP/global_step_N or returns non-zero
  local step=$1 target=$ADP/$1 staging=$ADP/.staging-$1-lane$LANE
  [[ -f "$target/adapter_model.safetensors" ]] && return 0
  local n=${step#global_step_}
  if [[ -n "$SOURCE" ]]; then  # copy the adapter another lane extracted; wait until it exists
    until [[ -f "$SOURCE/$step/adapter_model.safetensors" && -f "$SOURCE/$step/adapter_config.json" ]]; do sleep 60; done
    rm -rf "$staging"; mkdir -p "$staging"
    cp "$SOURCE/$step/adapter_config.json" "$SOURCE/$step/adapter_model.safetensors" "$staging/" \
      || { log "$step: copy from $SOURCE failed"; return 1; }
    mv "$staging" "$target"
    log "$step: adapter copied from $SOURCE"
    return 0
  fi
  until [[ -f "$CKPT/latest_checkpointed_iteration.txt" ]] && (( $(cat "$CKPT/latest_checkpointed_iteration.txt") >= n )); do
    sleep 60
  done
  rm -rf "$staging"
  CUDA_VISIBLE_DEVICES= nice -n 10 "$VENV/bin/python" -m verl.model_merger merge --backend fsdp \
    --local_dir "$CKPT/$step/actor" --target_dir "$staging" >> "$LOG" 2>&1 || { log "$step: model_merger failed"; return 1; }
  "$VENV/bin/python" - "$staging/lora_adapter" <<'EOF' >> "$LOG" 2>&1 || { log "$step: adapter check failed"; return 1; }
import sys
from safetensors import safe_open
f = safe_open(sys.argv[1] + "/adapter_model.safetensors", "pt")
b = [k for k in f.keys() if "lora_B" in k]
assert b and max(float(f.get_tensor(k).abs().max()) for k in b) > 0, "all lora_B are zero"
print(f"adapter ok: {len(list(f.keys()))} tensors")
EOF
  mv "$staging/lora_adapter" "$target" && rm -rf "$staging"
}

adapter_moves_logprobs() {  # served ckpt must differ from served base on a fixed prompt
  "$VENV/bin/python" - "$PORT" <<'EOF' >> "$LOG" 2>&1
import json, sys, urllib.request
def ask(model):
    body = {"model": model, "max_tokens": 32, "temperature": 0, "logprobs": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "List the apps you can use and how to find their APIs."}]}
    req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return [t["logprob"] for t in json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["logprobs"]["content"]]
b, c = ask("base"), ask("ckpt")
diff = max(abs(x - y) for x, y in zip(b, c))
print(f"adapter-check: max |logprob(base) - logprob(ckpt)| over {min(len(b), len(c))} tokens = {diff:.4f}")
sys.exit(0 if diff > 0 else 1)
EOF
}

release() {  # a failed item: drop its claim so a lane retries it, and count the failure
  exec 9<>"$OUT/.claims.lock"; flock 9
  grep -vx "$1" "$OUT/claims.txt" > "$OUT/claims.txt.tmp"; mv "$OUT/claims.txt.tmp" "$OUT/claims.txt"
  echo "$1" >> "$OUT/failures.txt"
  flock -u 9; exec 9>&-
  sleep 60
}

descendants() { for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done; }
stop_server() {  # the serve script execs vLLM; stop it and every process it started, by PID
  local all="$1 $(descendants "$1" | tr '\n' ' ')"
  kill -TERM $all 2>/dev/null; sleep 15; kill -KILL $all 2>/dev/null; wait "$1" 2>/dev/null
}

while true; do
  item=$(claim)
  [[ -z "$item" ]] && { log "queue empty; done"; break; }
  dest=$OUT/$item
  if [[ -f "$dest/result.json" ]]; then log "$item already evaluated"; continue; fi
  adapter=""
  if [[ "$item" != base ]]; then
    extract_adapter "$item" || { release "$item"; continue; }
    adapter=$ADP/$item
  fi
  mkdir -p "$dest"
  SERVE_LOG=$dest/vllm.log
  exec 8<>"$OUT/.startup.lock"; flock 8   # one engine initializes at a time on this GPU
  "$HERE/serve_eval_vllm.sh" "$GPU" "$PORT" "$SERVE_LOG" "$adapter" "$UTIL" < /dev/null > /dev/null 2>&1 &
  server=$!
  ready=0
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && { ready=1; break; }
    kill -0 "$server" 2>/dev/null || break
    sleep 5
  done
  flock -u 8; exec 8>&-
  if (( ! ready )); then
    log "$item: vLLM never became ready (see $SERVE_LOG)"; stop_server "$server"; release "$item"; continue
  fi
  model=base
  if [[ -n "$adapter" ]]; then
    model=ckpt
    adapter_moves_logprobs || { log "$item: served adapter does not change the base"; stop_server "$server"; release "$item"; continue; }
  fi
  log "$item: evaluating $SPLIT (model=$model, port $PORT)"
  "$VENV/bin/python" -u "$HERE/eval_appworld.py" --base-url "http://127.0.0.1:$PORT/v1" --model "$model" \
    --split "$SPLIT" --output "$dest" --port-base $((22000 + 200 * LANE)) --concurrency 12 \
    --tokenizer "$MODEL_PATH" >> "$dest/eval.log" 2>&1
  log "$item: $(tail -1 "$dest/eval.log")"
  stop_server "$server"
  [[ -f "$dest/result.json" ]] || release "$item"
done
