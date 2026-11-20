#!/usr/bin/env bash
# Build a Qwen3.5-4B model dir that stops at the chat turn end: symlinks to the HF snapshot plus a
# generation_config.json naming <|im_end|> (248046). The snapshot ships no generation_config.json and
# its config.json names only <|endoftext|> (248044), so an engine serving it never ends a turn.
# No sampling defaults on purpose: every request carries T=1.0 / top_p=1.0.
#
#   make_eosfix_model_dir.sh TARGET_DIR [SNAPSHOT_DIR]     (idempotent; TARGET on the host's local disk)
set -euo pipefail
TARGET="${1:?usage: $0 TARGET_DIR [SNAPSHOT_DIR]}"
SNAPSHOT="${2:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
[[ -f "$SNAPSHOT/model.safetensors.index.json" ]] || { echo "snapshot $SNAPSHOT incomplete" >&2; exit 1; }
[[ ! -e "$SNAPSHOT/generation_config.json" ]] || { echo "$SNAPSHOT already has a generation_config.json: check it instead of overriding it" >&2; exit 1; }
mkdir -p "$TARGET"
for path in "$SNAPSHOT"/*; do
  name=$(basename "$path")
  case "$name" in README.md|LICENSE|generation_config.json) continue ;; esac
  ln -sfn "$(readlink -f "$path")" "$TARGET/$name"
done
cat > "$TARGET/generation_config.json" <<'JSON'
{"eos_token_id": [248046, 248044], "_note": "Qwen3.5-4B @851bf6e8 ships no generation_config.json; config.json eos 248044 (<|endoftext|>) misses the chat turn end <|im_end|> 248046. No sampling defaults on purpose: requests carry T=1.0 only (top_p 1, top_k off), as the earlier AppWorld line."}
JSON
echo "[eosfix] $TARGET -> $SNAPSHOT"
