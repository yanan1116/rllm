#!/usr/bin/env bash
# Cap the address space of every AppWorld server process on this host (one per episode).
#
# A policy's code can make an AppWorld server grow without bound: on .29 (2026-10-09 05:47) one
# server reached 93.7 GB RSS and Ray's OOM killer took the GRPO trainer down with it. Under an
# RLIMIT_AS cap the runaway allocation fails inside the server: AppWorld returns the MemoryError
# to the policy as the output of that execute_code call and the episode goes on, so the protocol
# is unchanged for every episode that stays under the cap (normal servers: ~0.1 GB VSZ).
#
# Applies the cap to servers already running and to each new one within POLL_S seconds. Touches
# nothing else. Stop it by its PID (written to PID_FILE).
#
#   appworld_mem_cap.sh [CAP_GB=16] [POLL_S=2] [PID_FILE]
set -u
CAP_GB=${1:-16}
POLL_S=${2:-2}
PID_FILE=${3:-}
[[ -n "$PID_FILE" ]] && echo $$ > "$PID_FILE"
CAP_BYTES=$((CAP_GB * 1024 * 1024 * 1024))
LOG=$(dirname "$0")/runtime/appworld_mem_cap-$(hostname -s).log
mkdir -p "$(dirname "$LOG")"
echo "$(TZ=America/New_York date '+%F %T %Z') capping AppWorld servers at ${CAP_GB} GB address space (pid $$)" >> "$LOG"
declare -A capped=()
while true; do
  for pid in $(pgrep -u "$(id -u)" -f "appworld serve environment"); do
    [[ -n "${capped[$pid]:-}" ]] && continue
    if prlimit --pid "$pid" --as="$CAP_BYTES:$CAP_BYTES" 2>/dev/null; then
      capped[$pid]=1
    fi
  done
  # forget exited pids so the map does not grow without bound
  if (( ${#capped[@]} > 2000 )); then
    for pid in "${!capped[@]}"; do kill -0 "$pid" 2>/dev/null || unset "capped[$pid]"; done
  fi
  sleep "$POLL_S"
done
