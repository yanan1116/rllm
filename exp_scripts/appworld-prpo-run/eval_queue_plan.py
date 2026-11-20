"""Re-order a running eval_lane.sh queue without touching the lanes: whole epochs first, then half epochs.

eval_lane.sh claims, in ascending order, the first item that has no result.json and is not listed
in claims.txt (under the flock on .claims.lock). Running lanes cannot be edited, so this steers them
through claims.txt instead, under the same lock:

  phase 1  every step that is neither done, in progress, nor in PHASE1 is reserved (written to
           claims.txt and to reserved.txt), so the lanes next take PHASE1 in ascending order;
  phase 2  once every PHASE1 item is done or claimed by a lane, the PHASE2 reservations are
           released, so the lanes take PHASE2 in ascending order. Other steps stay reserved.

Items already done or being evaluated are left as they are.

  python eval_queue_plan.py --out DIR --phase1 6,12,...,60 --phase2 3,9,...,57 [--last-step 60]
"""

from __future__ import annotations

import argparse
import fcntl
import time
from pathlib import Path


def parse_steps(text: str) -> list[str]:
    return [f"global_step_{int(s)}" for s in text.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the queue's directory (eval_lane.sh OUT)")
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--phase2", required=True)
    parser.add_argument("--last-step", type=int, default=60)
    parser.add_argument("--poll-s", type=float, default=120)
    args = parser.parse_args()
    phase1, phase2 = parse_steps(args.phase1), parse_steps(args.phase2)
    every = [f"global_step_{s}" for s in range(1, args.last_step + 1)]
    claims, reserved_path = args.out / "claims.txt", args.out / "reserved.txt"

    def log(message: str) -> None:
        line = f"{time.strftime('%F %T %Z')} [queue-plan] {message}"
        print(line, flush=True)
        with (args.out / "queue_plan.log").open("a") as handle:
            handle.write(line + "\n")

    def edit(change) -> None:
        with open(args.out / ".claims.lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            lines = claims.read_text().split() if claims.exists() else []
            reserved = reserved_path.read_text().split() if reserved_path.exists() else []
            lines, reserved = change(lines, reserved)
            claims.write_text("".join(f"{item}\n" for item in lines))
            reserved_path.write_text("".join(f"{item}\n" for item in reserved))
            fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(lines, reserved):
        taken = set(lines)
        new = [item for item in every
               if item not in taken and item not in phase1 and not (args.out / item / "result.json").exists()]
        log(f"phase 1: reserving {len(new)} steps; lanes now take {[i for i in phase1 if i not in taken]}")
        return lines + new, reserved + new

    edit(reserve)

    while True:
        lines = set(claims.read_text().split()) if claims.exists() else set()
        reserved = set(reserved_path.read_text().split()) if reserved_path.exists() else set()
        pending = [item for item in phase1
                   if not (args.out / item / "result.json").exists() and (item not in lines or item in reserved)]
        if not pending:
            break
        time.sleep(args.poll_s)

    def release(lines, reserved):
        freed = [item for item in phase2 if item in reserved]
        log(f"phase 2: every phase-1 step is done or claimed; releasing {freed}")
        return [item for item in lines if item not in freed], [item for item in reserved if item not in freed]

    edit(release)
    log("done")


if __name__ == "__main__":
    main()
