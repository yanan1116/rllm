"""Re-queue the tasks of finished eval_gpu_pool.py models that only have infrastructure failures.

A task whose every recorded episode is an infrastructure failure (e.g. the client timeout on .36's
slow GPUs) was scored 0 without a real attempt. For each MODEL dir under OUT this script, under the
pool's own lock (model_dir/.lock), removes those task ids from claims.txt, moves their rows from
episodes.jsonl to episodes.infra-before-repair.jsonl, and moves result.json / result.INVALID.json
aside (to *.before-repair), so the next pool run on that model claims exactly those tasks again and
writes the result once they are re-run. Run it only on models no pool is evaluating right now.

  python eval_repair_infra.py OUT MODEL [MODEL ...]
"""

from __future__ import annotations

import fcntl
import json
import sys
from pathlib import Path


def repair(model_dir: Path) -> list[str]:
    rows: dict[str, list[str]] = {}
    path = model_dir / "episodes.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows.setdefault(row["task_id"], []).append(row["verifier_status"])
    only_infra = sorted(t for t, statuses in rows.items() if all(s == "infra_failure" for s in statuses))
    with open(model_dir / ".lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        claims_path = model_dir / "claims.txt"
        claims = claims_path.read_text().split() if claims_path.exists() else []
        claims_path.write_text("".join(t + "\n" for t in claims if t not in only_infra))
        if only_infra:
            # The pool counts a task as recorded once it has any row, infra failures included, and
            # writes result.json as soon as every task is recorded; the failed rows move aside so
            # the result waits for the re-runs.
            kept, moved = [], []
            for line in path.read_text().splitlines():
                (moved if json.loads(line)["task_id"] in only_infra else kept).append(line)
            with (model_dir / "episodes.infra-before-repair.jsonl").open("a") as handle:
                handle.writelines(line + "\n" for line in moved)
            path.write_text("".join(line + "\n" for line in kept))
            for name in ("result.json", "result.INVALID.json"):
                if (model_dir / name).exists():
                    (model_dir / name).rename(model_dir / f"{name}.before-repair")
    return only_infra


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for model in sys.argv[2:]:
        print(f"{model}: re-queued {repair(out / model)}")
