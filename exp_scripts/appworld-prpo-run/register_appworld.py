"""Register AppWorld's official splits as the rLLM dataset ``appworld``.

One row per task id from ``$APPWORLD_ROOT/data/datasets/<split>.txt``. The flow reads only
``task_id``: the policy's prompt comes from the AppWorld server's /initialize, exactly as in
Reef's SAO line. ``instruction`` is kept for logs and audits only.

  RLLM_HOME=<run rllm home> python register_appworld.py [--root APPWORLD_ROOT]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rllm.data.dataset import DatasetRegistry

SPLITS = {"train": 90, "dev": 57, "test_normal": 168, "test_challenge": 417}


def split_rows(root: Path, split: str) -> list[dict]:
    path = root / "data" / "datasets" / f"{split}.txt"
    task_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(task_ids) != SPLITS[split] or len(set(task_ids)) != len(task_ids):
        raise SystemExit(f"{path}: expected {SPLITS[split]} distinct task ids, found {len(task_ids)} "
                         f"({len(set(task_ids))} distinct). Wrong APPWORLD_ROOT or an edited split file.")
    rows = []
    for task_id in task_ids:
        specs = root / "data" / "tasks" / task_id / "specs.json"
        if not specs.is_file():
            raise SystemExit(f"{specs} missing: the AppWorld data under {root} is incomplete")
        rows.append({"id": task_id, "task_id": task_id, "split": split, "data_source": "appworld",
                     "instruction": json.loads(specs.read_text())["instruction"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.environ.get("APPWORLD_ROOT", "/home/yanan/appworld-root"))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    args = parser.parse_args()
    if not os.environ.get("RLLM_HOME"):
        raise SystemExit("RLLM_HOME is unset: refusing to write into the default ~/.rllm registry")
    root = Path(args.root)
    for split in args.splits:
        rows = split_rows(root, split)
        DatasetRegistry.register_dataset("appworld", rows, split=split, source=str(root),
                                         description="AppWorld react_code tasks (task ids only)", category="agent")
        print(f"[appworld] registered {split}: {len(rows)} tasks from {root}", flush=True)


if __name__ == "__main__":
    main()
