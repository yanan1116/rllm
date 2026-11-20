"""Summarize AppWorld evaluations: average score, TGC and SGC per model, from each model's episodes.jsonl.

Average score = mean corrected partial reward over the split's tasks. TGC (task goal completion)
= tasks whose AppWorld tests all pass. SGC (scenario goal completion, AppWorld's definition) =
scenarios (task ids sharing the prefix before the last "_") whose every task passes TGC. Only
models with every task recorded are complete; partial ones are reported with their progress.
Prints one JSON object per model (stdlib only, so it runs on any host).

  python3 summarize_eval.py SPLIT_DIR [SPLIT_DIR ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

SPLIT_SIZES = {"dev": 57, "test_normal": 168, "test_challenge": 417,
               "test_normal_scen1": 56, "test_challenge_scen1": 139}  # splits/: one task per scenario


def summarize(split_dir: Path) -> list[dict]:
    split = split_dir.name
    total = SPLIT_SIZES[split]
    out = []
    for model_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and (p / "episodes.jsonl").exists()):
        rows = {}
        for line in (model_dir / "episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["task_id"] not in rows or rows[row["task_id"]]["verifier_status"] == "infra_failure":
                rows[row["task_id"]] = row
        scenarios = defaultdict(list)
        for task_id, row in rows.items():
            scenarios[task_id.rsplit("_", 1)[0]].append(bool(row["tgc"]))
        complete_scenarios = {s: v for s, v in scenarios.items() if len(v) == 3}
        out.append({
            "split": split, "model": model_dir.name, "tasks_done": len(rows), "tasks_total": total,
            "complete": len(rows) >= total and (model_dir / "result.json").exists(),
            "avg_score": round(sum(float(r["reward"] or 0.0) for r in rows.values()) / max(1, len(rows)), 4),
            "tgc": sum(bool(r["tgc"]) for r in rows.values()),
            # SGC needs a scenario's 3 tasks; the *_scen1 splits hold one task per scenario.
            "sgc": None if split.endswith("_scen1") else sum(all(v) for v in complete_scenarios.values()),
            "scenarios": len(complete_scenarios),
            "infra_failures": sum(r["verifier_status"] == "infra_failure" for r in rows.values()),
        })
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        for item in summarize(Path(arg)):
            print(json.dumps(item))
