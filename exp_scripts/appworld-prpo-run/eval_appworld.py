"""Evaluate one served model on an AppWorld split with the training protocol, greedy.

The episode is the training one (appworld_react via appworld_flow.play: 50 executed calls, 24576
generated tokens per episode, a 32768-token context clamp, thinking off, the same reward and TGC),
against a plain vLLM endpoint instead of the training gateway. Sampling is greedy: temperature 0,
top_p 1.0, seed 1234. Each task runs once. An episode that fails for an infrastructure reason is
retried (up to 3 attempts); a task that still fails is kept and scored 0 with its error recorded,
never dropped. Finished tasks are skipped on a rerun into the same --output.

  python eval_appworld.py --base-url http://127.0.0.1:18210/v1 --model ckpt --split dev \
      --output DIR [--concurrency 12] [--tokenizer MODEL_DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="served model name: base, or ckpt for the mounted adapter")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test_normal", "test_challenge", "test_normal_scen1", "test_challenge_scen1"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--tokenizer", default="/works/yanan/models/qwen35-4b-851bf6e8-eosfix")
    parser.add_argument("--port-base", type=int, default=22000, help="first AppWorld server port of this lane")
    parser.add_argument("--attempts", type=int, default=3, help="tries per task for infrastructure failures")
    return parser.parse_args()


ARGS = parse_args()
ARGS.output.mkdir(parents=True, exist_ok=True)
# appworld_flow reads its settings at import.
os.environ.setdefault("APPWORLD_ROOT", "/home/yanan/appworld-root")
os.environ.setdefault("APPWORLD_VENV", "/home/yanan/agents/gitlab/tail/appworld/.venv")
os.environ["APPWORLD_RUN_DIR"] = str(ARGS.output)
os.environ["APPWORLD_PORT_BASE"] = str(ARGS.port_base)
os.environ["APPWORLD_MAX_EPISODES"] = str(ARGS.concurrency)
os.environ["APPWORLD_CONTEXT_TOKENS"] = "32768"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI  # noqa: E402

import appworld_flow as flow  # noqa: E402

SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 1234}
lock = threading.Lock()


def task_ids(split: str) -> list[str]:
    return (flow.APPWORLD_ROOT / "data" / "datasets" / f"{split}.txt").read_text().split()


def run_task(task_id: str, tokenizer) -> dict:
    errors = []
    for attempt in range(1, ARGS.attempts + 1):
        chat = flow.GatewayChat(OpenAI(base_url=ARGS.base_url, api_key="EMPTY", max_retries=0), ARGS.model,
                                tokenizer, context_tokens=flow.CONTEXT_TOKENS)
        chat.extra_body = {**chat.extra_body, **SAMPLING}
        started = time.time()
        try:
            episode = flow.play(chat, task_id, f"eval-{ARGS.model}")
        except Exception as error:  # noqa: BLE001 - infrastructure: retry, then keep the task scored 0
            errors.append(f"attempt={attempt}: {type(error).__name__}: {error}"[:500])
            time.sleep(5 * attempt)
            continue
        return {"task_id": task_id, "reward": episode.reward, "tgc": episode.tgc,
                "verifier_status": episode.verifier_status, "stop_reason": episode.stop_reason,
                "turns": len(chat.steps), "dispatches": len(episode.dispatches),
                "completion_tokens": episode.completion_tokens, "llm_errors": chat.llm_errors,
                "infra_errors": errors, "seconds": round(time.time() - started, 1),
                "messages": episode.messages}
    return {"task_id": task_id, "reward": 0.0, "tgc": False, "verifier_status": "infra_failure",
            "stop_reason": "infra_failure", "turns": 0, "dispatches": 0, "completion_tokens": 0,
            "llm_errors": [], "infra_errors": errors, "seconds": 0.0, "messages": []}


def preflight(task_id: str) -> None:
    """One AppWorld server must start and initialize a task, or nothing is evaluated: a broken
    install would otherwise score every task 0 as an infrastructure failure (.29, 2026-10-09)."""
    with flow.leased_port(ARGS.port_base, ARGS.concurrency) as port:
        server = flow.AppWorldServer(port, flow.APPWORLD_ROOT, flow.APPWORLD_VENV, ARGS.output / "preflight-server.log")
        experiment = f"rllm-appworld/eval-preflight/{os.getpid()}"
        try:
            server.start()
            server.post("/initialize", {"task_id": task_id, "experiment_name": experiment, "max_interactions": 1,
                                        "timeout_seconds": 600, "load_ground_truth": False, "random_seed": 100,
                                        "include_direct_functions": False})
        except Exception as error:  # noqa: BLE001
            raise SystemExit(f"[eval] preflight: an AppWorld server on {flow.APPWORLD_ROOT} cannot start or "
                             f"initialize {task_id}: {type(error).__name__}: {str(error)[:300]}. "
                             f"See {ARGS.output / 'preflight-server.log'}") from error
        finally:
            server.stop()
            import shutil
            shutil.rmtree(flow.APPWORLD_ROOT / "experiments" / "outputs" / experiment, ignore_errors=True)


MAX_INFRA_FRACTION = 0.05


def main() -> None:
    ids = task_ids(ARGS.split)
    preflight(ids[0])
    results_path = ARGS.output / "episodes.jsonl"
    done = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            row = json.loads(line)
            if row["verifier_status"] != "infra_failure":  # an infrastructure failure is run again
                done[row["task_id"]] = row
    todo = [task_id for task_id in ids if task_id not in done]
    # Longest first: order by each task's median duration in the other models already evaluated on
    # this split (sibling directories), so the few very long episodes (e.g. dev 383cbac, ~20-46 min
    # against an ~8 min median) start at once instead of trailing the run. Results are unaffected:
    # every task is still run once, independently; only the start order changes.
    seconds: dict[str, list[float]] = {}
    for sibling in ARGS.output.parent.glob("*/episodes.jsonl"):
        if sibling.parent == ARGS.output:
            continue
        for line in sibling.read_text().splitlines():
            row = json.loads(line)
            if row.get("seconds"):
                seconds.setdefault(row["task_id"], []).append(float(row["seconds"]))
    if seconds:
        median = {task_id: sorted(v)[len(v) // 2] for task_id, v in seconds.items()}
        todo.sort(key=lambda task_id: -median.get(task_id, 0.0))
    (ARGS.output / "protocol.json").write_text(json.dumps({
        "split": ARGS.split, "tasks": len(ids), "model": ARGS.model, "base_url": ARGS.base_url,
        "sampling": SAMPLING, "attempts": 1, "max_steps": flow.MAX_STEPS, "token_wall": flow.EPISODE_TOKEN_WALL,
        "context_tokens": flow.CONTEXT_TOKENS, "thinking": "off", "protocol": flow.PROTOCOL_VERSION,
        "concurrency": ARGS.concurrency}, indent=1))
    print(f"[eval] {ARGS.split}: {len(ids)} tasks, {len(done)} already done, model={ARGS.model}", flush=True)
    tokenizer = flow._load_tokenizer(ARGS.tokenizer)

    def one(task_id: str) -> None:
        row = run_task(task_id, tokenizer)
        with lock:
            with results_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            done[task_id] = row
            print(f"[eval] {len(done)}/{len(ids)} {task_id} reward={row['reward']} tgc={row['tgc']} "
                  f"stop={row['stop_reason']} turns={row['turns']}", flush=True)

    with ThreadPoolExecutor(ARGS.concurrency) as pool:
        list(pool.map(one, todo))

    rows = [done[task_id] for task_id in ids]
    rewards = [float(row["reward"] or 0.0) for row in rows]
    summary = {
        "split": ARGS.split, "model": ARGS.model, "tasks": len(rows),
        "mean_reward": round(sum(rewards) / len(rows), 6),
        "tgc_correct": sum(bool(row["tgc"]) for row in rows),
        "grader_failures": sum(row["reward"] is None for row in rows),
        "infra_failures": sum(row["verifier_status"] == "infra_failure" for row in rows),
        "stop_reasons": {reason: sum(row["stop_reason"] == reason for row in rows)
                         for reason in sorted({row["stop_reason"] for row in rows})},
    }
    summary["tgc_rate"] = round(summary["tgc_correct"] / len(rows), 6)
    if summary["infra_failures"] > MAX_INFRA_FRACTION * len(rows):
        (ARGS.output / "result.INVALID.json").write_text(json.dumps(summary, indent=1))
        raise SystemExit(f"[eval] {summary['infra_failures']}/{len(rows)} tasks failed for infrastructure reasons "
                         f"(> {MAX_INFRA_FRACTION:.0%}): no result.json written; see episodes.jsonl infra_errors")
    (ARGS.output / "result.json").write_text(json.dumps(summary, indent=1))
    print(f"[eval] COMPLETE {ARGS.split} mean_reward={summary['mean_reward']:.4f} "
          f"tgc={summary['tgc_correct']}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
