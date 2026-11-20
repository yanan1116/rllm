"""Evaluate a series of models on one AppWorld split with a pool of GPUs, one episode per GPU at a time.

One process per GPU. Models run in order (e.g. base, then one checkpoint per epoch); for each model
every GPU serves it with its own vLLM (LoRA mounted unmerged, model "ckpt"; the base is "base") and
claims tasks one at a time from a shared order: the split's task ids shuffled once with a fixed
seed, so tasks are picked at random, each exactly once, across all GPUs. A GPU that finds no
unclaimed task moves on to the next model; the model's result.json is written once all its tasks
are recorded.

The episode, sampling and records are eval_appworld.py's: the training protocol through
appworld_flow.play (50 executed calls, 24576 generated tokens, 32768-token context clamp, thinking
off), greedy (temperature 0, top_p 1.0, seed 1234), up to 3 attempts for infrastructure failures,
then the task is kept and scored 0 with its errors; result.json is withheld when more than 5% of
the tasks failed for infrastructure reasons. An AppWorld server must start and initialize a task
before any episode runs.

  python eval_gpu_pool.py --gpu 3 --port 18603 --split test_challenge --output ROOT \
      --models base,global_step_6,...,global_step_60 --adapter-source DIR --model-path DIR \
      [--dtype float16] [--gpu-mem-util 0.90] [--seed 20261009]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True, help="this GPU's vLLM port")
    parser.add_argument("--split", default="test_challenge", choices=["train", "dev", "test_normal", "test_challenge", "test_normal_scen1", "test_challenge_scen1"])
    parser.add_argument("--output", type=Path, required=True, help="one sub-directory per model")
    parser.add_argument("--models", required=True, help="comma list in evaluation order: base and/or global_step_N")
    parser.add_argument("--adapter-source", type=Path, required=True, help="DIR/global_step_N/adapter_*.{json,safetensors}")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", default="auto", help="float16 on sm75 (no bf16)")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--attention-backend", default="",
                        help="vLLM --attention-backend, e.g. TRITON_ATTN on sm75 (vLLM's default there is FlashInfer, "
                             "which JIT-compiles with nvcc); empty: vLLM's choice")
    parser.add_argument("--seed", type=int, default=20261009, help="task-order shuffle seed, shared by all GPUs")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--allow-noop-adapter", action="store_true",
                        help="evaluate an adapter that does not change the base (e.g. a critic-only warmup step) "
                             "instead of refusing it; off: refuse, as before")
    parser.add_argument("--episodes-per-gpu", type=int, default=1,
                        help="episodes this GPU runs at once against its one vLLM (threads claiming from the shared order)")
    return parser.parse_args()


ARGS = parse_args()
ARGS.output.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("APPWORLD_ROOT", "/home/yanan/appworld-root")
os.environ.setdefault("APPWORLD_VENV", "/home/yanan/agents/gitlab/tail/appworld/.venv")
os.environ["APPWORLD_RUN_DIR"] = str(ARGS.output / f"gpu{ARGS.gpu}")
os.environ["APPWORLD_PORT_BASE"] = str(23000 + 10 * int(ARGS.gpu))
os.environ["APPWORLD_MAX_EPISODES"] = str(ARGS.episodes_per_gpu + 4)  # candidate AppWorld ports per GPU
os.environ["APPWORLD_CONTEXT_TOKENS"] = "32768"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI  # noqa: E402

import appworld_flow as flow  # noqa: E402

SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 1234}  # = eval_appworld.SAMPLING
MAX_INFRA_FRACTION = 0.05                                     # = eval_appworld.MAX_INFRA_FRACTION
LOG = ARGS.output / f"gpu{ARGS.gpu}.log"


def log(message: str) -> None:
    line = f"{time.strftime('%F %T %Z')} [gpu{ARGS.gpu}] {message}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


@contextmanager
def locked(path: Path):
    with open(path, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def task_order(model_dir: Path) -> list[str]:
    """The split shuffled once per model directory with the shared seed (same order for every model)."""
    path = model_dir / "order.txt"
    with locked(model_dir / ".lock"):
        if not path.exists():
            ids = (flow.APPWORLD_ROOT / "data" / "datasets" / f"{ARGS.split}.txt").read_text().split()
            random.Random(ARGS.seed).shuffle(ids)
            path.write_text("\n".join(ids) + "\n")
    return path.read_text().split()


def recorded(model_dir: Path) -> dict[str, dict]:
    rows = {}
    path = model_dir / "episodes.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["verifier_status"] != "infra_failure":
                rows[row["task_id"]] = row
            else:
                rows.setdefault(row["task_id"], row)
    return rows


def claim(model_dir: Path, order: list[str]) -> str | None:
    """Next task in the shared random order that no GPU has claimed yet."""
    with locked(model_dir / ".lock"):
        claims_path = model_dir / "claims.txt"
        claimed = set(claims_path.read_text().split()) if claims_path.exists() else set()
        for task_id in order:
            if task_id not in claimed:
                with claims_path.open("a") as handle:
                    handle.write(task_id + "\n")
                return task_id
    return None


def unclaim(model_dir: Path, task_id: str) -> None:
    """Give back a claimed task that was never run (the model could not be served or checked)."""
    with locked(model_dir / ".lock"):
        claims_path = model_dir / "claims.txt"
        claims = claims_path.read_text().split() if claims_path.exists() else []
        if task_id in claims:
            claims.remove(task_id)
            claims_path.write_text("".join(t + "\n" for t in claims))


def run_task(task_id: str, tokenizer) -> dict:
    """eval_appworld.run_task, verbatim in behaviour."""
    errors = []
    for attempt in range(1, ARGS.attempts + 1):
        chat = flow.GatewayChat(OpenAI(base_url=f"http://127.0.0.1:{ARGS.port}/v1", api_key="EMPTY", max_retries=0),
                                MODEL_NAME, tokenizer, context_tokens=flow.CONTEXT_TOKENS)
        chat.extra_body = {**chat.extra_body, **SAMPLING}
        started = time.time()
        try:
            episode = flow.play(chat, task_id, f"eval-{MODEL_NAME}-gpu{ARGS.gpu}")
        except Exception as error:  # noqa: BLE001
            errors.append(f"attempt={attempt}: {type(error).__name__}: {error}"[:500])
            time.sleep(5 * attempt)
            continue
        return {"task_id": task_id, "reward": episode.reward, "tgc": episode.tgc,
                "verifier_status": episode.verifier_status, "stop_reason": episode.stop_reason,
                "turns": len(chat.steps), "dispatches": len(episode.dispatches),
                "completion_tokens": episode.completion_tokens, "llm_errors": chat.llm_errors,
                "infra_errors": errors, "seconds": round(time.time() - started, 1), "gpu": ARGS.gpu,
                "messages": episode.messages}
    return {"task_id": task_id, "reward": 0.0, "tgc": False, "verifier_status": "infra_failure",
            "stop_reason": "infra_failure", "turns": 0, "dispatches": 0, "completion_tokens": 0,
            "llm_errors": [], "infra_errors": errors, "seconds": 0.0, "gpu": ARGS.gpu, "messages": []}


def summarize(model_dir: Path, model: str, order: list[str]) -> None:
    """Write result.json once every task has a row; first GPU to see the model complete writes it."""
    with locked(model_dir / ".lock"):
        if (model_dir / "result.json").exists():
            return
        rows = recorded(model_dir)
        if len(rows) < len(order):
            return
        rows = [rows[task_id] for task_id in sorted(order)]
        rewards = [float(row["reward"] or 0.0) for row in rows]
        summary = {
            "split": ARGS.split, "model": model, "tasks": len(rows), "mean_reward": round(sum(rewards) / len(rows), 6),
            "tgc_correct": sum(bool(row["tgc"]) for row in rows),
            "grader_failures": sum(row["reward"] is None for row in rows),
            "infra_failures": sum(row["verifier_status"] == "infra_failure" for row in rows),
            "stop_reasons": {r: sum(row["stop_reason"] == r for row in rows) for r in sorted({row["stop_reason"] for row in rows})},
            "dtype": ARGS.dtype, "attention_backend": ARGS.attention_backend or "vllm default",
            "sampling": SAMPLING, "protocol": flow.PROTOCOL_VERSION,
        }
        summary["tgc_rate"] = round(summary["tgc_correct"] / len(rows), 6)
        name = "result.json"
        if summary["infra_failures"] > MAX_INFRA_FRACTION * len(rows):
            name = "result.INVALID.json"
        (model_dir / name).write_text(json.dumps(summary, indent=1))
        log(f"{model}: COMPLETE {ARGS.split} mean_reward={summary['mean_reward']:.4f} "
            f"tgc={summary['tgc_correct']}/{len(rows)} infra_failures={summary['infra_failures']}"
            + (" (INVALID: too many infrastructure failures)" if name != "result.json" else ""))


def start_vllm(adapter: Path | None, model_dir: Path) -> subprocess.Popen:
    venv_bin = Path(sys.executable).parent
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(ARGS.gpu), VLLM_USE_FLASHINFER_SAMPLER="0", HF_HUB_OFFLINE="1")
    command = [str(venv_bin / "vllm"), "serve", ARGS.model_path, "--served-model-name", "base",
               "--host", "127.0.0.1", "--port", str(ARGS.port), "--dtype", ARGS.dtype,
               "--max-model-len", "32768", "--gpu-memory-utilization", str(ARGS.gpu_mem_util),
               # one running sequence per episode (at least 4, the original value)
               "--max-num-seqs", str(max(4, ARGS.episodes_per_gpu)),
               "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
               "--limit-mm-per-prompt", '{"image": 0, "video": 0}']
    if ARGS.attention_backend:
        command += ["--attention-backend", ARGS.attention_backend]
    if adapter is not None:
        command += ["--enable-lora", "--max-lora-rank", "32", "--max-loras", "1", "--lora-modules", f"ckpt={adapter}"]
    log_file = (model_dir / f"vllm-gpu{ARGS.gpu}.log").open("ab")
    server = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              start_new_session=True)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{ARGS.port}/v1/models", timeout=5)
            return server
        except Exception:  # noqa: BLE001
            if server.poll() is not None:
                break
            time.sleep(5)
    stop_vllm(server)
    raise RuntimeError(f"vLLM on GPU {ARGS.gpu} never became ready (see {log_file.name})")


def stop_vllm(server: subprocess.Popen) -> None:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()


def adapter_moves_logprobs() -> float:
    def ask(model: str) -> list[float]:
        body = {"model": model, "max_tokens": 32, "temperature": 0, "logprobs": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "List the apps you can use and how to find their APIs."}]}
        request = urllib.request.Request(f"http://127.0.0.1:{ARGS.port}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        content = json.load(urllib.request.urlopen(request, timeout=300))["choices"][0]["logprobs"]["content"]
        return [token["logprob"] for token in content]
    base, ckpt = ask("base"), ask("ckpt")
    return max(abs(a - b) for a, b in zip(base, ckpt))


def local_adapter(model: str) -> Path:
    """Copy the extracted adapter next to the results (read once from the source, served locally)."""
    target = ARGS.output / "adapters" / model
    with locked(ARGS.output / ".adapters.lock"):
        if not (target / "adapter_model.safetensors").exists():
            source = ARGS.adapter_source / model
            while not ((source / "adapter_model.safetensors").exists() and (source / "adapter_config.json").exists()):
                log(f"{model}: waiting for the adapter at {source}")
                time.sleep(120)
            staging = target.with_name(f".staging-{model}")
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                shutil.copy2(source / name, staging / name)
            staging.rename(target)
            log(f"{model}: adapter copied from {source}")
    return target


def preflight(order: list[str]) -> None:
    with flow.leased_port(23000 + 10 * int(ARGS.gpu), 5) as port:
        server = flow.AppWorldServer(port, flow.APPWORLD_ROOT, flow.APPWORLD_VENV, ARGS.output / f"preflight-gpu{ARGS.gpu}.log")
        experiment = f"rllm-appworld/eval-preflight/gpu{ARGS.gpu}-{os.getpid()}"
        try:
            server.start()
            server.post("/initialize", {"task_id": order[0], "experiment_name": experiment, "max_interactions": 1,
                                        "timeout_seconds": 600, "load_ground_truth": False, "random_seed": 100,
                                        "include_direct_functions": False})
        finally:
            server.stop()
            shutil.rmtree(flow.APPWORLD_ROOT / "experiments" / "outputs" / experiment, ignore_errors=True)


MODEL_NAME = "base"


def main() -> None:
    global MODEL_NAME
    tokenizer = flow._load_tokenizer(ARGS.model_path)
    models = [m.strip() for m in ARGS.models.split(",") if m.strip()]
    first = True
    for model in models:
        model_dir = ARGS.output / model
        model_dir.mkdir(parents=True, exist_ok=True)
        if (model_dir / "result.json").exists():
            continue
        order = task_order(model_dir)
        if first:
            preflight(order)
            log(f"preflight ok: AppWorld at {flow.APPWORLD_ROOT}")
            first = False
        task = claim(model_dir, order)
        if task is None:
            summarize(model_dir, model, order)
            continue
        adapter = None if model == "base" else local_adapter(model)
        MODEL_NAME = "base" if adapter is None else "ckpt"
        for attempt in range(1, 4):  # a just-freed GPU or port can make one start fail
            try:
                server = start_vllm(adapter, model_dir)
                break
            except RuntimeError as error:
                if attempt == 3:
                    unclaim(model_dir, task)
                    raise
                log(f"{model}: {error}; retrying in 30 s (attempt {attempt}/3)")
                time.sleep(30)
        try:
            if adapter is not None:
                diff = adapter_moves_logprobs()
                log(f"{model}: adapter-check max |logprob(base) - logprob(ckpt)| = {diff:.4f}")
                if diff <= 0 and ARGS.allow_noop_adapter:
                    log(f"{model}: the served adapter does not change the base; evaluating it (--allow-noop-adapter)")
                elif diff <= 0:
                    unclaim(model_dir, task)  # never run: a refused model must not use up its tasks
                    raise SystemExit(f"{model}: the served adapter does not change the base")
            log(f"{model}: evaluating {ARGS.split} on GPU {ARGS.gpu} (model={MODEL_NAME}, dtype={ARGS.dtype}, "
                f"episodes_per_gpu={ARGS.episodes_per_gpu})")

            def episodes(first: str | None) -> None:
                current = first
                while current is not None:
                    row = run_task(current, tokenizer)
                    with locked(model_dir / ".lock"):
                        with (model_dir / "episodes.jsonl").open("a") as handle:
                            handle.write(json.dumps(row) + "\n")
                    log(f"{model}: {current} reward={row['reward']} tgc={row['tgc']} stop={row['stop_reason']} turns={row['turns']}")
                    current = claim(model_dir, order)

            firsts = [task] + [claim(model_dir, order) for _ in range(ARGS.episodes_per_gpu - 1)]
            with ThreadPoolExecutor(ARGS.episodes_per_gpu) as pool:
                list(pool.map(episodes, [t for t in firsts if t is not None]))
        finally:
            stop_vllm(server)
        summarize(model_dir, model, order)
    log("all models done on this GPU")


if __name__ == "__main__":
    main()
