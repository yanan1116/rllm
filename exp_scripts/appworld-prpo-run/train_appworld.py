"""Experiment-local AppWorld training entrypoint (synchronous rLLM, PRPO or GRPO).

Loads the registered ``appworld`` train/dev splits, checks the environment the flow depends on,
and delegates every training mechanic to rLLM's public AgentTrainer. No trainer, loss, sampler or
checkpoint logic lives here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

import appworld_flow
from appworld_flow import appworld_evaluator, appworld_react
from appworld_react.episode import REACT_PROMPT_PATH, TOOLS, initial_messages
from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer

EXPECTED_ROWS = {"train": 90, "dev": 57}
HERE = Path(__file__).resolve().parent


def _fail(what: str, expected: str, actual: str, fix: str) -> None:
    raise SystemExit(f"[appworld preflight] {what}\n  expected: {expected}\n  actual:   {actual}\n  fix:      {fix}")


def _preflight(config: DictConfig, train_rows: list[dict], val_rows: list[dict]) -> None:
    for split, rows in (("train", train_rows), ("dev", val_rows)):
        if len(rows) != EXPECTED_ROWS[split] or any(not row.get("task_id") for row in rows):
            _fail(f"appworld/{split} rows", f"{EXPECTED_ROWS[split]} rows with task_id", f"{len(rows)} rows",
                  "rerun register_appworld.py with the run's RLLM_HOME")

    check = subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS"], cwd=HERE / "appworld_react",
                           capture_output=True, text=True)
    if check.returncode != 0:
        _fail("appworld_react is not the environment Reef's SAO line runs", "sha256sum -c SHA256SUMS clean",
              (check.stdout + check.stderr).strip()[:500], "restore the files from reef exp_scripts/appworld/appworld_react")

    cli = appworld_flow.APPWORLD_VENV / "bin" / "appworld"
    if not cli.is_file():
        _fail("AppWorld CLI", str(cli), "missing", "set APPWORLD_VENV to a venv with appworld installed")
    for sub in ("data/datasets/train.txt", "data/tasks", "data/base_dbs"):
        if not (appworld_flow.APPWORLD_ROOT / sub).exists():
            _fail("AppWorld data", str(appworld_flow.APPWORLD_ROOT / sub), "missing", "set APPWORLD_ROOT")

    rollout = config.actor_rollout_ref.rollout
    context = int(rollout.max_model_len)
    if context != appworld_flow.CONTEXT_TOKENS:
        _fail("context window", f"rollout.max_model_len == APPWORLD_CONTEXT_TOKENS ({appworld_flow.CONTEXT_TOKENS})",
              str(context), "set both from the launcher's --context")
    if int(config.rllm.data.max_prompt_length) < context:
        _fail("prompt bound", f"rllm.data.max_prompt_length >= {context}", str(config.rllm.data.max_prompt_length),
              "a row opened by a prefix break starts mid-conversation and rLLM left-truncates its prompt silently")
    if not appworld_flow.TOKENIZER_PATH:
        _fail("local prompt count", "APPWORLD_TOKENIZER = the policy model dir", "unset",
              "the launcher sets it to the model path")

    # A row whose prompt is the task prompt has max_response_length tokens after it: every first prompt
    # must leave the whole context room. Rendered from each task's specs.json (what /initialize returns).
    tokenizer = appworld_flow._load_tokenizer(appworld_flow.TOKENIZER_PATH)
    template = REACT_PROMPT_PATH.read_text()
    first, min_first = 0, 10**9
    for row in train_rows + val_rows:
        specs = json.loads((appworld_flow.APPWORLD_ROOT / "data" / "tasks" / row["task_id"] / "specs.json").read_text())
        length = appworld_flow.count_prompt_tokens(tokenizer, initial_messages(specs, template), TOOLS)
        first, min_first = max(first, length), min(min_first, length)
    if min_first + int(config.rllm.data.max_response_length) < context:
        _fail("response bound", f"shortest first prompt + rllm.data.max_response_length >= {context}",
              f"{min_first} + {config.rllm.data.max_response_length}", "raise rllm.data.max_response_length")
    print(f"[appworld preflight] rows 90/57, SHA256SUMS ok, context {context}, first prompts {min_first}-{first} tokens", flush=True)


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig) -> None:
    train_dataset = DatasetRegistry.load_dataset("appworld", "train")
    val_dataset = DatasetRegistry.load_dataset("appworld", "dev")
    if train_dataset is None or val_dataset is None:
        raise SystemExit("appworld train/dev are not registered in the active RLLM_HOME: run register_appworld.py")
    _preflight(config, train_dataset.get_data(), val_dataset.get_data())

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=appworld_react,
        evaluator=appworld_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    sys.exit(main())
