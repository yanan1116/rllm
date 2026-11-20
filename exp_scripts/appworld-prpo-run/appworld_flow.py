"""AppWorld react_code AgentFlow and evaluator for rLLM synchronous training (PRPO / GRPO).

The environment is ``appworld_react`` -- the same files, byte for byte, as Reef's SAO AppWorld
line (``sha256sum -c appworld_react/SHA256SUMS``). This module only adds what rLLM needs around it:

* ``GatewayChat``: the episode's ``ChatModel`` over rLLM's gateway (``config.base_url``). Thinking is
  off per request (``chat_template_kwargs``); no reasoning parser may be configured on the engine.
  Each successful call appends exactly one ``Step`` whose ``chat_completions`` end with the same
  assistant dict ``run_episode`` appends, so the gateway traces line up 1:1 with the agent steps.
* One ``appworld serve environment`` process per episode, on a port held by an flock for the
  episode's lifetime; its world output is deleted after grading.
* Context handling as Reef's ``ReefChat``: a local chat-template count clamps ``max_completion_tokens``
  to the room left in ``APPWORLD_CONTEXT_TOKENS``; no room (or an engine context rejection) ends the
  episode as ``context_overflow`` and the world is graded as it stands -- the model's doing.
* A grader failure (``reward is None``) is never turned into a reward: the flow raises, so rLLM
  retries the rollout and, if it keeps failing, stops the run (no train90 / dev57 episode measured
  so far was ungradable).

Configuration comes from the launcher's environment (defaults are .16's local AppWorld install):
  APPWORLD_ROOT            AppWorld data root (data/, experiments/)
  APPWORLD_VENV            venv that provides the ``appworld`` CLI (the rllm venv has no appworld)
  APPWORLD_RUN_DIR         where server logs and per-episode records go
  APPWORLD_PORT_BASE       first port of the episode server pool          (default 20000)
  APPWORLD_MAX_EPISODES    concurrent episodes == pool size == threads     (default 32)
  APPWORLD_CONTEXT_TOKENS  serving context the clamp assumes               (default 32768)
  APPWORLD_TOKENIZER       model dir for the local prompt count (unset: no clamp, engine decides)
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import logging
import os
import shutil
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from appworld_react.episode import (
    EPISODE_TOKEN_WALL,
    MAX_STEPS,
    AppWorldServer,
    ChatModel,
    ContextOverflow,
    run_episode,
)
from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, OpenAI

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "appworld-react-code-v1"
FLOW_NAME = "appworld-react"

APPWORLD_ROOT = Path(os.environ.get("APPWORLD_ROOT", "/home/yanan/appworld-root"))
APPWORLD_VENV = Path(os.environ.get("APPWORLD_VENV", "/home/yanan/agents/gitlab/tail/appworld/.venv"))
RUN_DIR = Path(os.environ.get("APPWORLD_RUN_DIR", str(Path(__file__).resolve().parent / "runtime")))
PORT_BASE = int(os.environ.get("APPWORLD_PORT_BASE", "20000"))
MAX_EPISODES = int(os.environ.get("APPWORLD_MAX_EPISODES", "32"))
CONTEXT_TOKENS = int(os.environ.get("APPWORLD_CONTEXT_TOKENS", "32768"))
TOKENIZER_PATH = os.environ.get("APPWORLD_TOKENIZER", "")

CONTEXT_MARGIN = 64  # tolerance between the local chat-template count and the engine's (as ReefChat)
CONTEXT_OVERFLOW_MARKERS = ("context length", "context_length", "maximum context", "longer than the maximum", "too long")
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
# Per-call client timeout. 600 s unless APPWORLD_LLM_TIMEOUT_S says otherwise: .36's fp16 sm75 GPUs,
# three episodes each, need longer for a long completion (a timeout is an infrastructure failure).
LLM_TIMEOUT_SECONDS = float(os.environ.get("APPWORLD_LLM_TIMEOUT_S", "600"))
LLM_ATTEMPTS = 2
PORT_LOCK_DIR = Path(f"/tmp/appworld-port-locks-{os.getuid()}")


class GraderFailure(RuntimeError):
    """The world could not be graded: an infrastructure fault, never a policy outcome."""


# ---------------------------------------------------------------------------------------------
# Ports: one flock per port, held for the whole episode, shared by every process on the host.
# ---------------------------------------------------------------------------------------------

def _port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


@contextmanager
def leased_port(base: int = PORT_BASE, count: int = MAX_EPISODES, wait_s: float = 600.0):
    """Lease a port in [base, base+count): flock first, then confirm nothing else listens on it."""
    PORT_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_s
    while True:
        for port in range(base, base + count):
            handle = open(PORT_LOCK_DIR / f"{port}.lock", "a+")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                handle.close()
                if error.errno in (errno.EAGAIN, errno.EACCES):
                    continue
                raise
            if not _port_is_free(port):
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
                continue
            try:
                yield port
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"no free AppWorld port in [{base}, {base + count}) after {wait_s:.0f}s")
        time.sleep(1.0)


# ---------------------------------------------------------------------------------------------
# The model side of the episode.
# ---------------------------------------------------------------------------------------------

class _ProcessLocal(dict):
    """Pickles as empty: Ray cloudpickles this module's globals by value into the TaskRunner, and a lock
    or thread pool cannot cross (nor should the driver's tokenizer); each process builds its own."""

    def __reduce__(self):
        return (_ProcessLocal, ())


_RUNTIME: dict[str, Any] = _ProcessLocal()


def _runtime() -> dict[str, Any]:
    """Called from the event-loop thread before any episode thread exists, so creation cannot race."""
    if not _RUNTIME:
        _RUNTIME["tokenizer_lock"] = threading.Lock()
        _RUNTIME["executor"] = ThreadPoolExecutor(max_workers=MAX_EPISODES, thread_name_prefix="appworld-episode")
    return _RUNTIME


def _load_tokenizer(path: str):
    runtime = _runtime()
    with runtime["tokenizer_lock"]:
        if "tokenizer" not in runtime:
            from transformers import AutoTokenizer

            runtime["tokenizer"] = AutoTokenizer.from_pretrained(path)
    return runtime["tokenizer"]


def _template_view(messages: list[dict]) -> list[dict]:
    """Messages as the engine renders them: tool-call arguments as mappings (vLLM parses the JSON string)."""
    view = []
    for message in messages:
        calls = message.get("tool_calls")
        if not calls:
            view.append(message)
            continue
        parsed = []
        for call in calls:
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    function["arguments"] = {}
            parsed.append({**call, "function": function})
        view.append({**message, "tool_calls": parsed})
    return view


def count_prompt_tokens(tokenizer, messages: list[dict], tools: list[dict]) -> int:
    encoded = tokenizer.apply_chat_template(_template_view(messages), tools=tools, add_generation_prompt=True,
                                            tokenize=True, return_dict=True, **CHAT_TEMPLATE_KWARGS)
    return len(encoded["input_ids"])  # transformers 5: a BatchEncoding, whose len() counts keys


def _tool_calls_of(message: Any) -> list[dict]:
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        calls.append({"id": call.id, "type": call.type or "function",
                      "function": {"name": call.function.name, "arguments": call.function.arguments}})
    return calls


def assistant_dict(reply: dict) -> dict:
    """Exactly the message run_episode appends for a reply (and so what the next prompt carries)."""
    calls = reply.get("tool_calls") or []
    return {"role": "assistant", "content": reply.get("content") or "", **({"tool_calls": calls} if calls else {})}


class GatewayChat(ChatModel):
    """One AppWorld model call through rLLM's gateway; sampling is the gateway session's."""

    def __init__(self, client: OpenAI, model: str, tokenizer=None, context_tokens: int = CONTEXT_TOKENS) -> None:
        self.client, self.model, self.tokenizer, self.context_tokens = client, model, tokenizer, context_tokens
        self.steps: list[Step] = []
        self.llm_errors: list[str] = []
        self.extra_body: dict = {"chat_template_kwargs": CHAT_TEMPLATE_KWARGS}
        self.responses: list[Any] = []  # raw responses, kept only when record_responses is set (smoke)
        self.record_responses = False

    def complete(self, messages: list[dict], tools: list[dict], max_tokens: int) -> tuple[dict, dict]:
        budget, prompt_tokens = max_tokens, None
        if self.tokenizer is not None:
            prompt_tokens = count_prompt_tokens(self.tokenizer, messages, tools)
            room = self.context_tokens - prompt_tokens - CONTEXT_MARGIN
            if room <= 0:
                raise ContextOverflow(f"prompt of {prompt_tokens} tokens leaves no room in a {self.context_tokens}-token context")
            budget = min(max_tokens, room)
        response = None
        for attempt in range(1, LLM_ATTEMPTS + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools, max_completion_tokens=budget,
                    extra_body=self.extra_body, timeout=LLM_TIMEOUT_SECONDS)
                break
            except BadRequestError as error:
                if any(marker in str(error).lower() for marker in CONTEXT_OVERFLOW_MARKERS):
                    raise ContextOverflow(str(error)) from error
                raise
            except (APIConnectionError, APITimeoutError, InternalServerError) as error:
                self.llm_errors.append(f"attempt={attempt}:{type(error).__name__}:{error}"[:500])
                if attempt == LLM_ATTEMPTS:
                    raise
                time.sleep(2.0 * attempt)
        if self.record_responses:
            self.responses.append(response)
        choice = response.choices[0]
        reply = {"content": choice.message.content or "", "tool_calls": _tool_calls_of(choice.message)}
        self.steps.append(Step(chat_completions=[*messages, assistant_dict(reply)], model_response=reply["content"],
                               action=reply["content"], thought=reply["content"]))
        usage = response.usage
        return reply, {
            "finish_reason": choice.finish_reason,
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "prompt_tokens_local": prompt_tokens,
            "max_tokens": budget,
        }


# ---------------------------------------------------------------------------------------------
# The flow.
# ---------------------------------------------------------------------------------------------

def _record(entry: dict) -> None:
    path = RUN_DIR / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


def play(chat: ChatModel, task_id: str, label: str, max_steps: int = MAX_STEPS,
         token_wall: int = EPISODE_TOKEN_WALL):
    """One episode on its own AppWorld server; the server and the world it wrote are removed afterwards."""
    experiment = f"rllm-appworld/{label}/{uuid.uuid4().hex[:12]}"
    with leased_port() as port:
        server = AppWorldServer(port, APPWORLD_ROOT, APPWORLD_VENV, RUN_DIR / "appworld_servers" / f"{port}.log")
        try:
            server.start()
            return run_episode(chat, server, task_id, experiment, max_steps=max_steps, token_wall=token_wall)
        finally:
            server.stop()
            shutil.rmtree(APPWORLD_ROOT / "experiments" / "outputs" / experiment, ignore_errors=True)


def _run_sync(task_id: str, base_url: str, model: str, label: str) -> tuple[Any, GatewayChat]:
    if "APPWORLD_RUN_DIR" not in os.environ:
        raise RuntimeError("APPWORLD_RUN_DIR is unset in the rollout process: the launcher's environment did not "
                           "reach it, so every APPWORLD_* setting here is a default. Launch via train_appworld.sh.")
    tokenizer = _load_tokenizer(TOKENIZER_PATH) if TOKENIZER_PATH else None
    chat = GatewayChat(OpenAI(base_url=base_url, api_key="EMPTY", max_retries=0), model, tokenizer)
    return play(chat, task_id, label), chat


@rllm.rollout(name=FLOW_NAME)
async def appworld_react(task: Task, config: AgentConfig) -> Episode:
    meta = task.metadata or {}
    task_id = str(meta.get("task_id") or "")
    if not task_id:
        raise ValueError(f"{PROTOCOL_VERSION}: task {task.id!r} has no task_id in its metadata")
    label = "val" if config.is_validation else "train"
    loop = asyncio.get_running_loop()
    episode, chat = await loop.run_in_executor(_runtime()["executor"], _run_sync, task_id, config.base_url, config.model, label)

    summary = {
        "task_id": task_id,
        "split": meta.get("split"),
        "validation": config.is_validation,
        "reward": episode.reward,
        "tgc": episode.tgc,
        "verifier_status": episode.verifier_status,
        "stop_reason": episode.stop_reason,
        "task_completed": episode.task_completed,
        "turns": len(chat.steps),
        "dispatches": len(episode.dispatches),
        "completion_tokens": episode.completion_tokens,
        "last_prompt_tokens": next((t.get("prompt_tokens") for t in reversed(episode.turns) if t.get("prompt_tokens")), None),
        "llm_errors": chat.llm_errors,
        "elapsed_s": episode.elapsed_s,
        "recorded_at": time.time(),
    }
    _record(summary)
    if episode.reward is None:
        raise GraderFailure(f"{task_id}: {episode.verifier_status} {episode.verifier_components}")
    if not chat.steps:
        raise RuntimeError(f"{task_id}: episode ended before its first model call ({episode.stop_reason})")

    return Episode(
        trajectories=[Trajectory(name=FLOW_NAME, steps=chat.steps)],
        artifacts={**summary, "protocol_version": PROTOCOL_VERSION, "max_steps": MAX_STEPS,
                   "token_wall": EPISODE_TOKEN_WALL, "context_tokens": CONTEXT_TOKENS},
    )


@rllm.evaluator
def appworld_evaluator(task: dict, episode: Episode) -> EvalOutput:
    """Reward = the corrected partial score computed in the flow; is_correct = the official TGC."""
    artifacts = episode.artifacts or {}
    reward = artifacts.get("reward")
    if reward is None:
        raise GraderFailure(f"episode for {artifacts.get('task_id')!r} carries no reward")
    tgc = bool(artifacts.get("tgc"))
    return EvalOutput(
        reward=float(reward),
        is_correct=tgc,
        signals=[Signal(name="tgc", value=float(tgc)), Signal(name="partial_score", value=float(reward))],
        metadata={key: artifacts.get(key) for key in ("stop_reason", "task_completed", "turns", "dispatches",
                                                      "completion_tokens", "last_prompt_tokens", "verifier_status")},
    )
