"""CPU tests for the rLLM AppWorld flow: fake model client, fake AppWorld server, no GPU, no network.

  cd exp_scripts/appworld-prpo-run && ../../.venv/bin/python -m unittest -v test_appworld_flow
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("APPWORLD_RUN_DIR", "/tmp/appworld-flow-test-unused")

import appworld_flow as flow  # noqa: E402
from appworld_react.episode import TOOLS, ContextOverflow, run_episode  # noqa: E402
from rllm.types import AgentConfig, Task  # noqa: E402

SPECS = {"instruction": "Say hi.", "supervisor": {"first_name": "A", "last_name": "B", "email": "a@b.c",
                                                  "phone_number": "1"}, "datetime": "2023-05-18T12:00:00"}


def _call(code: str, call_id: str = "c1"):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name="execute_code", arguments=json.dumps({"code": code})))


def _response(content: str, calls=None, completion_tokens: int = 7, finish: str = "tool_calls"):
    message = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                           usage=SimpleNamespace(completion_tokens=completion_tokens, prompt_tokens=100))


class FakeCompletions:
    def __init__(self, script):
        self.script, self.requests = list(script), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(script):
    completions = FakeCompletions(script)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


class FakeServer:
    """The four routes run_episode uses; the tracker says one of two scored tests passed."""

    def __init__(self, root: Path, success: bool = False):
        self.root, self.success, self.executed = root, success, []

    def post(self, route, payload, timeout=300.0):
        if route == "/initialize":
            return SPECS
        if route == "/execute":
            self.executed.append(payload["code"])
            return "ok"
        if route == "/task_completed":
            return True
        if route == "/evaluate":
            return {"passes": [{"requirement": "r1"}] + ([{"requirement": "r2"}] if self.success else []),
                    "failures": [] if self.success else [{"requirement": "r2"}], "num_tests": 2,
                    "success": self.success}
        raise AssertionError(route)


class CountingTokenizer:
    """Stands in for the chat template: one token per 4 characters of the JSON messages."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tools, add_generation_prompt, tokenize, return_dict, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": [0] * (len(json.dumps(messages)) // 4)}


class GatewayChatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="appworld-flow-test-"))

    def test_one_step_per_call_and_steps_match_episode_messages(self):
        client, completions = _client([
            _response("look", [_call("print(1)")]),
            _response("done", [_call("apis.supervisor.complete_task()", "c2")]),
            _response("finished", None, finish="stop"),
        ])
        chat = flow.GatewayChat(client, "policy")
        episode = run_episode(chat, FakeServer(self.tmp), "t_1", "exp")
        self.assertEqual(len(chat.steps), 3)
        self.assertEqual(len(episode.turns), 3)
        # Each Step's conversation is exactly the episode's messages up to and including that reply.
        assistant_positions = [i for i, m in enumerate(episode.messages) if m["role"] == "assistant"]
        for step, position in zip(chat.steps, assistant_positions, strict=True):
            self.assertEqual(step.chat_completions, episode.messages[: position + 1])
        self.assertEqual((episode.stop_reason, episode.reward, episode.tgc), ("no_tool_call", 0.5, False))
        for request in completions.requests:
            self.assertEqual(request["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual((request["tools"], request["model"]), (TOOLS, "policy"))
        self.assertEqual(completions.requests[0]["max_completion_tokens"], flow.EPISODE_TOKEN_WALL)
        self.assertEqual(completions.requests[1]["max_completion_tokens"], flow.EPISODE_TOKEN_WALL - 7)

    def test_clamp_to_context_room_and_overflow_without_a_call(self):
        tokenizer = CountingTokenizer()
        client, completions = _client([_response("hi", [_call("print(1)")])] * 5)
        chat = flow.GatewayChat(client, "policy", tokenizer, context_tokens=10_000)
        messages = [{"role": "user", "content": "x" * 4000}]
        chat.complete(messages, TOOLS, 24576)
        self.assertEqual(tokenizer.calls[0], {"enable_thinking": False})
        prompt = len(json.dumps(messages)) // 4
        self.assertEqual(completions.requests[0]["max_completion_tokens"], 10_000 - prompt - flow.CONTEXT_MARGIN)
        with self.assertRaises(ContextOverflow):
            chat.complete([{"role": "user", "content": "x" * 40_000}], TOOLS, 24576)
        self.assertEqual((len(completions.requests), len(chat.steps)), (1, 1))  # the overflow made no call, no Step

    def test_engine_context_rejection_becomes_context_overflow(self):
        import httpx
        from openai import BadRequestError

        response = httpx.Response(400, request=httpx.Request("POST", "http://x/v1/chat/completions"))
        error = BadRequestError("This model's maximum context length is 32768 tokens", response=response, body=None)
        client, _ = _client([error])
        chat = flow.GatewayChat(client, "policy")
        with self.assertRaises(ContextOverflow):
            chat.complete([{"role": "user", "content": "x"}], TOOLS, 100)
        self.assertEqual(chat.steps, [])

    def test_template_view_parses_arguments_without_touching_the_episode(self):
        messages = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "execute_code", "arguments": '{"code": "1"}'}}]}]
        view = flow._template_view(messages)
        self.assertEqual(view[0]["tool_calls"][0]["function"]["arguments"], {"code": "1"})
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["arguments"], '{"code": "1"}')

    def test_leased_ports_are_exclusive(self):
        with mock.patch.object(flow, "PORT_LOCK_DIR", self.tmp):
            with flow.leased_port(base=21500, count=2) as first, flow.leased_port(base=21500, count=2) as second:
                self.assertNotEqual(first, second)
            with self.assertRaises(RuntimeError):
                with flow.leased_port(base=21500, count=1), flow.leased_port(base=21500, count=1, wait_s=0):
                    pass


class FlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="appworld-flow-test-"))

    def _patched(self, success: bool, grader_failure: bool = False):
        tmp = self.tmp

        def fake_play(chat, task_id, label, **_):
            episode = run_episode(chat, FakeServer(tmp, success), task_id, "exp")
            if grader_failure:
                episode.reward = None
            return episode

        script = [_response("go", [_call("apis.supervisor.complete_task()")]), _response("done", None, finish="stop")]
        patches = [mock.patch.object(flow, "play", fake_play), mock.patch.object(flow, "RUN_DIR", tmp),
                   mock.patch.object(flow, "TOKENIZER_PATH", ""),
                   mock.patch.object(flow, "OpenAI", lambda **_: _client(script)[0])]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def _run_flow():
        task = Task(id="uid-1", instruction="", metadata={"task_id": "t_1", "split": "train"})
        config = AgentConfig(base_url="http://gateway/v1", model="policy", session_uid="s")
        runner = getattr(flow.appworld_react, "arun", flow.appworld_react)
        return asyncio.run(runner(task, config))

    def test_flow_and_evaluator(self):
        self._patched(success=True)
        episode = self._run_flow()
        self.assertEqual(len(episode.trajectories[0].steps), 2)
        self.assertEqual(episode.artifacts["turns"], 2)
        evaluate = getattr(flow.appworld_evaluator, "evaluate", flow.appworld_evaluator)
        result = evaluate({}, episode)
        self.assertEqual((result.reward, result.is_correct), (1.0, True))
        self.assertEqual({s.name: s.value for s in result.signals}, {"tgc": 1.0, "partial_score": 1.0})
        record = json.loads((self.tmp / "episodes.jsonl").read_text().splitlines()[-1])
        self.assertEqual((record["reward"], record["stop_reason"]), (1.0, "no_tool_call"))

    def test_grader_failure_is_raised_never_scored(self):
        self._patched(success=False, grader_failure=True)
        with self.assertRaises(flow.GraderFailure):
            self._run_flow()


if __name__ == "__main__":
    unittest.main()
