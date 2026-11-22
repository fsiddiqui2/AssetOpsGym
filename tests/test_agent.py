"""The OpenAI-style tool loop with a scripted client and a fake
gym: it stops and scores when the conversation outgrows the
model."""

from types import SimpleNamespace as NS

import httpx
import pytest

pytest.importorskip("openai")
from openai import BadRequestError  # noqa: E402

from assetopsgym import agent as A  # noqa: E402


def bad_request(message: str) -> BadRequestError:
    req = httpx.Request("POST", "http://model/v1/chat/completions")
    return BadRequestError(message, body=None,
                           response=httpx.Response(400, request=req))


class Model:
    """Calls advance_time until `fail_at`, then raises `error`."""

    def __init__(self, fail_at: int, error: Exception):
        self.fail_at, self.error, self.n = fail_at, error, 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model, messages, tools, temperature):
        assert temperature == 0
        self.n += 1
        if self.n > self.fail_at:
            raise self.error
        call = NS(id=f"c{self.n}", function=NS(
            name="advance_time", arguments='{"hours": 24}'))
        msg = NS(tool_calls=[call], model_dump=lambda exclude_none:
                 {"role": "assistant", "tool_calls": [call.id]})
        return NS(choices=[NS(message=msg)],
                  usage=NS(prompt_tokens=50, completion_tokens=5))


class Env:
    def __init__(self):
        self.calls = []

    def reset(self, variant, seed):
        return NS(observation=NS(metadata={
            "instruction": "Keep the chiller running.",
            "episode_id": "ep", "seed": 7}))

    def list_tools(self):
        return [NS(name="advance_time", description="wait",
                   input_schema=None)]

    def step(self, action):
        self.calls.append(action.tool_name)
        done = action.tool_name == "finish"
        meta = {"result": {"variant": "fake"}} if done else {}
        return NS(done=done, reward=0.25 if done else 0.0,
                  observation=NS(error=None, metadata=meta,
                                 result={"data": {"ok": True}}))


def test_a_full_context_ends_the_episode_with_a_score():
    full = bad_request("This model's maximum context length is "
                       "32768 tokens. However, you requested 40000.")
    env = Env()
    out = A.run_episode(env, Model(2, full), "m", None, 10)
    assert env.calls == ["advance_time", "advance_time", "finish"]
    assert out["stopped"] == "context full" and out["score"] == 0.25
    assert out["seed"] == 7 and out["episode_id"] == "ep"


def test_any_other_bad_request_is_raised():
    with pytest.raises(BadRequestError):
        A.run_episode(Env(), Model(0, bad_request("no such tool")),
                      "m", None, 10)
