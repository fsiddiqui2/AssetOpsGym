"""The Claude tool loop with a scripted client and a fake gym."""

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("anthropic")

from assetopsgym import claude_agent as ca  # noqa: E402


def use(i: int, name: str, args: dict) -> NS:
    return NS(type="tool_use", id=f"tu{i}", name=name, input=args)


class Model:
    """Replies in order; keeps the requests it was sent."""

    def __init__(self, replies: list):
        self.replies, self.sent = list(replies), []
        self.messages = NS(create=self.create)

    def create(self, model, max_tokens, system, tools, messages,
               output_config, cache_control):
        self.sent.append(NS(system=system, tools=tools,
                            messages=list(messages),
                            effort=output_config["effort"]))
        content, stop = self.replies.pop(0)
        return NS(content=content, stop_reason=stop, usage=NS(
            input_tokens=10, cache_read_input_tokens=80,
            cache_creation_input_tokens=5, output_tokens=7))


class Env:
    def __init__(self):
        self.calls = []

    def reset(self, variant, seed):
        return NS(observation=NS(metadata={
            "instruction": "Keep the chiller running.",
            "episode_id": "ep"}))

    def list_tools(self):
        return [NS(name="advance_time", description="wait",
                   input_schema={"type": "object", "properties": {
                       "hours": {"type": "number"}}}),
                NS(name="finish", description="end",
                   input_schema=None)]

    def step(self, action):
        self.calls.append((action.tool_name, action.arguments))
        done = action.tool_name == "finish"
        meta = {"result": {"variant": "fake"}} if done else {}
        return NS(done=done, reward=0.5 if done else 0.0,
                  observation=NS(error=None, metadata=meta,
                                 result={"data": {"ok": True}}))


def test_tool_results_go_back_in_one_message_per_turn():
    think = NS(type="thinking", thinking="")
    model = Model([
        ([think, use(1, "advance_time", {"hours": 24}),
          use(2, "advance_time", {"hours": 48})], "tool_use"),
        ([NS(type="text", text="Looks fine.")], "end_turn"),
        ([use(3, "finish", {}), use(4, "advance_time", {"hours": 1})],
         "tool_use")])
    env = Env()
    out = ca.run_episode(env, model, "m", None, 10)
    assert env.calls == [("advance_time", {"hours": 24}),
                         ("advance_time", {"hours": 48}),
                         ("finish", {})]
    assert out["score"] == 0.5 and out["agent"] == ca.NAME
    first, second, third = model.sent
    assert first.system.startswith("Keep the chiller running.")
    assert [t["name"] for t in first.tools] == ["advance_time",
                                                "finish"]
    assert first.tools[1]["input_schema"]["type"] == "object"
    assert first.effort == ca.EFFORT
    # the assistant turn as it came, then both results together
    assert second.messages[1]["content"][0] is think
    back = second.messages[2]["content"]
    assert [b["tool_use_id"] for b in back] == ["tu1", "tu2"]
    assert not back[0]["is_error"]
    assert third.messages[-1] == {"role": "user",
                                  "content": ca.NUDGE}
    d = out["profile"]["derived"]
    assert d["llm_calls"] == 3 and d["prompt_tokens"] == 3 * 95
    assert d["completion_tokens"] == 21 and d["tool_calls"] == 3


def test_a_refusal_or_running_out_of_turns_ends_with_finish():
    env = Env()
    ca.run_episode(env, Model([([], "refusal")]), "m", None, 10)
    assert env.calls == [("finish", {})]
    env = Env()
    forever = [([use(i, "advance_time", {"hours": 1})], "tool_use")
               for i in range(5)]
    ca.run_episode(env, Model(forever), "m", None, 2)
    assert [c[0] for c in env.calls] == ["advance_time",
                                         "advance_time", "finish"]
