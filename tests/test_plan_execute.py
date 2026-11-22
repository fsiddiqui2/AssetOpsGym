"""Plan-Execute rounds with a scripted model and a fake client.

Needs AssetOpsBench's environment (see plan_execute.py);
skipped in the gym's own.
"""

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("agent.plan_execute.runner")

from assetopsgym import plan_execute as pe  # noqa: E402

PLAN = """#Task1: Look at the alerts
#Server1: assetopsgym
#Tool1: list_alerts
#Dependency1: None
#ExpectedOutput1: the alerts so far

#Task2: Wait a day
#Server2: assetopsgym
#Tool2: advance_time
#Dependency2: {dep}
#ExpectedOutput2: the new time
"""


class Model:
    """Answers AssetOpsBench's three prompts; keeps them."""

    def __init__(self, dep: str = "#S1"):
        self.dep, self.prompts = dep, []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model, temperature, max_completion_tokens,
               messages):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if prompt.startswith("You are a planning assistant"):
            text = PLAN.format(dep=self.dep)
        elif prompt.startswith("Generate the JSON arguments"):
            text = ('{"hours": 24}' if "Tool: advance_time" in prompt
                    else "{}")
        else:
            text = f"note {len(self.prompts)}"
        return NS(choices=[NS(message=NS(content=text))],
                  usage=NS(prompt_tokens=100, completion_tokens=10))


class Env:
    """A clock that advance_time moves; done after `days`."""

    def __init__(self, days: int):
        self.days, self.day, self.calls = days, 0, []

    def reset(self, variant, seed):
        return NS(observation=NS(metadata={
            "instruction": "Keep the chiller running.",
            "episode_id": "ep", "sim_time": "day 0"}))

    def list_tools(self):
        hours = {"type": "object", "required": ["hours"],
                 "properties": {"hours": {"type": "number"}}}
        return [NS(name="list_alerts", description="alerts",
                   input_schema=None),
                NS(name="advance_time", description="wait",
                   input_schema=hours),
                NS(name="finish", description="end",
                   input_schema=None)]

    def step(self, action):
        name, args = action.tool_name, action.arguments
        self.calls.append(name)
        data = {"alerts": []}
        if name == "advance_time":
            self.day += args["hours"] / 24
            data = {"sim_time": f"day {self.day:g}",
                    "sim_hours_advanced": args["hours"]}
        done = name == "finish" or self.day >= self.days
        meta = {"result": {"variant": "fake"}} if done else {}
        return NS(done=done, reward=0.5 if done else 0.0,
                  observation=NS(error=None, metadata=meta,
                                 result={"data": data}))


def test_rounds_carry_the_clock_and_notes_until_done():
    env, model = Env(days=3), Model()
    out = pe.run_episode(env, model, "m", None, max_rounds=10)
    assert env.calls == ["list_alerts", "advance_time"] * 3
    assert out["score"] == 0.5 and out["agent"] == "plan_execute"
    d = out["profile"]["derived"]
    # per round: a plan, two argument calls, a summary
    assert d["llm_calls"] == 12 and d["tool_calls"] == 6
    assert d["sim_hours"] == 72 and d["prompt_tokens"] == 1200
    assert out["profile"]["by_name"]["agent.round"]["count"] == 3
    plans = [p for p in model.prompts
             if p.startswith("You are a planning assistant")]
    assert "It is now day 0." in plans[0]
    assert "previous round: none yet." in plans[0]
    assert "It is now day 1." in plans[1]
    assert "previous round: note 4" in plans[1]
    assert "Keep the chiller running." in plans[2]


def test_out_of_rounds_finishes_and_a_bad_plan_costs_a_round():
    env, model = Env(days=99), Model()
    out = pe.run_episode(env, model, "m", None, max_rounds=2)
    assert env.calls[-1] == "finish" and out["score"] == 0.5
    assert env.calls.count("advance_time") == 2

    env, model = Env(days=99), Model(dep="#S7")  # no step 7
    pe.run_episode(env, model, "m", None, max_rounds=2)
    assert env.calls == ["finish"]
    assert "plan was rejected" in model.prompts[1]
