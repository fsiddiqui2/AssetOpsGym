"""AssetOpsBench's Plan-Execute agent on AssetOpsGym, zero-shot.

Plan-Execute plans once per question, so an episode is a
series of rounds. Each round is one PlanExecuteRunner.run()
on a question that carries the task, the clock and the
previous round's answer. The planner, executor and prompts
are AssetOpsBench's, unchanged; its two MCP helpers are
pointed at the gym's step(), so the reward flows.

Runs in AssetOpsBench's environment with the gym on top
(`mise run plan-execute`):

    uv run --no-sync --project $AOB_DIR --with-editable . \\
      python -P -m assetopsgym.plan_execute \\
      --variant refleak_mid --model Qwen/Qwen2.5-3B-Instruct

-P keeps this directory off sys.path: AssetOpsBench's agent
package is also called `agent`. OPENAI_BASE_URL picks the
model server, as for assetopsgym.agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Any

from agent.plan_execute import executor
from agent.plan_execute.runner import PlanExecuteRunner
from llm import LLMBackend, LLMResult
from openai import OpenAI

from assetopsgym import AssetOpsGymEnv
from assetopsgym.agent import (GUIDE, begin, call_tool, report,
                               tool_data, tool_text)
from assetopsgym.profiling import Profiler

NAME = "plan_execute"  # this agent, in profiles and rollouts.py
DEFAULT_LIMIT = 40  # rounds
SERVER = "assetopsgym"  # the one server the planner sees
MAX_NOTE_CHARS = 2000
MAX_TOKENS = 2048  # AssetOpsBench's OpenAICompatBackend
STUCK = (" The clock did not move last round; only "
         "advance_time moves it.")

ROUND = """{task}

{guide}

Decision round {n}. It is now {now}.
Notes from your previous round: {notes}

Plan this round only. Read what you need, act if the \
evidence supports it, and make the last step advance_time \
(or finish), choosing how long to wait before you look again."""


@dataclass
class Session:
    """One episode's tools. The executor gets it where it
    expects a server path."""
    env: Any  # the sync OpenEnv client
    prof: Profiler
    tools: list[dict]
    now: str  # simulated time, ISO 8601
    step: Any = None  # the last StepResult


async def _list_tools(s: Session) -> list[dict]:
    return s.tools


async def _call_tool(s: Session, name: str, args: dict) -> str:
    if s.step is not None and s.step.done:
        return json.dumps({"error": "the episode is over"})
    s.step = call_tool(s.env, s.prof, name, args)
    obs = s.step.observation
    data = tool_data(obs) if obs.error is None else None
    if isinstance(data, dict) and "sim_time" in data:
        s.now = data["sim_time"]
    return tool_text(obs)


# The executor reaches every tool through these two helpers
# (stdio, one new server process per call). Ours go through
# step() instead.
executor._list_tools = _list_tools
executor._call_tool = _call_tool


def describe(tools: list) -> list[dict]:
    """OpenEnv tools in the shape the executor's helpers
    return."""
    out = []
    for t in tools:
        schema = t.input_schema or {}
        required = schema.get("required", [])
        out.append({
            "name": t.name, "description": t.description or "",
            "parameters": [
                {"name": k, "type": v.get("type", "any"),
                 "required": k in required}
                for k, v in schema.get("properties", {}).items()]})
    return out


class EndpointLLM(LLMBackend):
    """An OpenAI-compatible endpoint as an AssetOpsBench
    backend. Each request is an llm.chat span."""

    def __init__(self, client: OpenAI, model: str,
                 prof: Profiler) -> None:
        self._client, self._model_id = client, model
        self._prof = prof

    def generate(self, prompt: str,
                 temperature: float = 0.0) -> str:
        return self.generate_with_usage(prompt, temperature).text

    def generate_with_usage(self, prompt: str,
                            temperature: float = 0.0) -> LLMResult:
        with self._prof.span("llm.chat") as sp:
            r = self._client.chat.completions.create(
                model=self._model_id, temperature=temperature,
                max_completion_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}])
            u = r.usage
            used = ((u.prompt_tokens, u.completion_tokens)
                    if u else (0, 0))
            sp.set(prompt_tokens=used[0], completion_tokens=used[1])
        return LLMResult(r.choices[0].message.content or "", *used)


def run_episode(env, llm: OpenAI, model: str,
                variant: str | None, max_rounds: int,
                prof: Profiler | None = None,
                seed: int | None = None) -> dict:
    prof = prof or Profiler()
    return run_rounds(env, EndpointLLM(llm, model, prof), model,
                      variant, max_rounds, prof, seed, NAME)


def run_rounds(env, backend: LLMBackend, model: str,
               variant: str | None, max_rounds: int,
               prof: Profiler, seed: int | None,
               agent: str) -> dict:
    """An episode on any AssetOpsBench backend; `agent` names it
    in the report."""
    meta = begin(env, prof, variant, seed)
    s = Session(env, prof, describe(env.list_tools()),
                meta["sim_time"])
    runner = PlanExecuteRunner(backend, {SERVER: s})
    notes, rounds = "none yet.", []
    prof.meta["stopped"] = "round limit"  # unless the episode ends
    for n in range(1, max_rounds + 1):
        before = s.now
        question = ROUND.format(task=meta["instruction"],
                                guide=GUIDE, n=n, now=s.now,
                                notes=notes)
        with prof.span("agent.round", round=n) as sp:
            try:
                out = asyncio.run(runner.run(question))
            except ValueError as e:  # the plan did not parse
                sp.set(error=str(e))
                notes = f"Your last plan was rejected: {e}"
                rounds.append({"round": n, "question": question,
                               "rejected": str(e)})
                continue
            sp.set(steps=len(out.plan.steps))
        rounds.append({"round": n, "question": question,
                       "plan": out.plan.raw, "answer": out.answer,
                       "steps": [asdict(x) for x in out.trajectory]})
        if s.step is not None and s.step.done:
            prof.meta["stopped"] = None
            break
        notes = (out.answer or "none.")[:MAX_NOTE_CHARS]
        if s.now == before:
            notes += STUCK
    if s.step is None or not s.step.done:
        # out of rounds: score the state reached
        s.step = call_tool(env, prof, "finish", {})
    return report(s.step, prof, model, agent, rounds)


def make_client() -> OpenAI:
    """OPENAI_BASE_URL and OPENAI_API_KEY pick the model server."""
    return OpenAI()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--variant", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--max-rounds", type=int,
                   default=DEFAULT_LIMIT)
    a = p.parse_args()
    with AssetOpsGymEnv(base_url=a.url,
                        message_timeout_s=300).sync() as env:
        print(json.dumps(run_episode(
            env, make_client(), a.model, a.variant, a.max_rounds),
            indent=2))


if __name__ == "__main__":
    main()
