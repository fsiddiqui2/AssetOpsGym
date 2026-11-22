"""AssetOpsBench's Plan-Execute agent on Claude's Messages API.

plan_execute.py's rounds with a Claude backend. Runs in
AssetOpsBench's environment with the gym on top
(`mise run claude-plan-execute`):

    uv run --no-sync --project $AOB_DIR \\
      --with-editable '.[claude]' \\
      python -P -m assetopsgym.claude_plan_execute \\
      --variant refleak_mid

Credentials as for claude_agent.py. Each planner or argument
prompt is one message with no shared block to cache.
"""

from __future__ import annotations

import argparse
import json

import anthropic
from llm import LLMBackend, LLMResult

from assetopsgym import AssetOpsGymEnv
from assetopsgym.claude_agent import (EFFORT, MAX_TOKENS, MODEL,
                                      make_client, record)
from assetopsgym.plan_execute import DEFAULT_LIMIT, run_rounds
from assetopsgym.profiling import Profiler

NAME = "claude_plan_execute"  # in profiles and rollouts.py


class ClaudeLLM(LLMBackend):
    """Claude as an AssetOpsBench backend. Each request is an
    llm.chat span."""

    def __init__(self, client: anthropic.Anthropic, model: str,
                 prof: Profiler) -> None:
        self._client, self._model_id = client, model
        self._prof = prof

    def generate(self, prompt: str,
                 temperature: float = 0.0) -> str:
        return self.generate_with_usage(prompt, temperature).text

    def generate_with_usage(self, prompt: str,
                            temperature: float = 0.0) -> LLMResult:
        # the planner's temperature is dropped: Claude's current
        # models reject sampling parameters
        with self._prof.span("llm.chat") as sp:
            r = self._client.messages.create(
                model=self._model_id, max_tokens=MAX_TOKENS,
                output_config={"effort": EFFORT},
                messages=[{"role": "user", "content": prompt}])
            prompt_tokens = record(sp, r)
        text = "".join(b.text for b in r.content if b.type == "text")
        return LLMResult(text, prompt_tokens, r.usage.output_tokens)


def run_episode(env, client: anthropic.Anthropic, model: str,
                variant: str | None, max_rounds: int,
                prof: Profiler | None = None,
                seed: int | None = None) -> dict:
    prof = prof or Profiler()
    return run_rounds(env, ClaudeLLM(client, model, prof), model,
                      variant, max_rounds, prof, seed, NAME)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--variant", default=None)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--max-rounds", type=int, default=DEFAULT_LIMIT)
    a = p.parse_args()
    with AssetOpsGymEnv(base_url=a.url,
                        message_timeout_s=300).sync() as env:
        print(json.dumps(run_episode(
            env, make_client(), a.model, a.variant, a.max_rounds),
            indent=2))


if __name__ == "__main__":
    main()
