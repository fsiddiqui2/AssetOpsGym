"""The zero-shot tool loop on Claude's Messages API.

The same agent as agent.py, a tool-calling loop that keeps the
whole conversation, for running the gym on Claude:

    uv sync --extra claude
    uv run python -m assetopsgym.claude_agent --variant refleak_mid

Credentials are the Anthropic SDK's: ANTHROPIC_API_KEY, or an
`ant auth login` profile. The conversation prefix is cached, so
each turn pays full price only for what the last turn added.
No fallback model is configured: a refusal ends the episode, so a
score is always the named model's own.
"""

from __future__ import annotations

import argparse
import json

import anthropic

from assetopsgym import AssetOpsGymEnv
from assetopsgym.agent import (GUIDE, NUDGE, call_tool, report,
                               tool_text)
from assetopsgym.profiling import Profiler, Span

NAME = "claude_tool_loop"  # this agent, in profiles and rollouts.py
DEFAULT_LIMIT = 60  # turns
MODEL = "claude-opus-5-5"
EFFORT = "medium"  # the model's default, set so it is on record
MAX_TOKENS = 16000
OVER = "the episode is over"


def record(sp: Span, r: anthropic.types.Message) -> int:
    """The request's token counts on its llm.chat span; returns the
    whole prompt's size (cached or not)."""
    u = r.usage
    read = u.cache_read_input_tokens or 0
    written = u.cache_creation_input_tokens or 0
    prompt = u.input_tokens + read + written
    sp.set(prompt_tokens=prompt, completion_tokens=u.output_tokens,
           cache_read_tokens=read, cache_write_tokens=written,
           stop_reason=r.stop_reason)
    return prompt


def run_episode(env, client: anthropic.Anthropic, model: str,
                variant: str | None, max_turns: int,
                prof: Profiler | None = None,
                seed: int | None = None) -> dict:
    prof = prof or Profiler()
    with prof.span("client.reset"):
        reset = env.reset(variant=variant, seed=seed)
    meta = reset.observation.metadata
    prof.episode_id = meta.get("episode_id")
    tools = [{"name": t.name, "description": t.description or "",
              "input_schema": t.input_schema
              or {"type": "object", "properties": {}}}
             for t in env.list_tools()]
    system = meta["instruction"] + "\n\n" + GUIDE
    messages = [{"role": "user", "content": "Your shift starts now."}]
    nudges, step = 0, None
    for turn in range(max_turns):
        with prof.span("llm.chat", turn=turn,
                       messages=len(messages)) as sp:
            r = client.messages.create(
                model=model, max_tokens=MAX_TOKENS, system=system,
                tools=tools, messages=messages,
                output_config={"effort": EFFORT},
                cache_control={"type": "ephemeral"})
            record(sp, r)
        # the content as it came: thinking blocks must go back whole
        messages.append({"role": "assistant", "content": r.content})
        if r.stop_reason == "refusal":
            break
        calls = [b for b in r.content if b.type == "tool_use"]
        if not calls:
            nudges += 1
            if nudges > 2:
                break
            messages.append({"role": "user", "content": NUDGE})
            continue
        results = []
        for c in calls:  # every tool_use needs its tool_result
            if step is not None and step.done:
                results.append({
                    "type": "tool_result", "tool_use_id": c.id,
                    "content": OVER, "is_error": True})
                continue
            step = call_tool(env, prof, c.name, c.input)
            results.append({
                "type": "tool_result", "tool_use_id": c.id,
                "content": tool_text(step.observation),
                "is_error": step.observation.error is not None})
        messages.append({"role": "user", "content": results})
        if step.done:
            break
    if step is None or not step.done:
        # out of turns: score the state reached
        step = call_tool(env, prof, "finish", {})
    return report(step, prof, model, NAME)


def make_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--variant", default=None)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--max-turns", type=int, default=DEFAULT_LIMIT)
    a = p.parse_args()
    with AssetOpsGymEnv(base_url=a.url,
                        message_timeout_s=300).sync() as env:
        print(json.dumps(run_episode(
            env, make_client(), a.model, a.variant, a.max_turns),
            indent=2))


if __name__ == "__main__":
    main()
