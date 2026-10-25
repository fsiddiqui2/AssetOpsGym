"""Zero-shot LLM agent for AssetOpsGym over any OpenAI-compatible endpoint.

    uv run server &                                   # the gym, on :8000
    OPENAI_BASE_URL=https://api.tokenrouter.com/v1 OPENAI_API_KEY=... \\
      uv run python -m assetopsgym.agent --variant refleak_mid --model openai/gpt-5-mini

Point OPENAI_BASE_URL at a vLLM server to run a local model. Every tool call goes
through env.step(), so the final observation carries the reward. Prints a JSON
summary with the score and the wall-time split between the LLM and the environment.
"""

from __future__ import annotations

import argparse
import json
import time

from openai import OpenAI

from assetopsgym import AssetOpsGymEnv, CallToolAction

MAX_TOOL_CHARS = 6000
GUIDE = ("Work like an engineer on shift: check the data, decide how far to advance time, and act on evidence. "
         "Tool results longer than a few thousand characters are cut off, so prefer sensor_stats over a time "
         "window, and pass sensors and a small limit to history.")


def _content(obs) -> str:
    if obs.error is not None:
        return json.dumps({"error": obs.error.message})
    result = obs.result
    data = result.get("data") if isinstance(result, dict) else getattr(result, "data", result)
    text = json.dumps(data, default=str)
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + f"... [truncated {len(text) - MAX_TOOL_CHARS} chars; narrow the query]"
    return text


def run_episode(env, llm: OpenAI, model: str, variant: str | None, max_turns: int) -> dict:
    reset = env.reset(variant=variant)
    tools = [{"type": "function", "function": {"name": t.name, "description": t.description[:1000],
                                               "parameters": t.input_schema or {"type": "object", "properties": {}}}}
             for t in env.list_tools()]
    messages = [{"role": "system", "content": reset.observation.metadata["instruction"] + "\n\n" + GUIDE},
                {"role": "user", "content": "Your shift starts now."}]
    t_llm = t_env = 0.0
    tokens = {"prompt": 0, "completion": 0}
    calls, nudges, step = 0, 0, None
    for _ in range(max_turns):
        t = time.perf_counter()
        resp = llm.chat.completions.create(model=model, messages=messages, tools=tools)
        t_llm += time.perf_counter() - t
        if resp.usage:
            tokens["prompt"] += resp.usage.prompt_tokens
            tokens["completion"] += resp.usage.completion_tokens
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            nudges += 1
            if nudges > 2:
                break
            messages.append({"role": "user", "content": "Keep going: use the tools, advance time, or call finish."})
            continue
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = {}
            t = time.perf_counter()
            step = env.step(CallToolAction(tool_name=tc.function.name, arguments=args))
            t_env += time.perf_counter() - t
            calls += 1
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": _content(step.observation)})
            if step.done:
                break
        if step is not None and step.done:
            break
    if step is None or not step.done:  # out of turns: score the state reached, as FDE-Gym does
        t = time.perf_counter()
        step = env.step(CallToolAction(tool_name="finish", arguments={}))
        t_env += time.perf_counter() - t
    result = step.observation.metadata.get("result", {})
    return {"variant": result.get("variant"), "model": model, "score": step.reward,
            "failure_prevented": result.get("failure_prevented"), "false_dispatches": result.get("false_dispatches"),
            "detection_delay_hours": result.get("detection_delay_hours"), "cost_total": result.get("cost_total"),
            "end_reason": result.get("end_reason"), "tool_calls": calls, "tokens": tokens,
            "seconds": {"llm": round(t_llm, 1), "env": round(t_env, 1)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--variant", default=None)
    p.add_argument("--model", default="openai/gpt-5-mini")
    p.add_argument("--max-turns", type=int, default=60)
    a = p.parse_args()
    with AssetOpsGymEnv(base_url=a.url, message_timeout_s=300).sync() as env:
        print(json.dumps(run_episode(env, OpenAI(), a.model, a.variant, a.max_turns), indent=2))


if __name__ == "__main__":
    main()
