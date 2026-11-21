"""Zero-shot LLM agent for AssetOpsGym over any OpenAI-compatible endpoint.

    uv run server &                                   # the gym, on :8000
    OPENAI_BASE_URL=https://api.tokenrouter.com/v1 OPENAI_API_KEY=... \\
      uv run python -m assetopsgym.agent --variant refleak_mid --model openai/gpt-5-mini

Point OPENAI_BASE_URL at a vLLM server (infra/serve.sh:
http://localhost:8001/v1) to run a local model. Every tool call goes
through env.step(), so the final observation carries the reward. Prints a JSON
summary with the score, the agent-side profile (LLM vs environment wall time,
tokens per simulated hour) and the server-side profile the environment attaches
to its last observation. Set AOG_PROFILE_DIR or WANDB_PROJECT to keep them
(see assetopsgym.profiling).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import BadRequestError, OpenAI

from assetopsgym import AssetOpsGymEnv, CallToolAction
from assetopsgym.profiling import Profiler

NAME = "tool_loop"  # this agent, in profiles and rollouts.py
DEFAULT_LIMIT = 60  # turns
MAX_TOOL_CHARS = 6000
NUDGE = "Keep going: use the tools, advance time, or call finish."
GUIDE = ("Work like an engineer on shift: check the data, decide how far to advance time, and act on evidence. "
         "Tool results longer than a few thousand characters are cut off, so prefer sensor_stats over a time "
         "window, and pass sensors and a small limit to history.")


def tool_data(obs):
    """A tool observation's payload (no error)."""
    result = obs.result
    return (result.get("data") if isinstance(result, dict)
            else getattr(result, "data", result))


def tool_text(obs) -> str:
    """What the model reads back from a tool call."""
    if obs.error is not None:
        return json.dumps({"error": obs.error.message})
    text = json.dumps(tool_data(obs), default=str)
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + f"... [truncated {len(text) - MAX_TOOL_CHARS} chars; narrow the query]"
    return text


def _sim_hours(obs) -> float:
    """Simulated hours an advance_time call actually moved, from its own result."""
    if obs.error is not None:
        return 0.0
    data = tool_data(obs)
    return float(data.get("sim_hours_advanced", 0.0)) if isinstance(data, dict) else 0.0


def call_tool(env, prof: Profiler, name: str, args: dict):
    with prof.span("client.step", tool=name) as sp:
        step = env.step(CallToolAction(tool_name=name, arguments=args))
        if name == "advance_time":
            sp.set(sim_hours=_sim_hours(step.observation))
    return step


def begin(env, prof: Profiler, variant: str | None,
          seed: int | None) -> dict:
    """Reset the gym; the episode's id and seed go on the profile.
    Returns the reset's metadata."""
    with prof.span("client.reset"):
        reset = env.reset(variant=variant, seed=seed)
    meta = reset.observation.metadata
    prof.episode_id = meta.get("episode_id")
    prof.meta["seed"] = meta.get("seed")
    return meta


def run_episode(env, llm: OpenAI, model: str, variant: str | None, max_turns: int,
                prof: Profiler | None = None,
                seed: int | None = None) -> dict:
    prof = prof or Profiler()
    meta = begin(env, prof, variant, seed)
    tools = [{"type": "function", "function": {"name": t.name, "description": t.description[:1000],
                                               "parameters": t.input_schema or {"type": "object", "properties": {}}}}
             for t in env.list_tools()]
    messages = [{"role": "system", "content": meta["instruction"] + "\n\n" + GUIDE},
                {"role": "user", "content": "Your shift starts now."}]
    nudges, step = 0, None
    prof.meta["stopped"] = "turn limit"  # unless something else does
    for turn in range(max_turns):
        with prof.span("llm.chat", turn=turn, messages=len(messages)) as sp:
            try:
                resp = llm.chat.completions.create(model=model, messages=messages, tools=tools)
            except BadRequestError as e:
                # the conversation outgrew the model: score the
                # state reached, like running out of turns
                if "context length" not in str(e):
                    raise
                prof.meta["stopped"] = "context full"
                break
            if resp.usage:
                sp.set(prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            nudges += 1
            if nudges > 2:
                prof.meta["stopped"] = "no tool calls"
                break
            messages.append({"role": "user", "content": NUDGE})
            continue
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = {}
            step = call_tool(env, prof, tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_text(step.observation)})
            if step.done:
                break
        if step is not None and step.done:
            prof.meta["stopped"] = None  # the episode ended itself
            break
    if step is None or not step.done:  # out of turns: score the state reached, as FDE-Gym does
        step = call_tool(env, prof, "finish", {})
    return report(step, prof, model, NAME, messages)


def report(step, prof: Profiler, model: str, agent: str,
           transcript: list) -> dict:
    """Emit the agent-side profile of a finished episode, save the
    episode (see save) and return its summary. `agent` names the
    agent in the sinks; prof.meta["stopped"] says why the agent's
    loop stopped, if the episode did not end by itself."""
    result = step.observation.metadata.get("result", {})
    profile = prof.emit(side="agent", agent=agent,
                        variant=result.get("variant"), model=model, score=step.reward,
                        end_reason=result.get("end_reason"),
                        episode_id=prof.episode_id,
                        server=(result.get("profile") or {}).get("derived"))
    d = profile["derived"]
    summary = {
            "episode_id": prof.episode_id, "seed": prof.meta.get("seed"),
            "stopped": prof.meta.get("stopped"),
            "variant": result.get("variant"), "agent": agent,
            "model": model, "score": step.reward,
            "failure_prevented": result.get("failure_prevented"), "false_dispatches": result.get("false_dispatches"),
            "detection_delay_hours": result.get("detection_delay_hours"), "cost_total": result.get("cost_total"),
            "end_reason": result.get("end_reason"), "tool_calls": d["tool_calls"],
            "tokens": {"prompt": d["prompt_tokens"], "completion": d["completion_tokens"]},
            "seconds": {"llm": round(d["llm_s"], 1), "env": round(d["env_s"], 1)},
            "tokens_per_sim_hour": d["tokens_per_sim_hour"], "sim_hours_per_wall_s": d["sim_hours_per_wall_s"],
            "result": {k: v for k, v in result.items() if k != "profile"},
            "profile": profile, "server_profile": result.get("profile")}
    save(summary, transcript)
    return summary


def _plain(x):
    """SDK message objects, for json.dumps."""
    return x.model_dump() if hasattr(x, "model_dump") else str(x)


def save(summary: dict, transcript: list) -> None:
    """With AOG_PROFILE_DIR set: the summary as a line of
    episodes.jsonl, and what the agent saw and said as
    transcripts/<episode id>.json."""
    d = os.environ.get("AOG_PROFILE_DIR")
    if not d:
        return
    out = Path(d) / "transcripts"
    out.mkdir(parents=True, exist_ok=True)
    with open(Path(d) / "episodes.jsonl", "a") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    (out / f"{summary['episode_id']}.json").write_text(
        json.dumps(transcript, default=_plain))


def make_client() -> OpenAI:
    """OPENAI_BASE_URL and OPENAI_API_KEY pick the model server."""
    return OpenAI()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--variant", default=None)
    p.add_argument("--model", default="openai/gpt-5-mini")
    p.add_argument("--max-turns", type=int, default=DEFAULT_LIMIT)
    a = p.parse_args()
    with AssetOpsGymEnv(base_url=a.url, message_timeout_s=300).sync() as env:
        print(json.dumps(run_episode(env, make_client(), a.model, a.variant, a.max_turns), indent=2))


if __name__ == "__main__":
    main()
