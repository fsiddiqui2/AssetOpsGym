"""Many episodes at once, and rollouts per GPU-hour.

    MAX_CONCURRENT_ENVS=16 uv run server &
    uv run python -m assetopsgym.rollouts --agent tool_loop \\
      --envs 16 --episodes 64 --variant refleak_mid --model ...

--envs worker threads each keep one session open (one
AssetOpsBench server pair) and take episodes from a shared
queue; episode i runs variant i mod V with seed
seed0 + i div V. Every episode still emits its own profile;
this adds the batch line. For --agent plan_execute run it
in AssetOpsBench's environment (see plan_execute.py).

GPU-hours are this host's GPUs times the wall time, so run
it on the vLLM host; elsewhere only rollouts per hour is
reported.
"""

from __future__ import annotations

import argparse
import importlib
import json
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Callable, ContextManager

from assetopsgym.profiling import (GPUSampler, Profiler, flatten,
                                   wandb_run)

# agent name -> its module (run_episode, NAME, DEFAULT_LIMIT)
AGENTS = {"tool_loop": "assetopsgym.agent",
          "plan_execute": "assetopsgym.plan_execute"}


@dataclass(frozen=True, slots=True)
class Job:
    variant: str | None  # None: the server picks by seed
    seed: int


@dataclass(frozen=True, slots=True)
class Batch:
    agent: str
    model: str
    envs: int
    episodes: int  # finished
    errors: tuple[str, ...]  # one per episode that raised
    wall_s: float  # includes opening the sessions
    rollouts_per_hour: float
    rollouts_per_gpu_hour: float | None  # None: no GPU here
    mean_score: float | None
    llm_s: float  # summed over episodes
    env_s: float
    prompt_tokens: int
    completion_tokens: int
    gpu: dict  # GPUSampler.summary()


def jobs(variants: list[str | None], episodes: int,
         seed0: int) -> list[Job]:
    v = len(variants)
    return [Job(variants[i % v], seed0 + i // v)
            for i in range(episodes)]


def run_batch(todo: list[Job], envs: int,
              open_env: Callable[[], ContextManager],
              run_one: Callable[[Any, Job], dict]
              ) -> tuple[list[dict], list[str], float]:
    """The episode summaries, the errors and the wall seconds.
    An episode that raises is recorded and its worker goes on."""
    q: queue.SimpleQueue[Job] = queue.SimpleQueue()
    for job in todo:
        q.put(job)
    done: list[dict] = []
    errors: list[str] = []

    def work() -> None:
        with open_env() as env:
            while True:
                try:
                    job = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    done.append(run_one(env, job))
                except Exception as e:  # recorded, not skipped
                    errors.append(f"{job}: {e!r}")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(envs) as pool:
        for f in [pool.submit(work) for _ in range(envs)]:
            f.result()
    return done, errors, time.perf_counter() - t0


def fold(agent: str, model: str, envs: int, done: list[dict],
         errors: list[str], wall_s: float, gpu: dict) -> Batch:
    n = len(done)
    per_hour = 3600 * n / wall_s
    scores = [d["score"] for d in done if d["score"] is not None]
    return Batch(
        agent=agent, model=model, envs=envs, episodes=n,
        errors=tuple(errors), wall_s=round(wall_s, 2),
        rollouts_per_hour=round(per_hour, 2),
        rollouts_per_gpu_hour=(round(per_hour / gpu["gpus"], 2)
                               if gpu.get("gpus") else None),
        mean_score=(round(sum(scores) / len(scores), 4)
                    if scores else None),
        llm_s=round(sum(d["seconds"]["llm"] for d in done), 1),
        env_s=round(sum(d["seconds"]["env"] for d in done), 1),
        prompt_tokens=sum(d["tokens"]["prompt"] for d in done),
        completion_tokens=sum(d["tokens"]["completion"]
                              for d in done),
        gpu=gpu)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=list(AGENTS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--envs", type=int, default=1)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--variant", nargs="*", default=[None])
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None,
                   help="turns (tool_loop) or rounds (plan_execute)")
    p.add_argument("--url", default="http://localhost:8000")
    a = p.parse_args()

    from openai import OpenAI

    from assetopsgym import AssetOpsGymEnv
    mod = importlib.import_module(AGENTS[a.agent])
    if mod.NAME != a.agent:
        raise RuntimeError(f"{AGENTS[a.agent]} is {mod.NAME!r}")
    limit = a.limit or mod.DEFAULT_LIMIT
    llm = OpenAI()

    def open_env():
        return AssetOpsGymEnv(base_url=a.url,
                              message_timeout_s=300).sync()

    def run_one(env, job: Job) -> dict:
        return mod.run_episode(env, llm, a.model, job.variant,
                               limit, Profiler(), job.seed)

    gpu = GPUSampler().start()
    done, errors, wall_s = run_batch(
        jobs(a.variant, a.episodes, a.seed0), a.envs,
        open_env, run_one)
    gpu.stop()
    batch = asdict(fold(a.agent, a.model, a.envs, done, errors,
                        wall_s, gpu.summary()))
    run = wandb_run()
    if run is not None:
        run.log(flatten({"batch": batch}))
    print(json.dumps(batch, indent=2))


if __name__ == "__main__":
    main()
