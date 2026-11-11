"""The batch runner: every job once, in parallel, errors kept."""

import time
from contextlib import contextmanager

from assetopsgym.rollouts import Job, fold, jobs, run_batch


def test_jobs_cycle_variants_then_seeds():
    assert jobs(["a", "b"], 5, 1000) == [
        Job("a", 1000), Job("b", 1000), Job("a", 1001),
        Job("b", 1001), Job("a", 1002)]


def test_batch_runs_each_job_once_in_parallel_and_keeps_errors():
    opened = []

    @contextmanager
    def open_env():
        opened.append(1)
        yield len(opened)

    def run_one(env, job):
        time.sleep(0.05)
        if job.seed == 3:
            raise RuntimeError("model down")
        return {"score": job.seed / 10, "seed": job.seed,
                "seconds": {"llm": 1.0, "env": 0.5},
                "tokens": {"prompt": 100, "completion": 10}}

    todo = jobs(["v"], 8, 0)
    done, errors, wall = run_batch(todo, 4, open_env, run_one)
    assert len(opened) == 4  # one session per worker
    assert sorted(d["seed"] for d in done) == [0, 1, 2, 4, 5, 6, 7]
    assert len(errors) == 1 and "model down" in errors[0]
    assert wall < 0.3  # 8 x 0.05 s in sequence is 0.4 s

    b = fold("tool_loop", "m", 4, done, errors, 36.0, {"gpus": 2})
    assert b.episodes == 7 and b.rollouts_per_hour == 700
    assert b.rollouts_per_gpu_hour == 350
    assert b.llm_s == 7.0 and b.prompt_tokens == 700
    assert b.mean_score == round(2.5 / 7, 4)
    none = fold("tool_loop", "m", 4, done, errors, 36.0,
                {"available": False, "samples": 0})
    assert none.rollouts_per_gpu_hour is None
