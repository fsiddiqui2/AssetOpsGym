"""The profiler folds spans into the report's numbers; the sinks only fire when configured."""

import json
import threading
import time

import pytest

from assetopsgym.profiling import (GPUSampler, Profiler, flatten,
                                   gpu_summary, read_gpu_log,
                                   vllm_work)


def test_spans_nest_per_thread_and_fold_by_name():
    p = Profiler(episode_id="ep1")
    with p.span("env.step", tool="advance_time"):
        with p.span("sim.advance", sim_hours=24) as s:
            time.sleep(0.01)
            s.set(rows=96)
        with p.span("couch.write_rows", rows=96):
            pass
    with p.span("env.step", tool="history"):
        with p.span("mcp.aob", server="iot", tool="history"):
            pass
    names = {s.name for s in p.spans}
    assert names == {"env.step", "sim.advance", "couch.write_rows", "mcp.aob"}
    inner = next(s for s in p.spans if s.name == "sim.advance")
    assert inner.parent == "env.step" and inner.attrs == {"sim_hours": 24, "rows": 96}
    s = p.summary()
    assert s["episode_id"] == "ep1" and s["spans"] == 5 and s["dropped"] == 0
    assert s["by_name"]["env.step"]["count"] == 2
    assert s["by_name"]["sim.advance"]["seconds"] >= 0.01
    assert s["derived"]["sim_hours"] == 24 and s["derived"]["tool_calls"] == 2
    assert s["derived"]["env_s"] >= s["derived"]["sim_s"] > 0
    assert s["derived"]["aob_s"] >= 0 and s["derived"]["llm_s"] == 0
    json.dumps(s)  # serializable as is



def test_finish_fast_forward_is_not_simulated_time():
    p = Profiler()
    with p.span("sim.advance", sim_hours=24):
        pass
    with p.span("sim.run_to_end", sim_hours_skipped=1416):
        pass
    d = p.summary()["derived"]
    assert d["sim_hours"] == 24 and d["sim_hours_skipped"] == 1416

def test_agent_side_derived_metrics():
    p = Profiler()
    for hours in (24, 48, 96):
        with p.span("llm.chat", prompt_tokens=1000, completion_tokens=50):
            time.sleep(0.002)
        with p.span("client.step", tool="advance_time", sim_hours=hours):
            time.sleep(0.001)
    d = p.summary()["derived"]
    assert d["llm_calls"] == 3 and d["tool_calls"] == 3
    assert d["prompt_tokens"] == 3000 and d["completion_tokens"] == 150
    assert d["sim_hours"] == 168
    assert d["tokens_per_sim_hour"] == round(3150 / 168, 1)
    assert 0 < d["llm_fraction"] < 1 and d["llm_s"] > d["env_s"]
    assert d["sim_hours_per_wall_s"] > 0


def test_timed_decorator_add_and_cap():
    p = Profiler(max_spans=3)

    @p.timed("sim.tick")
    def tick():
        return 1

    assert tick() == 1
    p.add("llm.chat", 0.5, prompt_tokens=10, completion_tokens=1)
    with p.span("env.step"):
        pass
    with p.span("env.step"):  # fourth span: dropped, counted
        pass
    s = p.summary()
    assert s["spans"] == 3 and s["dropped"] == 1
    assert s["by_name"]["llm.chat"]["seconds"] == 0.5 and s["derived"]["prompt_tokens"] == 10


def test_threads_keep_their_own_stacks():
    p = Profiler()

    def worker():
        with p.span("sim.advance", sim_hours=1):
            time.sleep(0.005)

    with p.span("env.step", tool="advance_time"):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    inner = next(s for s in p.spans if s.name == "sim.advance")
    assert inner.parent is None  # a different thread: no inherited parent
    assert p.total("env.step") >= inner.seconds


def test_emit_writes_jsonl_only_when_configured(tmp_path, monkeypatch):
    p = Profiler(episode_id="e")
    with p.span("env.step"):
        pass
    monkeypatch.delenv("AOG_PROFILE_DIR", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    p.emit(variant="healthy")
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("AOG_PROFILE_DIR", str(tmp_path / "runs"))
    out = p.emit(variant="healthy", model="m")
    assert out["meta"] == {"variant": "healthy", "model": "m"}
    lines = (tmp_path / "runs" / "profile.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["episode_id"] == "e"


def test_flatten_and_gpu_sampler_without_a_gpu():
    assert flatten({"a": {"b": 1, "c": {"d": "x"}}, "e": None, "f": [1]}) == {"a/b": 1, "a/c/d": "x", "e": None}
    g = GPUSampler(interval_s=0.01)
    g.available = False  # whatever the host has, exercise the absent path
    g.start()
    g.stop()
    assert g.summary() == {"available": False, "samples": 0}


def test_gpu_log_is_cut_to_the_batch_window(tmp_path):
    log = tmp_path / "gpu.csv"
    log.write_text("100, 0, 0, 500, 46068\n"      # before the batch
                   "110, 0, 90, 21000, 46068\n"
                   "111, 0, 70, 22000, 46068\n"
                   "130, 0, 5, 22000, 46068\n"     # after it
                   "131, 0, 9")                    # still being written
    rows = read_gpu_log(log, 105, 120)
    assert [r["t"] for r in rows] == [110, 111]
    assert gpu_summary(rows) == {
        "available": True, "samples": 2, "gpus": 1,
        "util_mean": 80.0, "util_max": 90, "mem_used_mb_mean": 21500,
        "mem_used_mb_max": 22000, "mem_total_mb": 46068}
    log.write_text("110, 0, ninety, 1, 2\n")
    with pytest.raises(ValueError):
        read_gpu_log(log, 0, 200)


def test_vllm_work_is_the_difference_of_two_scrapes():
    def page(n, queue, prefill, decode, ttft, e2e, p, c, g):
        return {"vllm:e2e_request_latency_seconds_count": n,
                "vllm:e2e_request_latency_seconds_sum": e2e,
                "vllm:request_queue_time_seconds_sum": queue,
                "vllm:request_prefill_time_seconds_sum": prefill,
                "vllm:request_decode_time_seconds_sum": decode,
                "vllm:time_to_first_token_seconds_sum": ttft,
                "vllm:prompt_tokens_total": p,
                "vllm:prompt_tokens_cached_total": c,
                "vllm:generation_tokens_total": g}
    w = vllm_work(page(10, 1, 2, 5, 1, 9, 1000, 100, 50),
                  page(30, 3, 6, 15, 5, 49, 9000, 6100, 1050))
    assert (w.requests, w.queue_s, w.prefill_s, w.decode_s) == (
        20, 2, 4, 10)
    assert (w.ttft_mean_s, w.e2e_mean_s) == (0.2, 2.0)
    assert (w.prompt_tokens, w.cached_prompt_tokens) == (8000, 6000)
    assert w.generation_tokens == 1000
    assert w.decode_tokens_per_s == 100
    idle = vllm_work(page(10, 1, 2, 5, 1, 9, 1, 1, 1),
                     page(10, 1, 2, 5, 1, 9, 1, 1, 1))
    assert idle.requests == 0 and idle.ttft_mean_s is None
