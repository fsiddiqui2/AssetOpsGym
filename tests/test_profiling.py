"""The profiler folds spans into the report's numbers; the sinks only fire when configured."""

import json
import threading
import time

from assetopsgym.profiling import GPUSampler, Profiler, flatten


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
