"""Timing instrumentation for AssetOpsGym rollouts (T10).

Every stage of a rollout is a Span: the environment step, each MCP tool call (a
gym tool or a proxied AssetOpsBench server call), the simulator's tick loop, the
CouchDB writes that mirror it, and on the agent side each LLM request. A Profiler
collects the spans of one episode and folds them into the numbers the report
needs: wall time per stage, tokens per simulated hour, simulated hours per wall
second, and the LLM / environment split.

    from assetopsgym.profiling import Profiler

    prof = Profiler(episode_id="...")
    with prof.span("llm.chat", prompt_tokens=812, completion_tokens=40):
        ...
    prof.summary()          # dict, JSON-serializable
    prof.emit(variant="refleak_mid", model="...")   # JSONL + W&B sinks, when configured

Always on: a span costs two perf_counter reads and a list append, so the server
and the agent wrap their stages unconditionally. Sinks are opt-in by environment:

    AOG_PROFILE_DIR=runs/     append one JSON line per episode to <dir>/profile.jsonl
    WANDB_PROJECT=assetopsgym log the flattened summary to Weights & Biases
                              (needs `uv sync --extra profile`; WANDB_RUN_NAME, WANDB_MODE
                              and the rest of wandb's own environment apply)

Span names are dotted; the first segment is the stage. The ones the code emits:

    env.reset   env.step   env.end          server side, per OpenEnv call
    tool.<name>                             the gym's own tool bodies (advance_time, ...)
    sim.advance                             the pure simulator (attrs: sim_hours, rows)
    sim.run_to_end                          finish()'s fast-forward (attrs: sim_hours_skipped)
    couch.write_rows  couch.sync_wo  couch.flush_wo
    mcp.aob                                 a proxied AssetOpsBench call (attrs: server, tool)
    llm.chat                                agent side (attrs: prompt_tokens, completion_tokens)
    client.step                             agent side, one env.step over HTTP (attrs: tool)

GPU utilisation and memory come from `GPUSampler`, which polls nvidia-smi on the
serving host; on a machine without one it reports `available: False`.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


@dataclass
class Span:
    name: str
    start: float
    end: float = 0.0
    parent: str | None = None
    thread: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return self.end - self.start

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


def _pct(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


class Profiler:
    """Collects spans for one episode; thread-safe; nesting is tracked per thread."""

    def __init__(self, episode_id: str | None = None, max_spans: int = 200_000):
        self.max_spans = max_spans
        self._lock = threading.Lock()
        self._local = threading.local()
        self.reset(episode_id)

    def reset(self, episode_id: str | None = None) -> None:
        with self._lock:
            self.episode_id = episode_id
            self.spans: list[Span] = []
            self.dropped = 0
            self.t0 = time.perf_counter()
            self.meta: dict[str, Any] = {}

    # ---- recording ----------------------------------------------------------
    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        s = Span(name, time.perf_counter(), parent=stack[-1] if stack else None,
                 thread=threading.get_ident(), attrs=attrs)
        stack.append(name)
        try:
            yield s
        finally:
            s.end = time.perf_counter()
            stack.pop()
            with self._lock:
                if len(self.spans) < self.max_spans:
                    self.spans.append(s)
                else:
                    self.dropped += 1

    def timed(self, name: str, **attrs: Any) -> Callable:
        """Decorator form of span()."""
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*a, **kw):
                with self.span(name, **attrs):
                    return fn(*a, **kw)
            return wrapper
        return deco

    def add(self, name: str, seconds: float, **attrs: Any) -> None:
        """Record a span measured elsewhere (e.g. a server-reported latency)."""
        now = time.perf_counter()
        with self._lock:
            if len(self.spans) < self.max_spans:
                self.spans.append(Span(name, now - seconds, now, attrs=attrs))
            else:
                self.dropped += 1

    # ---- folding ------------------------------------------------------------
    def total(self, prefix: str) -> float:
        with self._lock:
            return sum(s.seconds for s in self.spans if s.name == prefix or s.name.startswith(prefix + "."))

    def attr_sum(self, key: str) -> float:
        with self._lock:
            return sum(float(s.attrs[key]) for s in self.spans if isinstance(s.attrs.get(key), (int, float)))

    def summary(self) -> dict:
        with self._lock:
            spans = list(self.spans)
            wall = (max((s.end for s in spans), default=self.t0) - self.t0)
            meta, episode_id, dropped = dict(self.meta), self.episode_id, self.dropped
        by_name: dict[str, dict] = {}
        groups: dict[str, list[float]] = {}
        for s in spans:
            groups.setdefault(s.name, []).append(s.seconds)
        for name, vals in sorted(groups.items()):
            vals.sort()
            by_name[name] = {"count": len(vals), "seconds": round(sum(vals), 4),
                             "mean_ms": round(1000 * sum(vals) / len(vals), 2),
                             "p50_ms": round(1000 * _pct(vals, 0.5), 2),
                             "p95_ms": round(1000 * _pct(vals, 0.95), 2),
                             "max_ms": round(1000 * vals[-1], 2)}

        def tot(prefix: str) -> float:
            return sum(s.seconds for s in spans if s.name == prefix or s.name.startswith(prefix + "."))

        def asum(key: str) -> float:
            return sum(float(s.attrs[key]) for s in spans if isinstance(s.attrs.get(key), (int, float)))

        llm_s, env_s = tot("llm"), (tot("client.step") or tot("env.step"))
        sim_hours = asum("sim_hours")
        prompt, completion = asum("prompt_tokens"), asum("completion_tokens")
        tokens = prompt + completion
        derived = {
            "wall_s": round(wall, 3),
            "llm_s": round(llm_s, 3), "env_s": round(env_s, 3),
            "sim_s": round(tot("sim"), 3), "couch_s": round(tot("couch"), 3), "aob_s": round(tot("mcp"), 3),
            "llm_fraction": round(llm_s / (llm_s + env_s), 3) if llm_s + env_s else None,
            "sim_hours": round(sim_hours, 2),
            # fast-forwarded by finish(); no agent turns spent on it
            "sim_hours_skipped": round(asum("sim_hours_skipped"), 2),
            "sim_hours_per_wall_s": round(sim_hours / wall, 3) if wall else None,
            "llm_calls": sum(1 for s in spans if s.name.startswith("llm.")),
            "tool_calls": sum(1 for s in spans if s.name in ("client.step", "env.step")),
            "prompt_tokens": int(prompt), "completion_tokens": int(completion),
            "tokens_per_sim_hour": round(tokens / sim_hours, 1) if sim_hours else None,
            "tokens_per_llm_s": round(tokens / llm_s, 1) if llm_s else None,
        }
        return {"episode_id": episode_id, "meta": meta, "derived": derived, "by_name": by_name,
                "spans": len(spans), "dropped": dropped}

    # ---- sinks --------------------------------------------------------------
    def emit(self, **extra: Any) -> dict:
        """Fold, then write to every configured sink. Returns the summary."""
        self.meta.update(extra)
        s = self.summary()
        d = os.environ.get("AOG_PROFILE_DIR")
        if d:
            path = Path(d)
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "profile.jsonl", "a") as f:
                f.write(json.dumps(s, default=str) + "\n")
        run = wandb_run()
        if run is not None:
            run.log(flatten({"profile": {"derived": s["derived"], "by_name": s["by_name"], **s["meta"]}}))
        return s


def flatten(d: dict, prefix: str = "", sep: str = "/") -> dict:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
    return out


_wandb_run: Any = None
_wandb_tried = False


def wandb_run():
    """The process-wide W&B run, started on first use when WANDB_PROJECT is set; else None."""
    global _wandb_run, _wandb_tried
    if _wandb_run is not None or _wandb_tried:
        return _wandb_run
    _wandb_tried = True
    if not os.environ.get("WANDB_PROJECT"):
        return None
    try:
        import wandb
    except ImportError:
        return None
    _wandb_run = wandb.run or wandb.init(project=os.environ["WANDB_PROJECT"], job_type="profile",
                                         config={k: v for k, v in os.environ.items() if k.startswith("AOG_")})
    return _wandb_run


class GPUSampler:
    """Polls nvidia-smi for utilisation and memory on a background thread.

        g = GPUSampler(interval_s=1.0).start()
        ...
        g.stop(); g.summary()  # {"available": True, "samples": n, "util_mean": .., "mem_used_mb_max": ..}

    W&B's own system metrics cover the same thing when the run lives on the GPU host; this
    is for runs where the agent and the environment sit on a machine other than the server.
    """

    QUERY = "--query-gpu=index,utilization.gpu,memory.used,memory.total"

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self.available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> list[dict]:
        if not self.available:
            return []
        try:
            out = subprocess.run(["nvidia-smi", self.QUERY, "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            self.available = False
            return []
        rows = []
        for line in out.strip().splitlines():
            try:
                idx, util, used, total = (float(x) for x in line.split(","))
            except ValueError:
                continue
            rows.append({"t": time.time(), "gpu": int(idx), "util": util, "mem_used_mb": used, "mem_total_mb": total})
        return rows

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.extend(self.sample_once())
            self._stop.wait(self.interval_s)

    def start(self) -> "GPUSampler":
        if self.available and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.interval_s + 5)

    def summary(self) -> dict:
        if not self.samples:
            return {"available": self.available, "samples": 0}
        util = [s["util"] for s in self.samples]
        mem = [s["mem_used_mb"] for s in self.samples]
        return {"available": True, "samples": len(self.samples), "gpus": len({s["gpu"] for s in self.samples}),
                "util_mean": round(sum(util) / len(util), 1), "util_max": max(util),
                "mem_used_mb_mean": round(sum(mem) / len(mem)), "mem_used_mb_max": max(mem),
                "mem_total_mb": self.samples[-1]["mem_total_mb"]}
