"""AssetOpsGym as an OpenEnv MCPEnvironment.

The agent sees two kinds of MCP tools, all routed through step():
  * AssetOpsBench's own IoT and work-order tools, proxied to real server
    processes reading this episode's CouchDB databases;
  * the gym's tools: current_time, advance_time, finish;
  * the alert tools: list_alerts, acknowledge_alert, shelve_alert.

The world is frozen between calls. advance_time is the only thing that moves the
clock: the simulator runs, new telemetry is written to CouchDB (only the past is
ever stored), and the maintenance organization acts on the work orders the agent
filed through the WO server. The episode ends at the 60-day horizon or on
finish(); the final observation carries the score as its reward.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
import uuid
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Optional

from fastmcp import FastMCP
from fastmcp.tools import Tool as FastMCPTool
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

from ..profiling import Profiler
from .aob import AOBBridge, Couch
from .scenarios import VARIANTS, reference_costs, score
from .sim import (ASSET, FAILURE_CODES, INSTALLED_SENSORS, MAX_SHELVE_H,
                  SITE, Sim, Variant, iso)

IOT_TOOLS = ("sites", "asset_ids", "asset_detail", "assets", "installed_sensors", "measured_sensors",
             "stream_extent", "history", "latest_reading", "sensor_stats", "sensor_coverage")
WO_TOOLS = ("list_workorders", "get_workorder", "get_failure_codes", "generate_work_order",
            "update_workorder", "cancel_workorder")
MAX_ADVANCE_H = 168

INSTRUCTION = """You are the reliability engineer on call for site {site}. {asset} (chilled-water plant) streams its sensors every 15 minutes into the IoT store; history starts {history_start}.

It is now {now}. Your shift covers the next {days} days, until {end}. Simulated time is frozen between your actions and moves only when you call advance_time(hours), at most {max_h} hours per call. Reading data is free.

The building management system raises alerts, listed by list_alerts and summarized by advance_time. Operator alarms fire once a rule's condition has held for 2-3 hours; analytics alerts fire instantly and can be noisy.

acknowledge_alert marks an alert as seen; shelve_alert hides a rule's new alerts for up to {max_shelve} hours, and shelved alerts are never shown to you.

If you believe a fault is developing, open a corrective work order for asset {asset} at site {site} with generate_work_order and set its failure code with update_workorder (get_failure_codes lists them). The maintenance planner approves work orders and dispatches a technician; follow progress with get_workorder. The technician performs the work named by the failure code. Every technician visit costs money, and an unplanned chiller trip costs far more than planned work.

When you are done, call finish; the rest of the {days} days is then simulated without you."""


class GymState(State):
    variant: str = ""
    seed: int = 0  # replays the run: reset(variant, seed)
    sim_time: str = ""
    finished: bool = False
    score: Optional[float] = None


def pick_variant(variant: Optional[str],
                 seed: Optional[int]) -> Variant:
    """Neither given: a seed from the OS, then as if given."""
    if variant is None and seed is None:
        seed = secrets.randbelow(2**31)
    if variant is None:
        ids = sorted(VARIANTS)
        variant = ids[seed % len(ids)]
    v = VARIANTS[variant]
    if seed is None:
        return v
    return replace(v, seed=seed)  # also keys the reference_costs cache


def _wo_time(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S+00:00")  # the WO server's own date format


class _ForwardTool(FastMCPTool):
    """An AssetOpsBench tool re-exposed by the gym's own MCP server, so it is reachable
    both through step() and through OpenEnv's direct /mcp route."""

    forward: SkipJsonSchema[Any] = Field(exclude=True)

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        data = await asyncio.to_thread(self.forward, self.name, arguments)
        return ToolResult(content=[TextContent(type="text", text=json.dumps(data, default=str))],
                          structured_content={"result": data})


_WRAPPED_OUTPUT = {"type": "object", "properties": {"result": {}}, "required": ["result"],
                   "x-fastmcp-wrap-result": True}


def _replace_dates(obj: Any, value: str) -> Any:
    if isinstance(obj, dict):
        return {k: value if k in ("reportdate", "status_date") and v else _replace_dates(v, value)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_dates(x, value) for x in obj]
    return obj


class AssetOpsGymEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, max_steps: int = 400):
        mcp = FastMCP("assetopsgym")
        env = self

        # The clock tools run in a worker thread so CouchDB writes don't block the server's
        # event loop (other sessions keep stepping); the lock keeps this episode's simulator
        # single-threaded.
        @mcp.tool
        async def current_time() -> dict:
            """Current simulated time and the end of your shift (ISO 8601, UTC)."""
            return await asyncio.to_thread(env._locked, env._clock)

        @mcp.tool
        async def advance_time(hours: float) -> dict:
            """Let simulated time pass, between 0.25 and 168 hours. New telemetry lands in the IoT
            store and the maintenance organization works its queue. Returns the new time, the alerts
            raised meanwhile and any work-order updates."""
            return await asyncio.to_thread(env._locked, env._advance, hours)

        @mcp.tool
        async def list_alerts(since: Optional[str] = None,
                        kind: Optional[str] = None, limit: int = 50,
                        unacked_only: bool = False) -> dict:
            """Building-management alerts raised so far, oldest
            first. Shelved alerts are hidden.

            Args:
                since: optional ISO 8601 lower bound on alert time.
                kind: optional filter, "operator_alarm" (condition held 2-3 h) or "analytics" (instant, noisy).
                limit: maximum alerts returned, newest kept (1-200).
                unacked_only: only alerts not yet acknowledged.
            """
            return await asyncio.to_thread(
                env._locked, env._alerts, since, kind, limit,
                unacked_only)

        @mcp.tool
        async def acknowledge_alert(alert_id: str) -> dict:
            """Mark an alert as seen, by its id (e.g. "AL17")."""
            return await asyncio.to_thread(env._locked, env._ack, alert_id)

        @mcp.tool
        async def shelve_alert(rule: str, hours: float) -> dict:
            """Hide new alerts of one rule (e.g.
            "low_condenser_water_flow") for 0.25-168 hours. The rule
            keeps evaluating; its alerts are recorded, not shown."""
            return await asyncio.to_thread(
                env._locked, env._shelve, rule, hours)

        @mcp.tool
        async def finish() -> dict:
            """End your shift now. The rest of the horizon is simulated without you and the episode is scored."""
            return await asyncio.to_thread(env._locked, env._finish)

        super().__init__(mcp)
        self.max_steps = max_steps
        self._lock = threading.Lock()
        self.prof = Profiler()  # one episode at a time; reset() starts a new collection
        with self.prof.span("env.boot"):
            self.couch = Couch(f"aog_{uuid.uuid4().hex[:10]}")
            self.couch.recreate()
            self.bridge = AOBBridge(self.couch.server_env())
        self.proxied: dict[str, str] = {}  # tool name -> AssetOpsBench server
        for server, allow in (("iot", IOT_TOOLS), ("wo", WO_TOOLS)):
            for t in self.bridge.list_tools(server):
                if t.name in allow:
                    self.proxied[t.name] = server
                    mcp.add_tool(_ForwardTool(name=t.name, description=t.description or "",
                                              parameters=t.inputSchema, output_schema=_WRAPPED_OUTPUT,
                                              forward=self._call_proxied))
        self.sim: Sim | None = None
        self._result: dict | None = None
        self._state = GymState(episode_id=str(uuid.uuid4()))

    # ---- OpenEnv API ------------------------------------------------------
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              variant: Optional[str] = None, **kwargs: Any) -> Observation:
        v = pick_variant(variant, seed)
        episode_id = episode_id or str(uuid.uuid4())
        boot = self.prof.total("env.boot")
        self.prof.reset(episode_id)
        self.prof.meta.update(variant=v.id, boot_s=round(boot, 3))
        with self.prof.span("env.reset", variant=v.id):
            with self.prof.span("sim.history"):
                self.sim = Sim(v)
            self._result = None
            with self.prof.span("couch.recreate"):
                self.couch.recreate()
                self.couch.bulk("asset", [_asset_doc()])
                self.couch.bulk("fc", [{"_id": f"fc:{c.code}", "code": c.code, "description": c.description,
                                        "dataset": "failure_code"} for c in FAILURE_CODES.values()])
            with self.prof.span("couch.write_rows", rows=len(self.sim.history)):
                self._write_rows(self.sim.history)
            with self.prof.span("sim.reference_runs"):
                reference_costs(v)  # warm the score's reference runs now, not at the last step
        self._state = GymState(
            episode_id=episode_id, step_count=0, variant=v.id,
            seed=v.seed, sim_time=iso(self.sim.now))
        text = INSTRUCTION.format(site=SITE, asset=ASSET, now=iso(self.sim.now), end=iso(v.end),
                                  days=v.horizon_days, max_h=MAX_ADVANCE_H,
                                  max_shelve=MAX_SHELVE_H,
                                  history_start=iso(v.start - timedelta(days=v.history_days)))
        return Observation(done=False, reward=0.0, metadata={
            "instruction": text, "site": SITE, "asset": ASSET, "episode_id": episode_id,
            "seed": v.seed, "sim_time": iso(self.sim.now),
            "shift_end": iso(v.end)})

    # Rewards and termination only flow through step(), OpenEnv's simulation path;
    # direct /mcp calls reach the same tools without them.
    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        with self.prof.span("env.step", tool=getattr(action, "tool_name", type(action).__name__)):
            obs = super().step(action, timeout_s=timeout_s, **kwargs)
        return self._finalize(obs)

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        with self.prof.span("env.step", tool=getattr(action, "tool_name", type(action).__name__)):
            obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        return self._finalize(obs)

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return Observation(done=False, reward=0.0, metadata={
            "error": f"Unknown action type {type(action).__name__}; use ListToolsAction or CallToolAction."})

    @property
    def state(self) -> GymState:
        return self._state

    def close(self) -> None:
        self.bridge.close()
        self.couch.drop()
        super().close()

    # ---- tools ------------------------------------------------------------
    def _clock(self) -> dict:
        if self.sim is None:
            return {"error": "call reset first"}
        left = (self.sim.v.end - self.sim.now).total_seconds() / 3600
        return {"sim_time": iso(self.sim.now), "shift_end": iso(self.sim.v.end), "hours_left": left}

    def _advance(self, hours) -> dict:
        if self.sim is None:
            return {"error": "call reset first"}
        if self._result is not None:
            return {"error": "the episode is over"}
        try:
            hours = float(hours)
        except (TypeError, ValueError):
            return {"error": "hours must be a number"}
        if not 0.25 <= hours <= MAX_ADVANCE_H:
            return {"error": f"hours must be between 0.25 and {MAX_ADVANCE_H}"}
        with self.prof.span("tool.advance_time", hours=hours):
            with self.prof.span("couch.sync_wo"):
                self._sync_work_orders()
            seen, before = len(self.sim.alerts), self.sim.now
            with self.prof.span("sim.advance") as sp:
                rows = self.sim.advance(hours)
                sp.set(sim_hours=(self.sim.now - before).total_seconds() / 3600, rows=len(rows))
            with self.prof.span("couch.write_rows", rows=len(rows)):
                self._write_rows(rows)
            with self.prof.span("couch.flush_wo"):
                updates = self._flush_changes()
        alerts = [a.public() for a in self.sim.alerts[seen:] if not a.shelved]
        out = {"sim_time": iso(self.sim.now), "shift_end": iso(self.sim.v.end),
               "sim_hours_advanced": (self.sim.now - before).total_seconds() / 3600,
               "new_telemetry_rows": len(rows), "new_alert_count": len(alerts), "new_alerts": alerts[-30:],
               "work_order_updates": updates, "shift_over": self.sim.done}
        self._state.sim_time = iso(self.sim.now)
        if self.sim.done:
            self._end("horizon reached")
        return out

    def _alerts(self, since, kind, limit, unacked_only) -> dict:
        if self.sim is None:
            return {"error": "call reset first"}
        items = [a.public() for a in self.sim.visible_alerts()
                 if (kind is None or a.kind == kind)
                 and (since is None or iso(a.time) >= since)
                 and not (unacked_only and a.acked_at)]
        limit = max(1, min(200, int(limit)))
        return {"total": len(items), "returned": min(limit, len(items)), "alerts": items[-limit:]}

    def _ack(self, alert_id: str) -> dict:
        if self.sim is None:
            return {"error": "call reset first"}
        try:
            a = self.sim.acknowledge(str(alert_id))
        except KeyError as e:
            return {"error": str(e)}
        return {"id": a.id, "acked_at": iso(a.acked_at)}

    def _shelve(self, rule: str, hours: float) -> dict:
        if self.sim is None:
            return {"error": "call reset first"}
        try:
            until = self.sim.shelve(str(rule), float(hours))
        except (KeyError, ValueError, TypeError) as e:
            return {"error": str(e)}
        return {"rule": rule, "shelved_until": iso(until)}

    def _finish(self) -> dict:
        if self.sim is None:
            return {"error": "call reset first"}
        if self._result is None:
            with self.prof.span("tool.finish"):
                self._end("agent finished")
        return {"message": "Shift ended; the remaining time was simulated.", "score": self._result["score"]}

    # ---- plumbing ---------------------------------------------------------
    def _locked(self, fn, *args):
        with self._lock:
            return fn(*args)

    def _call_proxied(self, name: str, arguments: dict) -> Any:
        if self.sim is None or self._result is not None:
            raise RuntimeError("no running episode; call reset")
        with self._lock, self.prof.span("mcp.aob", server=self.proxied[name], tool=name):
            data = self.bridge.call(self.proxied[name], name, arguments)
            if name == "update_workorder" and arguments.get("notes") is not None:
                self._keep_tech_log(arguments, data)
            return self._fix_clock(name, data)

    def _keep_tech_log(self, arguments: dict, data: Any) -> None:
        """The WO server's update_workorder(notes=...) replaces the long description, which is
        also where the planner and technicians write. Rebuild it: their log, then the agent's note."""
        wonum = arguments.get("wonum")
        wo = self.sim.wos.get(wonum)
        doc = self.couch.get("wo", f"wo:{(arguments.get('site_id') or SITE).upper()}:{wonum}")
        if doc is None:
            return
        log = "\n".join(wo.notes) if wo else ""
        doc["description_longdescription"] = (log + "\n" if log else "") + \
            f"[{iso(self.sim.now)}] Agent: {arguments['notes']}"
        self.couch.put("wo", doc)
        if isinstance(data, dict) and isinstance(data.get("work_order"), dict):
            data["work_order"]["description_longdescription"] = doc["description_longdescription"]

    def _fix_clock(self, name: str, data: Any) -> Any:
        """AssetOpsBench servers read the wall clock; rewrite to simulated time."""
        if not isinstance(data, dict):
            return data
        now = self.sim.now
        if name == "latest_reading" and data.get("timestamp"):
            data["age_seconds"] = (now - datetime.fromisoformat(data["timestamp"])).total_seconds()
        if name in ("generate_work_order", "cancel_workorder") and data.get("wonum"):
            doc = self.couch.get("wo", f"wo:{(data.get('siteid') or SITE).upper()}:{data['wonum']}")
            if doc:
                key = "reportdate" if name == "generate_work_order" else "status_date"
                doc[key] = _wo_time(now)
                self.couch.put("wo", doc)
            data = _replace_dates(data, _wo_time(now))
        return data

    def _sync_work_orders(self) -> None:
        """Feed the agent's work-order writes (made through the WO server) into the simulator."""
        for doc in self.couch.find("wo", {"type": "workorder"}):
            wonum, cancel = doc["wonum"], doc.get("status") == "CAN"
            if wonum in self.sim.wos:
                self.sim.update_wo(wonum, code=doc.get("failurecode"), priority=doc.get("wopriority"),
                                   cancel=cancel)
            elif not cancel and doc.get("assetnum") == ASSET:
                self.sim.file_wo(doc.get("failurecode"), doc.get("wopriority", 3), wonum=wonum)

    def _flush_changes(self) -> list[dict]:
        """Write the maintenance organization's work-order updates back to CouchDB."""
        updates = []
        for c in self.sim.changes:
            doc_id = f"wo:{SITE}:{c['wonum']}"
            if c.get("new"):
                doc = {"_id": doc_id, "type": "workorder", "schema_version": "1.0.0", "wonum": c["wonum"],
                       "siteid": SITE, "description": c["description"], "assetnum": ASSET,
                       "wopriority": c["priority"], "worktype": c["work_type"], "status": c["status"],
                       "reportdate": _wo_time(c["reportdate"]), "reportedby": c["reported_by"],
                       "failurecode": c["failurecode"]}
                self.couch.put("wo", doc)
                updates.append({"wonum": c["wonum"], "status": c["status"], "note": c["description"]})
                continue
            doc = self.couch.get("wo", doc_id)
            if doc is None:
                continue
            doc["status"], doc["wopriority"] = c["status"], c["priority"]
            doc["status_date"] = _wo_time(c["time"])
            if "labor" in c:
                tech, start = c["labor"]
                doc.setdefault("wplabor", []).append({"laborcode": tech, "craft": "HVAC", "startdate": _wo_time(start)})
            if "finish" in c:
                doc["actfinish"], doc["actlabhrs"] = _wo_time(c["finish"]), c["hours"]
            if c.get("failurecode"):
                doc["failurecode"] = c["failurecode"]
            if c.get("notes"):
                log = doc.get("description_longdescription") or ""
                doc["description_longdescription"] = (log + "\n" if log else "") + f"[{iso(c['time'])}] {c['notes']}"
            self.couch.put("wo", doc)
            updates.append({"wonum": c["wonum"], "status": c["status"], "time": iso(c["time"]), "note": c.get("notes")})
        self.sim.changes.clear()
        return updates

    def _write_rows(self, rows: list[dict]) -> None:
        self.couch.bulk("iot", [{"_id": f"iot:{ASSET}:{r['timestamp']}", "asset_id": ASSET,
                                 "timestamp": r["timestamp"], **r["values"]} for r in rows])

    def _end(self, reason: str) -> None:
        with self.prof.span("env.end", reason=reason):
            with self.prof.span("couch.sync_wo"):
                self._sync_work_orders()
            before = self.sim.now
            with self.prof.span("sim.run_to_end") as sp:
                self.sim.run_to_end()  # ponytail: rows after an early finish are not written; nobody can read them
                sp.set(sim_hours_skipped=(self.sim.now - before)
                       .total_seconds() / 3600)
            self.sim.changes.clear()
            r = self.sim.result()
            reactive, best = reference_costs(self.sim.v)
            r.update(score=score(self.sim.v, r["cost_total"]), end_reason=reason,
                     reference_costs={"reactive_practice": reactive, "oracle": best})
        self._result = r
        self._state.finished, self._state.score = True, r["score"]
        self._state.sim_time = iso(self.sim.now)

    def _finalize(self, obs: Observation) -> Observation:
        # _end expects the lock held (it runs inside locked tools)
        with self._lock:
            if (self.sim is not None and self._result is None
                    and self._state.step_count >= self.max_steps):
                self._end("step limit")
        if self._result is not None:
            if "profile" not in self._result:  # first observation after the end: fold and emit once
                self._result["profile"] = self.prof.emit(side="server", episode_id=self._state.episode_id,
                                                         steps=self._state.step_count, score=self._result["score"],
                                                         end_reason=self._result["end_reason"])
            obs.done, obs.reward = True, self._result["score"]
            obs.metadata = {**obs.metadata, "result": self._result}
        return obs


def _asset_doc() -> dict:
    return {"_id": f"asset:{ASSET}", "assetnum": ASSET, "description": f"{ASSET} (chilled-water plant)",
            "siteid": SITE, "orgid": "RIVPLANT", "status": "OPERATING", "assettype": "CHILLER",
            "location": "MAIN-MECH-CH6", "sensors": INSTALLED_SENSORS, "dataset": "asset"}
