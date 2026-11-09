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
import logging
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
from pydantic_core import to_jsonable_python
from pydantic.json_schema import SkipJsonSchema
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

from ..profiling import Profiler
from ..schemas import (AckResult, AdvanceTimeResult, AlertKind,
                       ClockResult, EpisodeResult, ErrorCode,
                       ErrorResult, FinishResult, GymError,
                       ListAlertsResult, ServiceRequestEvent,
                       ShelveResult, WorkOrderUpdate)
from . import mirror
from .aob import AOBBridge, Couch
from .scenarios import VARIANTS, reference_costs, score
from .alerts import MAX_SHELVE_H
from .prompt import INSTRUCTION
from .sim import Sim
from .telemetry import ASSET, SITE
from .tools import MANIFEST, _errors_as_results, check_tools
from .variant import Variant, iso

MAX_ADVANCE_H = 168
logger = logging.getLogger(__name__)


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
        data = to_jsonable_python(await asyncio.to_thread(
            self.forward, self.name, arguments))
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
        mcp = FastMCP(MANIFEST.name)
        env = self

        # The clock tools run in a worker thread so CouchDB writes don't block the server's
        # event loop (other sessions keep stepping); the lock keeps this episode's simulator
        # single-threaded.
        async def current_time() -> ClockResult | ErrorResult:
            """Current simulated time and the end of your shift (ISO 8601, UTC)."""
            return await asyncio.to_thread(env._locked, env._clock)

        async def advance_time(
                hours: float) -> AdvanceTimeResult | ErrorResult:
            """Let simulated time pass, between 0.25 and 168 hours. New telemetry lands in the IoT
            store and the maintenance organization works its queue. Returns the new time, the alerts
            raised meanwhile and any work-order updates."""
            return await asyncio.to_thread(env._locked, env._advance, hours)

        async def list_alerts(
                since: Optional[str] = None,
                kind: Optional[AlertKind] = None,
                limit: int = 50, unacked_only: bool = False
                ) -> ListAlertsResult | ErrorResult:
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

        async def acknowledge_alert(
                alert_id: str) -> AckResult | ErrorResult:
            """Mark an alert as seen, by its id (e.g. "AL17")."""
            return await asyncio.to_thread(env._locked, env._ack, alert_id)

        async def shelve_alert(
                rule: str,
                hours: float) -> ShelveResult | ErrorResult:
            """Hide new alerts of one rule (e.g.
            "low_condenser_water_flow") for 0.25-168 hours. The rule
            keeps evaluating; its alerts are recorded, not shown."""
            return await asyncio.to_thread(
                env._locked, env._shelve, rule, hours)

        async def finish() -> FinishResult | ErrorResult:
            """End your shift now. The rest of the horizon is simulated without you and the episode is scored."""
            return await asyncio.to_thread(env._locked, env._finish)

        gym = (current_time, advance_time, list_alerts,
               acknowledge_alert, shelve_alert, finish)
        for fn in gym:
            mcp.tool(fn)
        super().__init__(mcp)
        self.max_steps = max_steps
        self._lock = threading.Lock()
        self.prof = Profiler()  # one episode at a time; reset() starts a new collection
        with self.prof.span("env.boot"):
            self.couch = Couch(f"aog_{uuid.uuid4().hex[:10]}")
            self.couch.recreate()
            self.bridge = AOBBridge(self.couch.server_env())
        self.proxied: dict[str, str] = {}  # tool name -> AssetOpsBench server
        own = {fn.__name__ for fn in gym}
        for server in self.bridge.sessions:
            for t in self.bridge.list_tools(server):
                if t.name in MANIFEST.tools and t.name not in own:
                    self.proxied[t.name] = server
                    mcp.add_tool(_ForwardTool(name=t.name, description=t.description or "",
                                              parameters=t.inputSchema, output_schema=_WRAPPED_OUTPUT,
                                              forward=self._call_proxied))
        check_tools([*own, *self.proxied], MANIFEST.tools)
        self.sim: Sim | None = None
        self._result: EpisodeResult | None = None
        self._profile: dict | None = None
        self._state = GymState(episode_id=str(uuid.uuid4()))

    # ---- OpenEnv API ------------------------------------------------------
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              variant: Optional[str] = None, **kwargs: Any) -> Observation:
        return self._locked(self._reset, seed, episode_id, variant)

    def _reset(self, seed: Optional[int], episode_id: Optional[str],
               variant: Optional[str]) -> Observation:
        v = pick_variant(variant, seed)
        episode_id = episode_id or str(uuid.uuid4())
        boot = self.prof.total("env.boot")
        self.prof.reset(episode_id)
        self.prof.meta.update(variant=v.id, boot_s=round(boot, 3))
        with self.prof.span("env.reset", variant=v.id):
            with self.prof.span("sim.history"):
                self.sim = Sim(v)
            self._result, self._profile = None, None
            self._wo_seen = 0  # events already mirrored to CouchDB
            with self.prof.span("couch.recreate"):
                self.couch.recreate()
                mirror.seed(self.couch)
            with self.prof.span("couch.write_rows", rows=len(self.sim.history)):
                mirror.write_rows(self.couch, self.sim.history)
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
        with self.prof.span("env.step", tool=getattr(action, "tool_name", type(action).__name__)):
            obs = super().step(action, timeout_s=timeout_s, **kwargs)
        return self._finalize(obs)

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
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
    def _running_sim(self) -> Sim:
        if self.sim is None:
            raise GymError(ErrorCode.NO_EPISODE, "call reset first")
        return self.sim

    @_errors_as_results
    def _clock(self) -> ClockResult:
        sim = self._running_sim()
        left = (sim.v.end - sim.now).total_seconds() / 3600
        return ClockResult(sim_time=sim.now, shift_end=sim.v.end,
                           hours_left=left)

    @_errors_as_results
    def _advance(self, hours: float) -> AdvanceTimeResult:
        self._running_sim()
        if self._result is not None:
            raise GymError(ErrorCode.EPISODE_OVER,
                           "the episode is over")
        if not 0.25 <= hours <= MAX_ADVANCE_H:
            raise GymError(
                ErrorCode.BAD_ARGUMENT,
                f"hours must be between 0.25 and {MAX_ADVANCE_H}")
        with self.prof.span("tool.advance_time", hours=hours):
            with self.prof.span("couch.sync_wo"):
                self._sync_work_orders()
            seen, before = len(self.sim.alerts), self.sim.now
            with self.prof.span("sim.advance") as sp:
                rows = self.sim.advance(hours)
                sp.set(sim_hours=(self.sim.now - before).total_seconds() / 3600, rows=len(rows))
            with self.prof.span("couch.write_rows", rows=len(rows)):
                mirror.write_rows(self.couch, rows)
            with self.prof.span("couch.flush_wo"):
                updates = self._flush_changes()
        alerts = [a.public() for a in self.sim.alerts[seen:] if not a.shelved]
        out = AdvanceTimeResult(
            sim_time=self.sim.now, shift_end=self.sim.v.end,
            sim_hours_advanced=(self.sim.now - before)
            .total_seconds() / 3600,
            new_telemetry_rows=len(rows), new_alert_count=len(alerts),
            new_alerts=alerts[-30:], work_order_updates=updates,
            shift_over=self.sim.done)
        self._state.sim_time = iso(self.sim.now)
        if self.sim.done:
            self._end("horizon reached")
        return out

    @_errors_as_results
    def _alerts(self, since: Optional[str], kind: Optional[AlertKind],
                limit: int, unacked_only: bool) -> ListAlertsResult:
        items = [a.public() for a in self._running_sim().visible_alerts()
                 if (kind is None or a.kind == kind)
                 and (since is None or iso(a.time) >= since)
                 and not (unacked_only and a.acked_at)]
        limit = max(1, min(200, limit))
        return ListAlertsResult(total=len(items),
                                returned=min(limit, len(items)),
                                alerts=items[-limit:])

    @_errors_as_results
    def _ack(self, alert_id: str) -> AckResult:
        try:
            a = self._running_sim().acknowledge(alert_id)
        except KeyError as e:
            raise GymError(ErrorCode.NOT_FOUND, e.args[0]) from e
        return AckResult(id=a.id, acked_at=a.acked_at)

    @_errors_as_results
    def _shelve(self, rule: str, hours: float) -> ShelveResult:
        try:
            until = self._running_sim().shelve(rule, hours)
        except KeyError as e:
            raise GymError(ErrorCode.UNKNOWN_RULE, e.args[0]) from e
        except ValueError as e:
            raise GymError(ErrorCode.BAD_ARGUMENT, str(e)) from e
        return ShelveResult(rule=rule, shelved_until=until)

    @_errors_as_results
    def _finish(self) -> FinishResult:
        self._running_sim()
        if self._result is None:
            with self.prof.span("tool.finish"):
                self._end("agent finished")
        return FinishResult(
            message="Shift ended; the remaining time was simulated.",
            score=self._result.score)

    # ---- plumbing ---------------------------------------------------------
    def _locked(self, fn, *args):
        with self._lock:
            return fn(*args)

    @_errors_as_results
    def _call_proxied(self, name: str, arguments: dict) -> Any:
        self._running_sim()
        if self._result is not None:
            raise GymError(ErrorCode.EPISODE_OVER,
                           "the episode is over; call reset")
        with self._lock, self.prof.span("mcp.aob", server=self.proxied[name], tool=name):
            data = self.bridge.call(self.proxied[name], name, arguments)
            if name == "update_workorder" and arguments.get("notes") is not None:
                self._keep_tech_log(arguments, data)
            return self._fix_clock(name, data)

    def _keep_tech_log(self, arguments: dict, data: Any) -> None:
        """The WO server's update_workorder(notes=...) replaces the long description, which is
        also where the planner and technicians write. Rebuild it: their log, then the agent's note."""
        wonum = arguments.get("wonum")
        doc = self.couch.get("wo", f"wo:{(arguments.get('site_id') or SITE).upper()}:{wonum}")
        if doc is None:
            return
        log = "\n".join(
            _wo_log(e) for e in self.sim.events
            if isinstance(e, ServiceRequestEvent) and e.wonum == wonum
            and e.actor != "agent")
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
            wo = self.sim.wos.get(wonum)
            try:
                if wo is None:
                    if not cancel and doc.get("assetnum") == ASSET:
                        self.sim.file_wo(doc.get("failurecode"),
                                         doc.get("wopriority", 3),
                                         wonum=wonum)
                elif cancel:
                    if not wo.canceled:
                        self.sim.cancel_workorder(wonum, SITE)
                else:
                    self.sim.update_workorder(
                        wonum, SITE, priority=doc.get("wopriority"),
                        failure_code=doc.get("failurecode"))
            except GymError as e:  # ponytail: wave 3b returns it to the agent
                logger.warning("work order %s not applied: %s",
                               wonum, e.message)

    def _flush_changes(self) -> list[WorkOrderUpdate]:
        """Mirror the planner's, crew's and BMS's work-order events to
        CouchDB; the agent's own writes went through the WO server."""
        new = self.sim.events[self._wo_seen:]
        self._wo_seen = len(self.sim.events)
        updates = []
        for e in new:
            if (not isinstance(e, ServiceRequestEvent)
                    or e.actor == "agent"):
                continue
            self.couch.put("wo", self._wo_doc(e))
            updates.append(WorkOrderUpdate(
                wonum=e.wonum, status=e.status, time=e.time,
                note=e.note))
        return updates

    def _wo_doc(self, e: ServiceRequestEvent) -> dict:
        doc = self.couch.get("wo", f"wo:{e.site}:{e.wonum}") or {
            "_id": f"wo:{e.site}:{e.wonum}", "type": "workorder",
            "schema_version": "1.0.0", "wonum": e.wonum,
            "siteid": e.site, "description": e.description,
            "assetnum": e.asset, "worktype": e.work_type,
            "reportdate": _wo_time(e.time),
            "reportedby": e.reported_by}
        doc["status"], doc["wopriority"] = e.status, e.priority
        doc["status_date"] = _wo_time(e.time)
        if e.code:
            doc["failurecode"] = e.code
        if e.arrive_at:
            doc["wplabor"] = [{"laborcode": e.tech, "craft": "HVAC",
                               "startdate": _wo_time(e.arrive_at)}]
        if e.labor_hours is not None:
            doc["actfinish"] = _wo_time(e.time)
            doc["actlabhrs"] = e.labor_hours
        if e.note:
            old = doc.get("description_longdescription")
            doc["description_longdescription"] = (
                f"{old}\n" if old else "") + _wo_log(e)
        return doc

    def _end(self, reason: str) -> None:
        with self.prof.span("env.end", reason=reason):
            with self.prof.span("couch.sync_wo"):
                self._sync_work_orders()
            before = self.sim.now
            with self.prof.span("sim.run_to_end") as sp:
                self.sim.run_to_end()  # ponytail: rows after an early finish are not written; nobody can read them
                sp.set(sim_hours_skipped=(self.sim.now - before)
                       .total_seconds() / 3600)
            r = self.sim.result()
            reactive, best = reference_costs(self.sim.v)
            r = replace(r, score=score(self.sim.v, r.cost_total),
                        end_reason=reason, reference_costs={
                            "reactive_practice": reactive,
                            "oracle": best})
        self._result = r
        self._state.finished, self._state.score = True, r.score
        self._state.sim_time = iso(self.sim.now)

    def _finalize(self, obs: Observation) -> Observation:
        # _end expects the lock held (it runs inside locked tools)
        with self._lock:
            self._state.step_count += 1
            if (self.sim is not None and self._result is None
                    and self._state.step_count >= self.max_steps):
                self._end("step limit")
            r = self._result
            if r is not None and self._profile is None:
                self._profile = self.prof.emit(
                    side="server", episode_id=self._state.episode_id,
                    steps=self._state.step_count, score=r.score,
                    end_reason=r.end_reason)
            profile = self._profile
        if r is not None:
            obs.done, obs.reward = True, r.score
            obs.metadata = {**obs.metadata, "result": {
                **to_jsonable_python(r), "profile": profile}}
        return obs


def _wo_log(e: ServiceRequestEvent) -> str:
    return f"[{iso(e.time)}] {e.note}"
