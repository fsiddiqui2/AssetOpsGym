"""Chiller 6 world for AssetOpsGym.

Telemetry, one latent fault, BMS alerts, the maintenance organization (planner
and technicians) and the cost ledger. Pure Python and deterministic for a given
Variant: the OpenEnv environment mirrors telemetry and work orders into CouchDB
for the AssetOpsBench MCP servers, while baselines and the score's reference
runs drive the Sim directly.

Time only moves in advance(). Rows are stamped on 15-minute ticks, and at sim
time `now` only rows strictly before `now` exist.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..schemas import (AlertActionEvent, AlertEvent, CostEvent, DowntimeEvent,
                       FailureEvent, LedgerEvent, ServiceRequestEvent)

TICK = timedelta(minutes=15)
TICK_H = 0.25
SITE, ASSET = "MAIN", "Chiller 6"
CAPACITY_TONS = 800.0
DESIGN_FLOW_GPM = 2400.0
SETPOINT_F = 44.0


def _s(name: str) -> str:
    return f"{ASSET} {name}"


COND_RET = _s("Condenser Water Return To Tower Temperature")
EFFICIENCY = _s("Chiller Efficiency")
TONNAGE = _s("Tonnage")
SUPPLY = _s("Supply Temperature")
RETURN = _s("Return Temperature")
FLOW = _s("Condenser Water Flow")
SCHEDULE = _s("Schedule")
POWER = _s("Power Input")
LOADED = _s("Chiller % Loaded")
REFRIG = _s("Liquid Refrigerant Evaporator Temperature")
SETPOINT = _s("Setpoint Temperature")
RUN = _s("Run Status")

# Same installed-vs-measured gap as AssetOpsBench: Oil Pressure is installed, never measured.
INSTALLED_SENSORS = [COND_RET, EFFICIENCY, TONNAGE, SUPPLY, RETURN, FLOW, SCHEDULE,
                     POWER, LOADED, REFRIG, SETPOINT, _s("Oil Pressure")]


@dataclass(frozen=True)
class FailureCode:
    code: str
    description: str
    repair_cost: float
    repair_hours: float
    parts_hours: float = 0.0


FAILURE_CODES = {fc.code: fc for fc in [
    FailureCode("CH-CWFLOW", "condenser water flow restriction (clogged strainer or fouled condenser tubes)", 2500, 8),
    FailureCode("CH-REFLEAK", "refrigerant leak or low refrigerant charge", 5000, 12),
    FailureCode("CH-COMPWEAR", "compressor mechanical wear", 12000, 16, parts_hours=48),
    FailureCode("CH-EVAPFOUL", "evaporator tube fouling", 3000, 8),
    FailureCode("CH-SENSOR", "sensor drift or failed transmitter", 800, 3),
    FailureCode("CH-CONTROLS", "chiller control or setpoint fault", 1200, 4),
]}
FAULT_MODES = ("CH-CWFLOW", "CH-REFLEAK", "CH-COMPWEAR")

# Cost model in USD. ponytail: flat priors; sweep them once the scenario suite settles.
DISPATCH_COST = 1500.0
PLANNED_DOWNTIME_PER_H = 400.0
UNPLANNED_DOWNTIME_PER_H = 2000.0
EMERGENCY_MULTIPLIER = 2.0
ENERGY_PRICE_PER_KWH = 0.15  # charged on the extra kW a degraded chiller draws
INSPECTION_HOURS = 3.0
RESPONSE_HOURS = {1: 4, 2: 12, 3: 36, 4: 72, 5: 120}
NOSHOW_DELAY_H = 24
CONTRACTOR_DELAY_H = 48  # nobody on the crew has the skill: an outside contractor is called in
SLA_GRACE_H = 4  # response later than the priority's target plus this grace is an SLA breach
OPEN = ("WAPPR", "APPR")


@dataclass(frozen=True)
class Technician:
    id: str
    skills: tuple[str, ...] | None = None  # failure codes this technician can repair; None = all
    travel_h: float = 0.0

    def can(self, code: str | None) -> bool:
        return self.skills is None or code is None or code in self.skills


DEFAULT_CREW = (Technician("TECH-01"), Technician("TECH-02"))

# BMS rules: metric (higher is worse) and operator threshold (needs the
# 2-hour on-delay). Analytics alerts fire when the metric is Z_ALERT sigma
# above its own last 24 h. Evaluated while running.
ON_DELAY_TICKS = 8
ANALYTICS_COOLDOWN = timedelta(hours=6)
Z_WINDOW_TICKS = 96
Z_ALERT = 3.0
MAX_SHELVE_H = 168
RULES = [
    ("low_condenser_water_flow", FLOW,
     lambda v: 1 - v[FLOW] / DESIGN_FLOW_GPM, 0.34,
     "condenser water flow below design"),
    ("high_evaporator_approach", REFRIG, lambda v: v[SUPPLY] - v[REFRIG], 11.0,
     "evaporator approach (supply minus refrigerant temperature) high"),
    ("compressor_load_mismatch", LOADED,
     lambda v: v[LOADED] - 100 * v[TONNAGE] / CAPACITY_TONS, 19.0,
     "compressor % loaded high for the tonnage delivered"),
    ("supply_above_setpoint", SUPPLY, lambda v: v[SUPPLY] - v[SETPOINT], 3.0,
     "chilled water supply above setpoint"),
    ("high_kw_per_ton", EFFICIENCY,
     lambda v: v[EFFICIENCY] if v[LOADED] > 50 else 0.0, 0.95,
     "kW per ton high at load"),
]


@dataclass(frozen=True)
class Variant:
    id: str
    mode: str | None
    onset_day: float = 0.0
    pf_days: float = 20.0
    blips_per_day: float = 1.0
    noshow: float = 0.05
    seed: int = 0
    start: datetime = datetime(2026, 6, 1)
    history_days: int = 14
    horizon_days: int = 60
    crew: tuple[Technician, ...] = DEFAULT_CREW

    @property
    def onset(self) -> datetime | None:
        return self.start + timedelta(days=self.onset_day) if self.mode else None

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=self.horizon_days)


def iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


@dataclass
class Alert:
    id: str
    time: datetime
    kind: str  # "operator_alarm" | "analytics"
    rule: str
    sensor: str
    value: float
    message: str
    shelved: bool = False  # raised while its rule was shelved: hidden
    acked_at: datetime | None = None

    def public(self) -> dict:
        return {"id": self.id, "time": iso(self.time), "kind": self.kind,
                "rule": self.rule, "site": SITE, "asset": ASSET,
                "sensor": self.sensor, "value": round(self.value, 3),
                "message": self.message, "acked": self.acked_at is not None}


@dataclass
class ZWindow:
    """The last Z_WINDOW_TICKS values of one metric, as running sums."""
    xs: deque = field(default_factory=deque)
    total: float = 0.0
    sq: float = 0.0

    def z(self, x: float) -> float | None:
        n = len(self.xs)
        if n < Z_WINDOW_TICKS:
            return None
        mu = self.total / n
        sd = math.sqrt(max(0.0, self.sq / n - mu * mu))
        return (x - mu) / max(sd, 1e-9)

    def push(self, x: float) -> None:
        self.xs.append(x)
        self.total, self.sq = self.total + x, self.sq + x * x
        if len(self.xs) > Z_WINDOW_TICKS:
            y = self.xs.popleft()
            self.total, self.sq = self.total - y, self.sq - y * y


@dataclass
class RuleState:
    on: int = 0  # consecutive ticks above the operator threshold
    off: int = 0
    active: bool = False
    last_analytics: datetime | None = None
    window: ZWindow = field(default_factory=ZWindow)


@dataclass
class WorkOrder:
    wonum: str
    code: str | None
    priority: int
    created: datetime
    rng: random.Random
    emergency: bool = False
    status: str = "WAPPR"
    approve_at: datetime | None = None
    jitter_h: float = 0.0
    arrive_at: datetime | None = None
    start_at: datetime | None = None
    finish_at: datetime | None = None
    started: bool = False
    failed_at_start: bool = False
    dispatched: bool = False
    tech: str | None = None
    target_arrive: datetime | None = None  # the priority's response target; later is an SLA breach
    sla_breach: bool = False
    notes: list[str] = field(default_factory=list)  # the technician / planner log, as written to the WO


class Sim:
    def __init__(self, variant: Variant):
        self.v = variant
        self.rng = random.Random(variant.seed)
        self.now = variant.start - timedelta(days=variant.history_days)
        # The active fault. Starts as the scenario's; inject_fault() can replace it (T6).
        self.mode, self.onset, self.pf_days = variant.mode, variant.onset, variant.pf_days
        self.fault_active = variant.mode is not None
        self.failed = False
        self.ever_failed = False
        self.failed_at: datetime | None = None
        self._offline_since: datetime | None = None
        self._weather = 0.0
        self._excess_kw = 0.0
        self._blip: tuple[int, int] | None = None  # (kind, ticks left)
        self.rules = {r[0]: RuleState() for r in RULES}
        self.shelved_until: dict[str, datetime] = {}
        self.alerts: list[Alert] = []
        self.wos: dict[str, WorkOrder] = {}
        self.changes: list[dict] = []  # work-order updates for CouchDB, drained by the env
        # Totals drive the score; the event ledger (schemas.LedgerEvent) is the audit trail
        # and the five KPIs are folds over it. Downtime and excess energy are emitted as
        # spans and daily sums, so the ledger stays a few hundred events, not one per tick.
        self.ledger = {"dispatch": 0.0, "repair": 0.0, "planned_downtime": 0.0, "unplanned_downtime": 0.0,
                       "excess_energy": 0.0}
        self.hours = {"planned_downtime": 0.0, "unplanned_downtime": 0.0}
        self.events: list[LedgerEvent] = []
        self._down: tuple[str, datetime] | None = None  # (planned|unplanned, span start)
        self._energy_acc, self._energy_day = 0.0, self.now.date()
        self._closed = False
        self.history = self._run_until(variant.start)

    # ---- public API -------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.now >= self.v.end

    def advance(self, hours: float) -> list[dict]:
        ticks = max(1, round(hours / TICK_H))
        return self._run_until(min(self.now + ticks * TICK, self.v.end))

    def run_to_end(self) -> list[dict]:
        return self._run_until(self.v.end)

    def severity(self, t: datetime) -> float:
        onset = self.onset
        if not self.fault_active or onset is None or t < onset:
            return 0.0
        return min(1.0, ((t - onset) / timedelta(days=self.pf_days)) ** 1.6)

    def inject_fault(self, mode: str, onset: datetime, pf_days: float) -> None:
        """Start (or replace) the latent fault at run time. The onset may be in the past."""
        if mode not in FAULT_MODES:
            raise ValueError(f"unknown fault mode {mode!r}; one of {FAULT_MODES}")
        self.mode, self.onset, self.pf_days = mode, onset, pf_days
        self.fault_active = True

    def file_wo(self, code: str | None, priority: int = 3, wonum: str | None = None) -> WorkOrder:
        idx = len(self.wos)
        rng = random.Random(self.v.seed * 7919 + idx)
        wo = WorkOrder(wonum or f"SIM{idx + 1}", _norm_code(code), _clamp_prio(priority), self.now, rng)
        wo.approve_at = self.now + rng.randint(8, 48) * TICK
        wo.jitter_h = rng.randint(0, 16) * TICK_H
        self.wos[wo.wonum] = wo
        self._event(ServiceRequestEvent, wo, actor="agent", note="Service request filed.")
        return wo

    def update_wo(self, wonum: str, code: str | None = None, priority: int | None = None,
                  cancel: bool = False) -> None:
        wo = self.wos.get(wonum)
        if wo is None or wo.status not in OPEN:
            return  # ponytail: changes after the technician arrives are ignored
        if cancel:
            wo.status = "CAN"
            self._event(ServiceRequestEvent, wo, actor="agent", note="Cancelled by the agent.")
            return
        changed = False
        if code is not None and _norm_code(code) != wo.code:
            wo.code, changed = _norm_code(code), True
        if priority is not None and _clamp_prio(priority) != wo.priority:
            wo.priority, changed = _clamp_prio(priority), True
            if wo.status == "APPR":
                self._schedule(wo)
        if changed:
            self._event(ServiceRequestEvent, wo, actor="agent", note="Updated by the agent.")

    def acknowledge(self, alert_id: str) -> Alert:
        a = next((a for a in self.alerts if a.id == alert_id), None)
        if a is None:
            raise KeyError(f"unknown alert {alert_id!r}")
        if a.acked_at is None:
            a.acked_at = self.now
            self.events.append(AlertActionEvent(
                time=self.now, action="ack", rule=a.rule, alert_id=a.id))
        return a

    def shelve(self, rule: str, hours: float) -> datetime:
        if rule not in self.rules:
            raise KeyError(f"unknown rule {rule!r}; one of {list(self.rules)}")
        if not 0.25 <= hours <= MAX_SHELVE_H:
            raise ValueError(f"hours must be between 0.25 and {MAX_SHELVE_H}")
        until = self.now + timedelta(hours=hours)
        self.shelved_until[rule] = until
        self.events.append(AlertActionEvent(
            time=self.now, action="shelve", rule=rule, hours=hours))
        return until

    def visible_alerts(self) -> list[Alert]:
        return [a for a in self.alerts if not a.shelved]

    def state(self) -> dict:
        """schemas.SimState. Ground truth included: for the environment and tests, never the agent."""
        return {"sim_time": self.now, "start": self.v.start, "end": self.v.end,
                "running": not self.failed and not self._repairing(), "failed": self.failed,
                "fault_active": self.fault_active, "severity": self.severity(self.now),
                "work_orders": [{"wonum": w.wonum, "code": w.code, "priority": w.priority, "status": w.status,
                                 "emergency": w.emergency, "created": w.created, "tech": w.tech,
                                 "arrive_at": w.arrive_at, "start_at": w.start_at, "finish_at": w.finish_at,
                                 "dispatched": w.dispatched} for w in self.wos.values()]}

    def kpis(self) -> dict:
        """The proposal's five KPIs, folded over the ledger (schemas.KPIs)."""
        dispatched = [w for w in self.wos.values() if w.dispatched]
        return {"downtime_hours": round(sum(self.hours.values()), 2),
                "missed_failures": int(self.ever_failed),
                "false_dispatches": sum(self.mode is None or w.code != self.mode for w in dispatched),
                "alert_load": len(self.alerts),
                "sla_breaches": sum(w.sla_breach for w in self.wos.values())}

    def result(self) -> dict:
        v, mode, onset = self.v, self.mode, self.onset
        dispatched = [w for w in self.wos.values() if w.dispatched]
        false = [w for w in dispatched if mode is None or w.code != mode]
        agent_hits = [w for w in self.wos.values() if not w.emergency and w.code == mode
                      and onset is not None and w.created >= onset and w.status != "CAN"]
        delay = min(((w.created - onset).total_seconds() / 3600 for w in agent_hits), default=None)
        return {
            "variant": v.id, "mode": mode, "onset": iso(onset) if onset else None,
            "failed": self.ever_failed, "failed_at": iso(self.failed_at) if self.failed_at else None,
            "failure_prevented": bool(mode) and not self.ever_failed,
            "cost_total": round(sum(self.ledger.values()), 2),
            "cost_breakdown": {k: round(x, 2) for k, x in self.ledger.items()},
            "downtime_hours": dict(self.hours),
            "dispatches": len(dispatched), "false_dispatches": len(false),
            "work_orders": len(self.wos),
            "detection_delay_hours": None if delay is None else round(delay, 2),
            "alerts": {k: sum(a.kind == k for a in self.alerts) for k in ("operator_alarm", "analytics")},
            "kpis": self.kpis(), "events": len(self.events),
        }

    # ---- simulation -------------------------------------------------------
    def _run_until(self, target: datetime) -> list[dict]:
        rows = []
        while self.now < target:
            rows.append(self._tick(self.now))
            self.now += TICK
        if self.now >= self.v.end and not self._closed:
            self._closed = True
            self._close_downtime(self.now)
            self._flush_energy(self.now)
        return rows

    def _repairing(self) -> bool:
        return any(w.started and w.status == "INPRG" and w.code for w in self.wos.values())

    def _tick(self, t: datetime) -> dict:
        for wo in list(self.wos.values()):
            self._service(wo, t)
        if not self.failed and self.severity(t) >= 1.0:
            self._fail(t)
        repairing = self._repairing()
        running = not self.failed and not repairing
        if running:
            self._offline_since = None
        elif self._offline_since is None:
            self._offline_since = t
        kind = "unplanned" if self.failed else "planned" if repairing else None
        if self._down and self._down[0] != kind:
            self._close_downtime(t)
        if kind and not self._down:
            self._down = (kind, t)
        if self.failed:
            self._charge("unplanned_downtime", UNPLANNED_DOWNTIME_PER_H)
        elif repairing:
            self._charge("planned_downtime", PLANNED_DOWNTIME_PER_H)
        values = self._values(t, running)
        if t.date() != self._energy_day:
            self._flush_energy(t)
        if running:
            energy = self._excess_kw * TICK_H * ENERGY_PRICE_PER_KWH
            self.ledger["excess_energy"] += energy
            self._energy_acc += energy
            self._evaluate_rules(t, values)
        return {"timestamp": iso(t), "values": values}

    def _charge(self, kind: str, per_hour: float) -> None:
        self.ledger[kind] += per_hour * TICK_H
        self.hours[kind] += TICK_H

    def _close_downtime(self, t: datetime) -> None:
        if not self._down:
            return
        kind, start = self._down
        self._down = None
        hours = (t - start).total_seconds() / 3600
        if hours <= 0:
            return
        rate = UNPLANNED_DOWNTIME_PER_H if kind == "unplanned" else PLANNED_DOWNTIME_PER_H
        self.events.append(DowntimeEvent(time=t, kind=kind, hours=hours, start=start))
        self.events.append(CostEvent(time=t, kind=f"{kind}_downtime", amount=hours * rate))

    def _flush_energy(self, t: datetime) -> None:
        if self._energy_acc > 0:
            self.events.append(CostEvent(time=t, kind="excess_energy", amount=self._energy_acc))
        self._energy_acc, self._energy_day = 0.0, t.date()

    def _alert(self, t: datetime, kind: str, rule: str, sensor: str, value: float, message: str) -> None:
        aid = f"AL{len(self.alerts) + 1}"
        shelved = t < self.shelved_until.get(rule, t)
        self.alerts.append(Alert(aid, t, kind, rule, sensor, value,
                                 message, shelved))
        self.events.append(AlertEvent(
            time=t, kind=kind, rule=rule, sensor=sensor, value=value,
            message=message, id=aid, shelved=shelved))

    def _event(self, cls, wo: WorkOrder, **fields) -> None:
        self.events.append(cls(time=self.now, wonum=wo.wonum, status=wo.status, priority=wo.priority,
                               code=wo.code, emergency=wo.emergency, tech=wo.tech, sla_breach=wo.sla_breach,
                               **fields))

    def _values(self, t: datetime, running: bool) -> dict:
        r = self.rng
        n = [r.gauss(0, 1) for _ in range(10)]  # fixed draw count keeps runs aligned across policies
        blip_u, blip_kind, blip_len = r.random(), r.randrange(5), r.randint(1, 3)
        h = t.hour + t.minute / 60
        weekend = t.weekday() >= 5
        lo, hi = (8, 18) if weekend else (6, 22)
        occupied = lo <= h < hi
        shape = math.sin(math.pi * (h - lo) / (hi - lo)) if occupied else 0.0
        base = 0.25 + (0.35 if weekend else 0.55) * shape if occupied else 0.20
        self._weather = 0.995 * self._weather + 0.01 * n[0]
        load = min(1.0, max(0.12, base + self._weather + 0.02 * n[1]))
        amb = 3 * math.sin(2 * math.pi * (h - 9) / 24)

        if self._blip is None and running and blip_u < self.v.blips_per_day * TICK_H / 24:
            self._blip = (blip_kind, blip_len)

        if not running:
            off_h = (t - self._offline_since).total_seconds() / 3600 if self._offline_since else 0.0
            supply = min(SETPOINT_F + 2 * off_h, 62.0) + 0.3 * n[2]
            v = {COND_RET: 80 + amb + 0.5 * n[6], EFFICIENCY: 0.0, TONNAGE: 0.0, SUPPLY: supply,
                 RETURN: supply + 0.5 + 0.2 * n[3], FLOW: 0.0, POWER: 4 + 0.5 * n[7], LOADED: 0.0,
                 REFRIG: supply - 1 + 0.3 * n[9], RUN: 0}
        else:
            s = self.severity(t)
            sf, sl, sw = (s if self.mode == m else 0.0 for m in FAULT_MODES)
            supply = SETPOINT_F + 0.3 + 0.25 * n[2] + 6 * sl ** 2 * load + 4 * sf ** 3 * load
            tons = CAPACITY_TONS * load * (1 + 0.01 * n[4])
            flow = DESIGN_FLOW_GPM * (1 - 0.45 * sf) * (1 + 0.02 * n[5])
            fault_kwpt = 0.30 * sf + 0.12 * sl + 0.25 * sw
            self._excess_kw = tons * fault_kwpt
            kwpt = 0.56 + 0.18 * (1 - load) ** 2 + fault_kwpt + 0.015 * n[7]
            loaded = 100 * load + 25 * sw + 1.5 * n[8]
            refrig = supply - (3 + 2 * load + 9 * sl) + 0.4 * n[9]
            if self._blip:  # transient sensor glitch, shorter than the operator on-delay
                kind = self._blip[0]
                flow *= 0.68 if kind == 0 else 1.0
                refrig -= 6.5 if kind == 1 else 0.0
                loaded += 16 if kind == 2 else 0.0
                supply += 3.5 if kind == 3 else 0.0
                kwpt += 0.4 if kind == 4 else 0.0
                left = self._blip[1] - 1
                self._blip = (kind, left) if left > 0 else None
            v = {COND_RET: 84 + amb + 9.5 * load * DESIGN_FLOW_GPM / max(flow, 1.0) + 0.5 * n[6],
                 EFFICIENCY: kwpt, TONNAGE: tons, SUPPLY: supply, RETURN: supply + 10 * load + 0.4 * n[3],
                 FLOW: flow, POWER: tons * kwpt, LOADED: loaded, REFRIG: refrig, RUN: 1}
        v[SCHEDULE] = 1 if occupied else 0
        v[SETPOINT] = SETPOINT_F
        return {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}

    def _evaluate_rules(self, t: datetime, v: dict) -> None:
        for rule, sensor, metric, op_thr, msg in RULES:
            m = metric(v)
            st = self.rules[rule]
            z = st.window.z(m)
            st.window.push(m)
            last = st.last_analytics
            cool = last is None or t - last >= ANALYTICS_COOLDOWN
            if z is not None and z > Z_ALERT and cool:
                self._alert(t, "analytics", rule, sensor, m,
                            f"Analytics: {msg}, {z:.1f} sigma"
                            " above its 24 h mean")
                st.last_analytics = t
            if m > op_thr:
                st.on, st.off = st.on + 1, 0
            else:
                st.on, st.off = 0, st.off + 1
            if not st.active and st.on >= ON_DELAY_TICKS:
                st.active = True
                self._alert(t, "operator_alarm", rule, sensor, m, f"Operator alarm: {msg} for 2 hours")
            elif st.active and st.off >= ON_DELAY_TICKS:
                st.active = False

    def _fail(self, t: datetime) -> None:
        self.failed, self.ever_failed, self.failed_at = True, True, t
        self.events.append(FailureEvent(time=t, mode=self.mode))
        self._alert(t, "operator_alarm", "chiller_trip", RUN, 0.0, "Operator alarm: Chiller 6 tripped offline")
        # The plant's current reactive practice: a trip always gets an emergency work order.
        mode = self.mode
        if any(w.code == mode and w.status == "INPRG" for w in self.wos.values()):
            return
        pending = [w for w in self.wos.values() if w.code == mode and w.status in OPEN]
        if pending:
            w = pending[0]
            w.priority, w.emergency = 1, True  # a tripped chiller always gets emergency response
            if w.status == "WAPPR":
                w.approve_at = min(w.approve_at, t)
            else:
                w.arrive_at = min(w.arrive_at, t + timedelta(hours=RESPONSE_HOURS[1]))
                w.target_arrive = min(w.target_arrive, t + timedelta(hours=RESPONSE_HOURS[1] + SLA_GRACE_H))
            self._change(w, actor="planner", notes="Escalated to priority 1 after chiller trip.")
            return
        idx = len(self.wos)
        w = WorkOrder(f"EM{idx + 1}", mode, 1, t, random.Random(self.v.seed * 7919 + idx), emergency=True)
        w.approve_at = t
        self.wos[w.wonum] = w
        self.changes.append({"wonum": w.wonum, "new": True, "status": "WAPPR", "priority": 1,
                             "work_type": "EM", "reportdate": t, "failurecode": mode,
                             "description": "EMERGENCY: Chiller 6 tripped offline",
                             "reported_by": "BMS"})
        self._event(ServiceRequestEvent, w, actor="bms", note="Emergency work order after chiller trip.")

    def _assign(self, wo: WorkOrder) -> tuple[Technician, float]:
        """Pick a technician for the work order: one with the skill if the crew has one, else anyone
        plus the contractor delay. Returns the technician and the extra hours before arrival."""
        crew = [t for t in self.v.crew if t.can(wo.code)]
        if crew:
            tech = wo.rng.choice(crew)
            return tech, tech.travel_h
        tech = wo.rng.choice(list(self.v.crew))
        return tech, tech.travel_h + CONTRACTOR_DELAY_H

    def _schedule(self, wo: WorkOrder) -> None:
        """(Re)compute the arrival time and SLA target from the approval time and priority.
        The technician is picked once, at approval; a priority change keeps them."""
        if wo.tech is None:
            tech, extra = self._assign(wo)
            wo.tech = tech.id
        else:
            tech = next(t for t in self.v.crew if t.id == wo.tech)
            extra = tech.travel_h + (0 if tech.can(wo.code) else CONTRACTOR_DELAY_H)
        response = RESPONSE_HOURS[wo.priority] + extra + (0 if wo.emergency else wo.jitter_h)
        wo.arrive_at = wo.approve_at + timedelta(hours=response)
        wo.target_arrive = wo.approve_at + timedelta(hours=RESPONSE_HOURS[wo.priority] + tech.travel_h + SLA_GRACE_H)

    def _service(self, wo: WorkOrder, t: datetime) -> None:
        while True:
            if wo.status == "WAPPR" and t >= wo.approve_at:
                wo.status = "APPR"
                self._schedule(wo)
                skilled = any(tech.id == wo.tech and tech.can(wo.code) for tech in self.v.crew)
                how = "" if skilled else " (outside contractor; nobody on the crew has this skill)"
                self._change(wo, actor="planner", labor=(wo.tech, wo.arrive_at),
                             notes=f"Approved by planner; {wo.tech} scheduled for {iso(wo.arrive_at)}{how}.")
            elif wo.status == "APPR" and t >= wo.arrive_at:
                if not wo.emergency and wo.rng.random() < self.v.noshow:
                    wo.arrive_at += timedelta(hours=NOSHOW_DELAY_H)
                    self._change(wo, actor="planner",
                                 notes=f"Technician did not arrive; rescheduled to {iso(wo.arrive_at)}.")
                    return
                wo.status, wo.dispatched = "INPRG", True
                wo.sla_breach = wo.target_arrive is not None and t > wo.target_arrive
                self.ledger["dispatch"] += DISPATCH_COST
                self.events.append(CostEvent(time=t, kind="dispatch", amount=DISPATCH_COST, wonum=wo.wonum))
                fc = FAILURE_CODES.get(wo.code)
                if fc:
                    wo.start_at = wo.arrive_at + timedelta(hours=fc.parts_hours)
                    wo.finish_at = wo.start_at + timedelta(hours=fc.repair_hours)
                    wait = f" Waiting {fc.parts_hours:.0f} h for parts." if fc.parts_hours else ""
                    self._change(wo, actor="technician", notes=f"{wo.tech} on site for {fc.description}.{wait}")
                else:
                    wo.start_at = wo.arrive_at
                    wo.finish_at = wo.arrive_at + timedelta(hours=INSPECTION_HOURS)
                    self._change(wo, actor="technician",
                                 notes=f"{wo.tech} on site; no failure code given, general inspection only.")
            elif wo.status == "INPRG" and not wo.started and t >= wo.start_at:
                wo.started, wo.failed_at_start = True, self.failed
            elif wo.status == "INPRG" and wo.started and t >= wo.finish_at:
                self._complete(wo, t)
            else:
                return

    def _complete(self, wo: WorkOrder, t: datetime) -> None:
        wo.status = "COMP"
        fc = FAILURE_CODES.get(wo.code)
        if fc is None:
            self._change(wo, actor="technician", finish=t, hours=INSPECTION_HOURS,
                         notes="General inspection complete: no actionable fault identified.")
            return
        cost = fc.repair_cost * (EMERGENCY_MULTIPLIER if wo.failed_at_start else 1.0)
        self.ledger["repair"] += cost
        self.events.append(CostEvent(time=t, kind="repair", amount=cost, wonum=wo.wonum))
        fixed = wo.code == self.mode and self.severity(t) > 0
        if fixed:
            self.fault_active = False
            self.failed = False
            finding = f"Confirmed {fc.description}; repaired and returned to service."
        else:
            finding = f"Performed {fc.description} work as requested; no matching fault found."
        self._change(wo, actor="technician", finish=t, hours=fc.repair_hours, failurecode=wo.code, notes=finding)

    def _change(self, wo: WorkOrder, actor: str = "planner", **fields) -> None:
        """Record a work-order transition: for CouchDB (changes), the ledger (events) and the
        work order's own log (notes), which the environment uses to keep the technician's
        entries when the agent writes notes of its own."""
        self.changes.append({"wonum": wo.wonum, "status": wo.status, "priority": wo.priority,
                             "time": self.now, **fields})
        if fields.get("notes"):
            wo.notes.append(f"[{iso(self.now)}] {fields['notes']}")
        self._event(ServiceRequestEvent, wo, actor=actor, note=fields.get("notes"))


def _norm_code(code: str | None) -> str | None:
    c = (code or "").strip().upper()
    return c if c in FAILURE_CODES else None


def _clamp_prio(p) -> int:
    try:
        return min(5, max(1, int(p)))
    except (TypeError, ValueError):
        return 3
