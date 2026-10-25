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
from dataclasses import dataclass, field
from datetime import datetime, timedelta

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
TECHS = ("TECH-01", "TECH-02")
OPEN = ("WAPPR", "APPR")

# BMS rules: metric (higher is worse), analytics threshold (instant, noisy),
# operator threshold (needs the 2-hour on-delay). Evaluated while running.
ON_DELAY_TICKS = 8
ANALYTICS_COOLDOWN = timedelta(hours=6)
RULES = [
    ("low_condenser_water_flow", FLOW, lambda v: 1 - v[FLOW] / DESIGN_FLOW_GPM, 0.20, 0.34,
     "condenser water flow below design"),
    ("high_evaporator_approach", REFRIG, lambda v: v[SUPPLY] - v[REFRIG], 7.8, 11.0,
     "evaporator approach (supply minus refrigerant temperature) high"),
    ("compressor_load_mismatch", LOADED, lambda v: v[LOADED] - 100 * v[TONNAGE] / CAPACITY_TONS, 11.0, 19.0,
     "compressor % loaded high for the tonnage delivered"),
    ("supply_above_setpoint", SUPPLY, lambda v: v[SUPPLY] - v[SETPOINT], 2.2, 3.0,
     "chilled water supply above setpoint"),
    ("high_kw_per_ton", EFFICIENCY, lambda v: v[EFFICIENCY] if v[LOADED] > 50 else 0.0, 0.85, 0.95,
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
    time: datetime
    kind: str  # "operator_alarm" | "analytics"
    rule: str
    sensor: str
    value: float
    message: str

    def public(self) -> dict:
        return {"time": iso(self.time), "kind": self.kind, "rule": self.rule, "site": SITE,
                "asset": ASSET, "sensor": self.sensor, "value": round(self.value, 3),
                "message": self.message}


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
    notes: list[str] = field(default_factory=list)


class Sim:
    def __init__(self, variant: Variant):
        self.v = variant
        self.rng = random.Random(variant.seed)
        self.now = variant.start - timedelta(days=variant.history_days)
        self.fault_active = variant.mode is not None
        self.failed = False
        self.ever_failed = False
        self.failed_at: datetime | None = None
        self._offline_since: datetime | None = None
        self._weather = 0.0
        self._excess_kw = 0.0
        self._blip: tuple[int, int] | None = None  # (kind, ticks left)
        self._alarm = {r[0]: {"on": 0, "off": 0, "active": False, "last_an": None} for r in RULES}
        self.alerts: list[Alert] = []
        self.wos: dict[str, WorkOrder] = {}
        self.changes: list[dict] = []  # work-order updates for CouchDB, drained by the env
        self.ledger = {"dispatch": 0.0, "repair": 0.0, "planned_downtime": 0.0, "unplanned_downtime": 0.0,
                       "excess_energy": 0.0}
        self.hours = {"planned_downtime": 0.0, "unplanned_downtime": 0.0}
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
        onset = self.v.onset
        if not self.fault_active or onset is None or t < onset:
            return 0.0
        return min(1.0, ((t - onset) / timedelta(days=self.v.pf_days)) ** 1.6)

    def file_wo(self, code: str | None, priority: int = 3, wonum: str | None = None) -> WorkOrder:
        idx = len(self.wos)
        rng = random.Random(self.v.seed * 7919 + idx)
        wo = WorkOrder(wonum or f"SIM{idx + 1}", _norm_code(code), _clamp_prio(priority), self.now, rng)
        wo.approve_at = self.now + rng.randint(8, 48) * TICK
        wo.jitter_h = rng.randint(0, 16) * TICK_H
        self.wos[wo.wonum] = wo
        return wo

    def update_wo(self, wonum: str, code: str | None = None, priority: int | None = None,
                  cancel: bool = False) -> None:
        wo = self.wos.get(wonum)
        if wo is None or wo.status not in OPEN:
            return  # ponytail: changes after the technician arrives are ignored
        if cancel:
            wo.status = "CAN"
            return
        if code is not None:
            wo.code = _norm_code(code)
        if priority is not None and _clamp_prio(priority) != wo.priority:
            wo.priority = _clamp_prio(priority)
            if wo.status == "APPR":
                wo.arrive_at = wo.approve_at + timedelta(hours=RESPONSE_HOURS[wo.priority] + wo.jitter_h)

    def result(self) -> dict:
        v = self.v
        dispatched = [w for w in self.wos.values() if w.dispatched]
        false = [w for w in dispatched if v.mode is None or w.code != v.mode]
        agent_hits = [w for w in self.wos.values() if not w.emergency and w.code == v.mode
                      and v.onset is not None and w.created >= v.onset and w.status != "CAN"]
        delay = min(((w.created - v.onset).total_seconds() / 3600 for w in agent_hits), default=None)
        return {
            "variant": v.id, "mode": v.mode, "onset": iso(v.onset) if v.onset else None,
            "failed": self.ever_failed, "failed_at": iso(self.failed_at) if self.failed_at else None,
            "failure_prevented": bool(v.mode) and not self.ever_failed,
            "cost_total": round(sum(self.ledger.values()), 2),
            "cost_breakdown": {k: round(x, 2) for k, x in self.ledger.items()},
            "downtime_hours": dict(self.hours),
            "dispatches": len(dispatched), "false_dispatches": len(false),
            "work_orders": len(self.wos),
            "detection_delay_hours": None if delay is None else round(delay, 2),
            "alerts": {k: sum(a.kind == k for a in self.alerts) for k in ("operator_alarm", "analytics")},
        }

    # ---- simulation -------------------------------------------------------
    def _run_until(self, target: datetime) -> list[dict]:
        rows = []
        while self.now < target:
            rows.append(self._tick(self.now))
            self.now += TICK
        return rows

    def _tick(self, t: datetime) -> dict:
        for wo in list(self.wos.values()):
            self._service(wo, t)
        if not self.failed and self.severity(t) >= 1.0:
            self._fail(t)
        repairing = any(w.started and w.status == "INPRG" and w.code for w in self.wos.values())
        running = not self.failed and not repairing
        if running:
            self._offline_since = None
        elif self._offline_since is None:
            self._offline_since = t
        if self.failed:
            self._charge("unplanned_downtime", UNPLANNED_DOWNTIME_PER_H)
        elif repairing:
            self._charge("planned_downtime", PLANNED_DOWNTIME_PER_H)
        values = self._values(t, running)
        if running:
            self.ledger["excess_energy"] += self._excess_kw * TICK_H * ENERGY_PRICE_PER_KWH
            self._evaluate_rules(t, values)
        return {"timestamp": iso(t), "values": values}

    def _charge(self, kind: str, per_hour: float) -> None:
        self.ledger[kind] += per_hour * TICK_H
        self.hours[kind] += TICK_H

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
            sf, sl, sw = (s if self.v.mode == m else 0.0 for m in FAULT_MODES)
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
        for rule, sensor, metric, an_thr, op_thr, msg in RULES:
            m = metric(v)
            st = self._alarm[rule]
            if m > an_thr and (st["last_an"] is None or t - st["last_an"] >= ANALYTICS_COOLDOWN):
                self.alerts.append(Alert(t, "analytics", rule, sensor, m, f"Analytics: {msg}"))
                st["last_an"] = t
            if m > op_thr:
                st["on"], st["off"] = st["on"] + 1, 0
            else:
                st["on"], st["off"] = 0, st["off"] + 1
            if not st["active"] and st["on"] >= ON_DELAY_TICKS:
                st["active"] = True
                self.alerts.append(Alert(t, "operator_alarm", rule, sensor, m,
                                         f"Operator alarm: {msg} for 2 hours"))
            elif st["active"] and st["off"] >= ON_DELAY_TICKS:
                st["active"] = False

    def _fail(self, t: datetime) -> None:
        self.failed, self.ever_failed, self.failed_at = True, True, t
        self.alerts.append(Alert(t, "operator_alarm", "chiller_trip", RUN, 0.0,
                                 "Operator alarm: Chiller 6 tripped offline"))
        # The plant's current reactive practice: a trip always gets an emergency work order.
        mode = self.v.mode
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
            self._change(w, notes="Escalated to priority 1 after chiller trip.")
            return
        idx = len(self.wos)
        w = WorkOrder(f"EM{idx + 1}", mode, 1, t, random.Random(self.v.seed * 7919 + idx), emergency=True)
        w.approve_at = t
        self.wos[w.wonum] = w
        self.changes.append({"wonum": w.wonum, "new": True, "status": "WAPPR", "priority": 1,
                             "work_type": "EM", "reportdate": t, "failurecode": mode,
                             "description": "EMERGENCY: Chiller 6 tripped offline",
                             "reported_by": "BMS"})

    def _service(self, wo: WorkOrder, t: datetime) -> None:
        while True:
            if wo.status == "WAPPR" and t >= wo.approve_at:
                wo.status = "APPR"
                wo.tech = wo.rng.choice(TECHS)
                response = RESPONSE_HOURS[wo.priority] + (0 if wo.emergency else wo.jitter_h)
                wo.arrive_at = wo.approve_at + timedelta(hours=response)
                self._change(wo, labor=(wo.tech, wo.arrive_at),
                             notes=f"Approved by planner; {wo.tech} scheduled for {iso(wo.arrive_at)}.")
            elif wo.status == "APPR" and t >= wo.arrive_at:
                if not wo.emergency and wo.rng.random() < self.v.noshow:
                    wo.arrive_at += timedelta(hours=NOSHOW_DELAY_H)
                    self._change(wo, notes=f"Technician did not arrive; rescheduled to {iso(wo.arrive_at)}.")
                    return
                wo.status, wo.dispatched = "INPRG", True
                self.ledger["dispatch"] += DISPATCH_COST
                fc = FAILURE_CODES.get(wo.code)
                if fc:
                    wo.start_at = wo.arrive_at + timedelta(hours=fc.parts_hours)
                    wo.finish_at = wo.start_at + timedelta(hours=fc.repair_hours)
                    wait = f" Waiting {fc.parts_hours:.0f} h for parts." if fc.parts_hours else ""
                    self._change(wo, notes=f"{wo.tech} on site for {fc.description}.{wait}")
                else:
                    wo.start_at = wo.arrive_at
                    wo.finish_at = wo.arrive_at + timedelta(hours=INSPECTION_HOURS)
                    self._change(wo, notes=f"{wo.tech} on site; no failure code given, general inspection only.")
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
            self._change(wo, finish=t, hours=INSPECTION_HOURS,
                         notes="General inspection complete: no actionable fault identified.")
            return
        self.ledger["repair"] += fc.repair_cost * (EMERGENCY_MULTIPLIER if wo.failed_at_start else 1.0)
        fixed = wo.code == self.v.mode and self.severity(t) > 0
        if fixed:
            self.fault_active = False
            self.failed = False
            finding = f"Confirmed {fc.description}; repaired and returned to service."
        else:
            finding = f"Performed {fc.description} work as requested; no matching fault found."
        self._change(wo, finish=t, hours=fc.repair_hours, failurecode=wo.code, notes=finding)

    def _change(self, wo: WorkOrder, **fields) -> None:
        self.changes.append({"wonum": wo.wonum, "status": wo.status, "priority": wo.priority,
                             "time": self.now, **fields})


def _norm_code(code: str | None) -> str | None:
    c = (code or "").strip().upper()
    return c if c in FAILURE_CODES else None


def _clamp_prio(p) -> int:
    try:
        return min(5, max(1, int(p)))
    except (TypeError, ValueError):
        return 3
