"""BMS rules and alerts: operator alarms, z-score analytics, ack, shelve.

An operator alarm fires when metric > threshold has held for hold_ticks in
a row with the chiller in the rule's run state (once per such run); an
analytics alert when a running chiller's metric is Z_ALERT sigma above
its own last 24 h.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from ..schemas import AlertActionEvent, AlertEvent, LedgerEvent
from .telemetry import (ASSET, CAPACITY_TONS, DESIGN_FLOW_GPM, EFFICIENCY,
                        FLOW, LOADED, POWER, REFRIG, SETPOINT, SITE, SUPPLY,
                        TONNAGE)
from .variant import TICK_H, iso

ANALYTICS_COOLDOWN = timedelta(hours=6)
Z_WINDOW_TICKS = 96
Z_ALERT = 3.0
MAX_SHELVE_H = 168


@dataclass(frozen=True)
class Rule:
    name: str
    sensor: str
    metric: Callable[[dict], float]  # higher is worse
    threshold: float
    message: str
    hold_ticks: int = 8  # 2 h
    running: bool = True  # the run state the rule watches


RULES = [
    Rule("low_condenser_water_flow", FLOW,
         lambda v: 1 - v[FLOW] / DESIGN_FLOW_GPM, 0.34,
         "condenser water flow below design"),
    Rule("high_evaporator_approach", REFRIG,
         lambda v: v[SUPPLY] - v[REFRIG], 11.0,
         "evaporator approach (supply minus refrigerant temperature) high"),
    Rule("compressor_load_mismatch", LOADED,
         lambda v: v[LOADED] - 100 * v[TONNAGE] / CAPACITY_TONS, 19.0,
         "compressor % loaded high for the tonnage delivered"),
    Rule("supply_above_setpoint", SUPPLY,
         lambda v: v[SUPPLY] - v[SETPOINT], 3.0,
         "chilled water supply above setpoint"),
    Rule("high_kw_per_ton", EFFICIENCY,
         lambda v: v[EFFICIENCY] if v[LOADED] > 50 else 0.0, 0.95,
         "kW per ton high at load"),
    # the next two are from AssetOpsBench's chiller monitoring rules
    Rule("supply_below_setpoint", SUPPLY,
         lambda v: v[SETPOINT] - v[SUPPLY], 2.0,
         "chilled water supply below setpoint"),
    Rule("power_while_off", POWER, lambda v: v[POWER], 5.0,
         "power input above 5 kW while the chiller is off",
         hold_ticks=12, running=False),
]


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
    run: int = 0  # consecutive running ticks above the threshold
    last_analytics: datetime | None = None
    window: ZWindow = field(default_factory=ZWindow)


@dataclass
class AlertState:
    rules: dict[str, RuleState] = field(
        default_factory=lambda: {r.name: RuleState() for r in RULES})
    shelved_until: dict[str, datetime] = field(default_factory=dict)
    alerts: list[Alert] = field(default_factory=list)


def raise_alert(st: AlertState, events: list[LedgerEvent], t: datetime,
                kind: str, rule: str, sensor: str, value: float,
                message: str) -> None:
    aid = f"AL{len(st.alerts) + 1}"
    shelved = t < st.shelved_until.get(rule, t)
    st.alerts.append(Alert(aid, t, kind, rule, sensor, value,
                           message, shelved))
    events.append(AlertEvent(
        time=t, kind=kind, rule=rule, sensor=sensor, value=value,
        message=message, id=aid, shelved=shelved))


def evaluate(st: AlertState, events: list[LedgerEvent], t: datetime,
             v: dict, running: bool) -> None:
    for r in RULES:
        rs = st.rules[r.name]
        if running != r.running:
            rs.run = 0
            continue
        m = r.metric(v)
        if running:
            _analytics(st, events, t, r, rs, m)
        rs.run = rs.run + 1 if m > r.threshold else 0
        if rs.run == r.hold_ticks:
            hours = r.hold_ticks * TICK_H
            raise_alert(st, events, t, "operator_alarm", r.name, r.sensor,
                        m, f"Operator alarm: {r.message} for {hours:g} hours")


def _analytics(st: AlertState, events: list[LedgerEvent], t: datetime,
               r: Rule, rs: RuleState, m: float) -> None:
    z = rs.window.z(m)
    rs.window.push(m)
    last = rs.last_analytics
    cool = last is None or t - last >= ANALYTICS_COOLDOWN
    if z is not None and z > Z_ALERT and cool:
        raise_alert(st, events, t, "analytics", r.name, r.sensor, m,
                    f"Analytics: {r.message}, {z:.1f} sigma"
                    " above its 24 h mean")
        rs.last_analytics = t


def acknowledge(st: AlertState, events: list[LedgerEvent], now: datetime,
                alert_id: str) -> Alert:
    a = next((a for a in st.alerts if a.id == alert_id), None)
    if a is None:
        raise KeyError(f"unknown alert {alert_id!r}")
    if a.acked_at is None:
        a.acked_at = now
        events.append(AlertActionEvent(
            time=now, action="ack", rule=a.rule, alert_id=a.id))
    return a


def shelve(st: AlertState, events: list[LedgerEvent], now: datetime,
           rule: str, hours: float) -> datetime:
    if rule not in st.rules:
        raise KeyError(f"unknown rule {rule!r}; one of {list(st.rules)}")
    if not 0.25 <= hours <= MAX_SHELVE_H:
        raise ValueError(f"hours must be between 0.25 and {MAX_SHELVE_H}")
    until = now + timedelta(hours=hours)
    st.shelved_until[rule] = until
    events.append(AlertActionEvent(
        time=now, action="shelve", rule=rule, hours=hours))
    return until


def visible(st: AlertState) -> list[Alert]:
    return [a for a in st.alerts if not a.shelved]
