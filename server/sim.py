"""Chiller 6 world for AssetOpsGym: wires the parts and runs the clock.

Telemetry, one latent fault, BMS alerts, the maintenance organization
(planner and technicians) and the cost ledger, each in its own module.
Pure Python and deterministic for a given Variant. Time only moves in
advance(). Rows are stamped on 15-minute ticks, and at sim time `now`
only rows strictly before `now` exist.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from ..schemas import EpisodeResult, KPIs, SimState, TelemetryRow
from . import alerts, faults, ledger, maintenance, report, telemetry
from .variant import TICK, TICK_H, Variant


class Sim:
    def __init__(self, variant: Variant):
        self.v = variant
        self.rng = random.Random(variant.seed)
        self.now = variant.start - timedelta(days=variant.history_days)
        self.fault = faults.Fault(variant.mode, variant.onset,
                                  variant.pf_days, variant.exponent,
                                  variant.mode is not None)
        self.plant, self.bms = telemetry.Plant(), alerts.AlertState()
        self.maint = maintenance.Maintenance()
        self.book = ledger.Ledger(self.now.date())
        self.history = self._run_until(variant.start)

    # flat views read by the environment, baselines and tests
    alerts = property(lambda s: s.bms.alerts)
    wos = property(lambda s: s.maint.wos)
    changes = property(lambda s: s.maint.changes)
    events = property(lambda s: s.book.events)
    ever_failed = property(lambda s: s.fault.ever_failed)
    failed_at = property(lambda s: s.fault.failed_at)
    done = property(lambda s: s.now >= s.v.end)

    def advance(self, hours: float) -> list[TelemetryRow]:
        ticks = max(1, round(hours / TICK_H))
        return self._run_until(min(self.now + ticks * TICK, self.v.end))

    def run_to_end(self) -> list[TelemetryRow]:
        return self._run_until(self.v.end)

    def severity(self, t: datetime) -> float:
        return faults.severity(self.fault, t)

    def inject_fault(self, mode: str, onset: datetime,
                     pf_days: float) -> None:
        faults.inject_fault(self.fault, mode, onset, pf_days)

    def file_wo(self, code: str | None, priority: int = 3,
                wonum: str | None = None) -> maintenance.WorkOrder:
        return maintenance.file_wo(self.maint, self.book, self.v,
                                   self.now, code, priority, wonum)

    def update_wo(self, wonum: str, code: str | None = None,
                  priority: int | None = None,
                  cancel: bool = False) -> None:
        maintenance.update_wo(self.maint, self.book, self.v, self.now,
                              wonum, code, priority, cancel)

    def acknowledge(self, alert_id: str) -> alerts.Alert:
        return alerts.acknowledge(self.bms, self.events, self.now,
                                  alert_id)

    def shelve(self, rule: str, hours: float) -> datetime:
        return alerts.shelve(self.bms, self.events, self.now, rule, hours)

    def visible_alerts(self) -> list[alerts.Alert]:
        return alerts.visible(self.bms)

    def state(self) -> SimState:
        return report.state(self.v, self.now, self.fault, self.maint)

    def kpis(self) -> KPIs:
        return report.kpis(self.book, self.fault, self.maint, self.alerts)

    def result(self) -> EpisodeResult:
        return report.result(self.v, self.book, self.fault, self.maint,
                             self.alerts)

    def _run_until(self, target: datetime) -> list[TelemetryRow]:
        rows = []
        while self.now < target:
            rows.append(self._tick(self.now))
            self.now += TICK
        if self.done and not self.book.closed:
            ledger.close(self.book, self.now)
        return rows

    def _tick(self, t: datetime) -> TelemetryRow:
        f, events = self.fault, self.events
        maintenance.service(self.maint, self.book, f, self.v, t)
        s = faults.severity(f, t)
        if not f.failed and s >= 1.0:
            faults.trip(f, events, t)
            alerts.raise_alert(self.bms, events, t, "operator_alarm",
                               "chiller_trip", telemetry.RUN, 0.0,
                               "Operator alarm: Chiller 6 tripped offline")
            maintenance.respond_to_trip(self.maint, self.book, self.v, t,
                                        f.mode)
        repairing = maintenance.repairing(self.maint)
        running = not f.failed and not repairing
        ledger.downtime(self.book, t, f.failed, repairing)
        values = telemetry.sample(self.plant, self.rng, t, running, f.mode,
                                  s, self.v.blips_per_day)
        ledger.energy(self.book, t, running, self.plant.excess_kw)
        alerts.evaluate(self.bms, events, t, values, running)
        return TelemetryRow(t, values)
