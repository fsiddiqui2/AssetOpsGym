"""Read-only views of a run: ground-truth state, the KPIs, the result."""

from __future__ import annotations

from datetime import datetime

from ..schemas import EpisodeResult, KPIs, SimState, WorkOrderState
from .alerts import Alert
from .faults import Fault, severity
from .ledger import Ledger
from .maintenance import Maintenance, repairing
from .variant import Variant


def state(v: Variant, now: datetime, f: Fault,
          m: Maintenance) -> SimState:
    """Ground truth included: for the environment and tests, never the
    agent."""
    return SimState(
        sim_time=now, start=v.start, end=v.end,
        running=not f.failed and not repairing(m),
        failed=f.failed, fault_active=f.active,
        severity=severity(f, now),
        work_orders=[
            WorkOrderState(
                wonum=w.wonum, code=w.code, priority=w.priority,
                status=w.status, emergency=w.emergency,
                created=w.created, tech=w.tech, arrive_at=w.arrive_at,
                start_at=w.start_at, finish_at=w.finish_at,
                dispatched=w.dispatched)
            for w in m.wos.values()])


def kpis(l: Ledger, f: Fault, m: Maintenance,
         alerts: list[Alert]) -> KPIs:
    """The proposal's five KPIs, folded over the ledger."""
    dispatched = [w for w in m.wos.values() if w.dispatched]
    return KPIs(
        downtime_hours=round(sum(l.hours.values()), 2),
        missed_failures=int(f.ever_failed),
        false_dispatches=sum(f.mode is None or w.code != f.mode
                             for w in dispatched),
        alert_load=len(alerts),
        sla_breaches=sum(w.sla_breach for w in m.wos.values()))


def result(v: Variant, l: Ledger, f: Fault, m: Maintenance,
           alerts: list[Alert]) -> EpisodeResult:
    mode, onset = f.mode, f.onset
    dispatched = [w for w in m.wos.values() if w.dispatched]
    false = [w for w in dispatched if mode is None or w.code != mode]
    agent_hits = [w for w in m.wos.values()
                  if not w.emergency and w.code == mode
                  and onset is not None and w.created >= onset
                  and w.status != "CAN"]
    delay = min(((w.created - onset).total_seconds() / 3600
                 for w in agent_hits), default=None)
    return EpisodeResult(
        variant=v.id, mode=mode, onset=onset,
        failed=f.ever_failed, failed_at=f.failed_at,
        failure_prevented=bool(mode) and not f.ever_failed,
        cost_total=round(sum(l.cost.values()), 2),
        cost_breakdown={k: round(x, 2) for k, x in l.cost.items()},
        downtime_hours=dict(l.hours),
        dispatches=len(dispatched), false_dispatches=len(false),
        work_orders=len(m.wos),
        detection_delay_hours=(None if delay is None
                               else round(delay, 2)),
        alerts={k: sum(a.kind == k for a in alerts)
                for k in ("operator_alarm", "analytics")},
        kpis=kpis(l, f, m, alerts), events=len(l.events))
