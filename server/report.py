"""Read-only views of a run: ground-truth state, the KPIs, the result."""

from __future__ import annotations

from datetime import datetime, timedelta

from ..schemas import (AlertActionEvent, EpisodeResult, EventTerms,
                       KPIs, SimState, WorkOrderState)
from .alerts import Alert
from .faults import SENSOR_CODE, Fault, severity, trip_time
from .ledger import Ledger, costs, downtime_hours
from .maintenance import (Maintenance, WorkOrder, dispatched,
                          inspection, latest_safe_filing, repairing,
                          status)
from .variant import Variant

# Proposed, not yet agreed: a visit at this priority or more
# urgent, for a fault that cannot fail inside the horizon, is a
# false dispatch in the seven-term cost.
URGENT_PRIORITY = 2


def false_dispatch(v: Variant, f: Fault, w: WorkOrder) -> bool:
    """A visit nothing called for. An inspection never is, nor is
    a transmitter repair once the scenario's chatter has begun."""
    if inspection(w):
        return False
    began = v.start + timedelta(days=v.chatter_start_day)
    if (v.chatter_rule and w.code == SENSOR_CODE
            and w.created >= began):
        return False
    return f.mode is None or w.code != f.mode


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
                status=status(w), emergency=w.emergency,
                created=w.created, tech=w.tech, arrive_at=w.arrive_at,
                start_at=w.start_at, finish_at=w.finish_at,
                dispatched=dispatched(w))
            for w in m.wos.values()])


def kpis(v: Variant, l: Ledger, f: Fault, m: Maintenance,
         alerts: list[Alert]) -> KPIs:
    """The proposal's five KPIs and the inspection visits, folded
    over the ledger. An inspection is not a false dispatch."""
    sent = [w for w in m.wos.values() if dispatched(w)]
    looks = sum(inspection(w) for w in sent)
    return KPIs(
        downtime_hours=round(
            sum(downtime_hours(l.events).values()), 2),
        missed_failures=int(f.ever_failed),
        false_dispatches=sum(false_dispatch(v, f, w) for w in sent),
        investigations=looks,
        alert_load=len(alerts),
        sla_breaches=sum(w.sla_breach for w in m.wos.values()))


def event_terms(v: Variant, l: Ledger, f: Fault,
                m: Maintenance) -> EventTerms:
    """The seven-term cost's counts, folded over the run."""
    sent = [w for w in m.wos.values() if dispatched(w)]
    trip = trip_time(f)
    incipient = trip is not None and trip >= v.end
    urgent = sum(incipient and w.code == f.mode and not w.emergency
                 and w.priority <= URGENT_PRIORITY for w in sent)
    return EventTerms(
        unplanned_downtime_hours=round(
            downtime_hours(l.events)["unplanned_downtime"], 2),
        missed_failures=int(f.ever_failed),
        false_dispatches=urgent + sum(
            false_dispatch(v, f, w) for w in sent),
        acknowledgements=sum(
            isinstance(e, AlertActionEvent) and e.action == "ack"
            for e in l.events),
        investigations=sum(inspection(w) for w in sent),
        unaddressed_faults=int(_unaddressed(v, f, m)),
        sla_breaches=sum(w.sla_breach for w in m.wos.values()))


def _unaddressed(v: Variant, f: Fault, m: Maintenance) -> bool:
    """Proposed, not yet agreed: the fault can fail inside the
    horizon, an order could still have beaten it, and the agent
    filed none for it between its onset and the latest safe time."""
    trip = trip_time(f)
    if trip is None or trip >= v.end:
        return False
    deadline = latest_safe_filing(v, f.mode, trip)
    return deadline >= f.onset and not any(
        w.code == f.mode and not w.emergency and not w.canceled
        and f.onset <= w.created <= deadline
        for w in m.wos.values())


def result(v: Variant, l: Ledger, f: Fault, m: Maintenance,
           alerts: list[Alert]) -> EpisodeResult:
    mode, onset = f.mode, f.onset
    sent = [w for w in m.wos.values() if dispatched(w)]
    false = [w for w in sent if false_dispatch(v, f, w)]
    agent_hits = [w for w in m.wos.values()
                  if not w.emergency and w.code == mode
                  and onset is not None and w.created >= onset
                  and not w.canceled]
    cost = costs(l.events)
    delay = min(((w.created - onset).total_seconds() / 3600
                 for w in agent_hits), default=None)
    return EpisodeResult(
        variant=v.id, mode=mode, onset=onset,
        failed=f.ever_failed, failed_at=f.failed_at,
        failure_prevented=bool(mode) and not f.ever_failed,
        cost_total=round(sum(cost.values()), 2),
        cost_breakdown={k: round(x, 2) for k, x in cost.items()},
        downtime_hours=downtime_hours(l.events),
        dispatches=len(sent), false_dispatches=len(false),
        work_orders=len(m.wos),
        detection_delay_hours=(None if delay is None
                               else round(delay, 2)),
        alerts={k: sum(a.kind == k for a in alerts)
                for k in ("operator_alarm", "analytics")},
        kpis=kpis(v, l, f, m, alerts), events=len(l.events),
        event_terms=event_terms(v, l, f, m))
