"""Work orders, the planner and the crew: approval, scheduling, visits,
repair, and the change log the environment mirrors to CouchDB."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..schemas import CostEvent, ServiceRequestEvent
from .faults import FAILURE_CODES, Fault, severity
from .ledger import Ledger
from .variant import TICK, TICK_H, Technician, Variant, iso

DISPATCH_COST = 1500.0  # USD
EMERGENCY_MULTIPLIER = 2.0
INSPECTION_HOURS = 3.0
RESPONSE_HOURS = {1: 4, 2: 12, 3: 36, 4: 72, 5: 120}
NOSHOW_DELAY_H = 24
# nobody on the crew has the skill: an outside contractor is called in
CONTRACTOR_DELAY_H = 48
# response later than the priority's target plus this grace is an SLA
# breach
SLA_GRACE_H = 4
OPEN = ("WAPPR", "APPR")


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
    # the priority's response target; later is an SLA breach
    target_arrive: datetime | None = None
    sla_breach: bool = False
    # the technician / planner log, as written to the WO
    notes: list[str] = field(default_factory=list)


@dataclass
class Maintenance:
    wos: dict[str, WorkOrder] = field(default_factory=dict)
    # work-order updates for CouchDB, drained by the env
    changes: list[dict] = field(default_factory=list)


def file_wo(m: Maintenance, l: Ledger, v: Variant, now: datetime,
            code: str | None, priority: int,
            wonum: str | None) -> WorkOrder:
    idx = len(m.wos)
    rng = random.Random(v.seed * 7919 + idx)
    wo = WorkOrder(wonum or f"SIM{idx + 1}", _norm_code(code),
                   _clamp_prio(priority), now, rng)
    wo.approve_at = now + rng.randint(8, 48) * TICK
    wo.jitter_h = rng.randint(0, 16) * TICK_H
    m.wos[wo.wonum] = wo
    _event(l, now, ServiceRequestEvent, wo, actor="agent",
           note="Service request filed.")
    return wo


def update_wo(m: Maintenance, l: Ledger, v: Variant, now: datetime,
              wonum: str, code: str | None, priority: int | None,
              cancel: bool) -> None:
    wo = m.wos.get(wonum)
    if wo is None or wo.status not in OPEN:
        return  # ponytail: changes after the technician arrives are ignored
    if cancel:
        wo.status = "CAN"
        _event(l, now, ServiceRequestEvent, wo, actor="agent",
               note="Cancelled by the agent.")
        return
    changed = False
    if code is not None and _norm_code(code) != wo.code:
        wo.code, changed = _norm_code(code), True
    if priority is not None and _clamp_prio(priority) != wo.priority:
        wo.priority, changed = _clamp_prio(priority), True
        if wo.status == "APPR":
            _schedule(v.crew, wo)
    if changed:
        _event(l, now, ServiceRequestEvent, wo, actor="agent",
               note="Updated by the agent.")


def repairing(m: Maintenance) -> bool:
    return any(w.started and w.status == "INPRG" and w.code
               for w in m.wos.values())


def respond_to_trip(m: Maintenance, l: Ledger, v: Variant, t: datetime,
                    mode: str | None) -> None:
    """The plant's current reactive practice: a trip always gets an
    emergency work order."""
    if any(w.code == mode and w.status == "INPRG" for w in m.wos.values()):
        return
    pending = [w for w in m.wos.values()
               if w.code == mode and w.status in OPEN]
    if pending:
        w = pending[0]
        # a tripped chiller always gets emergency response
        w.priority, w.emergency = 1, True
        if w.status == "WAPPR":
            w.approve_at = min(w.approve_at, t)
        else:
            w.arrive_at = min(w.arrive_at,
                              t + timedelta(hours=RESPONSE_HOURS[1]))
            w.target_arrive = min(w.target_arrive, t + timedelta(
                hours=RESPONSE_HOURS[1] + SLA_GRACE_H))
        _change(m, l, t, w, actor="planner",
                notes="Escalated to priority 1 after chiller trip.")
        return
    idx = len(m.wos)
    w = WorkOrder(f"EM{idx + 1}", mode, 1, t,
                  random.Random(v.seed * 7919 + idx), emergency=True)
    w.approve_at = t
    m.wos[w.wonum] = w
    m.changes.append({"wonum": w.wonum, "new": True, "status": "WAPPR",
                      "priority": 1, "work_type": "EM", "reportdate": t,
                      "failurecode": mode,
                      "description": "EMERGENCY: Chiller 6 tripped offline",
                      "reported_by": "BMS"})
    _event(l, t, ServiceRequestEvent, w, actor="bms",
           note="Emergency work order after chiller trip.")


def service(m: Maintenance, l: Ledger, f: Fault, v: Variant,
            t: datetime) -> None:
    for wo in list(m.wos.values()):
        _service(m, l, f, v, wo, t)


def _assign(crew: tuple[Technician, ...],
            wo: WorkOrder) -> tuple[Technician, float]:
    """Pick a technician for the work order: one with the skill if the
    crew has one, else anyone plus the contractor delay. Returns the
    technician and the extra hours before arrival."""
    skilled = [t for t in crew if t.can(wo.code)]
    if skilled:
        tech = wo.rng.choice(skilled)
        return tech, tech.travel_h
    tech = wo.rng.choice(list(crew))
    return tech, tech.travel_h + CONTRACTOR_DELAY_H


def _schedule(crew: tuple[Technician, ...], wo: WorkOrder) -> None:
    """(Re)compute the arrival time and SLA target from the approval time
    and priority. The technician is picked once, at approval; a priority
    change keeps them."""
    if wo.tech is None:
        tech, extra = _assign(crew, wo)
        wo.tech = tech.id
    else:
        tech = next(t for t in crew if t.id == wo.tech)
        extra = tech.travel_h + (0 if tech.can(wo.code)
                                 else CONTRACTOR_DELAY_H)
    response = (RESPONSE_HOURS[wo.priority] + extra
                + (0 if wo.emergency else wo.jitter_h))
    wo.arrive_at = wo.approve_at + timedelta(hours=response)
    wo.target_arrive = wo.approve_at + timedelta(
        hours=RESPONSE_HOURS[wo.priority] + tech.travel_h + SLA_GRACE_H)


def _service(m: Maintenance, l: Ledger, f: Fault, v: Variant,
             wo: WorkOrder, t: datetime) -> None:
    while True:
        if wo.status == "WAPPR" and t >= wo.approve_at:
            wo.status = "APPR"
            _schedule(v.crew, wo)
            skilled = any(tech.id == wo.tech and tech.can(wo.code)
                          for tech in v.crew)
            how = ("" if skilled else " (outside contractor; nobody on "
                   "the crew has this skill)")
            _change(m, l, t, wo, actor="planner",
                    labor=(wo.tech, wo.arrive_at),
                    notes=f"Approved by planner; {wo.tech} scheduled for "
                          f"{iso(wo.arrive_at)}{how}.")
        elif wo.status == "APPR" and t >= wo.arrive_at:
            if not wo.emergency and wo.rng.random() < v.noshow:
                wo.arrive_at += timedelta(hours=NOSHOW_DELAY_H)
                _change(m, l, t, wo, actor="planner",
                        notes="Technician did not arrive; rescheduled to "
                              f"{iso(wo.arrive_at)}.")
                return
            wo.status, wo.dispatched = "INPRG", True
            wo.sla_breach = (wo.target_arrive is not None
                             and t > wo.target_arrive)
            l.events.append(CostEvent(time=t, kind="dispatch",
                                      amount=DISPATCH_COST,
                                      wonum=wo.wonum))
            fc = FAILURE_CODES.get(wo.code)
            if fc:
                wo.start_at = wo.arrive_at + timedelta(hours=fc.parts_hours)
                wo.finish_at = wo.start_at + timedelta(
                    hours=fc.repair_hours)
                wait = (f" Waiting {fc.parts_hours:.0f} h for parts."
                        if fc.parts_hours else "")
                _change(m, l, t, wo, actor="technician",
                        notes=f"{wo.tech} on site for "
                              f"{fc.description}.{wait}")
            else:
                wo.start_at = wo.arrive_at
                wo.finish_at = wo.arrive_at + timedelta(
                    hours=INSPECTION_HOURS)
                _change(m, l, t, wo, actor="technician",
                        notes=f"{wo.tech} on site; no failure code "
                              "given, general inspection only.")
        elif wo.status == "INPRG" and not wo.started and t >= wo.start_at:
            wo.started, wo.failed_at_start = True, f.failed
        elif wo.status == "INPRG" and wo.started and t >= wo.finish_at:
            _complete(m, l, f, wo, t)
        else:
            return


def _complete(m: Maintenance, l: Ledger, f: Fault, wo: WorkOrder,
              t: datetime) -> None:
    wo.status = "COMP"
    fc = FAILURE_CODES.get(wo.code)
    if fc is None:
        _change(m, l, t, wo, actor="technician", finish=t,
                hours=INSPECTION_HOURS,
                notes="General inspection complete: no actionable fault "
                      "identified.")
        return
    mult = EMERGENCY_MULTIPLIER if wo.failed_at_start else 1.0
    cost = fc.repair_cost * mult
    l.events.append(CostEvent(time=t, kind="repair", amount=cost,
                              wonum=wo.wonum))
    fixed = wo.code == f.mode and severity(f, t) > 0
    if fixed:
        f.active = False
        f.failed = False
        finding = (f"Confirmed {fc.description}; repaired and returned to "
                   "service.")
    else:
        finding = (f"Performed {fc.description} work as requested; no "
                   "matching fault found.")
    _change(m, l, t, wo, actor="technician", finish=t,
            hours=fc.repair_hours, failurecode=wo.code, notes=finding)


def _change(m: Maintenance, l: Ledger, now: datetime, wo: WorkOrder,
            actor: str = "planner", **fields) -> None:
    """Record a work-order transition: for CouchDB (changes), the ledger
    (events) and the work order's own log (notes), which the environment
    uses to keep the technician's entries when the agent writes notes of
    its own."""
    m.changes.append({"wonum": wo.wonum, "status": wo.status,
                      "priority": wo.priority, "time": now, **fields})
    if fields.get("notes"):
        wo.notes.append(f"[{iso(now)}] {fields['notes']}")
    _event(l, now, ServiceRequestEvent, wo, actor=actor,
           note=fields.get("notes"))


def _event(l: Ledger, now: datetime, cls, wo: WorkOrder,
           **fields) -> None:
    l.events.append(cls(time=now, wonum=wo.wonum, status=wo.status,
                        priority=wo.priority, code=wo.code,
                        emergency=wo.emergency, tech=wo.tech,
                        sla_breach=wo.sla_breach, **fields))


def _norm_code(code: str | None) -> str | None:
    c = (code or "").strip().upper()
    return c if c in FAILURE_CODES else None


def _clamp_prio(p) -> int:
    try:
        return min(5, max(1, int(p)))
    except (TypeError, ValueError):
        return 3
