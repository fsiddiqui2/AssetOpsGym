"""Work orders, the planner and the crew. An order is a set of
timestamps; a heap of due actions (approve, arrive, start, finish)
moves them, and every change is one ledger event."""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..schemas import (CostEvent, ErrorCode, GymError,
                       ServiceRequestEvent, WoStatus)
from .faults import (FAILURE_CODES, INSPECT_MIN_SEVERITY, SUBSYSTEMS,
                     Fault, severity)
from .ledger import Ledger
from .telemetry import ASSET, SITE
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
# AssetOpsBench's workorders.WORKTYPES
WORK_TYPES = ("CM", "PM", "EM", "PdM", "CAL", "INSP", "GEN")


class Action(Enum):
    APPROVE = 1
    ARRIVE = 2
    START = 3
    FINISH = 4


# status once an action has run; the first match wins
STATUS_AFTER: tuple[tuple[Action, WoStatus], ...] = (
    (Action.FINISH, "COMP"), (Action.ARRIVE, "INPRG"),
    (Action.APPROVE, "APPR"))


@dataclass
class WorkOrder:
    wonum: str
    code: str | None
    priority: int
    created: datetime
    rng: random.Random
    description: str
    reported_by: str | None = None
    work_type: str = "CM"
    location: str | None = None
    inspect: str | None = None  # the subsystem investigate() named
    emergency: bool = False
    canceled: bool = False
    approve_at: datetime | None = None
    jitter_h: float = 0.0
    arrive_at: datetime | None = None
    start_at: datetime | None = None
    finish_at: datetime | None = None
    failed_at_start: bool = False
    tech: str | None = None
    # the priority's response target; later is an SLA breach
    target_arrive: datetime | None = None
    sla_breach: bool = False
    # bumped by every edit; heap entries of older versions are stale
    version: int = 0
    # the tick each action ran at
    done: dict[Action, datetime] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, order=True)
class Due:
    time: datetime
    seq: int  # unique: ties never compare the fields after it
    wonum: str
    version: int
    action: Action


@dataclass
class Maintenance:
    wos: dict[str, WorkOrder] = field(default_factory=dict)
    heap: list[Due] = field(default_factory=list)
    seq: itertools.count = field(default_factory=itertools.count)


def status(w: WorkOrder) -> WoStatus:
    if w.canceled:
        return "CAN"
    return next((s for a, s in STATUS_AFTER if a in w.done), "WAPPR")


def dispatched(w: WorkOrder) -> bool:
    return Action.ARRIVE in w.done


def inspection(w: WorkOrder) -> bool:
    """Still an inspection: no failure code was added to it."""
    return w.inspect is not None and w.code is None


def repairing(m: Maintenance) -> bool:
    return any(w.code and Action.START in w.done
               and status(w) == "INPRG" for w in m.wos.values())


def file_wo(m: Maintenance, l: Ledger, v: Variant, now: datetime,
            code: str | None, priority: int,
            wonum: str | None) -> WorkOrder:
    """Shortcut for baselines and scenarios: a request for this
    site's chiller with the failure code set at filing."""
    wo = WorkOrder(_wonum(m, wonum, "SIM"), _code(code),
                   _priority(priority), now, _rng(m, v),
                   description="Service request")
    return _request(m, l, now, wo, "Service request filed.")


def generate_work_order(m: Maintenance, l: Ledger, v: Variant,
                        now: datetime, description: str,
                        asset_num: str, site_id: str, priority: int,
                        work_type: str, reported_by: str | None,
                        location: str | None, notes: str | None,
                        wonum: str | None) -> WorkOrder:
    _check_asset(asset_num, site_id)
    if not description:
        raise GymError(ErrorCode.BAD_ARGUMENT,
                       "description is required")
    if work_type not in WORK_TYPES:
        raise GymError(ErrorCode.BAD_ARGUMENT,
                       f"work_type must be one of {WORK_TYPES}")
    wo = WorkOrder(_wonum(m, wonum, "SIM"), None, _priority(priority),
                   now, _rng(m, v), description=description[:100],
                   reported_by=reported_by, work_type=work_type,
                   location=location)
    return _request(m, l, now, wo, notes or "Service request filed.")


def investigate(m: Maintenance, l: Ledger, v: Variant,
                now: datetime, subsystem: str, priority: int,
                notes: str | None) -> WorkOrder:
    """An inspection order for one subsystem. Nothing is repaired
    and the chiller keeps running; the finding is the technician's
    closing note."""
    if subsystem not in SUBSYSTEMS:
        raise GymError(
            ErrorCode.BAD_ARGUMENT,
            f"unknown subsystem {subsystem!r}; one of "
            f"{', '.join(SUBSYSTEMS)}")
    wo = WorkOrder(_wonum(m, None, "SIM"), None, _priority(priority),
                   now, _rng(m, v),
                   description=f"Inspect the {subsystem}",
                   work_type="INSP", inspect=subsystem)
    return _request(m, l, now, wo, notes or "Inspection requested.")


def update_workorder(m: Maintenance, l: Ledger, v: Variant,
                     now: datetime, wonum: str, site_id: str,
                     description: str | None, priority: int | None,
                     location: str | None, asset_num: str | None,
                     notes: str | None,
                     failure_code: str | None) -> WorkOrder:
    """An update that changes nothing is not an edit: it succeeds
    even after the technician arrived."""
    wo = _find(m, wonum, site_id)
    _check_asset(ASSET if asset_num is None else asset_num, site_id)
    code = wo.code if failure_code is None else _code(failure_code)
    prio = wo.priority if priority is None else _priority(priority)
    desc = wo.description if description is None else description[:100]
    loc = wo.location if location is None else location
    if notes is None and (code, prio, desc, loc) == (
            wo.code, wo.priority, wo.description, wo.location):
        return wo
    _check_open(wo)
    wo.code, wo.description, wo.location = code, desc, loc
    if prio != wo.priority:
        wo.priority = prio
        if Action.APPROVE in wo.done:
            _schedule(v.crew, wo)
    _bump(m, wo)
    _event(l, now, wo, "agent", notes or "Updated by the agent.")
    return wo


def cancel_workorder(m: Maintenance, l: Ledger, now: datetime,
                     wonum: str, site_id: str,
                     reason: str | None) -> WorkOrder:
    wo = _find(m, wonum, site_id)
    _check_open(wo)
    wo.canceled = True
    wo.version += 1  # its pending action goes stale
    _event(l, now, wo, "agent", "Cancelled by the agent."
           + (f" {reason}" if reason else ""))
    return wo


def respond_to_trip(m: Maintenance, l: Ledger, v: Variant, t: datetime,
                    mode: str | None) -> None:
    """The plant's current reactive practice: a trip always gets an
    emergency work order."""
    mine = [w for w in m.wos.values() if w.code == mode]
    if any(status(w) == "INPRG" for w in mine):
        return
    pending = [w for w in mine if status(w) in OPEN]
    if pending:
        w = pending[0]
        # a tripped chiller always gets emergency response
        w.priority, w.emergency = 1, True
        if Action.APPROVE in w.done:
            w.arrive_at = min(w.arrive_at,
                              t + timedelta(hours=RESPONSE_HOURS[1]))
            w.target_arrive = min(w.target_arrive, t + timedelta(
                hours=RESPONSE_HOURS[1] + SLA_GRACE_H))
        else:
            w.approve_at = min(w.approve_at, t)
        _bump(m, w)
        _event(l, t, w, "planner",
               "Escalated to priority 1 after chiller trip.")
        return
    w = WorkOrder(_wonum(m, None, "EM"), mode, 1, t, _rng(m, v),
                  description=f"EMERGENCY: {ASSET} tripped offline",
                  reported_by="BMS", work_type="EM", emergency=True,
                  approve_at=t)
    _insert(m, w, Action.APPROVE, t)
    _event(l, t, w, "bms", "Emergency work order after chiller trip.")


def run_due(m: Maintenance, l: Ledger, f: Fault, v: Variant,
            t: datetime) -> None:
    """Run every action due by tick t, in (time, push) order."""
    while m.heap and m.heap[0].time <= t:
        d = heapq.heappop(m.heap)
        wo = m.wos[d.wonum]
        if d.version == wo.version:
            HANDLERS[d.action](m, l, f, v, wo, t)


def _approve(m: Maintenance, l: Ledger, f: Fault, v: Variant,
             wo: WorkOrder, t: datetime) -> None:
    wo.done[Action.APPROVE] = t
    _schedule(v.crew, wo)
    _push(m, wo, Action.ARRIVE, wo.arrive_at)
    skilled = any(tech.id == wo.tech and tech.can(wo.code)
                  for tech in v.crew)
    how = ("" if skilled else " (outside contractor; nobody on "
           "the crew has this skill)")
    _event(l, t, wo, "planner",
           f"Approved by planner; {wo.tech} scheduled for "
           f"{iso(wo.arrive_at)}{how}.")


def _arrive(m: Maintenance, l: Ledger, f: Fault, v: Variant,
            wo: WorkOrder, t: datetime) -> None:
    if not wo.emergency and wo.rng.random() < v.noshow:
        wo.arrive_at += timedelta(hours=NOSHOW_DELAY_H)
        _push(m, wo, Action.ARRIVE, wo.arrive_at)
        _event(l, t, wo, "planner", "Technician did not arrive; "
               f"rescheduled to {iso(wo.arrive_at)}.")
        return
    wo.done[Action.ARRIVE] = t
    wo.sla_breach = (wo.target_arrive is not None
                     and t > wo.target_arrive)
    l.events.append(CostEvent(time=t, kind="dispatch",
                              amount=DISPATCH_COST, wonum=wo.wonum))
    fc = FAILURE_CODES.get(wo.code)
    parts, work = ((fc.parts_hours, fc.repair_hours) if fc
                   else (0.0, INSPECTION_HOURS))
    wo.start_at = wo.arrive_at + timedelta(hours=parts)
    wo.finish_at = wo.start_at + timedelta(hours=work)
    _push(m, wo, Action.START, wo.start_at)
    _push(m, wo, Action.FINISH, wo.finish_at)
    if inspection(wo):
        note = f"{wo.tech} on site to inspect the {wo.inspect}."
    elif fc is None:
        note = (f"{wo.tech} on site; no failure code given, general "
                "inspection only.")
    else:
        wait = (f" Waiting {parts:.0f} h for parts." if parts else "")
        note = f"{wo.tech} on site for {fc.description}.{wait}"
    _event(l, t, wo, "technician", note)


def _start(m: Maintenance, l: Ledger, f: Fault, v: Variant,
           wo: WorkOrder, t: datetime) -> None:
    wo.done[Action.START] = t
    wo.failed_at_start = f.failed


def _finish(m: Maintenance, l: Ledger, f: Fault, v: Variant,
            wo: WorkOrder, t: datetime) -> None:
    wo.done[Action.FINISH] = t
    fc = FAILURE_CODES.get(wo.code)
    if fc is None:
        _event(l, t, wo, "technician",
               _finding(f, wo.inspect, t) if wo.inspect else
               "General inspection complete: no actionable fault "
               "identified.", labor_hours=INSPECTION_HOURS)
        return
    mult = EMERGENCY_MULTIPLIER if wo.failed_at_start else 1.0
    l.events.append(CostEvent(time=t, kind="repair",
                              amount=fc.repair_cost * mult,
                              wonum=wo.wonum))
    if wo.code == f.mode and severity(f, t) > 0:
        f.active = False
        f.failed = False
        finding = (f"Confirmed {fc.description}; repaired and "
                   "returned to service.")
    else:
        finding = (f"Performed {fc.description} work as requested; "
                   "no matching fault found.")
    _event(l, t, wo, "technician", finding,
           labor_hours=fc.repair_hours)


def _finding(f: Fault, subsystem: str, t: datetime) -> str:
    fc = SUBSYSTEMS[subsystem]
    if fc.code == f.mode and severity(f, t) >= INSPECT_MIN_SEVERITY:
        return (f"Inspected the {subsystem}: found {fc.description} "
                f"(failure code {fc.code}).")
    return f"Inspected the {subsystem}: nothing abnormal found."


HANDLERS = {Action.APPROVE: _approve, Action.ARRIVE: _arrive,
            Action.START: _start, Action.FINISH: _finish}


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


def _request(m: Maintenance, l: Ledger, now: datetime, wo: WorkOrder,
             note: str) -> WorkOrder:
    wo.approve_at = now + wo.rng.randint(8, 48) * TICK
    wo.jitter_h = wo.rng.randint(0, 16) * TICK_H
    _insert(m, wo, Action.APPROVE, wo.approve_at)
    _event(l, now, wo, "agent", note)
    return wo


def _insert(m: Maintenance, wo: WorkOrder, action: Action,
            time: datetime) -> None:
    m.wos[wo.wonum] = wo
    _push(m, wo, action, time)


def _push(m: Maintenance, wo: WorkOrder, action: Action,
          time: datetime) -> None:
    heapq.heappush(m.heap, Due(time, next(m.seq), wo.wonum,
                               wo.version, action))


def _bump(m: Maintenance, wo: WorkOrder) -> None:
    """After an edit: stale the pending action, push it afresh."""
    wo.version += 1
    if Action.APPROVE in wo.done:
        _push(m, wo, Action.ARRIVE, wo.arrive_at)
    else:
        _push(m, wo, Action.APPROVE, wo.approve_at)


def _event(l: Ledger, now: datetime, wo: WorkOrder, actor: str,
           note: str, labor_hours: float | None = None) -> None:
    l.events.append(ServiceRequestEvent(
        time=now, wonum=wo.wonum, status=status(wo),
        priority=wo.priority, code=wo.code, emergency=wo.emergency,
        actor=actor, tech=wo.tech, sla_breach=wo.sla_breach,
        note=note, description=wo.description, site=SITE, asset=ASSET,
        work_type=wo.work_type, location=wo.location,
        reported_by=wo.reported_by, arrive_at=wo.arrive_at,
        labor_hours=labor_hours))


def _rng(m: Maintenance, v: Variant) -> random.Random:
    return random.Random(v.seed * 7919 + len(m.wos))


def _wonum(m: Maintenance, wonum: str | None, prefix: str) -> str:
    wonum = wonum or f"{prefix}{len(m.wos) + 1}"
    if wonum in m.wos:
        raise GymError(ErrorCode.BAD_ARGUMENT,
                       f"work order {wonum} already exists")
    return wonum


def _find(m: Maintenance, wonum: str, site_id: str) -> WorkOrder:
    _check_asset(ASSET, site_id)
    if wonum not in m.wos:
        raise GymError(ErrorCode.NOT_FOUND,
                       f"work order {wonum!r} not found")
    return m.wos[wonum]


def _check_asset(asset_num: str, site_id: str) -> None:
    if asset_num != ASSET or site_id.upper() != SITE:
        raise GymError(
            ErrorCode.UNKNOWN_ASSET,
            f"no asset {asset_num!r} at site {site_id!r}; this "
            f"site's chiller is asset_num={ASSET!r}, site_id={SITE!r}")


def _check_open(wo: WorkOrder) -> None:
    if status(wo) not in OPEN:
        raise GymError(
            ErrorCode.WORK_ORDER_LOCKED,
            f"work order {wo.wonum} is {status(wo)}; it can change "
            "only before the technician arrives")


def _code(code: str | None) -> str | None:
    if code is None:
        return None
    c = code.strip().upper()
    if c not in FAILURE_CODES:
        raise GymError(ErrorCode.UNKNOWN_FAILURE_CODE,
                       f"unknown failure code {code!r}; valid: "
                       f"{', '.join(FAILURE_CODES)}")
    return c


def _priority(p: int) -> int:
    if not 1 <= p <= 5:
        raise GymError(ErrorCode.BAD_ARGUMENT,
                       "priority must be between 1 and 5")
    return p
