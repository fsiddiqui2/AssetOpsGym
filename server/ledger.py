"""The cost ledger: totals, downtime spans, daily excess-energy sums."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from ..schemas import CostEvent, DowntimeEvent, LedgerEvent
from .variant import TICK_H

# Cost model in USD. ponytail: flat priors; sweep them once the scenario
# suite settles.
PLANNED_DOWNTIME_PER_H = 400.0
UNPLANNED_DOWNTIME_PER_H = 2000.0
# charged on the extra kW a degraded chiller draws
ENERGY_PRICE_PER_KWH = 0.15


@dataclass
class Ledger:
    """Totals drive the score; the event ledger (schemas.LedgerEvent) is
    the audit trail and the five KPIs are folds over it. Downtime and
    excess energy are emitted as spans and daily sums, so the ledger stays
    a few hundred events, not one per tick."""
    energy_day: date
    cost: dict[str, float] = field(default_factory=lambda: {
        "dispatch": 0.0, "repair": 0.0, "planned_downtime": 0.0,
        "unplanned_downtime": 0.0, "excess_energy": 0.0})
    hours: dict[str, float] = field(default_factory=lambda: {
        "planned_downtime": 0.0, "unplanned_downtime": 0.0})
    events: list[LedgerEvent] = field(default_factory=list)
    # (planned|unplanned, span start)
    down: tuple[str, datetime] | None = None
    energy_acc: float = 0.0
    closed: bool = False


def downtime(l: Ledger, t: datetime, failed: bool,
             repairing: bool) -> None:
    kind = "unplanned" if failed else "planned" if repairing else None
    if l.down and l.down[0] != kind:
        _close_downtime(l, t)
    if kind and not l.down:
        l.down = (kind, t)
    if failed:
        _charge(l, "unplanned_downtime", UNPLANNED_DOWNTIME_PER_H)
    elif repairing:
        _charge(l, "planned_downtime", PLANNED_DOWNTIME_PER_H)


def energy(l: Ledger, t: datetime, running: bool,
           excess_kw: float) -> None:
    if t.date() != l.energy_day:
        _flush_energy(l, t)
    if running:
        e = excess_kw * TICK_H * ENERGY_PRICE_PER_KWH
        l.cost["excess_energy"] += e
        l.energy_acc += e


def close(l: Ledger, t: datetime) -> None:
    l.closed = True
    _close_downtime(l, t)
    _flush_energy(l, t)


def _charge(l: Ledger, kind: str, per_hour: float) -> None:
    l.cost[kind] += per_hour * TICK_H
    l.hours[kind] += TICK_H


def _close_downtime(l: Ledger, t: datetime) -> None:
    if not l.down:
        return
    kind, start = l.down
    l.down = None
    hours = (t - start).total_seconds() / 3600
    if hours <= 0:
        return
    rate = (UNPLANNED_DOWNTIME_PER_H if kind == "unplanned"
            else PLANNED_DOWNTIME_PER_H)
    l.events.append(DowntimeEvent(time=t, kind=kind, hours=hours,
                                  start=start))
    l.events.append(CostEvent(time=t, kind=f"{kind}_downtime",
                              amount=hours * rate))


def _flush_energy(l: Ledger, t: datetime) -> None:
    if l.energy_acc > 0:
        l.events.append(CostEvent(time=t, kind="excess_energy",
                                  amount=l.energy_acc))
    l.energy_acc, l.energy_day = 0.0, t.date()
