"""The event ledger: the only record of a run. Costs, downtime and
KPIs are folds over it; fetch() serves it as JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import get_args

from ..schemas import CostEvent, CostKind, DowntimeEvent, LedgerEvent
from .variant import TICK_H

# Cost model in USD. ponytail: flat priors; sweep them once the scenario
# suite settles.
PLANNED_DOWNTIME_PER_H = 400.0
UNPLANNED_DOWNTIME_PER_H = 2000.0
# charged on the extra kW a degraded chiller draws
ENERGY_PRICE_PER_KWH = 0.15


@dataclass
class Ledger:
    """Downtime and excess energy are emitted as spans and daily sums,
    so the ledger stays a few hundred events, not one per tick. The
    other fields are only the open span and day being accumulated."""
    energy_day: date
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


def energy(l: Ledger, t: datetime, running: bool,
           excess_kw: float) -> None:
    if t.date() != l.energy_day:
        _flush_energy(l, t)
    if running:
        l.energy_acc += excess_kw * TICK_H * ENERGY_PRICE_PER_KWH


def close(l: Ledger, t: datetime) -> None:
    l.closed = True
    _close_downtime(l, t)
    _flush_energy(l, t)


def costs(events: list[LedgerEvent]) -> dict[CostKind, float]:
    """USD by kind, every kind present. Complete once closed."""
    out = dict.fromkeys(get_args(CostKind), 0.0)
    for e in events:
        if isinstance(e, CostEvent):
            out[e.kind] += e.amount
    return out


def downtime_hours(events: list[LedgerEvent]) -> dict[str, float]:
    out = {"planned_downtime": 0.0, "unplanned_downtime": 0.0}
    for e in events:
        if isinstance(e, DowntimeEvent):
            out[f"{e.kind}_downtime"] += e.hours
    return out


def fetch(events: list[LedgerEvent], since: int,
          types: tuple[type, ...] | None
          ) -> tuple[list[dict], int]:
    """Events from index `since` on, as JSON-ready dicts, and the
    cursor to pass next time."""
    picked = [asdict(e) for e in events[since:]
              if types is None or isinstance(e, types)]
    return (json.loads(json.dumps(picked, default=datetime.isoformat)),
            len(events))


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
