"""Scenario parameters (Variant, crew), table loading and the sim
clock."""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timedelta
from typing import Literal

TICK = timedelta(minutes=15)
TICK_H = 0.25


@dataclass(frozen=True, slots=True)
class Technician:
    id: str
    # failure codes this technician can repair; None = all
    skills: tuple[str, ...] | None = None
    travel_h: float = 0.0  # added to every response time

    def can(self, code: str | None) -> bool:
        return self.skills is None or code is None or code in self.skills


DEFAULT_CREW = (Technician("TECH-01"), Technician("TECH-02"))


@dataclass(frozen=True, slots=True)
class Variant:
    """One task variant: the only scenario type. `mode=None` is a
    healthy twin. A new setting is a field with a default."""
    id: str
    mode: str | None  # failure code, e.g. CH-REFLEAK
    onset_day: float = 0.0  # days after start when degradation begins
    pf_days: float = 20.0  # P-F interval: onset to trip
    # severity = ((t - onset) / pf_days) ** exponent
    shape: Literal["power"] = "power"
    exponent: float = 1.6
    blips_per_day: float = 1.0  # transient sensor glitches
    # chance a scheduled, non-emergency visit slips a day
    noshow: float = 0.05
    seed: int = 0
    start: datetime = datetime(2026, 6, 1)
    history_days: int = 14  # telemetry before the episode starts
    horizon_days: int = 60
    crew: tuple[Technician, ...] = DEFAULT_CREW
    load_profile: Literal["office"] = "office"
    # a BMS rule whose transmitter chatters across its alarm limit
    chatter_rule: str | None = None
    chatter_per_day: float = 40.0  # operator alarms a day
    chatter_start_day: float = 0.0
    chatter_end_day: float | None = None  # None: to the horizon
    held_out: bool = False  # kept out of the main suite (shift study)
    tags: tuple[str, ...] = ()

    @property
    def onset(self) -> datetime | None:
        return (self.start + timedelta(days=self.onset_day)
                if self.mode else None)

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=self.horizon_days)

    @property
    def scoring(self) -> Literal["faulty", "healthy"]:
        return "faulty" if self.mode else "healthy"


def from_dict(cls: type, d: dict) -> typing.Any:
    """A dataclass from a table row / JSON object: nested dataclasses,
    tuples and ISO datetimes by the field types. Unknown keys raise."""
    hints = typing.get_type_hints(cls)
    unknown = d.keys() - {f.name for f in fields(cls) if f.init}
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**{k: _coerce(hints[k], x) for k, x in d.items()})


def _coerce(tp: typing.Any, x: typing.Any) -> typing.Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if x is None:
        return x
    if origin in (typing.Union, types.UnionType):
        return _coerce(next(a for a in args if a is not type(None)), x)
    if origin is tuple:
        return tuple(_coerce(args[0], y) for y in x)
    if is_dataclass(tp):
        return from_dict(tp, x)
    if tp is datetime and isinstance(x, str):
        return datetime.fromisoformat(x)
    return x


def iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")
