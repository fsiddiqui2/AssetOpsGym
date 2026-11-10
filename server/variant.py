"""Scenario parameters (Variant, crew) and the sim clock."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

TICK = timedelta(minutes=15)
TICK_H = 0.25


@dataclass(frozen=True)
class Technician:
    id: str
    # failure codes this technician can repair; None = all
    skills: tuple[str, ...] | None = None
    travel_h: float = 0.0

    def can(self, code: str | None) -> bool:
        return self.skills is None or code is None or code in self.skills


DEFAULT_CREW = (Technician("TECH-01"), Technician("TECH-02"))


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
    crew: tuple[Technician, ...] = DEFAULT_CREW

    @property
    def onset(self) -> datetime | None:
        return (self.start + timedelta(days=self.onset_day)
                if self.mode else None)

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=self.horizon_days)


def iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")
