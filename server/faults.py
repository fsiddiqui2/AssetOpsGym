"""Failure codes, the latent fault's growth and the trip."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..schemas import FailureEvent, LedgerEvent


@dataclass(frozen=True)
class FailureCode:
    code: str
    subsystem: str  # the part of the chiller investigate() names
    description: str
    repair_cost: float
    repair_hours: float
    parts_hours: float = 0.0


FAILURE_CODES = {fc.code: fc for fc in [
    FailureCode("CH-CWFLOW", "condenser",
                "condenser water flow restriction (clogged "
                "strainer or fouled condenser tubes)", 2500, 8),
    FailureCode("CH-REFLEAK", "refrigerant circuit",
                "refrigerant leak or low refrigerant charge", 5000, 12),
    FailureCode("CH-COMPWEAR", "compressor",
                "compressor mechanical wear", 12000, 16,
                parts_hours=48),
    FailureCode("CH-EVAPFOUL", "evaporator",
                "evaporator tube fouling", 3000, 8),
    FailureCode("CH-SENSOR", "sensors",
                "sensor drift or failed transmitter", 800, 3),
    FailureCode("CH-CONTROLS", "controls",
                "chiller control or setpoint fault", 1200, 4),
]}
FAULT_MODES = ("CH-CWFLOW", "CH-REFLEAK", "CH-COMPWEAR")
SUBSYSTEMS = {fc.subsystem: fc for fc in FAILURE_CODES.values()}
# An inspection finds a fault from this severity on: about
# where its primary sensor has moved 3 sigma of its noise.
INSPECT_MIN_SEVERITY = 0.15


@dataclass
class Fault:
    """The active fault. Starts as the scenario's; inject_fault() can
    replace it (T6)."""
    mode: str | None
    onset: datetime | None
    pf_days: float
    exponent: float
    active: bool
    failed: bool = False
    ever_failed: bool = False
    failed_at: datetime | None = None


def severity(f: Fault, t: datetime) -> float:
    onset = f.onset
    if not f.active or onset is None or t < onset:
        return 0.0
    return min(1.0, ((t - onset) / timedelta(days=f.pf_days))
               ** f.exponent)


def inject_fault(f: Fault, mode: str, onset: datetime,
                 pf_days: float) -> None:
    """Start (or replace) the latent fault at run time. The onset may be
    in the past."""
    if mode not in FAULT_MODES:
        raise ValueError(
            f"unknown fault mode {mode!r}; one of {FAULT_MODES}")
    f.mode, f.onset, f.pf_days = mode, onset, pf_days
    f.active = True


def trip(f: Fault, events: list[LedgerEvent], t: datetime) -> None:
    f.failed, f.ever_failed, f.failed_at = True, True, t
    events.append(FailureEvent(time=t, mode=f.mode))
