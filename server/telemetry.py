"""Chiller 6 sensors and the per-tick sample: load, weather, faults,
blips."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime

from .faults import FAULT_MODES
from .variant import TICK_H

SITE, ASSET = "MAIN", "Chiller 6"
CAPACITY_TONS = 800.0
DESIGN_FLOW_GPM = 2400.0
SETPOINT_F = 44.0


def _s(name: str) -> str:
    return f"{ASSET} {name}"


COND_RET = _s("Condenser Water Return To Tower Temperature")
EFFICIENCY = _s("Chiller Efficiency")
TONNAGE = _s("Tonnage")
SUPPLY = _s("Supply Temperature")
RETURN = _s("Return Temperature")
FLOW = _s("Condenser Water Flow")
SCHEDULE = _s("Schedule")
POWER = _s("Power Input")
LOADED = _s("Chiller % Loaded")
REFRIG = _s("Liquid Refrigerant Evaporator Temperature")
SETPOINT = _s("Setpoint Temperature")
RUN = _s("Run Status")

# Same installed-vs-measured gap as AssetOpsBench: Oil Pressure is
# installed, never measured.
INSTALLED_SENSORS = [COND_RET, EFFICIENCY, TONNAGE, SUPPLY, RETURN, FLOW,
                     SCHEDULE, POWER, LOADED, REFRIG, SETPOINT,
                     _s("Oil Pressure")]


@dataclass
class Plant:
    weather: float = 0.0
    excess_kw: float = 0.0
    blip: tuple[int, int] | None = None  # (kind, ticks left)
    offline_since: datetime | None = None


def sample(p: Plant, rng: random.Random, t: datetime, running: bool,
           mode: str | None, s: float, blips_per_day: float) -> dict:
    """One tick of sensor values; `s` is the fault's severity at t."""
    if running:
        p.offline_since = None
    elif p.offline_since is None:
        p.offline_since = t
    r = rng
    # fixed draw count keeps runs aligned across policies
    n = [r.gauss(0, 1) for _ in range(10)]
    blip_u, blip_kind, blip_len = r.random(), r.randrange(5), r.randint(1, 3)
    h = t.hour + t.minute / 60
    weekend = t.weekday() >= 5
    lo, hi = (8, 18) if weekend else (6, 22)
    occupied = lo <= h < hi
    shape = math.sin(math.pi * (h - lo) / (hi - lo)) if occupied else 0.0
    base = 0.25 + (0.35 if weekend else 0.55) * shape if occupied else 0.20
    p.weather = 0.995 * p.weather + 0.01 * n[0]
    load = min(1.0, max(0.12, base + p.weather + 0.02 * n[1]))
    amb = 3 * math.sin(2 * math.pi * (h - 9) / 24)

    if (p.blip is None and running
            and blip_u < blips_per_day * TICK_H / 24):
        p.blip = (blip_kind, blip_len)

    if not running:
        off_h = ((t - p.offline_since).total_seconds() / 3600
                 if p.offline_since else 0.0)
        supply = min(SETPOINT_F + 2 * off_h, 62.0) + 0.3 * n[2]
        v = {COND_RET: 80 + amb + 0.5 * n[6], EFFICIENCY: 0.0,
             TONNAGE: 0.0, SUPPLY: supply,
             RETURN: supply + 0.5 + 0.2 * n[3], FLOW: 0.0,
             POWER: 4 + 0.5 * n[7], LOADED: 0.0,
             REFRIG: supply - 1 + 0.3 * n[9], RUN: 0}
    else:
        sf, sl, sw = (s if mode == m else 0.0 for m in FAULT_MODES)
        supply = (SETPOINT_F + 0.3 + 0.25 * n[2] + 6 * sl ** 2 * load
                  + 4 * sf ** 3 * load)
        tons = CAPACITY_TONS * load * (1 + 0.01 * n[4])
        flow = DESIGN_FLOW_GPM * (1 - 0.45 * sf) * (1 + 0.02 * n[5])
        fault_kwpt = 0.30 * sf + 0.12 * sl + 0.25 * sw
        p.excess_kw = tons * fault_kwpt
        kwpt = 0.56 + 0.18 * (1 - load) ** 2 + fault_kwpt + 0.015 * n[7]
        loaded = 100 * load + 25 * sw + 1.5 * n[8]
        refrig = supply - (3 + 2 * load + 9 * sl) + 0.4 * n[9]
        if p.blip:  # transient sensor glitch, shorter than the on-delay
            kind = p.blip[0]
            flow *= 0.68 if kind == 0 else 1.0
            refrig -= 6.5 if kind == 1 else 0.0
            loaded += 16 if kind == 2 else 0.0
            supply += 3.5 if kind == 3 else 0.0
            kwpt += 0.4 if kind == 4 else 0.0
            left = p.blip[1] - 1
            p.blip = (kind, left) if left > 0 else None
        cond = (84 + amb + 9.5 * load * DESIGN_FLOW_GPM / max(flow, 1.0)
                + 0.5 * n[6])
        v = {COND_RET: cond, EFFICIENCY: kwpt, TONNAGE: tons,
             SUPPLY: supply, RETURN: supply + 10 * load + 0.4 * n[3],
             FLOW: flow, POWER: tons * kwpt, LOADED: loaded,
             REFRIG: refrig, RUN: 1}
    v[SCHEDULE] = 1 if occupied else 0
    v[SETPOINT] = SETPOINT_F
    return {k: (round(x, 3) if isinstance(x, float) else x)
            for k, x in v.items()}
