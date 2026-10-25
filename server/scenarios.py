"""Scenario family and score, in the style of FDE-Gym.

One environment (Chiller 6, 60 days), several variants that change the key facts:
which fault, when it starts, how fast it grows, how noisy the BMS is, how reliable
the technicians are. Healthy twins test over-diagnosis.

Score, per FDE-Gym's normalized business uplift with total cost as the KPI:

    faulty variant:   U = (C_reactive - C_agent) / (C_reactive - C_oracle)
    healthy variant:  U = 1 - C_agent / HEALTHY_SCALE

C_reactive is the plant's current practice (do nothing; a trip gets an emergency
work order) and C_oracle files the right work order at fault onset. U is clipped to
[-1, 1], the floor FDE-Gym applies when aggregating. On healthy twins doing nothing
is the reference, the analogue of FDE-Gym's correct-and-complete sponsor request.
"""

from __future__ import annotations

from functools import lru_cache

from .sim import DISPATCH_COST, Sim, Variant

HEALTHY_SCALE = 5 * DISPATCH_COST

VARIANTS = {v.id: v for v in [
    Variant("cwflow_early", "CH-CWFLOW", onset_day=6, pf_days=24, seed=11),
    Variant("cwflow_noshow", "CH-CWFLOW", onset_day=18, pf_days=20, noshow=0.5, seed=12),
    Variant("refleak_mid", "CH-REFLEAK", onset_day=15, pf_days=18, seed=21),
    Variant("refleak_fast", "CH-REFLEAK", onset_day=9, pf_days=8, seed=22),
    Variant("compwear_late", "CH-COMPWEAR", onset_day=28, pf_days=22, seed=31),
    Variant("healthy", None, seed=41),
    Variant("healthy_noisy", None, blips_per_day=5, seed=42),
]}


def do_nothing(sim: Sim) -> None:
    sim.run_to_end()


def oracle(sim: Sim) -> None:
    """Knows the fault and its onset: files the right work order right after onset."""
    if sim.v.mode:
        sim.advance((sim.v.onset - sim.now).total_seconds() / 3600 + 0.25)
        sim.file_wo(sim.v.mode, priority=2)
    sim.run_to_end()


def run(variant: Variant, policy) -> dict:
    sim = Sim(variant)
    policy(sim)
    return sim.result()


@lru_cache(maxsize=None)
def reference_costs(variant: Variant) -> tuple[float, float]:
    return run(variant, do_nothing)["cost_total"], run(variant, oracle)["cost_total"]


def score(variant: Variant, cost: float) -> float:
    if variant.mode is None:
        u = 1 - cost / HEALTHY_SCALE
    else:
        reactive, best = reference_costs(variant)
        u = (reactive - cost) / (reactive - best)
    return max(-1.0, min(1.0, u))
