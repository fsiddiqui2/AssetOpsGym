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

from dataclasses import fields, replace
from functools import lru_cache
from typing import Callable

from ..schemas import EpisodeResult, EventScore, EventTerms
from .faults import trip_time
from .maintenance import DISPATCH_COST
from .sim import Sim
from .variant import Variant

HEALTHY_SCALE = 5 * DISPATCH_COST

# The seven-term score (logged beside the dollar score). Weights
# are the scenario sheet's provisional ones, in downtime-hour
# equivalents; the references are means over REFERENCE_SEEDS, the
# development seeds, so a variant where the threshold rule is late
# in only some seeds still has a gap to divide by.
WEIGHTS = EventTerms(
    unplanned_downtime_hours=1, missed_failures=24,
    false_dispatches=4, acknowledgements=0.005, investigations=0.5,
    unaddressed_faults=12, sla_breaches=2)
REFERENCE_SEEDS = range(1000, 1020)
EPSILON = 0.01  # a smaller threshold-oracle gap is no gap
# the BMS rule that points at each fault
RULE_CODE = {
    "low_condenser_water_flow": "CH-CWFLOW",
    "high_evaporator_approach": "CH-REFLEAK",
    "compressor_load_mismatch": "CH-COMPWEAR",
}

VARIANTS = {v.id: v for v in [
    Variant("cwflow_early", "CH-CWFLOW", onset_day=6, pf_days=24, seed=11),
    Variant("cwflow_noshow", "CH-CWFLOW", onset_day=18, pf_days=20, noshow=0.5, seed=12),
    Variant("refleak_mid", "CH-REFLEAK", onset_day=15, pf_days=18, seed=21),
    Variant("refleak_fast", "CH-REFLEAK", onset_day=9, pf_days=8, seed=22),
    Variant("compwear_late", "CH-COMPWEAR", onset_day=28, pf_days=22, seed=31),
    Variant("healthy", None, seed=41),
    Variant("healthy_noisy", None, blips_per_day=5, seed=42),
    Variant("healthy_chatter", None, seed=43,
            chatter_rule="low_condenser_water_flow",
            chatter_start_day=3, chatter_end_day=9),
]}


def do_nothing(sim: Sim) -> None:
    sim.run_to_end()


def oracle(sim: Sim) -> None:
    """Knows the fault and its onset: files the right work order
    right after onset, as planned work if the fault cannot fail
    inside the horizon."""
    if sim.v.mode:
        sim.advance((sim.v.onset - sim.now).total_seconds() / 3600 + 0.25)
        urgent = trip_time(sim.fault) < sim.v.end
        sim.file_wo(sim.v.mode, priority=2 if urgent else 4)
    sim.run_to_end()


def threshold_on_delay(sim: Sim, check_hours: float = 24) -> None:
    """Daily check; file a work order for each operator alarm that
    maps to a fault. The plant's alarm-driven practice, and the
    seven-term score's zero."""
    seen, filed = 0, set()
    while not sim.done:
        sim.advance(check_hours)
        for a in sim.alerts[seen:]:
            code = RULE_CODE.get(a.rule)
            if a.kind == "operator_alarm" and code and code not in filed:
                sim.file_wo(code, priority=2)
                filed.add(code)
        seen = len(sim.alerts)


def run(variant: Variant, policy) -> EpisodeResult:
    sim = Sim(variant)
    policy(sim)
    return sim.result()


@lru_cache(maxsize=None)
def reference_costs(variant: Variant) -> tuple[float, float]:
    return (run(variant, do_nothing).cost_total,
            run(variant, oracle).cost_total)


def score(variant: Variant, cost: float) -> float:
    if variant.mode is None:
        u = 1 - cost / HEALTHY_SCALE
    else:
        reactive, best = reference_costs(variant)
        u = (reactive - cost) / (reactive - best)
    return max(-1.0, min(1.0, u))


def event_cost(terms: EventTerms) -> float:
    return sum(getattr(WEIGHTS, f.name) * getattr(terms, f.name)
               for f in fields(EventTerms))


def event_references(variant: Variant) -> tuple[float, float]:
    """The variant's mean seven-term cost under the threshold rule
    and under the oracle, whatever seed this episode has."""
    return _event_references(replace(variant, seed=0))


@lru_cache(maxsize=None)
def _event_references(variant: Variant) -> tuple[float, float]:
    def mean(policy: Callable[[Sim], None]) -> float:
        return sum(event_cost(run(replace(variant, seed=s),
                                  policy).event_terms)
                   for s in REFERENCE_SEEDS) / len(REFERENCE_SEEDS)
    return mean(threshold_on_delay), mean(oracle)


def event_score(variant: Variant, r: EpisodeResult) -> EventScore:
    """Threshold rule = 0, oracle = 1, clipped to [-1, 1]; a
    failure caps it at 0. No score where the references tie."""
    threshold, best = event_references(variant)
    cost, gap = event_cost(r.event_terms), threshold - best
    raw = (threshold - cost) / gap if gap > EPSILON else None
    capped = None if raw is None else max(-1.0, min(1.0, raw))
    if capped is not None and r.failed:
        capped = min(0.0, capped)
    return EventScore(
        cost=round(cost, 4), threshold_cost=round(threshold, 4),
        oracle_cost=round(best, 4),
        raw=None if raw is None else round(raw, 4),
        score=None if capped is None else round(capped, 4))
