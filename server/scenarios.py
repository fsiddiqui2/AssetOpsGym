"""The scenario suite, the reference policies and the score.

SUITE is the evaluation: ten scenarios of three or four variants on
one chiller over 60 days, each variant one changed setting from its
scenario's base (`<scenario>.<variant>`). DEV holds the seven older
variants, for tests and quick runs; they are not in the suite.

Score: the scenario sheet's seven weighted event terms, in
downtime-hour equivalents, normalised per variant:

    S = (C_threshold - C_agent) / (C_threshold - C_oracle)

clipped to [-1, 1] and capped at 0 after a failure. C_threshold is
the plant's alarm-driven practice and C_oracle files the right work
order at fault onset; both are the variant's means over
REFERENCE_SEEDS, so a variant where the threshold rule is late in
only some seeds still has a gap to divide by. Where the two tie (a
healthy chiller, or a fault the alarm catches in time):

    S = 1 - (C_agent - C_oracle) / TIE_SCALE
"""

from __future__ import annotations

from dataclasses import fields, replace
from functools import lru_cache
from typing import Callable

from ..schemas import EpisodeResult, EventScore, EventTerms
from .faults import FAILURE_CODES, FAULT_MODES, trip_time
from .sim import Sim
from .variant import DEFAULT_CREW, Technician, Variant

# Weights are the scenario sheet's provisional ones.
WEIGHTS = EventTerms(
    unplanned_downtime_hours=1, missed_failures=24,
    false_dispatches=4, acknowledgements=0.005, investigations=0.5,
    unaddressed_faults=12, sla_breaches=2)
REFERENCE_SEEDS = range(1000, 1020)  # the development seeds
TEST_SEEDS = range(2000, 2020)  # unrun until scoring is frozen
EPSILON = 0.01  # a smaller threshold-oracle gap is a tie
# on a tie, five false dispatches' worth of cost is a score of 0
TIE_SCALE = 5 * WEIGHTS.false_dispatches
# the BMS rule that points at each fault
RULE_CODE = {
    "low_condenser_water_flow": "CH-CWFLOW",
    "high_evaporator_approach": "CH-REFLEAK",
    "compressor_load_mismatch": "CH-COMPWEAR",
}

_FLOW, _LEAK, _WEAR = FAULT_MODES
# nobody repairs leaks: a leak order waits for the contractor
_NO_LEAK = tuple(
    replace(t, skills=tuple(c for c in FAILURE_CODES if c != _LEAK))
    for t in DEFAULT_CREW)


def _far(hours: float) -> tuple[Technician, ...]:
    return tuple(replace(t, travel_h=hours) for t in DEFAULT_CREW)


_CHATTER = "low_condenser_water_flow"

SUITE = (
    # fast faults: the operator alarm leaves about a day
    Variant("cf_fast.base", _FLOW, onset_day=20, pf_days=8),
    Variant("cf_fast.onset8", _FLOW, onset_day=8, pf_days=8),
    Variant("cf_fast.onset40", _FLOW, onset_day=40, pf_days=8),
    Variant("cf_fast.pf6", _FLOW, onset_day=20, pf_days=6),
    Variant("rl_fast.base", _LEAK, onset_day=15, pf_days=8),
    Variant("rl_fast.onset35", _LEAK, onset_day=35, pf_days=8),
    Variant("rl_fast.pf6", _LEAK, onset_day=15, pf_days=6),
    Variant("rl_fast.pf10", _LEAK, onset_day=15, pf_days=10),
    # compressor parts take 48 h
    Variant("cw_parts_wait.base", _WEAR, onset_day=20, pf_days=22),
    Variant("cw_parts_wait.onset30", _WEAR, onset_day=30, pf_days=22),
    Variant("cw_parts_wait.pf18", _WEAR, onset_day=20, pf_days=18),
    Variant("cw_parts_wait.pf26", _WEAR, onset_day=20, pf_days=26),
    Variant("rl_no_skill.base", _LEAK, onset_day=15, pf_days=20,
            crew=_NO_LEAK),
    Variant("rl_no_skill.onset30", _LEAK, onset_day=30, pf_days=20,
            crew=_NO_LEAK),
    Variant("rl_no_skill.pf16", _LEAK, onset_day=15, pf_days=16,
            crew=_NO_LEAK),
    Variant("rl_no_skill.pf24", _LEAK, onset_day=15, pf_days=24,
            crew=_NO_LEAK),
    # a real leak among sensor glitches
    Variant("rl_masked_by_blips.base", _LEAK, onset_day=25,
            pf_days=10, blips_per_day=5),
    Variant("rl_masked_by_blips.blips8", _LEAK, onset_day=25,
            pf_days=10, blips_per_day=8),
    Variant("rl_masked_by_blips.onset12", _LEAK, onset_day=12,
            pf_days=10, blips_per_day=5),
    Variant("rl_masked_by_blips.pf12", _LEAK, onset_day=25,
            pf_days=12, blips_per_day=5),
    # the threshold rule is late in some seeds only
    Variant("cf_noshow.base", _FLOW, onset_day=20, pf_days=16,
            noshow=0.5),
    Variant("cf_noshow.onset35", _FLOW, onset_day=35, pf_days=16,
            noshow=0.5),
    Variant("cf_noshow.noshow30", _FLOW, onset_day=20, pf_days=16,
            noshow=0.3),
    Variant("cf_noshow.pf20", _FLOW, onset_day=20, pf_days=20,
            noshow=0.5),
    Variant("cf_remote_crew.base", _FLOW, onset_day=20, pf_days=16,
            crew=_far(12)),
    Variant("cf_remote_crew.travel18", _FLOW, onset_day=20,
            pf_days=16, crew=_far(18)),
    Variant("cf_remote_crew.onset35", _FLOW, onset_day=35,
            pf_days=16, crew=_far(12)),
    Variant("cf_remote_crew.pf20", _FLOW, onset_day=20, pf_days=20,
            crew=_far(12)),
    # cannot fail inside the 60 days: planned work, not a dispatch
    Variant("cf_incipient_defer.base", _FLOW, onset_day=20,
            pf_days=45),
    Variant("cf_incipient_defer.onset17", _FLOW, onset_day=17,
            pf_days=45),
    Variant("cf_incipient_defer.pf42", _FLOW, onset_day=20,
            pf_days=42),
    Variant("healthy_quiet.base", None),
    Variant("healthy_quiet.blips_half", None, blips_per_day=0.5),
    Variant("healthy_quiet.blips2", None, blips_per_day=2),
    Variant("healthy_chatter.base", None, chatter_rule=_CHATTER,
            chatter_start_day=3, chatter_end_day=9),
    Variant("healthy_chatter.rule", None,
            chatter_rule="high_evaporator_approach",
            chatter_start_day=3, chatter_end_day=9),
    Variant("healthy_chatter.rate20", None, chatter_rule=_CHATTER,
            chatter_per_day=20, chatter_start_day=3,
            chatter_end_day=9),
    Variant("healthy_chatter.long", None, chatter_rule=_CHATTER,
            chatter_start_day=3, chatter_end_day=20),
)

DEV = (
    Variant("cwflow_early", _FLOW, onset_day=6, pf_days=24, seed=11),
    Variant("cwflow_noshow", _FLOW, onset_day=18, pf_days=20,
            noshow=0.5, seed=12),
    Variant("refleak_mid", _LEAK, onset_day=15, pf_days=18, seed=21),
    Variant("refleak_fast", _LEAK, onset_day=9, pf_days=8, seed=22),
    Variant("compwear_late", _WEAR, onset_day=28, pf_days=22,
            seed=31),
    Variant("healthy", None, seed=41),
    Variant("healthy_noisy", None, blips_per_day=5, seed=42),
)

VARIANTS = {v.id: v for v in (*SUITE, *DEV)}


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
    score's zero."""
    seen, filed = 0, set()
    while not sim.done:
        sim.advance(check_hours)
        for a in sim.alerts[seen:]:
            code = RULE_CODE.get(a.rule)
            if a.kind == "operator_alarm" and code and code not in filed:
                sim.file_wo(code, priority=2)
                filed.add(code)
        seen = len(sim.alerts)


def run(variant: Variant, policy: Callable[[Sim], None]
        ) -> EpisodeResult:
    sim = Sim(variant)
    policy(sim)
    return sim.result()


def event_cost(terms: EventTerms) -> float:
    return sum(getattr(WEIGHTS, f.name) * getattr(terms, f.name)
               for f in fields(EventTerms))


def references(variant: Variant) -> tuple[float, float]:
    """The variant's mean cost under the threshold rule and under
    the oracle, whatever seed this episode has."""
    return _references(replace(variant, seed=0))


@lru_cache(maxsize=None)
def _references(variant: Variant) -> tuple[float, float]:
    def mean(policy: Callable[[Sim], None]) -> float:
        return sum(event_cost(run(replace(variant, seed=s),
                                  policy).event_terms)
                   for s in REFERENCE_SEEDS) / len(REFERENCE_SEEDS)
    return mean(threshold_on_delay), mean(oracle)


def score(variant: Variant, r: EpisodeResult) -> EventScore:
    threshold, best = references(variant)
    cost, gap = event_cost(r.event_terms), threshold - best
    raw = (threshold - cost) / gap if gap > EPSILON else None
    s = 1 - (cost - best) / TIE_SCALE if raw is None else raw
    s = max(-1.0, min(1.0, s))
    if r.failed:
        s = min(0.0, s)
    return EventScore(
        cost=round(cost, 4), threshold_cost=round(threshold, 4),
        oracle_cost=round(best, 4),
        raw=None if raw is None else round(raw, 4),
        score=round(s, 4))
