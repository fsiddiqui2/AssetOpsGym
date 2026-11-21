"""The seven-term cost and its score (logged, not the reward).
Pure."""

from dataclasses import replace
from datetime import timedelta

import pytest

from assetopsgym.server import scenarios
from assetopsgym.server.maintenance import latest_safe_filing
from assetopsgym.server.scenarios import (VARIANTS, do_nothing,
                                          event_cost, event_score,
                                          oracle, run,
                                          threshold_on_delay)
from assetopsgym.server.sim import Sim
from assetopsgym.server.variant import Variant

# onset day 9, 8 days to trip: the threshold rule is too late
LEAK = replace(VARIANTS["refleak_fast"], noshow=0.0)
SLOW = Variant("slow", "CH-CWFLOW", onset_day=20, pf_days=45,
               noshow=0.0)  # cannot fail inside the 60 days


@pytest.fixture
def four_seeds(monkeypatch):
    monkeypatch.setattr(scenarios, "REFERENCE_SEEDS",
                        range(1000, 1004))
    scenarios._event_references.cache_clear()
    yield scenarios.REFERENCE_SEEDS
    scenarios._event_references.cache_clear()


def test_doing_nothing_pays_downtime_failure_and_no_response():
    t = run(LEAK, do_nothing).event_terms
    assert (t.missed_failures, t.unaddressed_faults) == (1, 1)
    assert t.false_dispatches == t.investigations == 0
    assert t.unplanned_downtime_hours > 0
    assert event_cost(t) == pytest.approx(
        t.unplanned_downtime_hours + 24 + 12 + 2 * t.sla_breaches)
    assert event_cost(run(LEAK, oracle).event_terms) == 0


def filed_at(v: Variant, when, priority: int) -> Sim:
    sim = Sim(v)
    sim.advance((when - sim.now) / timedelta(hours=1))
    sim.file_wo(v.mode, priority=priority)
    sim.run_to_end()
    return sim


def test_the_latest_safe_time_separates_addressed_from_not():
    trip = LEAK.onset + timedelta(days=LEAK.pf_days)
    deadline = latest_safe_filing(LEAK, LEAK.mode, trip)
    # 12 h approval + 4 h response + 4 h late + 12 h repair
    assert trip - deadline == timedelta(hours=32)
    on_time = filed_at(LEAK, deadline, 1)
    assert not on_time.ever_failed
    assert on_time.result().event_terms.unaddressed_faults == 0
    late = filed_at(LEAK, deadline + timedelta(hours=1), 1)
    assert late.result().event_terms.unaddressed_faults == 1


def test_urgent_visit_for_a_fault_that_cannot_fail_is_false():
    when = SLOW.onset + timedelta(days=10)
    urgent = filed_at(SLOW, when, 2).result()
    planned = filed_at(SLOW, when, 4).result()
    assert urgent.event_terms.false_dispatches == 1
    assert planned.event_terms.false_dispatches == 0
    assert urgent.kpis.false_dispatches == 0  # the KPI is unchanged
    assert not urgent.failed and not planned.failed
    for r in (urgent, planned, run(SLOW, do_nothing)):
        assert r.event_terms.unaddressed_faults == 0
    assert event_cost(run(SLOW, oracle).event_terms) == 0


def test_acknowledgements_and_inspections_are_charged():
    sim = Sim(replace(VARIANTS["healthy_noisy"], noshow=0.0))
    sim.advance(24 * 5)
    for a in sim.alerts[-3:]:
        sim.acknowledge(a.id)
    sim.investigate("condenser", priority=1)
    sim.run_to_end()
    t = sim.result().event_terms
    assert (t.acknowledgements, t.investigations) == (3, 1)
    assert event_cost(t) == pytest.approx(3 * 0.005 + 0.5)


def test_threshold_is_zero_and_the_oracle_one(four_seeds):
    def mean_raw(policy):
        return sum(event_score(LEAK, run(replace(LEAK, seed=s),
                                         policy)).raw
                   for s in four_seeds) / len(four_seeds)
    assert mean_raw(threshold_on_delay) == pytest.approx(0, abs=1e-3)
    assert mean_raw(oracle) == pytest.approx(1, abs=1e-3)
    nothing = event_score(LEAK, run(LEAK, do_nothing))
    assert nothing.raw < 0 and nothing.score <= 0


def test_a_failure_caps_the_score_and_a_tie_has_none(four_seeds):
    # late, but before the threshold rule would be: fails, cheaply
    trip = LEAK.onset + timedelta(days=LEAK.pf_days)
    r = filed_at(LEAK, trip - timedelta(hours=20), 1).result()
    s = event_score(LEAK, r)
    assert r.failed and s.raw > 0 and s.score == 0
    healthy = VARIANTS["healthy"]
    tie = event_score(healthy, run(healthy, do_nothing))
    assert tie.raw is None and tie.score is None and tie.cost == 0
    chatter = VARIANTS["healthy_chatter"]  # the rule dispatches
    assert event_score(chatter, run(chatter, do_nothing)).score == 1
