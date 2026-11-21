"""The score's anchors, on each suite scenario's base variant with
the pure simulator (`python -m assetopsgym.baselines` prints every
variant).

    the oracle scores 1
    doing nothing: at most 0 where the fault can fail, 1 elsewhere
    every shortcut: at most 0.8
"""

from dataclasses import replace

import pytest

from assetopsgym.baselines import SHORTCUTS
from assetopsgym.server.scenarios import (DEV, SUITE, VARIANTS,
                                          do_nothing, oracle,
                                          references, run, score)
from assetopsgym.server.sim import Sim

BASES = [v for v in SUITE if v.id.endswith(".base")]


def test_the_suite_is_ten_scenarios_of_three_or_four_variants():
    sizes = {}
    for v in SUITE:
        sizes[v.scenario] = sizes.get(v.scenario, 0) + 1
    assert len(sizes) == len(BASES) == 10
    assert set(sizes.values()) <= {3, 4}
    assert not {v.scenario for v in DEV} & set(sizes)


@pytest.mark.parametrize("v", BASES, ids=lambda v: v.scenario)
def test_anchors(v):
    threshold, best = references(v)
    assert threshold >= best
    assert score(v, run(v, oracle)).score >= 0.95
    nothing = score(v, run(v, do_nothing)).score
    can_fail = v.mode and v.onset_day + v.pf_days < v.horizon_days
    assert nothing <= 0 if can_fail else nothing == 1
    for name, policy in SHORTCUTS.items():
        assert score(v, run(v, policy)).score <= 0.8, name


def test_no_alarms_or_costs_when_healthy():
    r = run(VARIANTS["healthy"], do_nothing)
    assert r.alerts["operator_alarm"] == 0 and r.cost_total == 0


def test_faults_reach_an_operator_alarm_before_failing():
    for v in DEV:
        if not v.mode:
            continue
        sim = Sim(v)
        sim.run_to_end()
        alarms = [a for a in sim.alerts if a.kind == "operator_alarm" and a.rule != "chiller_trip"]
        assert alarms and alarms[0].time < sim.failed_at, v.id


def test_reset_seed_varies_the_run():
    runs = [replace(VARIANTS["refleak_mid"], seed=s) for s in (0, 1, 2)]
    assert len({tuple(map(str, Sim(v).history[:96])) for v in runs}) == 3
    assert len({references(v) for v in runs}) == 1  # per variant


def test_deterministic():
    v = VARIANTS["refleak_mid"]
    assert Sim(v).history == Sim(v).history
    assert run(v, oracle) == run(v, oracle)


def test_unseeded_reset_is_replayable():
    from assetopsgym.server.environment import pick_variant
    v = pick_variant(None, None)
    assert pick_variant(None, v.seed) == v and v.id in {
        s.id for s in SUITE}
    assert pick_variant("healthy", None) == VARIANTS["healthy"]
