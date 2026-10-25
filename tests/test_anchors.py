"""FDE-Gym admission anchors, checked on every variant with the pure simulator.

    faulty:  do nothing = 0, oracle >= 0.95, every shortcut <= 0.8
    healthy: do nothing (the reference there) >= 0.95, every shortcut <= 0.8
"""

import pytest

from assetopsgym.baselines import SHORTCUTS
from assetopsgym.server.scenarios import VARIANTS, do_nothing, oracle, reference_costs, run, score
from assetopsgym.server.sim import Sim


@pytest.mark.parametrize("vid", sorted(VARIANTS))
def test_anchors(vid):
    v = VARIANTS[vid]
    nothing = score(v, run(v, do_nothing)["cost_total"])
    if v.mode:
        reactive, best = reference_costs(v)
        assert reactive > best
        assert nothing == 0.0
        assert score(v, run(v, oracle)["cost_total"]) >= 0.95
    else:
        assert nothing >= 0.95
    for name, policy in SHORTCUTS.items():
        assert score(v, run(v, policy)["cost_total"]) <= 0.8, name


def test_no_alarms_or_costs_when_healthy():
    r = run(VARIANTS["healthy"], do_nothing)
    assert r["alerts"]["operator_alarm"] == 0 and r["cost_total"] == 0


def test_faults_reach_an_operator_alarm_before_failing():
    for v in VARIANTS.values():
        if not v.mode:
            continue
        sim = Sim(v)
        sim.run_to_end()
        alarms = [a for a in sim.alerts if a.kind == "operator_alarm" and a.rule != "chiller_trip"]
        assert alarms and alarms[0].time < sim.failed_at, v.id


def test_deterministic():
    v = VARIANTS["refleak_mid"]
    assert Sim(v).history == Sim(v).history
    assert run(v, oracle) == run(v, oracle)
