"""The four interface schemas round-trip what the code produces today, and the ledger
is a complete account of the score."""

from datetime import timedelta

import pytest
import yaml

from assetopsgym.schemas import (AGENT_TOOLS, AlertEvent, CostEvent, EpisodeResult, Scenario, SimState,
                                 Technician)
from assetopsgym.server.scenarios import VARIANTS, do_nothing, oracle, run, score
from assetopsgym.server.sim import CONTRACTOR_DELAY_H, FAULT_MODES, Sim, Variant
from assetopsgym.server.sim import Technician as SimTech
from pathlib import Path


@pytest.mark.parametrize("vid", sorted(VARIANTS))
def test_scenario_round_trips_variant(vid):
    v = VARIANTS[vid]
    s = Scenario.from_variant(v)
    assert s.to_variant() == v
    assert s.scoring == ("faulty" if v.mode else "healthy")
    assert Scenario.model_validate_json(s.model_dump_json()) == s


@pytest.mark.parametrize("vid", sorted(VARIANTS))
def test_result_state_and_events_validate(vid):
    v = VARIANTS[vid]
    sim = Sim(v)
    sim.advance(24 * 30)
    SimState.model_validate(sim.state())
    sim.file_wo("CH-CWFLOW", priority=2)
    sim.run_to_end()
    r = EpisodeResult.model_validate(sim.result())
    assert r.events == len(sim.events)
    for a in sim.alerts:
        AlertEvent.model_validate(a.public())
    # Every dollar in the score has a cost event behind it, by kind.
    by_kind = {}
    for e in sim.events:
        if isinstance(e, CostEvent):
            by_kind[e.kind] = by_kind.get(e.kind, 0.0) + e.amount
    for kind, total in sim.ledger.items():
        assert abs(by_kind.get(kind, 0.0) - total) < 0.01, kind
    # Downtime spans account for the downtime hours.
    spans = sum(e.hours for e in sim.events if e.type == "downtime")
    assert abs(spans - sum(sim.hours.values())) < 1e-6


def test_tool_list_matches_manifest():
    manifest = yaml.safe_load((Path(__file__).resolve().parents[1] / "openenv.yaml").read_text())
    declared = manifest["validation"]["capabilities"]["declared_tools"]
    assert [t.name for t in AGENT_TOOLS] == declared


def test_inject_fault_at_run_time():
    v = VARIANTS["healthy"]
    sim = Sim(v)
    sim.advance(24 * 10)
    sim.inject_fault("CH-REFLEAK", sim.now, pf_days=10)
    sim.run_to_end()
    assert sim.ever_failed and sim.result()["mode"] == "CH-REFLEAK"
    with pytest.raises(ValueError):
        sim.inject_fault("CH-NOPE", sim.now, 1)


def test_crew_without_the_skill_waits_for_a_contractor():
    base = VARIANTS["compwear_late"]
    small = Variant(**{**base.__dict__, "id": "compwear_smallcrew",
                       "crew": (SimTech("TECH-01", skills=("CH-CWFLOW", "CH-REFLEAK")),)})
    wo_full = run_one(base)
    wo_small = run_one(small)
    assert wo_small.arrive_at - wo_full.arrive_at >= timedelta(hours=CONTRACTOR_DELAY_H)
    assert wo_small.sla_breach and not wo_full.sla_breach
    assert Scenario.from_variant(small).crew.technicians[0] == Technician(id="TECH-01",
                                                                         skills=["CH-CWFLOW", "CH-REFLEAK"])


def run_one(v):
    sim = Sim(v)
    sim.advance(24 * 30)
    wo = sim.file_wo("CH-COMPWEAR", priority=2)
    sim.run_to_end()
    return wo


def test_default_crew_leaves_reference_scores_alone():
    for v in VARIANTS.values():
        assert score(v, run(v, do_nothing)["cost_total"]) == (0.0 if v.mode else 1.0)
        assert score(v, run(v, oracle)["cost_total"]) >= 0.95
        assert all(m in FAULT_MODES for m in [v.mode] if m)
