"""The interface schemas hold what the code produces today, a variant loads
from a table row, and the ledger is a complete account of the score."""

import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta

import pytest

from assetopsgym.schemas import (AlertEvent, CostEvent, EpisodeResult, SimState,
                                 ErrorCode, ErrorResult, GymError)
from assetopsgym.server.scenarios import VARIANTS, do_nothing, oracle, run, score
from assetopsgym.server.faults import FAULT_MODES
from assetopsgym.server.maintenance import CONTRACTOR_DELAY_H
from assetopsgym.server.sim import Sim
from assetopsgym.server.variant import Technician, Variant, from_dict


@pytest.mark.parametrize("vid", sorted(VARIANTS))
def test_variant_loads_from_its_json(vid):
    v = VARIANTS[vid]
    row = json.loads(json.dumps(asdict(v), default=datetime.isoformat))
    assert from_dict(Variant, row) == v
    assert v.scoring == ("faulty" if v.mode else "healthy")


def test_from_dict_sparse_row_and_unknown_keys():
    v = from_dict(Variant, {
        "id": "x", "mode": None, "start": "2026-07-01T00:00:00",
        "crew": [{"id": "T1", "skills": ["CH-SENSOR"]}],
        "tags": ["shift"]})
    assert v.crew == (Technician("T1", ("CH-SENSOR",)),)
    assert v.start == datetime(2026, 7, 1) and v.tags == ("shift",)
    assert v.pf_days == Variant("y", None).pf_days
    with pytest.raises(KeyError):
        from_dict(Variant, {"id": "x", "mode": None, "onset": 3})


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


def test_tool_check_is_loud():
    from assetopsgym.server.environment import MANIFEST, check_tools

    assert MANIFEST.name == "assetopsgym" and "finish" in MANIFEST.tools
    check_tools(list(reversed(MANIFEST.tools)), MANIFEST.tools)
    for bad in (MANIFEST.tools[1:], [*MANIFEST.tools, "x"]):
        with pytest.raises(RuntimeError):
            check_tools(list(bad), MANIFEST.tools)


def test_gym_error_becomes_error_result():
    from assetopsgym.server.environment import _errors_as_results

    def bad(x: int) -> int:
        raise GymError(ErrorCode.BAD_ARGUMENT, f"no {x}")

    assert _errors_as_results(bad)(3) == ErrorResult(
        error="no 3", error_code=ErrorCode.BAD_ARGUMENT)
    assert _errors_as_results(abs)(-2) == 2


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
    small = replace(base, id="compwear_smallcrew", crew=(
        Technician("TECH-01", skills=("CH-CWFLOW", "CH-REFLEAK")),))
    wo_full = run_one(base)
    wo_small = run_one(small)
    assert wo_small.arrive_at - wo_full.arrive_at >= timedelta(hours=CONTRACTOR_DELAY_H)
    assert wo_small.sla_breach and not wo_full.sla_breach


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
