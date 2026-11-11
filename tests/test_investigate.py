"""investigate: an inspection finds the fault only in the right
subsystem and only once it has grown. Pure."""

from datetime import timedelta

import pytest

from assetopsgym.schemas import ErrorCode, GymError
from assetopsgym.server.faults import (INSPECT_MIN_SEVERITY,
                                       SUBSYSTEMS)
from assetopsgym.server.maintenance import status
from assetopsgym.server.scenarios import VARIANTS
from assetopsgym.server.sim import Sim

LEAK = VARIANTS["refleak_mid"]  # onset day 15, 18 days to trip


def finding(sim: Sim, subsystem: str) -> str:
    """Inspect now at priority 1; the technician's closing note."""
    wo = sim.investigate(subsystem, priority=1)
    while status(wo) != "COMP":
        sim.advance(6)
    assert sim.severity(sim.now) < 1  # before the trip
    return [e.note for e in sim.events if e.type != "cost"
            and getattr(e, "wonum", None) == wo.wonum][-1]


def at_day(day: float) -> Sim:
    sim = Sim(LEAK)
    sim.advance(24 * day)
    return sim


def test_a_grown_fault_is_found_in_its_own_subsystem_only():
    sim = at_day(24)
    assert sim.severity(sim.now) > INSPECT_MIN_SEVERITY
    assert finding(sim, "condenser").endswith(
        "nothing abnormal found.")
    found = finding(sim, "refrigerant circuit")
    assert "refrigerant leak" in found and "CH-REFLEAK" in found


def test_a_young_fault_and_a_healthy_chiller_show_nothing():
    sim = at_day(16)  # a day after onset
    assert 0 < sim.severity(sim.now) < INSPECT_MIN_SEVERITY
    assert finding(sim, "refrigerant circuit").endswith(
        "nothing abnormal found.")
    for part in SUBSYSTEMS:
        assert finding(Sim(VARIANTS["healthy"]), part).endswith(
            "nothing abnormal found.")


def test_an_inspection_costs_a_visit_and_nothing_else():
    sim = at_day(24)
    finding(sim, "refrigerant circuit")
    r = sim.result()
    assert r.kpis.investigations == 1 and r.dispatches == 1
    assert r.kpis.false_dispatches == 0 == r.false_dispatches
    assert r.downtime_hours["planned_downtime"] == 0
    assert sim.severity(sim.now) > 0  # not repaired
    sim.file_wo(None, priority=1)  # a vague request still counts
    sim.advance(48)
    assert sim.kpis().false_dispatches == 1


def test_unknown_subsystem_is_a_typed_error():
    with pytest.raises(GymError) as e:
        Sim(LEAK).investigate("gearbox")
    assert e.value.code == ErrorCode.BAD_ARGUMENT
    assert "condenser" in e.value.message


def test_adding_a_code_turns_the_inspection_into_a_repair():
    sim = at_day(24)
    wo = sim.investigate("refrigerant circuit", priority=1)
    sim.update_workorder(wo.wonum, "MAIN", failure_code="CH-REFLEAK")
    sim.advance(24 * 3)
    assert status(wo) == "COMP" and sim.severity(sim.now) == 0
    assert sim.kpis().investigations == 0
    assert sim.now - wo.created < timedelta(days=4)
