"""The work-order heap and the AssetOpsBench-shaped writes. Pure."""

from dataclasses import replace
from datetime import timedelta

import pytest

from assetopsgym.schemas import CostEvent, ErrorCode, GymError
from assetopsgym.server.maintenance import NOSHOW_DELAY_H, status
from assetopsgym.server.scenarios import VARIANTS
from assetopsgym.server.sim import Sim


def sim_with(noshow: float) -> Sim:
    return Sim(replace(VARIANTS["healthy"], noshow=noshow))


def notes(sim: Sim, wonum: str) -> list[str]:
    return [e.note for e in sim.events
            if getattr(e, "wonum", None) == wonum and e.type != "cost"]


def dispatches(sim: Sim) -> int:
    return sum(isinstance(e, CostEvent) and e.kind == "dispatch"
               for e in sim.events)


def test_noshow_reschedules_the_arrival():
    sim = sim_with(noshow=1.0)
    wo = sim.file_wo("CH-SENSOR", priority=1)
    sim.advance(72)
    approved = next(e for e in sim.events
                    if getattr(e, "status", None) == "APPR")
    missed = sum(n.startswith("Technician did not arrive")
                 for n in notes(sim, wo.wonum))
    assert missed >= 2 and status(wo) == "APPR"
    assert wo.arrive_at - approved.arrive_at == timedelta(
        hours=NOSHOW_DELAY_H * missed)
    assert dispatches(sim) == 0


def test_update_stales_the_old_arrival():
    sim = sim_with(noshow=0.0)
    wo = sim.file_wo("CH-SENSOR", priority=5)
    sim.advance(13)  # approval is 2-12 h after filing
    slow = wo.arrive_at
    sim.update_workorder(wo.wonum, "MAIN", priority=1)
    assert wo.arrive_at < slow
    while sim.now <= slow + timedelta(days=1):
        sim.advance(24)
    assert status(wo) == "COMP" and dispatches(sim) == 1


def test_bad_writes_raise():
    sim = sim_with(noshow=0.0)
    with pytest.raises(GymError) as e:
        sim.generate_work_order("check", "Chiller 9", "MAIN")
    assert e.value.code == ErrorCode.UNKNOWN_ASSET
    assert "'Chiller 6'" in e.value.message
    wo = sim.generate_work_order("check", "Chiller 6", "main")
    with pytest.raises(GymError) as e:
        sim.update_workorder(wo.wonum, "MAIN", failure_code="CH-NOPE")
    assert e.value.code == ErrorCode.UNKNOWN_FAILURE_CODE
    assert "CH-CWFLOW" in e.value.message
    with pytest.raises(GymError) as e:
        sim.cancel_workorder("WO-404", "MAIN")
    assert e.value.code == ErrorCode.NOT_FOUND
    sim.update_workorder(wo.wonum, "MAIN", failure_code="ch-sensor")
    sim.advance(72)
    assert status(wo) in ("INPRG", "COMP")
    sim.update_workorder(wo.wonum, "MAIN", priority=wo.priority)
    for write in (lambda: sim.update_workorder(wo.wonum, "MAIN",
                                               priority=1),
                  lambda: sim.cancel_workorder(wo.wonum, "MAIN")):
        with pytest.raises(GymError) as e:
            write()
        assert e.value.code == ErrorCode.WORK_ORDER_LOCKED
