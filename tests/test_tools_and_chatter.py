"""The rest of the agent's work-order tools on the simulator, and
the chattering transmitter. Pure."""

from dataclasses import replace
from datetime import timedelta

import pytest

from assetopsgym.schemas import ErrorCode, GymError
from assetopsgym.server.alerts import CHATTER_OVER, RULE
from assetopsgym.server.maintenance import (MAX_DEFER_H,
                                            RESPONSE_HOURS, status)
from assetopsgym.server.scenarios import VARIANTS, do_nothing, run
from assetopsgym.server.sim import Sim
from assetopsgym.server.variant import Technician

QUIET = replace(VARIANTS["healthy"], noshow=0.0)
CHATTER = replace(VARIANTS["healthy_chatter.base"], noshow=0.0)


def approved(sim: Sim, code: str, priority: int):
    wo = sim.file_wo(code, priority=priority)
    sim.advance(13)  # approval is 2-12 h after filing
    assert status(wo) == "APPR"
    return wo


def test_one_call_files_the_code_and_needs_a_description():
    sim = Sim(QUIET)
    wo = sim.file_wo("ch-sensor", 2, None, "Flow reads low", "why")
    assert wo.code == "CH-SENSOR" and wo.description == "Flow reads low"
    with pytest.raises(GymError) as e:
        sim.file_wo(None, 3, None, "")
    assert e.value.code == ErrorCode.BAD_ARGUMENT


def test_defer_moves_the_visit_and_its_target_together():
    sim = Sim(QUIET)
    wo = approved(sim, "CH-SENSOR", 4)
    arrive, target = wo.arrive_at, wo.target_arrive
    sim.defer(wo.wonum, 48, "not urgent")
    assert wo.arrive_at - arrive == timedelta(hours=48)
    assert wo.target_arrive - target == timedelta(hours=48)
    sim.update_workorder(wo.wonum, "MAIN", priority=1)  # keeps it
    assert wo.arrive_at - wo.approve_at >= timedelta(
        hours=48 + RESPONSE_HOURS[1])
    while status(wo) != "COMP":
        sim.advance(24)
    assert not wo.sla_breach and wo.done  # late by choice, no breach
    for bad in (0, MAX_DEFER_H + 1):
        with pytest.raises(GymError):
            sim.defer(sim.file_wo(None).wonum, bad)
    with pytest.raises(GymError) as e:
        sim.defer(wo.wonum, 24)
    assert e.value.code == ErrorCode.WORK_ORDER_LOCKED


def test_a_deferral_before_approval_still_applies():
    sim = Sim(QUIET)
    wo = sim.file_wo("CH-SENSOR", priority=1)
    sim.defer(wo.wonum, 72)
    sim.advance(48)
    assert status(wo) == "APPR"
    assert wo.arrive_at - wo.approve_at >= timedelta(hours=76)


def test_the_agents_technician_replaces_the_planners():
    far = Technician("FAR", travel_h=20)
    near = Technician("NEAR", skills=("CH-CWFLOW",))
    sim = Sim(replace(QUIET, crew=(far, near)))
    wo = sim.file_wo("CH-CWFLOW", priority=1)
    sim.assign_technician(wo.wonum, "FAR")  # before approval
    sim.advance(13)
    assert wo.tech == "FAR"
    slow = wo.arrive_at
    sim.assign_technician(wo.wonum, "NEAR")  # after it
    assert slow - wo.arrive_at == timedelta(hours=20)
    with pytest.raises(GymError) as e:
        sim.assign_technician(wo.wonum, "NOBODY")
    assert e.value.code == ErrorCode.NOT_FOUND


def test_chatter_alarms_at_its_rate_inside_its_window_only():
    sim = Sim(CHATTER)
    sim.run_to_end()
    rule = RULE[CHATTER.chatter_rule]
    days = [(a.time - CHATTER.start) / timedelta(days=1)
            for a in sim.alerts if a.kind == "operator_alarm"]
    assert days and all(3 <= d < 9 for d in days)
    assert 30 <= len(days) / 6 <= 50  # about 40 a day
    assert not [a for a in sim.alerts if a.kind == "operator_alarm"
                and a.rule != rule.name]
    over = [a.value for a in sim.alerts
            if a.kind == "operator_alarm"]
    assert all(abs(x - CHATTER_OVER * rule.threshold) < 1e-3
               for x in over)


def test_chatter_leaves_the_plants_own_noise_alone():
    plain = Sim(replace(CHATTER, chatter_rule=None))
    noisy = Sim(CHATTER)
    a, b = plain.advance(24 * 12), noisy.advance(24 * 12)
    sensor = RULE[CHATTER.chatter_rule].sensor
    for x, y in zip(a, b):
        assert {k: v for k, v in x.values.items() if k != sensor} \
            == {k: v for k, v in y.values.items() if k != sensor}
    changed = sum(x.values[sensor] != y.values[sensor]
                  for x, y in zip(a, b))
    assert 150 < changed < 350  # ~0.42 of the running ticks, 6 days


def test_inspecting_and_repairing_the_transmitter():
    sim = Sim(CHATTER)
    sim.advance(24 * 4)
    look = sim.investigate("sensors", priority=1)
    sim.advance(24)  # the inspection first, then the repair
    fix = sim.file_wo("CH-SENSOR", priority=1)
    sim.advance(24)
    notes = {e.wonum: e.note for e in sim.events
             if e.type == "service_request"}
    assert "faulty transmitter" in notes[look.wonum]
    assert "low_condenser_water_flow" in notes[look.wonum]
    assert notes[fix.wonum].startswith("Replaced the faulty")
    seen = len(sim.alerts)
    sim.advance(24 * 3)  # still inside the window, now quiet
    assert not [a for a in sim.alerts[seen:]
                if a.kind == "operator_alarm"]
    k = sim.kpis()
    assert k.false_dispatches == 0 and k.investigations == 1
    early = Sim(CHATTER)  # before the chatter: nothing to find
    early.file_wo("CH-SENSOR", priority=1)
    early.advance(48)
    assert early.kpis().false_dispatches == 1


def test_unknown_chatter_rule_is_refused():
    with pytest.raises(ValueError):
        Sim(replace(QUIET, chatter_rule="no_such_rule"))
    assert run(CHATTER, do_nothing).cost_total == 0
