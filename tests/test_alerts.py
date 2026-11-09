"""T7: z-score analytics alerts, acknowledge and shelve."""

import random
from datetime import timedelta

import pytest

from assetopsgym.schemas import AlertActionEvent
from assetopsgym.server.scenarios import VARIANTS
from assetopsgym.server.sim import (ANALYTICS_COOLDOWN, Z_WINDOW_TICKS,
                                    Sim, ZWindow)


def test_zwindow_flags_a_step_not_noise():
    r, w = random.Random(0), ZWindow()
    for _ in range(Z_WINDOW_TICKS):
        w.push(r.gauss(0, 1))
    assert abs(w.z(0.0)) < 1 and w.z(6.0) > 3


def test_analytics_respect_cooldown_and_are_deterministic():
    v = VARIANTS["healthy_noisy"]
    a, b = Sim(v), Sim(v)
    a.run_to_end(), b.run_to_end()
    assert [x.public() for x in a.alerts] == [x.public() for x in b.alerts]
    last = {}
    for x in a.alerts:
        if x.kind == "analytics":
            gap = x.time - last.get(x.rule, x.time - ANALYTICS_COOLDOWN)
            assert gap >= ANALYTICS_COOLDOWN
            last[x.rule] = x.time
    assert len(last) == 5


def test_shelve_hides_but_does_not_change_the_world():
    v = VARIANTS["healthy_noisy"]
    free = Sim(v)
    free.run_to_end()
    hit = next(a for a in free.alerts if a.time >= v.start + timedelta(days=3))
    t0 = hit.time - timedelta(hours=12)
    sim = Sim(v)
    sim.advance((t0 - sim.now).total_seconds() / 3600)
    until = sim.shelve(hit.rule, 24)
    sim.run_to_end()
    assert [(a.time, a.rule) for a in sim.alerts] == \
           [(a.time, a.rule) for a in free.alerts]
    for a in sim.alerts:
        assert a.shelved == (a.rule == hit.rule and t0 <= a.time < until)
    assert hit.id not in {a.id for a in sim.visible_alerts()}
    shelves = [e for e in sim.events if isinstance(e, AlertActionEvent)]
    assert [(e.action, e.rule, e.hours) for e in shelves] == \
           [("shelve", hit.rule, 24)]


def test_shelve_rejects_unknown_rule_and_bad_hours():
    sim = Sim(VARIANTS["healthy"])
    with pytest.raises(KeyError):
        sim.shelve("chiller_trip", 24)
    with pytest.raises(ValueError):
        sim.shelve("high_kw_per_ton", 169)


def test_acknowledge_once():
    sim = Sim(VARIANTS["healthy_noisy"])
    sim.advance(24 * 5)
    a = sim.alerts[-1]
    sim.acknowledge(a.id)
    sim.advance(1)
    sim.acknowledge(a.id)
    assert a.acked_at is not None and a.public()["acked"]
    acks = [e for e in sim.events if isinstance(e, AlertActionEvent)]
    assert [(e.action, e.alert_id) for e in acks] == [("ack", a.id)]
    with pytest.raises(KeyError):
        sim.acknowledge("AL0")
