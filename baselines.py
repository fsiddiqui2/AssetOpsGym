"""Scripted policies on the pure simulator, and the suite's table.

    uv run python -m assetopsgym.baselines     # about 3 minutes

Shortcut policies are the strategies FDE-Gym requires to score at most 0.8. The
threshold rule and the oracle (server/scenarios.py) are the score's 0 and 1.
"""

from __future__ import annotations

from assetopsgym.server.scenarios import (RULE_CODE, SUITE,
                                          do_nothing, oracle, run,
                                          score, threshold_on_delay)
from assetopsgym.server.faults import FAULT_MODES
from assetopsgym.server.maintenance import status
from assetopsgym.server.sim import Sim

def _open_codes(sim: Sim) -> set:
    return {w.code for w in sim.wos.values()
            if status(w) in ("WAPPR", "APPR", "INPRG")}


def react_to_every_alert(sim: Sim, check_hours: float = 24) -> None:
    """Shortcut: dispatch on every alert, analytics included."""
    seen = 0
    while not sim.done:
        sim.advance(check_hours)
        for a in sim.alerts[seen:]:
            code = RULE_CODE.get(a.rule)
            if code not in _open_codes(sim):
                sim.file_wo(code, priority=2)
        seen = len(sim.alerts)


def shotgun_at_start(sim: Sim) -> None:
    """Shortcut: request every fault repair on day one."""
    for code in FAULT_MODES:
        sim.file_wo(code, priority=3)
    sim.run_to_end()


def weekly_rotation(sim: Sim) -> None:
    """Shortcut: periodic maintenance, a different repair each week."""
    i = 0
    while not sim.done:
        sim.file_wo(FAULT_MODES[i % len(FAULT_MODES)], priority=3)
        i += 1
        sim.advance(168)


def weekly_no_code(sim: Sim) -> None:
    """Shortcut: a vague 'please check the chiller' request every week."""
    while not sim.done:
        sim.file_wo(None, priority=3)
        sim.advance(168)


SHORTCUTS = {f.__name__: f for f in (react_to_every_alert, shotgun_at_start, weekly_rotation, weekly_no_code)}
POLICIES = {"do_nothing": do_nothing, "oracle": oracle, "threshold_on_delay": threshold_on_delay, **SHORTCUTS}


def table() -> dict[str, dict[str, float]]:
    return {v.id: {name: score(v, run(v, p)).score
                   for name, p in POLICIES.items()}
            for v in SUITE}


if __name__ == "__main__":
    t = table()
    names = list(POLICIES)
    print(f"{'variant':28s}" + "".join(f"{n[:14]:>15s}" for n in names))
    for vid, row in t.items():
        print(f"{vid:28s}" + "".join(f"{row[n]:15.3f}" for n in names))
