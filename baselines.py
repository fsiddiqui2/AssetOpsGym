"""Reference policies on the pure simulator, and the admission-anchor table.

    uv run python -m assetopsgym.baselines

Shortcut policies are the strategies FDE-Gym requires to score at most 0.8. The
threshold-plus-on-delay rule is the non-LLM baseline an agent has to beat.
"""

from __future__ import annotations

from assetopsgym.server.scenarios import VARIANTS, do_nothing, oracle, run, score
from assetopsgym.server.faults import FAULT_MODES
from assetopsgym.server.sim import Sim

RULE_CODE = {
    "low_condenser_water_flow": "CH-CWFLOW",
    "high_evaporator_approach": "CH-REFLEAK",
    "compressor_load_mismatch": "CH-COMPWEAR",
}


def _open_codes(sim: Sim) -> set:
    return {w.code for w in sim.wos.values() if w.status in ("WAPPR", "APPR", "INPRG")}


def threshold_on_delay(sim: Sim, check_hours: float = 24) -> None:
    """Daily check; file a work order for each operator alarm that maps to a fault."""
    seen, filed = 0, set()
    while not sim.done:
        sim.advance(check_hours)
        for a in sim.alerts[seen:]:
            code = RULE_CODE.get(a.rule)
            if a.kind == "operator_alarm" and code and code not in filed:
                sim.file_wo(code, priority=2)
                filed.add(code)
        seen = len(sim.alerts)


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
    return {vid: {name: score(v, run(v, p)["cost_total"]) for name, p in POLICIES.items()}
            for vid, v in VARIANTS.items()}


if __name__ == "__main__":
    t = table()
    names = list(POLICIES)
    print(f"{'variant':16s}" + "".join(f"{n[:14]:>15s}" for n in names))
    for vid, row in t.items():
        print(f"{vid:16s}" + "".join(f"{row[n]:15.3f}" for n in names))
