"""End to end through OpenEnv and the real AssetOpsBench servers.

Needs CouchDB (AssetOpsBench's docker compose) and an AssetOpsBench checkout with
`uv sync` done (AOB_DIR, default ../AssetOpsBench). Skipped otherwise.
"""

import pytest
import requests

from assetopsgym.server.aob import AOB_DIR, Couch
from assetopsgym.server.scenarios import VARIANTS

try:
    _couch_up = requests.get(Couch("probe").url, timeout=2).ok
except requests.RequestException:
    _couch_up = False
pytestmark = pytest.mark.skipif(not (_couch_up and (AOB_DIR / ".venv").exists()),
                                reason="needs CouchDB and a synced AssetOpsBench checkout")


@pytest.fixture(scope="module")
def env():
    from assetopsgym.server.environment import AssetOpsGymEnvironment

    e = AssetOpsGymEnvironment()
    yield e
    e.close()


def call(env, name, **args):
    from openenv.core.env_server.mcp_types import CallToolAction

    obs = env.step(CallToolAction(tool_name=name, arguments=args))
    assert obs.error is None, obs.error
    return obs.result.structured_content["result"], obs


def test_only_the_past_is_stored(env):
    env.reset(variant="healthy")
    extent, _ = call(env, "stream_extent", site_name="MAIN", asset_id="Chiller 6")
    assert extent["end_time"] < "2026-06-01T00:00:00"
    call(env, "advance_time", hours=24)
    extent, _ = call(env, "stream_extent", site_name="MAIN", asset_id="Chiller 6")
    assert extent["end_time"] == "2026-06-01T23:45:00"
    latest, _ = call(env, "latest_reading", site_name="MAIN",
                     asset_id="Chiller 6")
    assert latest["age_seconds"] == 900  # sim time, not wall clock


def test_do_nothing_scores_zero(env):
    env.reset(variant="refleak_fast")
    while True:
        _, obs = call(env, "advance_time", hours=168)
        if obs.done:
            break
    assert obs.reward == 0.0 and obs.metadata["result"]["failed"]
    for name, args in (("advance_time", {"hours": 1}),
                       ("history", {"site_name": "MAIN",
                                    "asset_id": "Chiller 6"})):
        over, _ = call(env, name, **args)
        assert over["error_code"] == "EPISODE_OVER", over


def test_oracle_through_the_wo_server_scores_one(env):
    v = VARIANTS["cwflow_early"]
    env.reset(variant=v.id)
    hours = (v.onset - v.start).total_seconds() / 3600 + 0.25
    while hours > 0:
        call(env, "advance_time", hours=min(168, hours))
        hours -= 168
    wo, _ = call(env, "generate_work_order", description="Condenser flow dropping", asset_num="Chiller 6",
                 site_id="MAIN", priority=2)
    assert wo["work_order"]["reportdate"].startswith("2026-06-07")  # simulated, not wall-clock, time
    call(env, "update_workorder", wonum=wo["wonum"], site_id="MAIN", failure_code="CH-CWFLOW")
    _, obs = call(env, "finish")
    assert obs.done and obs.reward >= 0.95
    assert obs.metadata["result"]["failure_prevented"]


def test_gym_owns_the_work_order_writes(env):
    aob = {t.name for t in env.bridge.list_tools("wo")}
    writes = {"generate_work_order", "update_workorder",
              "cancel_workorder", "approve_workorder",
              "assign_technician", "close_workorder"}
    assert "get_workorder" in aob and not aob & writes
    assert not set(env.proxied) & writes
    env.reset(variant="healthy", seed=1)
    bad, _ = call(env, "generate_work_order", description="x",
                  asset_num="Chiller6", site_id="MAIN")
    assert bad["error_code"] == "UNKNOWN_ASSET", bad
    wo, _ = call(env, "generate_work_order", description="Check",
                 asset_num="Chiller 6", site_id="MAIN", priority=1)
    filed = "2026-06-01T00:00:00+00:00"
    assert wo["status"] == "WAPPR"
    assert wo["work_order"]["reportdate"] == filed
    num = wo["wonum"]
    up, _ = call(env, "update_workorder", wonum=num, site_id="MAIN",
                 failure_code="CH-SENSOR")
    assert up["work_order"]["failurecode"] == "CH-SENSOR"
    out, _ = call(env, "advance_time", hours=72)
    got, _ = call(env, "get_workorder", wonum=num, site_id="MAIN")
    w = got["work_order"]
    since = next(u["time"] for u in out["work_order_updates"]
                 if u["status"] == w["status"])
    assert w["status"] in ("INPRG", "COMP") and w["reportdate"] == filed
    assert w["status_date"] == since + "+00:00"
    listed, _ = call(env, "list_workorders", site_id="MAIN")
    assert [x["status"] for x in listed["work_orders"]] == [w["status"]]
    for name, args in (("update_workorder", {"priority": 5}),
                       ("cancel_workorder", {})):
        locked, _ = call(env, name, wonum=num, site_id="MAIN", **args)
        assert locked["error_code"] == "WORK_ORDER_LOCKED", locked


def test_alert_tools(env):
    env.reset(variant="healthy_noisy")
    out, _ = call(env, "advance_time", hours=72)
    alert = out["new_alerts"][0]
    acked, _ = call(env, "acknowledge_alert", alert_id=alert["id"])
    assert acked["id"] == alert["id"]
    left, _ = call(env, "list_alerts", unacked_only=True, limit=200)
    assert alert["id"] not in {a["id"] for a in left["alerts"]}
    shelf, _ = call(env, "shelve_alert", rule=alert["rule"], hours=168)
    assert shelf["rule"] == alert["rule"]
    out, _ = call(env, "advance_time", hours=168)
    assert alert["rule"] not in {a["rule"] for a in out["new_alerts"]}
    bad, _ = call(env, "shelve_alert", rule="chiller_trip", hours=1)
    assert bad["error_code"] == "UNKNOWN_RULE", bad


def test_step_limit_ends_the_episode(env):
    env.reset(variant="healthy")
    limit, env.max_steps = env.max_steps, 2
    try:
        _, first = call(env, "current_time")
        _, second = call(env, "current_time")
    finally:
        env.max_steps = limit
    assert not first.done and second.done
    r = second.metadata["result"]
    assert r["end_reason"] == "step limit", r
    assert env.state.step_count == 2


def test_investigate_reports_through_the_work_order(env):
    env.reset(variant="refleak_mid")
    for hours in (168, 168, 168, 72):  # day 24: the leak has grown
        call(env, "advance_time", hours=hours)
    made, _ = call(env, "investigate",
                   subsystem="refrigerant circuit", priority=1)
    out, _ = call(env, "advance_time", hours=36)
    assert "CH-REFLEAK" in out["work_order_updates"][-1]["note"]
    got, _ = call(env, "get_workorder", wonum=made["wonum"],
                  site_id="MAIN")
    wo = got["work_order"]
    assert wo["worktype"] == "INSP" and wo["failurecode"] is None
    assert "CH-REFLEAK" in wo["description_longdescription"]
