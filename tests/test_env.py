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
    return obs.result.data, obs


def test_only_the_past_is_stored(env):
    env.reset(variant="healthy")
    extent, _ = call(env, "stream_extent", site_name="MAIN", asset_id="Chiller 6")
    assert extent["end_time"] < "2026-06-01T00:00:00"
    call(env, "advance_time", hours=24)
    extent, _ = call(env, "stream_extent", site_name="MAIN", asset_id="Chiller 6")
    assert extent["end_time"] == "2026-06-01T23:45:00"


def test_do_nothing_scores_zero(env):
    env.reset(variant="refleak_fast")
    while True:
        _, obs = call(env, "advance_time", hours=168)
        if obs.done:
            break
    assert obs.reward == 0.0 and obs.metadata["result"]["failed"]


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
