"""Client for a running AssetOpsGym server.

    with AssetOpsGymEnv(base_url="http://localhost:8000").sync() as env:
        env.reset(variant="refleak_mid")
        env.call_tool("advance_time", hours=24)
"""

from openenv.core.mcp_client import MCPToolClient


class AssetOpsGymEnv(MCPToolClient):
    pass
