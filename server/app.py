"""FastAPI app serving AssetOpsGym over OpenEnv's HTTP and WebSocket protocol.

    uv run server            # or: uvicorn assetopsgym.server.app:app --port 8000
"""

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation

from .environment import AssetOpsGymEnvironment, GymState

app = create_app(
    AssetOpsGymEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="assetopsgym",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")),
    state_cls=GymState,
    reset_observation_cls=Observation,
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
