# AssetOpsGym (MVP)

An [OpenEnv](https://github.com/openenv/OpenEnv) environment where an LLM agent runs a degrading chiller for 60 simulated days through the real [AssetOpsBench](https://github.com/IBM/AssetOpsBench) MCP servers. It is scored, FDE-Gym style, on what happened to the plant. Team 33, HPML Fall 2026, project Dhaval_1.

## How an episode works

- **Clock.** The world is frozen between tool calls. Only `advance_time(hours)` (0.25–168 h) moves it. Reads are free.
- **Data.** Telemetry for Chiller 6 (AssetOpsBench sensor names, 15-minute rows) lands in this episode's own CouchDB as time advances. The database only ever holds the past.
- **Tools.**
  - AssetOpsBench IoT tools: `history`, `sensor_stats`, `latest_reading`, …
  - AssetOpsBench work-order tools: `generate_work_order`, `update_workorder`, `get_workorder`, …
  - Gym tools: `current_time`, `advance_time`, `list_alerts`, `finish`.
- **Faults.** One latent fault per faulty variant grows from its onset until the chiller trips. Its primary sensor:
  - condenser flow restriction: condenser water flow
  - refrigerant leak: evaporator approach
  - compressor wear: % loaded vs tonnage
- **Alerts.** The building management system raises operator alarms (2-hour on-delay, late) and analytics alerts (instant, noisy, also fired by sensor blips).
- **Maintenance.** The planner approves the agent's work orders and a technician comes. Technicians may not show up, and the compressor job waits 48 h for parts. The technician does the work named by the failure code, and everything is written back to the work-order store, where the agent can read it.
- **Ledger.** Every cost, downtime span, failure, alert and work-order transition is a typed event (`schemas.LedgerEvent`) in `Sim.events`; the score's totals and the proposal's five KPIs (downtime, missed failures, false dispatches, alert load, SLA breaches) are folds over it.
- **Crew.** A scenario names its technicians, their skills and travel time. A job nobody on the crew can do waits 48 h for a contractor. Arriving later than the priority's response target is an SLA breach.
- **Ending.** At day 60 or on `finish()`, the final `step()` observation carries `reward` = score and `metadata.result`, which has costs, failure prevented, false dispatches, detection delay and the KPIs.

## Score

KPI = total cost: dispatches, repairs (×2 if the chiller has already tripped), planned and unplanned downtime, and the extra energy a degraded chiller draws.

```
faulty:  U = (C_reactive − C_agent) / (C_reactive − C_oracle)     clipped to [−1, 1]
healthy: U = 1 − C_agent / (5 × dispatch cost)
```

- `C_reactive` is the plant's current practice: do nothing; a trip gets an emergency work order.
- `C_oracle` files the right work order at fault onset.
- On healthy twins, doing nothing *is* the reference.

## Variants and admission anchors

`uv run python -m assetopsgym.baselines`. `tests/test_anchors.py` enforces FDE-Gym's anchors: oracle ≥ 0.95, doing nothing = 0 on faulty variants, every shortcut ≤ 0.8.

| variant | do nothing | oracle | threshold + on-delay | react to every alert | shotgun day 1 | weekly rotation | weekly no-code |
|---|---|---|---|---|---|---|---|
| cwflow_early | 0.00 | 1.00 | 0.89 | −1.00 | −1.00 | −1.00 | −0.50 |
| cwflow_noshow | 0.00 | 1.00 | 0.92 | −1.00 | −1.00 | −1.00 | −0.53 |
| refleak_mid | 0.00 | 1.00 | 0.97 | −1.00 | −1.00 | −1.00 | −0.40 |
| refleak_fast | 0.00 | 1.00 | 0.72 | −1.00 | −1.00 | −1.00 | −0.41 |
| compwear_late | 0.00 | 1.00 | 0.83 | −1.00 | −0.24 | 0.45 | −0.09 |
| healthy | 1.00 | 1.00 | 1.00 | −1.00 | −1.00 | −1.00 | −0.80 |
| healthy_noisy | 1.00 | 1.00 | 1.00 | −1.00 | −1.00 | −1.00 | −0.80 |

First zero-shot run: `openai/gpt-5-mini` on `refleak_mid`, capped at 40 turns, scored **−0.04**. It made one false dispatch, the fault went undiagnosed, and the chiller tripped. LLM time was 149 s against 4.5 s in the environment, with 673k prompt tokens.

## Run it

Prerequisites: an AssetOpsBench checkout with `uv sync` done (`AOB_DIR`, default `../AssetOpsBench`), and CouchDB running (`docker compose -f src/couchdb/docker-compose.yaml up -d` in that checkout).

```bash
uv sync --extra agent --extra dev
uv run pytest -q                               # anchors + end-to-end through the AOB servers
uv run server                                  # OpenEnv server on :8000 (PORT to change)
OPENAI_BASE_URL=https://api.tokenrouter.com/v1 OPENAI_API_KEY=... \
  uv run python -m assetopsgym.agent --variant refleak_mid
uv run openenv validate . --level static       # manifest check
```

Client use:

```python
from assetopsgym import AssetOpsGymEnv, CallToolAction
with AssetOpsGymEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(variant="refleak_mid")           # or seed=k
    step = env.step(CallToolAction(tool_name="advance_time", arguments={"hours": 24}))
```

Rewards only flow through `step()`, which is OpenEnv's simulation path. `list_tools`/`call_tool` on the client use the direct `/mcp` route: same tools, no reward.

## Profiling (T10)

Every stage of a rollout is timed, always, by `profiling.Profiler`: the environment step, each gym tool, the pure simulator, the CouchDB writes, each proxied AssetOpsBench call, and on the agent side each LLM request. The last observation of an episode carries the server-side fold in `metadata.result.profile`; the agent prints its own fold next to it. Both report wall seconds per stage with p50/p95, the LLM vs environment split, simulated hours per wall second, and tokens per simulated hour.

```bash
AOG_PROFILE_DIR=runs uv run server                  # one JSON line per episode in runs/profile.jsonl
WANDB_PROJECT=assetopsgym uv run python -m assetopsgym.agent --variant refleak_mid   # same, logged to W&B
uv sync --extra profile                             # wandb
```

Add a stage with `with env.prof.span("sim.something", rows=n): ...`; names are dotted and the first segment groups them (`env`, `tool`, `sim`, `couch`, `mcp`, `llm`, `client`). `profiling.GPUSampler` polls nvidia-smi for utilization and memory on the serving host. For the inference side, run the vLLM server under `nsys profile -t cuda,nvtx -o vllm` and read the PyTorch Profiler trace vLLM writes when started with `VLLM_TORCH_PROFILER_DIR`; W&B's system metrics cover GPU utilization when the run lives on the GPU host.

## Layout

| file | what |
|---|---|
| `schemas.py` | the four team interfaces as pydantic models: scenario, simulator API, ledger events, agent tool list (draft; `tests/test_schemas.py` keeps them honest against the code) |
| `profiling.py` | span timing for every rollout stage, the per-episode fold (LLM/env split, tokens per simulated hour), JSONL and W&B sinks, GPU sampler |
| `server/sim.py` | chiller physics-lite, faults, blips, BMS rules, planner and technicians, cost ledger (pure, deterministic) |
| `server/scenarios.py` | variants, reference runs, score |
| `server/aob.py` | per-episode CouchDB databases and the stdio bridge to the AssetOpsBench servers |
| `server/environment.py` | the OpenEnv `MCPEnvironment`: tools, clock, CouchDB mirroring, wall-clock fixes, reward |
| `server/app.py`, `client.py`, `openenv.yaml` | OpenEnv packaging |
| `baselines.py`, `tests/` | reference policies, anchors, integration tests |
| `agent.py` | zero-shot agent over any OpenAI-compatible endpoint (TokenRouter, vLLM) |

## Known ceilings (deliberate, for the MVP)

Each line names its owner task on the team board.

- **One asset, one fault per episode**; telemetry is a hand-tuned model (T5, T6). The repo's Chiller 6 data is noise (`analysis/chiller6.md`), and Dhaval's MATLAB failure-mode list replaces our three modes.
- **Alerts live inside the gym** with fixed thresholds. The separate alert MCP server, with operator alarms vs analytics alerts, is T7.
- **Cost priors are flat guesses**, not swept (T2). The SLA target (priority response time plus 4 h grace) and the 48 h contractor delay for a skill the crew lacks are flat too.
- **Work-order edits** after the technician arrives are ignored. A no-code request gets an inspection that never finds anything. Technicians are never busy on another job (T8).
- **Each environment instance starts its own IoT and WO server processes** (about 10 s). The AssetOpsBench servers read their database names from the environment at start, so sharing one server pair across episodes needs a change upstream (T4, T11).
- **Wall-clock audit (T4).** Of the tools the gym exposes, only `generate_work_order` (reportdate), `cancel_workorder` (status_date) and `latest_reading` (age) read the real clock, and all three are rewritten to simulated time. The other clock readers in the WO server (`approve_workorder`, `assign_technician`, `close_workorder`, KPIs, calendar) are deliberately not exposed: the planner and technicians are simulated.
- **No Docker image yet.** It needs CouchDB and AssetOpsBench beside the gym (T11).
- **The agent keeps the whole conversation**, so prompt tokens grow quadratically with turns (T9). The profile's `llm.chat` spans carry `messages` and `prompt_tokens` per turn, so the curve is in the data.
- **No "before" number yet.** Last year's Plan-Execute profiling run is the baseline the report compares against; it has not been re-run on this gym (T10).
