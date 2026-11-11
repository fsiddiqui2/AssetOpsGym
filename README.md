# AssetOpsGym (MVP)

An [OpenEnv](https://github.com/openenv/OpenEnv) environment where an LLM agent runs a degrading chiller for 60 simulated days through the real [AssetOpsBench](https://github.com/IBM/AssetOpsBench) MCP servers. It is scored, FDE-Gym style, on what happened to the plant. Team 33, HPML Fall 2026, project Dhaval_1.

## How an episode works

- **Clock.** The world is frozen between tool calls. Only `advance_time(hours)` (0.25–168 h) moves it. Reads are free.
- **Data.** Telemetry for Chiller 6 (AssetOpsBench sensor names, 15-minute rows) lands in this episode's own CouchDB as time advances. The database only ever holds the past.
- **Tools.**
  - AssetOpsBench IoT tools: `history`, `sensor_stats`, `latest_reading`, …
  - AssetOpsBench work-order reads: `get_workorder`, `list_workorders`, `get_failure_codes` (its WO server runs with `AOB_READONLY=1`).
  - Gym tools (in-process): `current_time`, `advance_time`, `finish`, the alert tools `list_alerts`, `acknowledge_alert`, `shelve_alert`, AssetOpsBench's work-order writes `generate_work_order`, `update_workorder`, `cancel_workorder` (its names, arguments and result shape; `aob_source` left out), and the gym's own `investigate`. A wrong asset, an unknown failure code or an edit after the technician arrived comes back as `{"error", "error_code"}`.
  - The list the agent sees is `declared_tools` in `openenv.yaml`; the environment refuses to boot if its tools differ.
- **Faults.** One latent fault per faulty variant grows from its onset until the chiller trips. Its primary sensor:
  - condenser flow restriction: condenser water flow
  - refrigerant leak: evaporator approach
  - compressor wear: % loaded vs tonnage
- **Alerts.** The building management system raises operator alarms (fixed threshold held 2–3 h, late; 7 rules, 2 of them from AssetOpsBench's chiller rules) and analytics alerts (rule metric 3σ above its own last 24 h; about 1.5 a day when healthy, 3.7 with blips; they carry no fault signal). The agent can acknowledge an alert or shelve a rule for up to 168 h; shelved alerts are recorded but hidden, so shelving the fault's rule hides its alarm.
- **Maintenance.** The planner approves the agent's work orders and a technician comes. Technicians may not show up, and the compressor job waits 48 h for parts. The technician does the work named by the failure code. `investigate(subsystem)` opens an inspection order instead: a visit that repairs nothing, leaves the chiller running, and reports the fault and its failure code only if the fault is in that subsystem and has grown past severity 0.15 (about where its primary sensor has moved 3σ). The work-order documents in CouchDB are a fold over the ledger's events, rewritten after every write and every `advance_time`; AssetOpsBench's WO server reads them.
- **Ledger.** Every cost, downtime span, failure, alert and work-order transition is a typed event (`schemas.LedgerEvent`) in `Sim.events`; the score's totals and the proposal's five KPIs (downtime, missed failures, false dispatches, alert load, SLA breaches) are folds over it, as is the count of inspection visits, which are not false dispatches.
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

With [mise](https://mise.jdx.dev): `mise trust && mise install`, then `mise tasks` (setup, couch, test, server, agent, baselines, vllm, gcp, ...). Without it:

```bash
uv sync --extra agent --extra dev
uv run pytest -q                               # anchors + end-to-end through the AOB servers
uv run server                                  # OpenEnv server on :8000 (PORT to change)
OPENAI_BASE_URL=https://api.tokenrouter.com/v1 OPENAI_API_KEY=... \
  uv run python -m assetopsgym.agent --variant refleak_mid
mise run plan-execute -- --variant refleak_mid --model Qwen/Qwen2.5-3B-Instruct   # AssetOpsBench's Plan-Execute agent
infra/serve.sh                                 # vLLM on :8001 (GPU box); agent: OPENAI_BASE_URL=http://localhost:8001/v1
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
| `schemas.py` | the team interfaces as frozen dataclasses: simulator API, ledger events, agent tool results (draft; `tests/test_schemas.py` keeps them honest against the code) |
| `profiling.py` | span timing for every rollout stage, the per-episode fold (LLM/env split, tokens per simulated hour), JSONL and W&B sinks, GPU sampler |
| `server/sim.py` | the simulator (pure, deterministic): wires the parts below and runs the 15-min tick loop |
| `server/variant.py` | `Variant` (the only scenario type; `from_dict` loads a table row), crew, the clock |
| `server/telemetry.py` | sensors, load/weather, blips, the per-tick sample |
| `server/faults.py` | failure codes, fault severity, the trip |
| `server/alerts.py` | BMS rules, operator alarms, z-score analytics, ack/shelve |
| `server/maintenance.py` | work orders, planner and technicians |
| `server/ledger.py`, `server/report.py` | cost ledger; state, KPIs and result |
| `server/scenarios.py` | variants, reference runs, score |
| `server/aob.py` | per-episode CouchDB databases and the stdio bridge to the AssetOpsBench servers |
| `server/environment.py` | the OpenEnv `MCPEnvironment`: tools, clock, reward, `latest_reading`'s age |
| `server/tools.py` | the tool list (`openenv.yaml`), GymError → ErrorResult, AssetOpsBench tools re-exposed, the WO write signatures |
| `server/mirror.py` | CouchDB as a mirror: asset and failure-code catalogs, telemetry rows, WO documents folded from the events |
| `server/prompt.py` | the agent's instruction |
| `server/app.py`, `client.py`, `openenv.yaml` | OpenEnv packaging |
| `rollouts.py` | runs many episodes at once for either agent (`--envs` sessions, `--episodes`, seeds from `--seed0`) and reports rollouts per hour and per GPU-hour; the server needs `MAX_CONCURRENT_ENVS` ≥ `--envs` |
| `baselines.py`, `tests/` | reference policies, anchors, integration tests |
| `agent.py` | zero-shot agent over any OpenAI-compatible endpoint (TokenRouter, vLLM): a tool-calling loop that keeps the whole conversation |
| `plan_execute.py` | AssetOpsBench's Plan-Execute agent, zero-shot, one run per decision round: each round plans, executes and ends in `advance_time`; the previous round's answer is carried as notes. Runs in AssetOpsBench's environment (`mise run plan-execute`); its code is unchanged |

## Known ceilings (deliberate, for the MVP)

Each line names its owner task on the team board.

- **One asset, one fault per episode**; telemetry is a hand-tuned model (T5, T6). The repo's Chiller 6 data is noise (`analysis/chiller6.md`), and Dhaval's MATLAB failure-mode list replaces our three modes.
- **Alerts run in the gym's process**, not a separate MCP server. Acknowledge and shelve are free until T2 prices them; no chatter-rate knob yet (T7, d07).
- **Cost priors are flat guesses**, not swept (T2). The SLA target (priority response time plus 4 h grace) and the 48 h contractor delay for a skill the crew lacks are flat too.
- **Work-order edits** after the technician arrives are refused (`WORK_ORDER_LOCKED`). A no-code request gets an inspection that never finds anything. Technicians are never busy on another job (T8).
- **Each environment instance starts its own IoT and WO server processes** (about 10 s). The AssetOpsBench servers read their database names from the environment at start, so sharing one server pair across episodes needs a change upstream (T4, T11).
- **Wall-clock audit (T4).** The gym writes work orders itself, in simulated time. Of the AssetOpsBench tools it exposes, only `latest_reading` (age) reads the real clock, and the gym rewrites it to simulated time. The other clock readers in the WO server (KPIs, calendar) are deliberately not exposed; its writes are hidden by `AOB_READONLY=1`.
- **No Docker image yet.** It needs CouchDB and AssetOpsBench beside the gym (T11).
- **The agent keeps the whole conversation**, so prompt tokens grow quadratically with turns (T9). The profile's `llm.chat` spans carry `messages` and `prompt_tokens` per turn, so the curve is in the data.
- **No "before" number yet.** Last year's Plan-Execute profiling run is the baseline the report compares against; it has not been re-run on this gym (T10).
