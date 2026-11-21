# AssetOpsGym (MVP)

An [OpenEnv](https://github.com/openenv/OpenEnv) environment where an LLM agent runs a degrading chiller for 60 simulated days through the real [AssetOpsBench](https://github.com/IBM/AssetOpsBench) MCP servers. It is scored, FDE-Gym style, on what happened to the plant. Team 33, HPML Fall 2026, project Dhaval_1.

## How an episode works

- **Clock.** The world is frozen between tool calls. Only `advance_time(hours)` (0.25–168 h) moves it. Reads are free.
- **Data.** Telemetry for Chiller 6 (AssetOpsBench sensor names, 15-minute rows) lands in this episode's own CouchDB as time advances. The database only ever holds the past.
- **Tools.**
  - AssetOpsBench IoT tools: `history`, `sensor_stats`, `latest_reading`, …
  - AssetOpsBench work-order reads: `get_workorder`, `list_workorders`, `get_failure_codes` (its WO server runs with `AOB_READONLY=1`).
  - Gym tools (in-process): `current_time`, `advance_time`, `finish`, the alert tools `list_alerts`, `acknowledge_alert`, `suppress_alert`, and the work-order writes: `create_work_order` (one call, failure code optional), `investigate`, `escalate` (to priority 1), `defer` (visit later, target moves with it), `assign_technician` (instead of the planner's pick), and AssetOpsBench's `update_workorder` and `cancel_workorder` (its names, arguments and result shape). The gym also implements AssetOpsBench's `generate_work_order`; `openenv.yaml` leaves it out of the agent's list. A wrong asset, an unknown failure code or an edit after the technician arrived comes back as `{"error", "error_code"}`.
  - The list the agent sees is `declared_tools` in `openenv.yaml`; the environment refuses to boot if its tools differ.
- **Faults.** One latent fault per faulty variant grows from its onset until the chiller trips. Its primary sensor:
  - condenser flow restriction: condenser water flow
  - refrigerant leak: evaporator approach
  - compressor wear: % loaded vs tonnage
- **Alerts.** The building management system raises operator alarms (fixed threshold held 2–3 h, late; 7 rules, 2 of them from AssetOpsBench's chiller rules) and analytics alerts (rule metric 3σ above its own last 24 h; about 1.5 a day when healthy, 3.7 with blips; they carry no fault signal). The agent can acknowledge an alert or suppress (shelve) a rule for up to 168 h; shelved alerts are recorded but hidden, so shelving the fault's rule hides its alarm. A scenario can make one rule's transmitter chatter (`chatter_rule`, `chatter_per_day`, `chatter_start_day`, `chatter_end_day`): its reading spikes 15% past the alarm limit on that share of ticks and each spike raises an operator alarm at once, with no hold. `investigate("sensors")` finds the transmitter and a `CH-SENSOR` repair ends the chatter; neither is a false dispatch.
- **Maintenance.** The planner approves the agent's work orders and a technician comes. Technicians may not show up, and the compressor job waits 48 h for parts. The technician does the work named by the failure code. `investigate(subsystem)` opens an inspection order instead: a visit that repairs nothing, leaves the chiller running, and reports the fault and its failure code only if the fault is in that subsystem and has grown past severity 0.15 (about where its primary sensor has moved 3σ). The work-order documents in CouchDB are a fold over the ledger's events, rewritten after every write and every `advance_time`; AssetOpsBench's WO server reads them.
- **Ledger.** Every cost, downtime span, failure, alert and work-order transition is a typed event (`schemas.LedgerEvent`) in `Sim.events`; the score's totals and the proposal's five KPIs (downtime, missed failures, false dispatches, alert load, SLA breaches) are folds over it, as is the count of inspection visits, which are not false dispatches.
- **Crew.** A scenario names its technicians, their skills and travel time. A job nobody on the crew can do waits 48 h for a contractor. Arriving later than the priority's response target is an SLA breach.
- **Ending.** At day 60 or on `finish()`, the final `step()` observation carries `reward` = score and `metadata.result`, which has the score's terms and references, costs, failure prevented, false dispatches, detection delay and the KPIs.

## Score

The score is the scenario sheet's seven weighted event terms, in downtime-hour equivalents:

| term | weight |
|---|---|
| unplanned downtime | 1 per hour |
| failure (the chiller trips) | 24 |
| false dispatch (a mechanical visit nothing called for) | 4 |
| acknowledgement | 0.005 |
| investigation (an inspection visit) | 0.5 |
| unaddressed fault | 12 |
| SLA breach the agent could have prevented | 2 |

```
S = (C_threshold − C_agent) / (C_threshold − C_oracle)     clipped to [−1, 1], at most 0 after a failure
S = 1 − (C_agent − C_oracle) / 20                          where the two references tie
```

- `C_threshold` is the plant's alarm-driven practice: a daily check that files a work order for each operator alarm that maps to a fault. It is the score's zero.
- `C_oracle` files the right work order at fault onset (as planned work, if the fault cannot fail inside the 60 days). It is the score's one.
- Both are the variant's means over the 20 development seeds (`scenarios.REFERENCE_SEEDS`, 1000–1019), so a variant where the threshold rule is late in only some seeds still has a gap. The test seeds are 2000–2019.
- The references tie on a healthy chiller, and on a fault the alarm catches in time; there, 20 units of cost (five false dispatches) is a score of 0.
- A fault is unaddressed if it can fail inside the 60 days and no order for it was filed between its onset and the latest safe time: the last moment a priority-1 order still finishes before the trip, on the planner's slowest approval and arrival.
- A priority 1 or 2 visit for a fault that cannot fail inside the 60 days counts as a false dispatch.
- An SLA breach is charged only if the visit was late even without the technician's no-shows and without the contractor wait when nobody on the crew has the skill. In today's simulator that leaves almost nothing to charge: technicians are never busy, so the only late arrivals the agent causes are from assigning a technician without the skill.

Every result carries `event_terms` (the seven counts) and `event_score` (the cost, both references and the score). The dollar ledger (`cost_total`, `cost_breakdown`) and the KPIs are still logged; they are not the score.

## Scenarios and anchors

The suite (`scenarios.SUITE`) is ten scenarios of three or four variants, each variant one changed setting from its scenario's base, named `<scenario>.<variant>`. `reset()` without a variant draws from it. The seven older variants (`scenarios.DEV`: `refleak_mid`, `healthy`, ...) stay for tests and quick runs and are not in the suite.

`uv run python -m assetopsgym.baselines` prints every suite variant (about 3 minutes). `tests/test_anchors.py` enforces the anchors on each base variant: the oracle scores 1; doing nothing scores at most 0 where the fault can fail and 1 elsewhere; every shortcut scores at most 0.8. Scores below are on each base variant's own seed, so the threshold rule's column is near 0 rather than exactly 0.

| scenario (base) | what it tests | threshold cost | do nothing | oracle | threshold rule | react to every alert | shotgun day 1 | weekly rotation | weekly no-code |
|---|---|---|---|---|---|---|---|---|---|
| cf_fast | a fast flow restriction, before the alarm | 42.5 | −0.13 | 1.00 | −0.02 | −1.00 | −0.32 | 0.44 | −0.98 |
| rl_fast | a fast leak | 50.0 | −0.04 | 1.00 | −0.04 | −1.00 | −0.20 | −0.52 | −0.76 |
| cw_parts_wait | compressor wear, 48 h parts wait | 69.9 | −0.49 | 1.00 | 0.00 | −1.00 | −0.60 | 0.66 | −1.00 |
| rl_no_skill | a leak nobody on the crew repairs | 48.9 | −1.00 | 1.00 | −0.06 | −1.00 | −1.00 | 0.51 | −1.00 |
| rl_masked_by_blips | a leak among sensor glitches | 49.3 | −0.06 | 1.00 | −0.06 | −1.00 | −0.22 | 0.51 | −0.79 |
| cf_noshow | technicians that may not come | 25.4 | −0.89 | 1.00 | −0.23 | −1.00 | −1.00 | 0.05 | −1.00 |
| cf_remote_crew | a crew 12 h away | 9.6 | −1.00 | 1.00 | 1.00 | −1.00 | −1.00 | −1.00 | −1.00 |
| cf_incipient_defer | a fault too slow to fail in 60 days | 2.8 | 1.00 | 1.00 | −0.43 | −1.00 | −1.00 | −1.00 | −1.00 |
| healthy_quiet | a healthy chiller | 0.0 | 1.00 | 1.00 | 1.00 | −1.00 | 0.40 | −0.80 | −0.80 |
| healthy_chatter | a healthy chiller, one alarm chattering | 4.0 | 1.00 | 1.00 | 0.00 | −1.00 | −1.00 | −1.00 | −1.00 |

Known gap: a policy that inspects the three fault subsystems on a schedule and files what the technician finds scores 0.81 on `cw_parts_wait` (weekly) and 0.84 on `cw_parts_wait.pf26` (fortnightly), just over the 0.8 bound, because an inspection costs 0.5. It is not among the seven scripted policies.

First zero-shot run, under the earlier dollar score: `openai/gpt-5-mini` on `refleak_mid`, capped at 40 turns, scored −0.04. It made one false dispatch, the fault went undiagnosed, and the chiller tripped. LLM time was 149 s against 4.5 s in the environment, with 673k prompt tokens.

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
mise run claude-agent -- --variant rl_fast.base   # either agent on Claude: claude-agent, claude-plan-execute
infra/serve.sh                                 # vLLM on :8001 (GPU box); agent: OPENAI_BASE_URL=http://localhost:8001/v1
infra/mlx.sh                                   # or Qwen3-8B 4-bit on :8001 on an Apple-silicon laptop; --model mlx-community/Qwen3-8B-4bit
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

With `AOG_PROFILE_DIR` set, every agent also appends each episode's summary and full result to `episodes.jsonl` and writes what it saw and said to `transcripts/<episode id>.json`; the summary's `stopped` says why the agent's loop ended if the episode did not end by itself (turn limit, context full, no tool calls, refusal).

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
| `server/alerts.py` | BMS rules, operator alarms, z-score analytics, ack/shelve, chatter spikes |
| `server/maintenance.py` | work orders, planner and technicians |
| `server/ledger.py`, `server/report.py` | cost ledger; state, KPIs and result |
| `server/scenarios.py` | variants, reference runs, score |
| `server/aob.py` | per-episode CouchDB databases and the stdio bridge to the AssetOpsBench servers |
| `server/environment.py` | the OpenEnv `MCPEnvironment`: tools, clock, reward, `latest_reading`'s age |
| `server/tools.py` | the tool list (`openenv.yaml`), GymError → ErrorResult, AssetOpsBench tools re-exposed, the WO write signatures |
| `server/mirror.py` | CouchDB as a mirror: asset and failure-code catalogs, telemetry rows, WO documents folded from the events |
| `server/prompt.py` | the agent's instruction |
| `server/app.py`, `client.py`, `openenv.yaml` | OpenEnv packaging |
| `claude_agent.py`, `claude_plan_execute.py` | the same two agents on Claude's Messages API (`uv sync --extra claude`; `ANTHROPIC_API_KEY` or an `ant auth login` profile; default model `claude-opus-5-5`). The tool loop caches its conversation prefix. No fallback model: a refusal ends the episode |
| `rollouts.py` | runs many episodes at once for any agent (`--envs` sessions, `--episodes`, seeds from `--seed0`) and reports rollouts per hour and per GPU-hour; the server needs `MAX_CONCURRENT_ENVS` ≥ `--envs` |
| `baselines.py`, `tests/` | reference policies, anchors, integration tests |
| `agent.py` | zero-shot agent over any OpenAI-compatible endpoint (TokenRouter, vLLM): a tool-calling loop that keeps the whole conversation |
| `plan_execute.py` | AssetOpsBench's Plan-Execute agent, zero-shot, one run per decision round: each round plans, executes and ends in `advance_time`; the previous round's answer is carried as notes. Runs in AssetOpsBench's environment (`mise run plan-execute`); its code is unchanged |

## Known ceilings (deliberate, for the MVP)

Each line names its owner task on the team board.

- **One asset, one fault per episode**; telemetry is a hand-tuned model (T5, T6). The repo's Chiller 6 data is noise (`analysis/chiller6.md`), and Dhaval's MATLAB failure-mode list replaces our three modes.
- **Alerts run in the gym's process**, not a separate MCP server. Shelving is free; an acknowledgement costs 0.005.
- **The score's weights are the scenario sheet's provisional ones**, not swept (T2). The SLA target (priority response time plus 4 h grace) and the 48 h contractor delay for a skill the crew lacks are flat too.
- **Work-order edits** after the technician arrives are refused (`WORK_ORDER_LOCKED`). A no-code request gets an inspection that never finds anything. Technicians are never busy on another job (T8).
- **Each environment instance starts its own IoT and WO server processes** (about 10 s). The AssetOpsBench servers read their database names from the environment at start, so sharing one server pair across episodes needs a change upstream (T4, T11).
- **Wall-clock audit (T4).** The gym writes work orders itself, in simulated time. Of the AssetOpsBench tools it exposes, only `latest_reading` (age) reads the real clock, and the gym rewrites it to simulated time. The other clock readers in the WO server (KPIs, calendar) are deliberately not exposed; its writes are hidden by `AOB_READONLY=1`.
- **No Docker image yet.** It needs CouchDB and AssetOpsBench beside the gym (T11).
- **The agent keeps the whole conversation**, so prompt tokens grow quadratically with turns (T9). The profile's `llm.chat` spans carry `messages` and `prompt_tokens` per turn, so the curve is in the data.
- **No "before" number yet.** Last year's Plan-Execute profiling run is the baseline the report compares against; it has not been re-run on this gym (T10).
