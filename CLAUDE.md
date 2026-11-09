# AssetOpsGym

OpenEnv gym: an agent watches a simulated, degrading
chiller through AssetOpsBench's IoT and work-order MCP
servers and is scored over a 60-day window.

- `server/sim.py`: the simulator (telemetry, faults, WOs,
  crew, ledger). Pure, no I/O.
- `server/environment.py`: the OpenEnv environment
  (reset/step/state, `advance_time`, proxy to AOB).
- `server/aob.py`: CouchDB + AOB server bridge.
- `server/scenarios.py`: variants and the score.
- `infra/`: GPU + vLLM for the agent. `gcp.sh` (your own
  gcloud project), `insomnia.sbatch`; both run `serve.sh`.
- `schemas.py`: the team's interface contract (T2/T4/T7/T8
  all write to it). Change it deliberately, not in passing.

Run: `uv run pytest -q` (the `test_env.py` tests need
CouchDB and `../AssetOpsBench`, else they skip).
Server: `uv run server`. The same commands are mise tasks
(`mise.toml`, `mise tasks`).
W&B: `WANDB_ENTITY=hpml-projects WANDB_PROJECT=assetopsgym`
(host `api.forge.coreweave.com`, wandb ≥ 0.31's default).

## Code rules

- **Smallest diff that fixes it.** If 10 lines do it,
  don't write 200. Reuse what is in the repo before
  adding anything.
- **One function, one purpose.** A function that creates
  *or* updates *or* appends a log, picked by a flag, is
  three functions.
- **Explicit parameters.** Typed, named parameters. No
  `**kwargs` / `*args` pass-through and no `dict` option
  bags, except where a framework signature forces it
  (`reset(**kwargs)` for OpenEnv).
- **No stringly-typed data.** Data that crosses a function
  boundary is a pydantic model (at I/O boundaries) or a
  dataclass (inside the sim), not a `dict` with string
  keys. Raw strings only for true external names (a
  CouchDB field, an MCP tool name), each defined once.
  Counter-example: `_flush_changes` in `environment.py`
  reads `c["wonum"]`, `c.get("new")`, `"labor" in c`.
- **One source of truth.** Every list (tools, variants,
  sensors, failure codes) is defined once; everything
  else derives from it, and a test checks external copies
  (`openenv.yaml`). Counter-example: the agent's tool list
  lives in `IOT_TOOLS`/`WO_TOOLS`, `AGENT_TOOLS` and
  `openenv.yaml`.
- **No hand-written finite state machines** (state enum +
  transition table / `if state == ...` ladders) for
  control flow. External status fields (Maximo WO status
  `WAPPR`→`COMP`) are data we mirror, not a pattern to
  copy.
- **No OOP for its own sake.** No class with one method,
  no base class with one subclass, no factories. Plain
  functions + data classes.
- **No silent skips.** A missing document or unknown key
  raises or is logged, never `continue`d past.
- **Comments only when they add information** the name
  and types don't: a unit, an invariant, a cache-key side
  effect, why a constant has its value. Usually one line.
- **Line width:** aim for 60, never over 80. Applies to
  new and touched lines; don't reformat untouched code
  (teammates merge against it).
- **File size:** keep files under ~400 lines; split by
  purpose, not by size alone.

## Invariants (break these and the benchmark is wrong)

- **Sim time only.** Nothing under `server/` reads the
  wall clock (`datetime.now`, `time.time`) except
  profiling. Tools must never see the future.
- **All randomness comes from the seed.** Use the sim's
  `rng` / a `random.Random` derived from `Variant.seed`,
  never the module-level `random`. Same seed, same run.
- **Anchors hold:** do-nothing = 0, oracle ≥ 0.95, every
  shortcut ≤ 0.8 (`tests/test_anchors.py`).

## Workflow

- Never commit to `main`; branch per task
  (`rares/t3-seed`). Don't commit or push unless asked.
- Other people's branches (`farhaan`,
  `astrid/scenario-suite`) are read-only.
- Non-trivial logic leaves one runnable check behind (a
  test in `tests/`). Pure tests must not need CouchDB.
- Performance claims are measured with the profiler
  (`profiling.py` on `farhaan`, once merged), before and
  after, not estimated.
