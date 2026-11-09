# AssetOpsGym

OpenEnv gym: an agent watches a simulated, degrading
chiller through AssetOpsBench's IoT and work-order MCP
servers and is scored over a 60-day window.

- `server/sim.py`: the simulator's tick loop and public
  API; the parts live in `variant.py`, `telemetry.py`,
  `faults.py`, `alerts.py`, `maintenance.py`, `ledger.py`,
  `report.py`. Pure, no I/O.
- `server/environment.py`: the OpenEnv environment
  (reset/step/state, `advance_time`, proxy to AOB).
- `server/tools.py`: the tool manifest check, the error
  wrapper, AOB tools re-exposed, AOB's WO write
  signatures (the gym serves them).
- `server/mirror.py`: CouchDB as a mirror of the sim
  (catalogs, telemetry rows, WO documents from events).
- `server/aob.py`: CouchDB + AOB server bridge.
- `server/scenarios.py`: variants and the score.
- `infra/`: GPU + vLLM for the agent. `gcp.sh` (your own
  gcloud project), `insomnia.sbatch`; both run `serve.sh`.
  vLLM listens on :8001, the gym on :8000.
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
  boundary is a dataclass, not a `dict` with string
  keys. Raw strings only for true external names (a
  CouchDB field, an MCP tool name), each defined once.
  Counter-example: `_flush_changes` in `environment.py`
  reads `c["wonum"]`, `c.get("new")`, `"labor" in c`.
- **One source of truth.** Every list (tools, variants,
  sensors, failure codes) is defined once; everything
  else derives from it or is checked against it. Example:
  the agent's tool list is `declared_tools` in
  `openenv.yaml`; the environment reads it at import and
  refuses to boot if its registered tools differ.
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

## Design decisions (2026-11-09)

Implemented by the refactor stacked on `rares/mise`
(`rares/quick-fixes` → `rares/sim-split` → ... →
`rares/wo-tools`), except the parked items at the end.

- **The events are the record.** The ledger's event list
  is the only record of what happened. Cost totals, KPIs,
  the episode result and the CouchDB work-order documents
  are folds over it; nothing keeps a second copy. Other
  services read it with `fetch(since, types)`: JSON plus
  a cursor (`read()` is the typed in-process form).
- **Work orders are timed events, not a status ladder.**
  A heap of (time, action) entries (approve, arrive,
  start, finish); an edit bumps the order's version and
  stale entries are dropped. Status is derived from the
  timestamps, never stored.
- **`sim.py` only orchestrates** (~60 lines: the tick
  loop and the public API). Each concern is a module
  with a state dataclass plus plain functions: telemetry,
  faults, alerts, maintenance, ledger, variant.
- **Dataclasses, not pydantic.** `@dataclass(frozen=True,
  slots=True)`: 64 B per instance against 480 B for a
  pydantic model. Pydantic stays only where a framework
  forces it (OpenEnv's `State`/`Action`/`Observation`);
  fastmcp and openenv import it anyway, so dropping it
  saves per-object memory, not the library. FastMCP
  builds tool output schemas from dataclasses.
- **A new scenario setting is at most 5 lines.** One
  `Variant` dataclass is the only scenario type: add a
  field with a default, read it where it matters. A
  generic `from_dict` loads tables (Astrid's suite) with
  no parser change. Unused settings are fine.
- **`schemas.py` holds a type iff two modules use it.**
  Module-private types stay in their module.
- **The gym owns the work-order writes.** It implements
  AssetOpsBench's `generate_work_order`, `update_workorder`
  and `cancel_workorder` with AOB's signatures and result
  shape; CouchDB is a mirror and AOB's WO server serves
  the reads. This removes the WO date patching. The one
  wall-clock patch left is `latest_reading`'s age (AOB
  reads the real clock there); it stays in the gym.
- **Never change AssetOpsBench.** No PRs or patches to
  it (IBM may use it internally). Work around its
  behaviour at the gym boundary.
- **Errors: one exception class plus an enum.**
  `GymError(code: ErrorCode, message)` is the only
  exception subclass (the one allowed exception to "no
  OOP"). Tools return `ErrorResult(error, error_code)` as
  a normal result: AOB's `{"error": ...}` shape plus the
  code AOB drops. Unknown asset, unknown failure code and
  edits after a technician arrives are errors, never
  silently ignored.
- **The tool list lives only in `openenv.yaml`.** A plain
  loader reads it at start; which server owns a tool is
  derived at boot (gym registry vs AOB `list_tools`), and
  a mismatch raises.
- **Ports:** gym server 8000 (OpenEnv convention), vLLM
  8001.
- **`reset()` without a seed draws one and returns it**,
  so every run is replayable.
- Parked: a second architecture discussion (review item
  6), CI, the Dockerfile, and the package layout.

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
