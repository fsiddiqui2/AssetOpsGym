"""The four interfaces the team meets at, as pydantic models.

Drafted from what the MVP already does, so every model here round-trips an object
the code produces today (see tests/test_schemas.py). The simulator keeps its
dataclasses internally; these are the contracts T2, T4, T7 and T8 write to.

    1. Scenario          what one task variant specifies           (T1, T2)
    2. Simulator API     advance, inject a fault, repair, read state (T5, T6, T8)
    3. Ledger events     costs, downtime, failures, alerts, requests (T2, T7, T8)
    4. Agent tool results  what the gym's own tools return     (T3, T4, T9)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ----------------------------------------------------------------------------
# 1. Scenario format
# ----------------------------------------------------------------------------


class FaultSpec(BaseModel):
    """One latent fault. `mode` is a failure code the simulator knows (sim.FAILURE_CODES);
    kept as a string, not an enum, so Dhaval's MATLAB modes can be added as data."""

    model_config = ConfigDict(frozen=True)

    mode: str = Field(description="failure code, e.g. CH-REFLEAK")
    onset_day: float = Field(ge=0, description="days after the episode start when degradation begins")
    pf_days: float = Field(gt=0, description="P-F interval: days from onset to functional failure (trip)")
    shape: Literal["power"] = Field("power", description="severity curve family; only 'power' so far")
    exponent: float = Field(1.6, gt=0, description="severity = ((t - onset) / pf_days) ** exponent")


class Technician(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    skills: Optional[list[str]] = Field(None, description="failure codes this technician can repair; None = all")
    travel_h: float = Field(0.0, ge=0, description="added to every response time")


class CrewSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    technicians: list[Technician] = Field(min_length=1)
    noshow: float = Field(0.05, ge=0, le=1, description="chance a scheduled, non-emergency visit slips a day")


class Scenario(BaseModel):
    """What one task variant specifies. `fault=None` is a healthy twin."""

    model_config = ConfigDict(frozen=True)

    id: str
    fault: Optional[FaultSpec] = None
    blips_per_day: float = Field(1.0, ge=0, description="transient sensor glitches per day")
    crew: CrewSpec = CrewSpec(technicians=[Technician(id="TECH-01"), Technician(id="TECH-02")])
    load_profile: Literal["office"] = "office"
    seed: int = 0
    start: datetime = datetime(2026, 6, 1)
    history_days: int = Field(14, ge=0, description="telemetry available before the episode starts")
    horizon_days: int = Field(60, gt=0)
    held_out: bool = Field(False, description="kept out of the main suite for the shift study")
    tags: list[str] = Field(default_factory=list)

    @property
    def scoring(self) -> Literal["faulty", "healthy"]:
        return "faulty" if self.fault else "healthy"

    # Bridges to the simulator's own Variant, until Sim takes a Scenario directly.
    @classmethod
    def from_variant(cls, v) -> "Scenario":
        fault = FaultSpec(mode=v.mode, onset_day=v.onset_day, pf_days=v.pf_days) if v.mode else None
        crew = CrewSpec(technicians=[Technician(id=t.id, skills=None if t.skills is None else list(t.skills),
                                                travel_h=t.travel_h) for t in v.crew], noshow=v.noshow)
        return cls(id=v.id, fault=fault, blips_per_day=v.blips_per_day, crew=crew, seed=v.seed, start=v.start,
                   history_days=v.history_days, horizon_days=v.horizon_days)

    def to_variant(self):
        from .server.sim import Technician as _T, Variant

        crew = tuple(_T(t.id, None if t.skills is None else tuple(t.skills), t.travel_h)
                     for t in self.crew.technicians)
        f = self.fault
        return Variant(self.id, f.mode if f else None, onset_day=f.onset_day if f else 0.0,
                       pf_days=f.pf_days if f else 20.0, blips_per_day=self.blips_per_day,
                       noshow=self.crew.noshow, seed=self.seed, start=self.start,
                       history_days=self.history_days, horizon_days=self.horizon_days, crew=crew)


# ----------------------------------------------------------------------------
# 2. Simulator API
# ----------------------------------------------------------------------------


class TelemetryRow(BaseModel):
    timestamp: datetime
    values: dict[str, float]


class ServiceRequest(BaseModel):
    """What the agent (or a baseline) asks for. Maps to a work order."""

    code: Optional[str] = Field(None, description="failure code; None = 'please inspect'")
    priority: int = Field(3, ge=1, le=5)
    wonum: Optional[str] = Field(None, description="pin the id, e.g. the WO server's number")


class ServiceRequestUpdate(BaseModel):
    wonum: str
    code: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=5)
    cancel: bool = False


class FaultInjection(BaseModel):
    """Start a fault at run time (T6). The scenario's own fault is injected at construction."""

    mode: str
    onset: datetime
    pf_days: float = Field(gt=0)


class Repair(BaseModel):
    """What a completed repair fixes. The simulator applies this when a work order completes;
    it is public so T6 can define what each failure code's repair actually restores."""

    wonum: str
    code: str
    time: datetime
    fixed: bool = Field(description="True if the code matched the active fault")


class WorkOrderState(BaseModel):
    wonum: str
    code: Optional[str]
    priority: int
    status: Literal["WAPPR", "APPR", "INPRG", "COMP", "CAN"]
    emergency: bool = False
    created: datetime
    tech: Optional[str] = None
    arrive_at: Optional[datetime] = None
    start_at: Optional[datetime] = None
    finish_at: Optional[datetime] = None
    dispatched: bool = False


class SimState(BaseModel):
    """Everything the environment may read. `severity` and `fault_active` are ground truth,
    for scoring and tests only; never shown to the agent."""

    sim_time: datetime
    start: datetime
    end: datetime
    running: bool
    failed: bool
    fault_active: bool
    severity: float = Field(ge=0, le=1)
    work_orders: list[WorkOrderState]


class SimulatorAPI(Protocol):
    """The methods the environment, baselines and tests call. sim.Sim satisfies this."""

    def advance(self, hours: float) -> list[dict]: ...           # -> TelemetryRow dicts for CouchDB
    def inject_fault(self, fault: FaultInjection) -> None: ...
    def file_wo(self, code: Optional[str], priority: int = 3, wonum: Optional[str] = None): ...
    def update_wo(self, wonum: str, code: Optional[str] = None, priority: Optional[int] = None,
                  cancel: bool = False) -> None: ...
    def acknowledge(self, alert_id: str): ...
    def shelve(self, rule: str, hours: float) -> datetime: ...
    def state(self) -> SimState: ...
    def result(self) -> dict: ...                                 # -> EpisodeResult


# ----------------------------------------------------------------------------
# 3. Ledger events
# ----------------------------------------------------------------------------

CostKind = Literal["dispatch", "repair", "planned_downtime", "unplanned_downtime", "excess_energy"]


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True)
    time: datetime


class CostEvent(_Event):
    type: Literal["cost"] = "cost"
    kind: CostKind
    amount: float = Field(ge=0, description="USD")
    wonum: Optional[str] = None


class DowntimeEvent(_Event):
    """One contiguous span the chiller was off, emitted when the span closes."""

    type: Literal["downtime"] = "downtime"
    kind: Literal["planned", "unplanned"]
    hours: float = Field(gt=0)
    start: datetime


class FailureEvent(_Event):
    type: Literal["failure"] = "failure"
    mode: str


class AlertEvent(_Event):
    type: Literal["alert"] = "alert"
    kind: Literal["operator_alarm", "analytics"]
    rule: str
    sensor: str
    value: float
    message: str
    id: str
    shelved: bool = Field(
        False, description="raised while its rule was shelved; hidden")


class AlertActionEvent(_Event):
    """The agent acknowledged an alert or shelved a rule."""

    type: Literal["alert_action"] = "alert_action"
    action: Literal["ack", "shelve"]
    rule: str
    alert_id: Optional[str] = None
    hours: Optional[float] = None


class ServiceRequestEvent(_Event):
    """A work-order transition, whoever caused it."""

    type: Literal["service_request"] = "service_request"
    wonum: str
    status: Literal["WAPPR", "APPR", "INPRG", "COMP", "CAN"]
    priority: int
    code: Optional[str] = None
    emergency: bool = False
    actor: Literal["agent", "bms", "planner", "technician"]
    tech: Optional[str] = None
    sla_breach: bool = Field(False, description="technician arrived later than the priority's response target")
    note: Optional[str] = None


LedgerEvent = Annotated[Union[CostEvent, DowntimeEvent, FailureEvent,
                              AlertEvent, AlertActionEvent,
                              ServiceRequestEvent],
                        Field(discriminator="type")]


class KPIs(BaseModel):
    """The proposal's five KPIs, each a fold over the ledger. Logged beside the score, not in it."""

    downtime_hours: float
    missed_failures: int
    false_dispatches: int
    alert_load: int
    sla_breaches: int


class EpisodeResult(BaseModel):
    variant: str
    mode: Optional[str]
    onset: Optional[datetime]
    failed: bool
    failed_at: Optional[datetime]
    failure_prevented: bool
    cost_total: float
    cost_breakdown: dict[CostKind, float]
    downtime_hours: dict[Literal["planned_downtime", "unplanned_downtime"], float]
    dispatches: int
    false_dispatches: int
    work_orders: int
    detection_delay_hours: Optional[float]
    alerts: dict[Literal["operator_alarm", "analytics"], int]
    kpis: KPIs
    events: int = Field(description="ledger length; the events themselves are in Sim.events")
    score: Optional[float] = Field(None, ge=-1, le=1)
    end_reason: Optional[str] = None
    reference_costs: Optional[dict[str, float]] = None

    @model_validator(mode="after")
    def _breakdown_sums(self):
        if abs(sum(self.cost_breakdown.values()) - self.cost_total) > 0.05:
            raise ValueError("cost_breakdown does not sum to cost_total")
        return self


# ----------------------------------------------------------------------------
# 4. Agent tool results (the tool list is openenv.yaml)
# ----------------------------------------------------------------------------


class ErrorCode(str, Enum):
    NO_EPISODE = "NO_EPISODE"
    EPISODE_OVER = "EPISODE_OVER"
    BAD_ARGUMENT = "BAD_ARGUMENT"
    NOT_FOUND = "NOT_FOUND"
    UNKNOWN_RULE = "UNKNOWN_RULE"


class GymError(Exception):
    """A tool call the agent got wrong. The only exception
    subclass: kinds are ErrorCode values, not subclasses."""

    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ErrorResult:
    """AssetOpsBench's tool error shape, {"error": msg}, plus
    the code its envelope has but drops before replying."""

    error: str
    error_code: ErrorCode


class ClockResult(BaseModel):
    sim_time: datetime
    shift_end: datetime
    hours_left: float


class AdvanceTimeResult(BaseModel):
    sim_time: datetime
    shift_end: datetime
    new_telemetry_rows: int
    new_alert_count: int
    new_alerts: list[AlertEvent]
    work_order_updates: list[dict]
    shift_over: bool


class ListAlertsResult(BaseModel):
    total: int
    returned: int
    alerts: list[AlertEvent]

    @field_validator("alerts")
    @classmethod
    def _bounded(cls, v):
        if len(v) > 200:
            raise ValueError("limit is 200")
        return v


class FinishResult(BaseModel):
    message: str
    score: float = Field(ge=-1, le=1)
