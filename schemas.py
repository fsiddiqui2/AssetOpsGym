"""The four interfaces the team meets at, as frozen dataclasses
(FastMCP schemas the tool results from them).

Drafted from what the MVP already does: the simulator returns these
types (see tests/test_schemas.py) and keeps its own dataclasses
internally; these are the contracts T2, T4, T7 and T8 write to.

    1. Scenario          what one task variant specifies (Variant)  (T1, T2)
    2. Simulator API     advance, inject a fault, repair, read state (T5, T6, T8)
    3. Ledger events     costs, downtime, failures, alerts, requests (T2, T7, T8)
    4. Agent tool results  what the gym's own tools return     (T3, T4, T9)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal, Optional, Protocol, Union

WoStatus = Literal["WAPPR", "APPR", "INPRG", "COMP", "CAN"]
AlertKind = Literal["operator_alarm", "analytics"]

# ----------------------------------------------------------------------------
# 1. Scenario format: server/variant.py's Variant (frozen dataclass);
#    variant.from_dict loads a table row into it.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# 2. Simulator API
# ----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TelemetryRow:
    """One 15-minute tick; values is keyed by sensor name, the IoT
    document's own field names."""

    timestamp: datetime
    values: dict[str, float]


@dataclass(frozen=True, slots=True)
class ServiceRequest:
    """What the agent (or a baseline) asks for. Maps to a work order."""

    code: Optional[str] = None  # failure code; None = 'please inspect'
    priority: int = 3  # 1-5
    wonum: Optional[str] = None  # pin the id, e.g. the WO server's number


@dataclass(frozen=True, slots=True)
class ServiceRequestUpdate:
    wonum: str
    code: Optional[str] = None
    priority: Optional[int] = None  # 1-5
    cancel: bool = False


@dataclass(frozen=True, slots=True)
class FaultInjection:
    """Start a fault at run time (T6). The scenario's own fault is injected at construction."""

    mode: str
    onset: datetime
    pf_days: float  # > 0


@dataclass(frozen=True, slots=True)
class Repair:
    """What a completed repair fixes. The simulator applies this when a work order completes;
    it is public so T6 can define what each failure code's repair actually restores."""

    wonum: str
    code: str
    time: datetime
    fixed: bool  # True if the code matched the active fault


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkOrderState:
    wonum: str
    code: Optional[str]
    priority: int
    status: WoStatus
    emergency: bool = False
    created: datetime
    tech: Optional[str] = None
    arrive_at: Optional[datetime] = None
    start_at: Optional[datetime] = None
    finish_at: Optional[datetime] = None
    dispatched: bool = False


@dataclass(frozen=True, slots=True)
class SimState:
    """Everything the environment may read. `severity` (0-1) and
    `fault_active` are ground truth, for scoring and tests only;
    never shown to the agent."""

    sim_time: datetime
    start: datetime
    end: datetime
    running: bool
    failed: bool
    fault_active: bool
    severity: float
    work_orders: list[WorkOrderState]


class SimulatorAPI(Protocol):
    """The methods the environment, baselines and tests call. sim.Sim satisfies this."""

    def advance(self, hours: float) -> list[TelemetryRow]: ...
    def inject_fault(self, fault: FaultInjection) -> None: ...
    def file_wo(self, code: Optional[str], priority: int = 3, wonum: Optional[str] = None): ...
    # AssetOpsBench's WO write tools, same arguments; raise GymError
    def generate_work_order(
        self, description: str, asset_num: str, site_id: str,
        priority: int = 3, work_type: str = "CM",
        reported_by: Optional[str] = None,
        location: Optional[str] = None, notes: Optional[str] = None,
        wonum: Optional[str] = None): ...
    def update_workorder(
        self, wonum: str, site_id: str,
        description: Optional[str] = None,
        priority: Optional[int] = None,
        location: Optional[str] = None,
        asset_num: Optional[str] = None, notes: Optional[str] = None,
        failure_code: Optional[str] = None): ...
    def cancel_workorder(self, wonum: str, site_id: str,
                         reason: Optional[str] = None): ...
    def acknowledge(self, alert_id: str): ...
    def shelve(self, rule: str, hours: float) -> datetime: ...
    def state(self) -> SimState: ...
    def result(self) -> EpisodeResult: ...


# ----------------------------------------------------------------------------
# 3. Ledger events
# ----------------------------------------------------------------------------

CostKind = Literal["dispatch", "repair", "planned_downtime", "unplanned_downtime", "excess_energy"]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Event:
    time: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class CostEvent(_Event):
    type: Literal["cost"] = "cost"
    kind: CostKind
    amount: float  # USD, >= 0
    wonum: Optional[str] = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DowntimeEvent(_Event):
    """One contiguous span the chiller was off, emitted when the span closes."""

    type: Literal["downtime"] = "downtime"
    kind: Literal["planned", "unplanned"]
    hours: float
    start: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class FailureEvent(_Event):
    type: Literal["failure"] = "failure"
    mode: str


@dataclass(frozen=True, slots=True, kw_only=True)
class AlertEvent(_Event):
    type: Literal["alert"] = "alert"
    kind: AlertKind
    rule: str
    sensor: str
    value: float
    message: str
    id: str
    shelved: bool = False  # raised while its rule was shelved; hidden


@dataclass(frozen=True, slots=True, kw_only=True)
class AlertActionEvent(_Event):
    """The agent acknowledged an alert or shelved a rule."""

    type: Literal["alert_action"] = "alert_action"
    action: Literal["ack", "shelve"]
    rule: str
    alert_id: Optional[str] = None
    hours: Optional[float] = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ServiceRequestEvent(_Event):
    """A work-order change, whoever caused it. Carries the whole
    work-order header, so the last event per wonum rebuilds the
    WO document; the first one's time is its reportdate."""

    type: Literal["service_request"] = "service_request"
    wonum: str
    status: WoStatus
    priority: int
    code: Optional[str] = None
    emergency: bool = False
    actor: Literal["agent", "bms", "planner", "technician"]
    tech: Optional[str] = None
    # technician arrived later than the priority's response target
    sla_breach: bool = False
    note: Optional[str] = None
    description: str
    site: str
    asset: str
    work_type: str
    location: Optional[str] = None
    reported_by: Optional[str] = None
    arrive_at: Optional[datetime] = None  # technician's scheduled arrival
    labor_hours: Optional[float] = None  # on completion


LedgerEvent = Union[CostEvent, DowntimeEvent, FailureEvent,
                    AlertEvent, AlertActionEvent, ServiceRequestEvent]


@dataclass(frozen=True, slots=True)
class KPIs:
    """The proposal's five KPIs, each a fold over the ledger. Logged beside the score, not in it."""

    downtime_hours: float
    missed_failures: int
    false_dispatches: int
    alert_load: int
    sla_breaches: int


@dataclass(frozen=True, slots=True)
class EpisodeResult:
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
    alerts: dict[AlertKind, int]
    kpis: KPIs
    events: int  # ledger length; the events themselves are in Sim.events
    # set by the environment at the end of an episode
    score: Optional[float] = None  # [-1, 1]
    end_reason: Optional[str] = None
    reference_costs: Optional[dict[str, float]] = None

    def __post_init__(self):
        if abs(sum(self.cost_breakdown.values()) - self.cost_total) > 0.05:
            raise ValueError("cost_breakdown does not sum to cost_total")


# ----------------------------------------------------------------------------
# 4. Agent tool results (the tool list is openenv.yaml)
# ----------------------------------------------------------------------------


class ErrorCode(str, Enum):
    NO_EPISODE = "NO_EPISODE"
    EPISODE_OVER = "EPISODE_OVER"
    BAD_ARGUMENT = "BAD_ARGUMENT"
    NOT_FOUND = "NOT_FOUND"
    UNKNOWN_RULE = "UNKNOWN_RULE"
    UNKNOWN_ASSET = "UNKNOWN_ASSET"
    UNKNOWN_FAILURE_CODE = "UNKNOWN_FAILURE_CODE"
    WORK_ORDER_LOCKED = "WORK_ORDER_LOCKED"  # technician arrived


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


@dataclass(frozen=True, slots=True)
class ClockResult:
    sim_time: datetime
    shift_end: datetime
    hours_left: float


@dataclass(frozen=True, slots=True)
class WorkOrderUpdate:
    wonum: str
    status: WoStatus
    time: datetime
    note: Optional[str]


@dataclass(frozen=True, slots=True)
class PublicAlert:
    """What the agent sees of an alert (alerts.Alert.public())."""

    id: str
    time: datetime
    kind: AlertKind
    rule: str
    site: str
    asset: str
    sensor: str
    value: float
    message: str
    acked: bool


@dataclass(frozen=True, slots=True)
class AdvanceTimeResult:
    sim_time: datetime
    shift_end: datetime
    sim_hours_advanced: float
    new_telemetry_rows: int
    new_alert_count: int
    new_alerts: list[PublicAlert]  # newest 30
    work_order_updates: list[WorkOrderUpdate]
    shift_over: bool


@dataclass(frozen=True, slots=True)
class ListAlertsResult:
    total: int
    returned: int
    alerts: list[PublicAlert]  # at most 200


@dataclass(frozen=True, slots=True)
class AckResult:
    id: str
    acked_at: datetime


@dataclass(frozen=True, slots=True)
class ShelveResult:
    rule: str
    shelved_until: datetime


@dataclass(frozen=True, slots=True)
class FinishResult:
    message: str
    score: float
