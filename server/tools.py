"""The agent's tool boundary: which tools exist (openenv.yaml),
how a GymError becomes a normal ErrorResult, AssetOpsBench's
tools re-exposed, and its WO write signatures."""

from __future__ import annotations

import asyncio
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Awaitable, Callable, Optional, ParamSpec,
                    TypeVar)

import yaml
from fastmcp.tools import Tool as FastMCPTool
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent, Tool as MCPTool
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import to_jsonable_python

from ..schemas import ErrorResult, GymError, WorkOrderMutationResult
from .faults import SUBSYSTEMS
from .maintenance import MAX_DEFER_H, WorkOrder
from .sim import Sim
from .telemetry import SITE


@dataclass(frozen=True, slots=True)
class Manifest:
    name: str
    tools: tuple[str, ...]  # declared_tools: what the agent sees


def load_manifest(path: Path) -> Manifest:
    m = yaml.safe_load(path.read_text())
    tools = m["validation"]["capabilities"]["declared_tools"]
    return Manifest(name=m["name"], tools=tuple(tools))


MANIFEST = load_manifest(
    Path(__file__).resolve().parents[1] / "openenv.yaml")


def check_tools(registered: list[str],
                declared: tuple[str, ...]) -> None:
    if sorted(registered) != sorted(declared):
        twice = {n for n in registered if registered.count(n) > 1}
        raise RuntimeError(
            "registered tools != openenv.yaml declared_tools; "
            f"missing {sorted(set(declared) - set(registered))}, "
            f"undeclared {sorted(set(registered) - set(declared))}, "
            f"registered twice {sorted(twice)}")


P = ParamSpec("P")
R = TypeVar("R")


def _errors_as_results(
        fn: Callable[P, R]) -> Callable[P, R | ErrorResult]:
    @functools.wraps(fn)
    def run(*args: P.args, **kwargs: P.kwargs) -> R | ErrorResult:
        try:
            return fn(*args, **kwargs)
        except GymError as e:
            return ErrorResult(error=e.message, error_code=e.code)
    return run


class _ForwardTool(FastMCPTool):
    """An AssetOpsBench tool re-exposed by the gym's own MCP
    server, so it is reachable both through step() and through
    OpenEnv's direct /mcp route."""

    forward: SkipJsonSchema[Any] = Field(exclude=True)

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        data = to_jsonable_python(await asyncio.to_thread(
            self.forward, self.name, arguments))
        text = json.dumps(data, default=str)
        return ToolResult(
            content=[TextContent(type="text", text=text)],
            structured_content={"result": data})


_WRAPPED_OUTPUT = {"type": "object", "properties": {"result": {}},
                   "required": ["result"],
                   "x-fastmcp-wrap-result": True}


def forward_tool(t: MCPTool,
                 forward: Callable[[str, dict], Any]) -> FastMCPTool:
    return _ForwardTool(
        name=t.name, description=t.description or "",
        parameters=t.inputSchema, output_schema=_WRAPPED_OUTPUT,
        forward=forward)


Mutation = WorkOrderMutationResult | ErrorResult
# (verb for the message, the change to apply) -> the tool result
Write = Callable[[str, Callable[[Sim], WorkOrder]], Awaitable[Mutation]]


def wo_write_tools(write: Write) -> tuple[Callable, ...]:
    """AssetOpsBench's WO writes: its names, arguments and result
    shape (left out: generate's aob_source, a provenance dict).
    Then the gym's own, for this site's one chiller. The agent
    sees the ones openenv.yaml declares."""

    async def generate_work_order(
            description: str, asset_num: str, site_id: str,
            priority: int = 3, work_type: str = "CM",
            reported_by: Optional[str] = None,
            location: Optional[str] = None,
            notes: Optional[str] = None,
            wonum: Optional[str] = None) -> Mutation:
        """Create a work order (status WAPPR)."""
        return await write("created (WAPPR)", lambda sim:
                           sim.generate_work_order(
                               description, asset_num, site_id,
                               priority, work_type, reported_by,
                               location, notes, wonum))

    async def update_workorder(
            wonum: str, site_id: str,
            description: Optional[str] = None,
            priority: Optional[int] = None,
            location: Optional[str] = None,
            asset_num: Optional[str] = None,
            notes: Optional[str] = None,
            failure_code: Optional[str] = None) -> Mutation:
        """Update mutable fields on a work order. They lock once
        the technician arrives."""
        return await write("updated", lambda sim:
                           sim.update_workorder(
                               wonum, site_id, description, priority,
                               location, asset_num, notes,
                               failure_code))

    async def cancel_workorder(
            wonum: str, site_id: str,
            reason: Optional[str] = None) -> Mutation:
        """Cancel a work order (-> CAN), before the technician
        arrives."""
        return await write("cancelled (CAN)", lambda sim:
                           sim.cancel_workorder(wonum, site_id, reason))

    async def investigate(subsystem: str, priority: int = 3,
                          notes: Optional[str] = None) -> Mutation:
        """Send a technician to inspect one part of the chiller:
        {parts}. Opens an inspection work order; nothing is
        repaired and the chiller keeps running. The finding comes
        back as that order's update once the visit is done: the
        fault and its failure code, or nothing abnormal. A fault
        elsewhere, or one still too small to see, is not found."""
        return await write("created (WAPPR)", lambda sim:
                           sim.investigate(subsystem, priority, notes))

    investigate.__doc__ = investigate.__doc__.format(
        parts=", ".join(SUBSYSTEMS))

    async def create_work_order(
            description: str, failure_code: Optional[str] = None,
            priority: int = 3,
            notes: Optional[str] = None) -> Mutation:
        """Open a corrective work order for the chiller (status
        WAPPR). Give the failure code if you know it
        (get_failure_codes lists them): the technician does the
        work it names, and an order without one gets a general
        inspection that finds nothing. update_workorder sets or
        changes the code until the technician arrives."""
        return await write("created (WAPPR)", lambda sim:
                           sim.file_wo(failure_code, priority, None,
                                       description, notes))

    async def escalate(wonum: str,
                       reason: Optional[str] = None) -> Mutation:
        """Raise an open work order to priority 1, the fastest
        response, before the technician arrives."""
        return await write("escalated", lambda sim:
                           sim.update_workorder(
                               wonum, SITE, priority=1,
                               notes=reason or "Escalated by the "
                               "agent."))

    async def defer(wonum: str, hours: float,
                    reason: Optional[str] = None) -> Mutation:
        """Push an open work order's visit later by `hours` (0.25
        to {max}), before the technician arrives: for planned work
        that need not happen yet. Its response target moves too."""
        return await write("deferred", lambda sim:
                           sim.defer(wonum, hours, reason))

    defer.__doc__ = defer.__doc__.format(max=MAX_DEFER_H)

    async def assign_technician(wonum: str,
                                technician: str) -> Mutation:
        """Give an open work order to a named technician instead of
        the planner's pick, before the technician arrives. One
        without the skill for its failure code brings in an outside
        contractor, which is slower."""
        return await write("assigned", lambda sim:
                           sim.assign_technician(wonum, technician))

    return (generate_work_order, update_workorder,
            cancel_workorder, investigate, create_work_order,
            escalate, defer, assign_technician)
