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
from .maintenance import WorkOrder
from .sim import Sim


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
    shape. Left out: generate's aob_source (a provenance dict)."""

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

    return generate_work_order, update_workorder, cancel_workorder
