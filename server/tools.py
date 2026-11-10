"""The agent's tool boundary: which tools exist (openenv.yaml)
and how a GymError becomes a normal ErrorResult."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import yaml

from ..schemas import ErrorResult, GymError


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
        raise RuntimeError(
            "registered tools != openenv.yaml declared_tools; "
            f"missing {sorted(set(declared) - set(registered))}, "
            f"undeclared {sorted(set(registered) - set(declared))}")


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
