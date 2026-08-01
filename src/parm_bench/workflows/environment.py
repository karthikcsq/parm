from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


class EnvironmentNotImplementedError(LookupError):
    pass


class ToolInvocationError(ValueError):
    """Raised for a malformed call. The agent sees the message and may retry."""


@dataclass(frozen=True)
class ToolSpec:
    """One tool the agent may call, in OpenAI Responses function-tool shape."""

    name: str
    description: str
    parameters: dict[str, Any]
    mutating: bool = False

    def as_tool_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    text: str


@dataclass(frozen=True)
class TrajectoryStep:
    """One executed tool call and the observation it produced.

    ``step_index`` is the timing coordinate every workflow assertion uses: when
    the cue became visible, when memory was admitted, and when the decision-
    bearing action happened are all expressed as step indices on this log.
    """

    step_index: int
    tool_name: str
    arguments: dict[str, Any]
    ok: bool
    observation_text: str
    mutating: bool


class WorkflowEnvironment(Protocol):
    adapter_name: str
    adapter_version: str

    def tools(self) -> tuple[ToolSpec, ...]: ...

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...

    @property
    def trajectory(self) -> tuple[TrajectoryStep, ...]: ...

    def state(self) -> dict[str, Any]: ...

    def mutations(self) -> tuple[dict[str, Any], ...]: ...


EnvironmentBuilder = Callable[[dict[str, Any]], WorkflowEnvironment]

_ENVIRONMENTS: dict[str, EnvironmentBuilder] = {}


def register_environment(name: str, builder: EnvironmentBuilder) -> None:
    name = name.strip()
    if not name:
        raise ValueError("environment adapter name must be non-empty")
    if name in _ENVIRONMENTS:
        raise ValueError(f"environment adapter already registered: {name}")
    _ENVIRONMENTS[name] = builder


def available_environments() -> tuple[str, ...]:
    return tuple(sorted(_ENVIRONMENTS))


def get_environment(name: str, fixture: dict[str, Any]) -> WorkflowEnvironment:
    try:
        builder = _ENVIRONMENTS[name]
    except KeyError as exc:
        raise EnvironmentNotImplementedError(
            f"Environment adapter {name!r} is not implemented. "
            f"Available adapters: {', '.join(available_environments()) or 'none'}"
        ) from exc
    return builder(fixture)


def load_fixture(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class _TrajectoryRecorder:
    """Shared step log so every adapter reports timing the same way."""

    steps: list[TrajectoryStep] = field(default_factory=list)
    _mutations: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        *,
        mutating: bool,
    ) -> TrajectoryStep:
        step = TrajectoryStep(
            step_index=len(self.steps),
            tool_name=tool_name,
            arguments=dict(arguments),
            ok=result.ok,
            observation_text=result.text,
            mutating=mutating,
        )
        self.steps.append(step)
        return step

    def record_mutation(
        self, step_index: int, kind: str, detail: dict[str, Any]
    ) -> None:
        self._mutations.append({"step_index": step_index, "kind": kind, **detail})

    @property
    def next_step_index(self) -> int:
        return len(self.steps)

    @property
    def mutations(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._mutations)
