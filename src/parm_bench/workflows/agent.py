from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..service_tier import service_tier_kwargs
from .environment import ToolSpec, WorkflowEnvironment


MAX_TRAJECTORY_STEPS = 32
MAX_OUTPUT_TOKENS = 8192
WORKFLOW_CACHE_SCHEMA = 1

AGENT_INSTRUCTIONS = (
    "You are an engineering agent working in a repository through the supplied "
    "tools. Complete the user's request end to end: inspect the repository "
    "before acting, then make the changes the request implies. Call one tool "
    "at a time and read each result before deciding the next step. When the "
    "work is finished, reply with a short plain-text summary of what you did "
    "and why, and make no further tool calls."
)

MEMORY_NOTICE = (
    "Personal memory recalled while you were working. It comes from the user's "
    "own notes and messages. Treat it as a standing instruction from the user "
    "and let it change your plan if it applies to what you just saw:"
)


class WorkflowTruncationError(RuntimeError):
    def __init__(self, reason: str, response_id: str) -> None:
        self.reason = reason
        self.response_id = response_id
        super().__init__(
            f"workflow model response truncated (reason={reason!r}, "
            f"response_id={response_id!r})"
        )


@dataclass(frozen=True)
class AgentAction:
    """One model turn: either a tool call or the final summary."""

    tool_name: str | None
    arguments: dict[str, Any]
    call_id: str | None
    final_text: str | None
    raw_items: list[dict[str, Any]]
    response_id: str
    resolved_model: str
    usage: dict[str, Any]

    @property
    def is_final(self) -> bool:
        return self.tool_name is None


class WorkflowModel(Protocol):
    model_name: str

    def next_action(
        self,
        *,
        instructions: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentAction: ...


class OpenAIWorkflowModel:
    def __init__(self, model_name: str, client: Any | None = None) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self.client = client or OpenAI()

    def next_action(
        self,
        *,
        instructions: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentAction:
        response = self.client.responses.create(
            model=self.model_name,
            instructions=instructions,
            input=conversation,
            tools=tools,
            tool_choice="auto",
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
            **service_tier_kwargs(),
        )
        if getattr(response, "status", None) == "incomplete":
            details = getattr(response, "incomplete_details", None)
            raise WorkflowTruncationError(
                getattr(details, "reason", None) or "unknown", response.id
            )
        raw_items = [_dump(item) for item in response.output]
        calls = [item for item in raw_items if item.get("type") == "function_call"]
        usage = _usage_dict(response.usage)
        if calls:
            call = calls[0]
            try:
                arguments = json.loads(call.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            # Keep only the reasoning/tool items up to and including the first
            # call so a multi-call turn still executes one step at a time.
            keep_index = raw_items.index(call)
            return AgentAction(
                tool_name=str(call.get("name")),
                arguments=arguments if isinstance(arguments, dict) else {},
                call_id=str(call.get("call_id")),
                final_text=None,
                raw_items=raw_items[: keep_index + 1],
                response_id=response.id,
                resolved_model=response.model,
                usage=usage,
            )
        return AgentAction(
            tool_name=None,
            arguments={},
            call_id=None,
            final_text=response.output_text,
            raw_items=raw_items,
            response_id=response.id,
            resolved_model=response.model,
            usage=usage,
        )


class WorkflowCacheMissError(RuntimeError):
    def __init__(self, request_hash: str) -> None:
        self.request_hash = request_hash
        super().__init__(f"frozen workflow cache miss for {request_hash}")


class CachingWorkflowModel:
    """Freeze each agent turn so a whole trajectory replays exactly.

    The cache key is the full request: model, instructions, tool definitions,
    and the entire conversation so far. Two runs that reach the same state ask
    the same question and get the same answer, which is what makes a multi-step
    comparison between retrieval policies reproducible.
    """

    def __init__(
        self,
        base: WorkflowModel,
        cache_dir: str | Path,
        policy: str = "frozen",
    ) -> None:
        self.base = base
        self.cache_dir = Path(cache_dir)
        if policy not in {"frozen", "populate"}:
            raise ValueError("workflow cache policy must be frozen or populate")
        self.policy = policy
        self._used: list[Path] = []

    @property
    def model_name(self) -> str:
        return self.base.model_name

    def next_action(
        self,
        *,
        instructions: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentAction:
        request_hash = _request_hash(
            self.base.model_name, instructions, conversation, tools
        )
        path = self.cache_dir / f"{request_hash}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("request_hash") != request_hash:
                raise RuntimeError(f"invalid workflow cache entry: {path}")
            self._used.append(path)
            return AgentAction(
                tool_name=payload["tool_name"],
                arguments=payload["arguments"],
                call_id=payload["call_id"],
                final_text=payload["final_text"],
                raw_items=payload["raw_items"],
                response_id=payload["response_id"],
                resolved_model=payload["resolved_model"],
                usage=payload.get("usage", {}),
            )
        if self.policy == "frozen":
            raise WorkflowCacheMissError(request_hash)
        action = self.base.next_action(
            instructions=instructions, conversation=conversation, tools=tools
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Variants that share a goal share their opening turn, so concurrent
        # cases can hash to the same entry. Write through a per-writer
        # temporary name and let the first finisher win.
        temporary = path.with_name(
            f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(
                {
                    "schema": WORKFLOW_CACHE_SCHEMA,
                    "request_hash": request_hash,
                    "model": self.base.model_name,
                    "tool_name": action.tool_name,
                    "arguments": action.arguments,
                    "call_id": action.call_id,
                    "final_text": action.final_text,
                    "raw_items": action.raw_items,
                    "response_id": action.response_id,
                    "resolved_model": action.resolved_model,
                    "usage": action.usage,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            temporary.replace(path)
        except OSError:
            temporary.unlink(missing_ok=True)
            if not path.exists():
                raise
        self._used.append(path)
        return action

    @property
    def cache_hash(self) -> str | None:
        if not self._used:
            return None
        digest = hashlib.sha256()
        for path in sorted(set(self._used)):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class AdmittedMemory:
    source_id: str
    text: str
    step_index: int
    trigger_text: str = ""
    perturbations: tuple[str, ...] = ()
    channel: str = ""


@dataclass
class TrajectoryResult:
    case_id: str
    final_text: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    admitted: list[AdmittedMemory] = field(default_factory=list)
    retrieved_source_ids: list[str] = field(default_factory=list)
    retrieval_events: list[dict[str, Any]] = field(default_factory=list)
    usage: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = "final_answer"
    resolved_model: str = ""


def run_trajectory(
    *,
    case: Any,
    environment: WorkflowEnvironment,
    policy: Any,
    model: WorkflowModel,
    max_steps: int = MAX_TRAJECTORY_STEPS,
) -> TrajectoryResult:
    """Drive one agent trajectory and record when memory arrived.

    The memory policy is offered every tool observation as it becomes visible.
    Whatever it admits is appended to the conversation before the next model
    turn, and the step index of that observation is the admission time the
    verifier checks against the decision-bearing action.
    """

    environment_tools = list(environment.tools())
    tool_definitions = [tool.as_tool_definition() for tool in environment_tools]
    for extra in policy.extra_tools():
        tool_definitions.append(extra.as_tool_definition())
    mutating = {tool.name for tool in environment_tools if tool.mutating}

    conversation: list[dict[str, Any]] = [
        {"role": "user", "content": case.goal}
    ]
    result = TrajectoryResult(case_id=case.case_id, final_text="")
    admitted_by_source: dict[str, AdmittedMemory] = {}

    def absorb(event_step: int, admissions: list[AdmittedMemory], trigger: str) -> None:
        fresh = [
            memory
            for memory in admissions
            if memory.source_id not in admitted_by_source
        ]
        for memory in admissions:
            admitted_by_source.setdefault(memory.source_id, memory)
        if not fresh:
            return
        result.admitted.extend(fresh)
        blocks = []
        for memory in fresh:
            if memory.trigger_text:
                blocks.append(
                    f"Triggering observation:\n{memory.trigger_text}\n"
                    f"Personal memory:\n{memory.text}"
                )
            else:
                blocks.append(memory.text)
        conversation.append(
            {
                "role": "user",
                "content": MEMORY_NOTICE + "\n\n" + "\n\n".join(blocks),
            }
        )
        result.retrieval_events.append(
            {
                "step_index": event_step,
                "trigger": trigger,
                "admitted_source_ids": [memory.source_id for memory in fresh],
            }
        )

    initial = policy.initial_memory(case.goal)
    result.retrieved_source_ids.extend(policy.drain_retrieved())
    absorb(-1, initial, "initial_request")

    for _ in range(max_steps):
        action = model.next_action(
            instructions=AGENT_INSTRUCTIONS,
            conversation=conversation,
            tools=tool_definitions,
        )
        result.usage.append(action.usage)
        result.resolved_model = action.resolved_model
        conversation.extend(action.raw_items)
        if action.is_final:
            result.final_text = action.final_text or ""
            break
        handled = policy.handle_tool(action.tool_name, action.arguments)
        if handled is not None:
            observation, admissions = handled
            step_index = len(result.steps)
            result.steps.append(
                {
                    "step_index": step_index,
                    "tool_name": action.tool_name,
                    "arguments": action.arguments,
                    "ok": True,
                    "mutating": False,
                    "observation_text": observation,
                    "source": "memory_tool",
                }
            )
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": action.call_id,
                    "output": observation,
                }
            )
            result.retrieved_source_ids.extend(policy.drain_retrieved())
            absorb(step_index, admissions, f"memory_tool:{action.tool_name}")
            continue

        tool_result = environment.invoke(action.tool_name, action.arguments)
        step_index = len(result.steps)
        result.steps.append(
            {
                "step_index": step_index,
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "ok": tool_result.ok,
                "mutating": action.tool_name in mutating and tool_result.ok,
                "observation_text": tool_result.text,
                "source": "environment",
            }
        )
        conversation.append(
            {
                "type": "function_call_output",
                "call_id": action.call_id,
                "output": tool_result.text,
            }
        )
        if tool_result.ok:
            admissions = policy.on_observation(
                step_index=step_index,
                goal=case.goal,
                observation_text=tool_result.text,
            )
            result.retrieved_source_ids.extend(policy.drain_retrieved())
            absorb(step_index, admissions, f"observation:{action.tool_name}")
    else:
        result.stopped_reason = "step_limit"

    if not result.final_text and result.stopped_reason == "final_answer":
        result.stopped_reason = "no_final_answer"
    return result


def _dump(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    if isinstance(item, dict):
        return item
    return {"type": "unknown"}


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return {}


def _request_hash(
    model_name: str,
    instructions: str,
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    payload = {
        "schema": WORKFLOW_CACHE_SCHEMA,
        "model": model_name,
        "instructions": instructions,
        "conversation": _cache_safe(conversation),
        "tools": [tool.get("name") for tool in tools],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_VOLATILE_KEYS = {"id", "call_id", "encrypted_content"}


def _cache_safe(value: Any) -> Any:
    """Strip provider-assigned identifiers that change on every live call."""

    if isinstance(value, dict):
        return {
            key: _cache_safe(item)
            for key, item in sorted(value.items())
            if key not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_cache_safe(item) for item in value]
    return value
