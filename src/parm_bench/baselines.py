from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .backends import MemoryBackend


class BaselineNotImplementedError(LookupError):
    pass


class Baseline(Protocol):
    name: str

    def run(
        self,
        case: dict[str, Any],
        memory_backend: MemoryBackend,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PlannedBaseline:
    """Design inventory only. A planned baseline is not executable."""

    name: str
    research_question: str


PLANNED_BASELINES = (
    PlannedBaseline(
        "no_memory",
        "How does the response change when no personal-memory system is present?",
    ),
    PlannedBaseline(
        "input_rag",
        "What can retrieval triggered only from the original prompt recover?",
    ),
    PlannedBaseline(
        "prompted_memory_tool",
        "Will a real agent elect to use an available memory tool without being asked?",
    ),
    PlannedBaseline(
        "naive_output_rag",
        "What happens when the complete output is used for retrieval without cue selection?",
    ),
    PlannedBaseline(
        "all_entity_output_rag",
        "What happens when every extracted output entity becomes a retrieval query?",
    ),
    PlannedBaseline(
        "key_sentence_output_rag",
        "Does a deliberately developed key-sentence selector improve retrieval precision?",
    ),
    PlannedBaseline(
        "oracle_cue",
        "Is the gold memory recoverable when the correct cue is supplied?",
    ),
)

_BASELINES: dict[str, Baseline] = {}


def register_baseline(baseline: Baseline) -> None:
    """Register a reviewed implementation; planning metadata is insufficient."""
    name = baseline.name.strip()
    if not name:
        raise ValueError("baseline.name must be non-empty")
    if name in _BASELINES:
        raise ValueError(f"baseline already registered: {name}")
    _BASELINES[name] = baseline


def available_baselines() -> dict[str, Baseline]:
    return dict(_BASELINES)


def get_baseline(name: str) -> Baseline:
    try:
        return _BASELINES[name]
    except KeyError as exc:
        raise BaselineNotImplementedError(
            f"Baseline '{name}' is not implemented. "
            "No placeholder baseline will be executed."
        ) from exc
