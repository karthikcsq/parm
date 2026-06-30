from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .backends import MemoryBackend
from .models import LanguageModel


class BaselineNotImplementedError(LookupError):
    pass


@dataclass(frozen=True)
class BenchmarkInput:
    case_id: str
    prompt: str
    observation_kind: str
    observation_text: str


class Baseline(Protocol):
    name: str
    requires_memory: bool

    def run(
        self,
        case: BenchmarkInput,
        model: LanguageModel,
        memory_backend: MemoryBackend | None,
    ) -> dict[str, Any]: ...


class NoMemoryBaseline:
    name = "no_memory"
    requires_memory = False

    def run(
        self,
        case: BenchmarkInput,
        model: LanguageModel,
        memory_backend: MemoryBackend | None,
    ) -> dict[str, Any]:
        if memory_backend is not None:
            raise ValueError("no_memory baseline must not receive a memory backend")
        response = model.generate(
            prompt=case.prompt,
            observation_kind=case.observation_kind,
            observation_text=case.observation_text,
        )
        return {
            "case_id": case.case_id,
            "response_text": response.text,
            "requested_model": model.model_name,
            "resolved_model": response.resolved_model,
            "provider_response_id": response.response_id,
            "usage": response.usage,
            "trace": {
                "detected_cues": [],
                "retrieved_source_ids": [],
                "admitted_source_ids": [],
                "admitted_perturbations": {},
            },
        }


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


def benchmark_input(case: dict[str, Any]) -> BenchmarkInput:
    """Expose only model-visible benchmark fields to a baseline."""
    return BenchmarkInput(
        case_id=case["case_id"],
        prompt=case["prompt"],
        observation_kind=case["observation"]["kind"],
        observation_text=case["observation_text"],
    )


register_baseline(NoMemoryBaseline())
