from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .models import FINAL_ANSWER_INSTRUCTIONS, LanguageModel
from .retrieval import RetrievalRequest, Retriever


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
        retriever: Retriever | None,
    ) -> dict[str, Any]: ...


INPUT_RAG_INSTRUCTIONS = (
    "Follow the selection task using the supplied observation and retrieved "
    "personal memory. Return exactly one requested label or name and no "
    "explanation."
)


class NoMemoryBaseline:
    name = "no_memory"
    requires_memory = False

    def run(
        self,
        case: BenchmarkInput,
        model: LanguageModel,
        retriever: Retriever | None,
    ) -> dict[str, Any]:
        if retriever is not None:
            raise ValueError("no_memory baseline must not receive a retriever")
        response = model.generate(
            prompt=case.prompt,
            observation_kind=case.observation_kind,
            observation_text=case.observation_text,
            instructions=FINAL_ANSWER_INSTRUCTIONS,
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
class InputRagBaseline:
    retrieval_limit: int = 5
    name = "input_rag"
    requires_memory = True

    def __post_init__(self) -> None:
        if self.retrieval_limit < 1:
            raise ValueError("retrieval_limit must be at least 1")

    def run(
        self,
        case: BenchmarkInput,
        model: LanguageModel,
        retriever: Retriever | None,
    ) -> dict[str, Any]:
        if retriever is None:
            raise ValueError("input_rag baseline requires a retriever")
        retrieval = retriever.retrieve(
            RetrievalRequest(case.prompt, top_k=self.retrieval_limit)
        )
        hits = retrieval.hits
        memory_context = "\n\n".join(
            f"{index}. {hit.text}" for index, hit in enumerate(hits, start=1)
        )
        response = model.generate(
            prompt=case.prompt,
            observation_kind=case.observation_kind,
            observation_text=case.observation_text,
            instructions=(
                INPUT_RAG_INSTRUCTIONS
                if memory_context
                else FINAL_ANSWER_INSTRUCTIONS
            ),
            memory_context=memory_context or None,
        )
        source_ids = [hit.slug for hit in hits]
        page_ids = [hit.page_id for hit in hits]
        return {
            "case_id": case.case_id,
            "response_text": response.text,
            "requested_model": model.model_name,
            "resolved_model": response.resolved_model,
            "provider_response_id": response.response_id,
            "usage": response.usage,
            "trace": {
                "detected_cues": [],
                **retrieval.trace,
                "retrieved_page_ids": page_ids,
                "retrieved_source_ids": source_ids,
                "admitted_source_ids": source_ids,
                "admitted_perturbations": {
                    hit.slug: list(hit.perturbations)
                    for hit in hits
                    if hit.perturbations
                },
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

@dataclass(frozen=True)
class BaselineConfiguration:
    retrieval_limit: int = 5


BaselineBuilder = Callable[[BaselineConfiguration], Baseline]

_BASELINES: dict[str, BaselineBuilder] = {}


def register_baseline(name: str, builder: BaselineBuilder) -> None:
    """Register a reviewed implementation factory."""
    name = name.strip()
    if not name:
        raise ValueError("baseline name must be non-empty")
    if name in _BASELINES:
        raise ValueError(f"baseline already registered: {name}")
    _BASELINES[name] = builder


def available_baselines() -> dict[str, BaselineBuilder]:
    return dict(_BASELINES)


def get_baseline(
    name: str,
    configuration: BaselineConfiguration | None = None,
) -> Baseline:
    try:
        builder = _BASELINES[name]
    except KeyError as exc:
        raise BaselineNotImplementedError(
            f"Baseline '{name}' is not implemented. "
            "No placeholder baseline will be executed."
        ) from exc
    return builder(configuration or BaselineConfiguration())


def benchmark_input(case: dict[str, Any]) -> BenchmarkInput:
    """Expose only model-visible benchmark fields to a baseline."""
    return BenchmarkInput(
        case_id=case["case_id"],
        prompt=case["prompt"],
        observation_kind=case["observation"]["kind"],
        observation_text=case["observation_text"],
    )


register_baseline("no_memory", lambda _: NoMemoryBaseline())
register_baseline(
    "input_rag",
    lambda configuration: InputRagBaseline(configuration.retrieval_limit),
)
