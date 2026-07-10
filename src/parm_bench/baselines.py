from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Protocol

from .models import FINAL_ANSWER_INSTRUCTIONS, LanguageModel, ModelResponse
from .retrieval import EntityRetriever, RetrievalRequest, Retriever


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
    retrieval_resource: "RetrievalResourceKind"

    def run(
        self,
        case: BenchmarkInput,
        model: LanguageModel,
        retriever: Retriever | EntityRetriever | None,
    ) -> dict[str, Any]: ...


INPUT_RAG_INSTRUCTIONS = (
    "Follow the selection task using the supplied observation and retrieved "
    "personal memory. Return exactly one requested label or name and no "
    "explanation."
)


class OutputRagFlow(str, Enum):
    TOOL_OUTPUT_ONLY = "tool_output_only"
    MODEL_OUTPUT_ONLY = "model_output_only"
    TOOL_THEN_MODEL_OUTPUT = "tool_then_model_output"


class RetrievalResourceKind(str, Enum):
    NONE = "none"
    MODE_RETRIEVER = "mode_retriever"
    ENTITY_EXACT = "entity_exact"


class NoMemoryBaseline:
    name = "no_memory"
    requires_memory = False
    retrieval_resource = RetrievalResourceKind.NONE

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
    retrieval_resource = RetrievalResourceKind.MODE_RETRIEVER

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
class NaiveOutputRagBaseline:
    retrieval_limit: int = 5
    output_rag_flow: OutputRagFlow = OutputRagFlow.TOOL_THEN_MODEL_OUTPUT
    name = "naive_output_rag"
    requires_memory = True
    retrieval_resource = RetrievalResourceKind.MODE_RETRIEVER

    def __post_init__(self) -> None:
        if self.retrieval_limit < 1:
            raise ValueError("retrieval_limit must be at least 1")
        object.__setattr__(
            self,
            "output_rag_flow",
            OutputRagFlow(self.output_rag_flow),
        )

    def run(
        self,
        case: BenchmarkInput,
        model: LanguageModel,
        retriever: Retriever | None,
    ) -> dict[str, Any]:
        if retriever is None:
            raise ValueError("naive_output_rag baseline requires a retriever")

        retrievals = []
        model_passes = []
        admitted_hits = []
        final_instructions = FINAL_ANSWER_INSTRUCTIONS

        if self.output_rag_flow is OutputRagFlow.TOOL_OUTPUT_ONLY:
            tool_retrieval = _retrieve_phase(
                retriever,
                "tool_output",
                case.observation_text,
                self.retrieval_limit,
            )
            retrievals.append(tool_retrieval)
            admitted_hits.extend(tool_retrieval["hits"])
        elif self.output_rag_flow is OutputRagFlow.MODEL_OUTPUT_ONLY:
            first_response = model.generate(
                prompt=case.prompt,
                observation_kind=case.observation_kind,
                observation_text=case.observation_text,
                instructions=FINAL_ANSWER_INSTRUCTIONS,
            )
            model_passes.append(_model_pass_trace("first_model", first_response))
            model_retrieval = _retrieve_phase(
                retriever,
                "model_output",
                first_response.text,
                self.retrieval_limit,
            )
            retrievals.append(model_retrieval)
            admitted_hits.extend(model_retrieval["hits"])
        else:
            tool_retrieval = _retrieve_phase(
                retriever,
                "tool_output",
                case.observation_text,
                self.retrieval_limit,
            )
            retrievals.append(tool_retrieval)
            tool_hits = tool_retrieval["hits"]
            intermediate_context = _memory_context(tool_hits)
            intermediate_response = model.generate(
                prompt=case.prompt,
                observation_kind=case.observation_kind,
                observation_text=case.observation_text,
                instructions=(
                    INPUT_RAG_INSTRUCTIONS
                    if intermediate_context
                    else FINAL_ANSWER_INSTRUCTIONS
                ),
                memory_context=intermediate_context or None,
            )
            model_passes.append(
                _model_pass_trace("intermediate_model", intermediate_response)
            )
            model_retrieval = _retrieve_phase(
                retriever,
                "model_output",
                intermediate_response.text,
                self.retrieval_limit,
            )
            retrievals.append(model_retrieval)
            admitted_hits.extend(tool_hits)
            admitted_hits.extend(model_retrieval["hits"])

        deduped_hits = _dedupe_hits_by_source(admitted_hits)
        memory_context = _memory_context(deduped_hits)
        if memory_context:
            final_instructions = INPUT_RAG_INSTRUCTIONS
        response = model.generate(
            prompt=case.prompt,
            observation_kind=case.observation_kind,
            observation_text=case.observation_text,
            instructions=final_instructions,
            memory_context=memory_context or None,
        )
        source_ids = [hit.slug for hit in deduped_hits]
        page_ids = [hit.page_id for hit in deduped_hits]
        return {
            "case_id": case.case_id,
            "response_text": response.text,
            "requested_model": model.model_name,
            "resolved_model": response.resolved_model,
            "provider_response_id": response.response_id,
            "usage": response.usage,
            "trace": {
                "detected_cues": [],
                "output_rag_flow": self.output_rag_flow.value,
                "model_passes": model_passes,
                "retrieval_queries": [
                    retrieval["query"] for retrieval in retrievals
                ],
                "retrievals": [
                    {
                        "phase": retrieval["phase"],
                        "query": retrieval["query"],
                        "trace": retrieval["trace"],
                    }
                    for retrieval in retrievals
                ],
                "retrieved_page_ids": [
                    hit.page_id
                    for retrieval in retrievals
                    for hit in retrieval["hits"]
                ],
                "retrieved_source_ids": [
                    hit.slug
                    for retrieval in retrievals
                    for hit in retrieval["hits"]
                ],
                "admitted_page_ids": page_ids,
                "admitted_source_ids": source_ids,
                "admitted_perturbations": {
                    hit.slug: list(hit.perturbations)
                    for hit in deduped_hits
                    if hit.perturbations
                },
            },
        }


@dataclass(frozen=True)
class AllEntityOutputRagBaseline:
    retrieval_limit: int = 5
    name = "all_entity_output_rag"
    requires_memory = True
    retrieval_resource = RetrievalResourceKind.ENTITY_EXACT

    def __post_init__(self) -> None:
        if self.retrieval_limit < 1:
            raise ValueError("retrieval_limit must be at least 1")

    def run(
        self,
        case: BenchmarkInput,
        model: LanguageModel,
        retriever: Retriever | EntityRetriever | None,
    ) -> dict[str, Any]:
        if retriever is None or not hasattr(retriever, "retrieve_entities"):
            raise ValueError("all_entity_output_rag baseline requires an entity retriever")
        retrieval = retriever.retrieve_entities(
            case.observation_text,
            top_k=self.retrieval_limit,
        )
        admitted_hits = _dedupe_entity_hits(list(retrieval.hits))
        memory_context = _memory_context(admitted_hits)
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
        retrieved_source_ids = [hit.slug for hit in retrieval.hits]
        admitted_source_ids = [hit.slug for hit in admitted_hits]
        return {
            "case_id": case.case_id,
            "response_text": response.text,
            "requested_model": model.model_name,
            "resolved_model": response.resolved_model,
            "provider_response_id": response.response_id,
            "usage": response.usage,
            "trace": {
                "detected_cues": [
                    {
                        "seed_id": seed.seed_id,
                        "surface": seed.surface,
                        "normalized_surface": seed.normalized_surface,
                        "source": seed.source,
                        "span": [seed.span_start, seed.span_end],
                        "matched_page_ids": list(seed.matched_page_ids),
                    }
                    for seed in retrieval.seeds
                ],
                **retrieval.trace,
                "retrieved_page_ids": [hit.page_id for hit in retrieval.hits],
                "retrieved_source_ids": retrieved_source_ids,
                "admitted_page_ids": [hit.page_id for hit in admitted_hits],
                "admitted_source_ids": admitted_source_ids,
                "admitted_perturbations": {
                    hit.slug: list(hit.perturbations)
                    for hit in admitted_hits
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
    output_rag_flow: OutputRagFlow = OutputRagFlow.TOOL_THEN_MODEL_OUTPUT


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
register_baseline(
    "naive_output_rag",
    lambda configuration: NaiveOutputRagBaseline(
        configuration.retrieval_limit,
        configuration.output_rag_flow,
    ),
)
register_baseline(
    "all_entity_output_rag",
    lambda configuration: AllEntityOutputRagBaseline(
        configuration.retrieval_limit,
    ),
)


def retrieval_resource_kind(name: str) -> RetrievalResourceKind:
    return get_baseline(name).retrieval_resource


def _retrieve_phase(
    retriever: Retriever,
    phase: str,
    query: str,
    retrieval_limit: int,
) -> dict[str, Any]:
    retrieval = retriever.retrieve(
        RetrievalRequest(query, top_k=retrieval_limit)
    )
    return {
        "phase": phase,
        "query": query,
        "hits": list(retrieval.hits),
        "trace": retrieval.trace,
    }


def _memory_context(hits: list[Any]) -> str:
    return "\n\n".join(
        f"{index}. {hit.text}" for index, hit in enumerate(hits, start=1)
    )


def _dedupe_hits_by_source(hits: list[Any]) -> list[Any]:
    seen = set()
    deduped = []
    for hit in hits:
        if hit.slug in seen:
            continue
        seen.add(hit.slug)
        deduped.append(hit)
    return deduped


def _dedupe_entity_hits(hits: list[Any]) -> list[Any]:
    best_by_page = {}
    contributors: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        contributors.setdefault(hit.page_id, []).append(
            {
                "seed_id": hit.diagnostics.get("entity_seed_id"),
                "surface": hit.diagnostics.get("entity_surface"),
                "source": hit.diagnostics.get("entity_source"),
                "rank": hit.rank,
                "score": hit.score,
            }
        )
        previous = best_by_page.get(hit.page_id)
        if previous is None or (hit.rank, -hit.score, hit.page_id) < (
            previous.rank,
            -previous.score,
            previous.page_id,
        ):
            best_by_page[hit.page_id] = hit
    ordered = sorted(
        best_by_page.values(),
        key=lambda hit: (hit.rank, -hit.score, hit.page_id),
    )
    deduped = []
    for rank, hit in enumerate(ordered, start=1):
        diagnostics = dict(hit.diagnostics)
        diagnostics["entity_contributors"] = contributors[hit.page_id]
        deduped.append(replace(hit, rank=rank, diagnostics=diagnostics))
    return deduped


def _model_pass_trace(phase: str, response: ModelResponse) -> dict[str, Any]:
    return {
        "phase": phase,
        "response_text": response.text,
        "provider_response_id": response.response_id,
        "resolved_model": response.resolved_model,
        "usage": response.usage,
    }
