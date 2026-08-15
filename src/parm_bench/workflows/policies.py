from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..retrieval import RetrievalRequest
from .agent import AdmittedMemory
from .environment import ToolResult, ToolSpec


class MemoryPolicyNotImplementedError(LookupError):
    pass


class MemoryPolicy(Protocol):
    """When a system may search personal memory during a trajectory.

    The benchmark is retrieval-agnostic: a policy may use dense search, BM25, a
    graph, an agent-visible tool, or anything else. What the policy fixes is
    *when* retrieval is allowed to run and what text it is allowed to see.
    """

    name: str

    def initial_memory(self, goal: str) -> list[AdmittedMemory]: ...

    def on_observation(
        self, *, step_index: int, goal: str, observation_text: str
    ) -> list[AdmittedMemory]: ...

    def extra_tools(self) -> tuple[ToolSpec, ...]: ...

    def handle_tool(
        self, name: str, arguments: dict[str, Any], *, step_index: int
    ) -> tuple[str, list[AdmittedMemory]] | None: ...

    def drain_retrieved(self) -> list[str]: ...


@dataclass
class _BasePolicy:
    corpus_id: str = "fixture"
    retrieval_limit: int = 5
    _retrieved: list[str] = field(default_factory=list, init=False)

    def initial_memory(self, goal: str) -> list[AdmittedMemory]:
        return []

    def on_observation(
        self, *, step_index: int, goal: str, observation_text: str
    ) -> list[AdmittedMemory]:
        return []

    def extra_tools(self) -> tuple[ToolSpec, ...]:
        return ()

    def handle_tool(
        self, name: str, arguments: dict[str, Any], *, step_index: int
    ) -> tuple[str, list[AdmittedMemory]] | None:
        return None

    def drain_retrieved(self) -> list[str]:
        drained = list(self._retrieved)
        self._retrieved.clear()
        return drained

    def _admit(self, hits: Any, step_index: int, channel: str) -> list[AdmittedMemory]:
        admissions = []
        for hit in hits:
            self._retrieved.append(hit.slug)
            admissions.append(
                AdmittedMemory(
                    source_id=hit.slug,
                    text=hit.text,
                    step_index=step_index,
                    perturbations=tuple(hit.perturbations),
                    channel=channel,
                )
            )
        return admissions


@dataclass
class NoMemoryPolicy(_BasePolicy):
    name: str = "no_memory"


@dataclass
class InputRagPolicy(_BasePolicy):
    """Retrieve once from the original request and never look again.

    This is the policy the late-cue claim is about: the goal alone gives no
    reason to recall the memory, so a prompt-triggered system cannot find it no
    matter how good its ranker is.
    """

    name: str = "input_rag"
    retriever: Any = None

    def initial_memory(self, goal: str) -> list[AdmittedMemory]:
        if self.retriever is None:
            raise ValueError("input_rag requires a retriever")
        retrieval = self.retriever.retrieve(
            RetrievalRequest(goal, top_k=self.retrieval_limit, corpus_id=self.corpus_id)
        )
        return self._admit(retrieval.hits, -1, "input_rag")


@dataclass
class NaiveOutputRagPolicy(_BasePolicy):
    """Use every tool observation, whole, as a retrieval query."""

    name: str = "naive_output_rag"
    retriever: Any = None

    def on_observation(
        self, *, step_index: int, goal: str, observation_text: str
    ) -> list[AdmittedMemory]:
        if self.retriever is None:
            raise ValueError("naive_output_rag requires a retriever")
        if not observation_text.strip():
            return []
        retrieval = self.retriever.retrieve(
            RetrievalRequest(
                observation_text,
                top_k=self.retrieval_limit,
                corpus_id=self.corpus_id,
            )
        )
        return self._admit(retrieval.hits, step_index, "naive_output_rag")


@dataclass
class AllEntityOutputRagPolicy(_BasePolicy):
    """Turn every exact entity in an observation into a retrieval query."""

    name: str = "all_entity_output_rag"
    retriever: Any = None

    def on_observation(
        self, *, step_index: int, goal: str, observation_text: str
    ) -> list[AdmittedMemory]:
        if self.retriever is None or not hasattr(self.retriever, "retrieve_entities"):
            raise ValueError("all_entity_output_rag requires an entity retriever")
        if not observation_text.strip():
            return []
        retrieval = self.retriever.retrieve_entities(
            observation_text,
            top_k=self.retrieval_limit,
            corpus_id=self.corpus_id,
        )
        return self._admit(retrieval.hits, step_index, "all_entity_output_rag")


MEMORY_SEARCH_TOOL = ToolSpec(
    "search_personal_memory",
    "Search the user's personal notes and messages for facts, commitments, or "
    "preferences that may change what you should do next.",
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A concise standalone memory search query.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


@dataclass
class PromptedMemoryToolPolicy(_BasePolicy):
    """Give the agent a memory tool and let it decide when to search.

    This is the strongest "just ask the agent" baseline: the tool is described
    and always available, so any failure is a failure to notice that the memory
    might matter, not a missing capability.
    """

    name: str = "prompted_memory_tool"
    retriever: Any = None

    def extra_tools(self) -> tuple[ToolSpec, ...]:
        return (MEMORY_SEARCH_TOOL,)

    def handle_tool(
        self, name: str, arguments: dict[str, Any], *, step_index: int
    ) -> tuple[str, list[AdmittedMemory]] | None:
        if name != MEMORY_SEARCH_TOOL.name:
            return None
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return "error: query must be a non-empty string", []
        if self.retriever is None:
            raise ValueError("prompted_memory_tool requires a retriever")
        retrieval = self.retriever.retrieve(
            RetrievalRequest(
                query, top_k=self.retrieval_limit, corpus_id=self.corpus_id
            )
        )
        admissions = self._admit(retrieval.hits, step_index, "prompted_memory_tool")
        if not admissions:
            return "No personal memory matched that query.", []
        return (
            f"{len(admissions)} personal memory record(s) matched.",
            admissions,
        )


@dataclass
class PARMPolicy(_BasePolicy):
    """Search memory in parallel against every observation as it becomes visible.

    PARM sees the same observation stream and the same per-observation budget as
    the output-RAG policies. What differs is admission: it selects cue-sized
    regions inside the observation and may admit nothing at all, which is the
    behavior the cue-ablated control is built to measure.
    """

    name: str = "parm"
    retriever: Any = None

    def on_observation(
        self, *, step_index: int, goal: str, observation_text: str
    ) -> list[AdmittedMemory]:
        if self.retriever is None or not hasattr(
            self.retriever, "retrieve_observation"
        ):
            raise ValueError("parm requires a PARM observation retriever")
        if not observation_text.strip():
            return []
        retrieval = self.retriever.retrieve_observation(
            goal,
            observation_text,
            top_k=self.retrieval_limit,
            corpus_id=self.corpus_id,
        )
        regions = {
            region["region_id"]: region["text"]
            for region in retrieval.trace.get("regions", [])
        }
        admissions = []
        for hit in retrieval.hits:
            self._retrieved.append(hit.slug)
            admissions.append(
                AdmittedMemory(
                    source_id=hit.slug,
                    text=hit.text,
                    step_index=step_index,
                    trigger_text=regions.get(hit.diagnostics.get("region_id"), ""),
                    perturbations=tuple(hit.perturbations),
                    channel=str(hit.diagnostics.get("channel", "parm")),
                )
            )
        return admissions


PolicyBuilder = Callable[[str, int, Any], MemoryPolicy]

_POLICIES: dict[str, tuple[PolicyBuilder, str]] = {
    "no_memory": (
        lambda corpus_id, limit, retriever: NoMemoryPolicy(corpus_id, limit),
        "none",
    ),
    "input_rag": (
        lambda corpus_id, limit, retriever: InputRagPolicy(
            corpus_id, limit, retriever=retriever
        ),
        "mode_retriever",
    ),
    "naive_output_rag": (
        lambda corpus_id, limit, retriever: NaiveOutputRagPolicy(
            corpus_id, limit, retriever=retriever
        ),
        "mode_retriever",
    ),
    "prompted_memory_tool": (
        lambda corpus_id, limit, retriever: PromptedMemoryToolPolicy(
            corpus_id, limit, retriever=retriever
        ),
        "mode_retriever",
    ),
    "all_entity_output_rag": (
        lambda corpus_id, limit, retriever: AllEntityOutputRagPolicy(
            corpus_id, limit, retriever=retriever
        ),
        "entity_exact",
    ),
    "parm": (
        lambda corpus_id, limit, retriever: PARMPolicy(
            corpus_id, limit, retriever=retriever
        ),
        "parm_convergence",
    ),
}


def available_policies() -> tuple[str, ...]:
    return tuple(_POLICIES)


def policy_resource_kind(name: str) -> str:
    try:
        return _POLICIES[name][1]
    except KeyError as exc:
        raise MemoryPolicyNotImplementedError(
            f"Memory policy {name!r} is not implemented. "
            f"Available policies: {', '.join(available_policies())}"
        ) from exc


def get_policy(
    name: str, *, corpus_id: str, retrieval_limit: int, retriever: Any = None
) -> MemoryPolicy:
    try:
        builder = _POLICIES[name][0]
    except KeyError as exc:
        raise MemoryPolicyNotImplementedError(
            f"Memory policy {name!r} is not implemented. "
            f"Available policies: {', '.join(available_policies())}"
        ) from exc
    return builder(corpus_id, retrieval_limit, retriever)
