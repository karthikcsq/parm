from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..retrieval import (
    EntityExactRetriever,
    IndexRetriever,
    OpenAIEmbedder,
    PARMConvergenceRetriever,
    RetrievalIndex,
    RetrievalMode,
)
from ..semantic_parm import (
    CachedOpenAIAdmissionJudge,
    PARMSemanticJudgeRetriever,
    semantic_admission_cache_namespace,
)
from .agent import (
    MAX_TRAJECTORY_STEPS,
    TrajectoryResult,
    WorkflowModel,
    WorkflowTruncationError,
    run_trajectory,
)
from .case import WorkflowCase
from .policies import get_policy, policy_resource_kind
from .verify import evaluate_assertions


def build_retrieval_resource(
    policy_name: str,
    *,
    retrieval_index: str | None,
    retrieval_mode: str | None,
    parm_retriever: str = "semantic-judge",
    parm_admission_cache: str | None = None,
    parm_admission_policy: str = "populate",
) -> Any:
    kind = policy_resource_kind(policy_name)
    if kind == "none":
        return None
    if retrieval_index is None:
        raise ValueError(f"memory policy {policy_name!r} requires --retrieval-index")
    index = RetrievalIndex.load(retrieval_index)
    if kind == "entity_exact":
        return EntityExactRetriever(index)
    if kind == "parm_convergence":
        if parm_retriever == "convergence":
            return PARMConvergenceRetriever(index, OpenAIEmbedder())
        # The workflow corpora are ordinary personal histories with no link
        # graph and no review/reflection filename convention, which is the
        # shape the semantic-pair admission path was built for. It stays the
        # workflow default for the same reason it is the PersonaMem default.
        if parm_admission_cache is None:
            raise ValueError(
                "semantic-judge PARM requires --parm-admission-cache"
            )
        return PARMSemanticJudgeRetriever(
            index,
            OpenAIEmbedder(),
            CachedOpenAIAdmissionJudge(
                parm_admission_cache,
                parm_admission_policy,
                cache_namespace=semantic_admission_cache_namespace(
                    index.manifest_hash
                ),
            ),
        )
    if retrieval_mode is None:
        raise ValueError(f"memory policy {policy_name!r} requires --retrieval-mode")
    mode = RetrievalMode(retrieval_mode)
    if mode is RetrievalMode.ENHANCED:
        raise ValueError(
            "enhanced retrieval needs a frozen expansion cache and is not wired "
            "into the workflow suite yet; use dense or hybrid"
        )
    return IndexRetriever(index, mode, OpenAIEmbedder())


def assert_gold_reachable(cases: list[WorkflowCase], retrieval_resource: Any) -> None:
    """Refuse to run a memory policy against an index missing a case's gold.

    Without this, running a scenario whose commitment lives above the first
    corpus tier against the first tier's index produces a clean-looking zero:
    every policy fails the positive, and nothing distinguishes "the retrieval
    policy missed it" from "the memory was not in the index". That is the most
    expensive kind of wrong result, because it looks like evidence.
    """

    index = getattr(retrieval_resource, "index", None)
    if index is None:
        return
    available = {page.slug for page in index.pages}
    for case in cases:
        missing = sorted(set(case.gold_source_ids) - available)
        if missing:
            raise ValueError(
                f"{case.case_id} declares tier {case.corpus_tier!r} but the "
                f"retrieval index at {index.path} is missing its gold "
                f"record(s): {', '.join(missing)}"
            )


def run_workflow_case(
    case: WorkflowCase,
    *,
    policy_name: str,
    model: WorkflowModel,
    retrieval_resource: Any,
    retrieval_limit: int = 5,
    max_steps: int = MAX_TRAJECTORY_STEPS,
) -> dict[str, Any]:
    """Run one case and verify the environment it leaves behind.

    The environment is rebuilt from the tracked fixture here, so a positive and
    its cue-ablated twin never share mutable state.
    """

    environment = case.build_environment()
    # A scenario may declare a larger budget than the run-wide default when its
    # ordinary path is simply longer. Never smaller: the caller's limit is a
    # ceiling on cost, and a case should not be able to raise it below what the
    # operator asked for.
    max_steps = max(max_steps, case.step_budget or 0)
    policy = get_policy(
        policy_name,
        corpus_id=case.corpus_id,
        retrieval_limit=retrieval_limit,
        retriever=retrieval_resource,
    )
    truncated = False
    try:
        trajectory = run_trajectory(
            case=case,
            environment=environment,
            policy=policy,
            model=model,
            max_steps=max_steps,
        )
    except WorkflowTruncationError as error:
        truncated = True
        trajectory = _empty_trajectory(case.case_id, error)

    assertions = evaluate_assertions(
        case.data.get("assertions", []),
        state=environment.state(),
        mutations=environment.mutations(),
        trajectory=environment.trajectory,
    )
    admission_steps: dict[str, int] = {}
    perturbations: dict[str, list[str]] = {}
    for memory in trajectory.admitted:
        admission_steps.setdefault(memory.source_id, memory.step_index)
        if memory.perturbations:
            perturbations[memory.source_id] = list(memory.perturbations)
    return {
        "injected_memory_tokens": _injected_memory_tokens(trajectory.admitted),
        "case_id": case.case_id,
        "variant": case.variant,
        "policy": policy_name,
        "final_text": trajectory.final_text,
        "stopped_reason": trajectory.stopped_reason,
        "truncated": truncated,
        "requested_model": model.model_name,
        "resolved_model": trajectory.resolved_model,
        "steps": trajectory.steps,
        "assertions": assertions,
        "mutations": list(environment.mutations()),
        "usage": trajectory.usage,
        "trace": {
            "corpus_id": case.corpus_id,
            "retrieved_source_ids": list(
                dict.fromkeys(trajectory.retrieved_source_ids)
            ),
            "admitted_source_ids": list(admission_steps),
            "admission_steps": admission_steps,
            "admitted_perturbations": perturbations,
            "retrieval_events": trajectory.retrieval_events,
        },
    }


def _injected_memory_tokens(admitted: list[Any]) -> int:
    """Count the memory tokens a policy actually put in front of the model.

    Every policy gets the same top-k per observation, and none is capped across
    a trajectory, because capping the total would handicap a broad policy rather
    than measure it. What separates them is how much history they end up
    injecting to reach the same decision, so that cost is reported instead of
    constrained. Admissions are deduplicated by source before injection, so this
    counts each record once however many times it was retrieved.
    """

    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    seen: set[str] = set()
    total = 0
    for memory in admitted:
        if memory.source_id in seen:
            continue
        seen.add(memory.source_id)
        total += len(encoding.encode(memory.text))
        if memory.trigger_text:
            total += len(encoding.encode(memory.trigger_text))
    return total


def run_workflow_cases(
    cases: list[WorkflowCase],
    *,
    policy_name: str,
    model: WorkflowModel,
    retrieval_resource: Any,
    retrieval_limit: int = 5,
    max_steps: int = MAX_TRAJECTORY_STEPS,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Run cases, optionally in parallel.

    Parallel runs are safe because every case owns a freshly built environment
    and no case reads another case's state. They are not safe against a shared
    live external service, which is one reason the pilot environment is local.
    """

    def run_one(case: WorkflowCase) -> dict[str, Any]:
        return run_workflow_case(
            case,
            policy_name=policy_name,
            model=model,
            retrieval_resource=retrieval_resource,
            retrieval_limit=retrieval_limit,
            max_steps=max_steps,
        )

    if workers <= 1:
        return [run_one(case) for case in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, cases))


def _empty_trajectory(case_id: str, error: WorkflowTruncationError) -> TrajectoryResult:
    result = TrajectoryResult(case_id=case_id, final_text="")
    result.stopped_reason = f"truncated:{error.reason}"
    return result
