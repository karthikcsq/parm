from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .amara import normalize_amara
from .baselines import (
    BaselineConfiguration,
    BaselineNotImplementedError,
    OutputRagFlow,
    RetrievalResourceKind,
    benchmark_input,
    get_baseline,
    retrieval_resource_kind,
)
from .dataset import (
    VARIANTS,
    DatasetValidationError,
    load_cases,
    validate_cases,
)
from .models import (
    CachingLanguageModel,
    ModelTruncationError,
    OpenAIResponsesModel,
    ResponseCacheMissError,
    ResponsePolicy,
)
from .retrieval import (
    CANDIDATE_DEPTH,
    COSINE_WEIGHT,
    EXPANSION_MODEL,
    EXPANSION_PROMPT_VERSION,
    GRAPH_MIN_INBOUND,
    GRAPH_MULTIPLIER,
    OFFICIAL_TOP_K,
    RRF_K,
    RRF_WEIGHT,
    CachedOpenAIQueryExpander,
    EntityExactRetriever,
    ExpansionCacheMissError,
    ExpansionPolicy,
    IndexRetriever,
    OpenAIEmbedder,
    RetrievalIndex,
    RetrievalMode,
    RetrievalValidationError,
)
from .retrieval_export import export_gbrain_index
from .scoring import score_predictions
from .workbench import serve_workbench


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(prog="parm-bench")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-amara")
    prepare.add_argument("--source", default="data/amara-life-v1/source")
    prepare.add_argument("--out", default=".gbrain-local/corpus/amara-life-v1")

    export = commands.add_parser("export-retrieval-index")
    export.add_argument("--out", required=True)
    export.add_argument(
        "--gbrain-runtime"
    )
    export.add_argument("--gbrain-home")
    export.add_argument("--corpus-id", default="amara-life-v1")
    export.add_argument("--chunker-version", required=True)
    export.add_argument(
        "--provenance-source", default="data/amara-life-v1/source"
    )

    validate = commands.add_parser("validate")
    validate.add_argument("dataset_dir")

    inspect = commands.add_parser("inspect")
    inspect.add_argument("dataset_dir")
    inspect.add_argument("--case", dest="case_id")

    run = commands.add_parser("run")
    run.add_argument("dataset_dir")
    run.add_argument("--baseline", required=True)
    run.add_argument(
        "--output-rag-flow",
        choices=tuple(flow.value for flow in OutputRagFlow),
    )
    run.add_argument(
        "--retrieval-mode", choices=tuple(mode.value for mode in RetrievalMode)
    )
    run.add_argument("--retrieval-index")
    run.add_argument("--retrieval-limit", type=_positive_int)
    run.add_argument("--expansion-cache")
    run.add_argument(
        "--expansion-policy",
        choices=tuple(policy.value for policy in ExpansionPolicy),
    )
    run.add_argument("--response-cache")
    run.add_argument(
        "--response-policy",
        choices=tuple(policy.value for policy in ResponsePolicy),
    )
    run.add_argument(
        "--variant",
        action="append",
        dest="variants",
        choices=tuple(sorted(VARIANTS)),
        help="restrict the run to one or more variants (default: all)",
    )
    run.add_argument("--model")
    run.add_argument("--out", required=True)

    score = commands.add_parser("score")
    score.add_argument("results_jsonl")
    score.add_argument("--gold", required=True)
    score.add_argument("--out")

    workbench = commands.add_parser("serve-workbench")
    workbench.add_argument("--retrieval-index", required=True)
    workbench.add_argument("--dataset", default="data/benchmark_v1")
    workbench.add_argument("--host", default="127.0.0.1")
    workbench.add_argument("--port", type=_port, default=8765)
    workbench.add_argument("--model")
    workbench.add_argument("--expansion-cache")
    workbench.add_argument(
        "--expansion-policy",
        choices=tuple(policy.value for policy in ExpansionPolicy),
        default=ExpansionPolicy.FROZEN.value,
    )
    workbench.add_argument("--no-open", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-amara":
            count = normalize_amara(args.source, args.out)
            print(f"Prepared {count} Amara pages in {args.out}")
            return 0
        if args.command == "export-retrieval-index":
            manifest = export_gbrain_index(
                args.out,
                runtime=_from_repo_root(
                    args.gbrain_runtime
                    or os.environ.get("PARM_GBRAIN_CWD")
                    or ".gbrain-local/runtime/gbrain"
                ),
                gbrain_home=_from_repo_root(
                    args.gbrain_home
                    or os.environ.get("GBRAIN_HOME")
                    or ".gbrain-local/home"
                ),
                corpus_id=args.corpus_id,
                chunker_version=args.chunker_version,
                provenance_source=_from_repo_root(args.provenance_source),
            )
            print(
                f"Exported {manifest['counts']['pages']} pages to {args.out}"
            )
            return 0
        if args.command == "validate":
            cases = load_cases(args.dataset_dir)
            validate_cases(cases)
            print(f"Validated {len(cases)} cases")
            return 0
        if args.command == "inspect":
            return _inspect(args.dataset_dir, args.case_id)
        if args.command == "run":
            _validate_retrieval_args(args)
            return _run(
                args.dataset_dir,
                args.baseline,
                args.output_rag_flow,
                args.retrieval_mode,
                args.retrieval_index,
                args.retrieval_limit or OFFICIAL_TOP_K,
                args.expansion_cache,
                args.expansion_policy,
                args.response_cache,
                args.response_policy,
                args.model,
                args.out,
                args.variants,
            )
        if args.command == "score":
            return _score(args.results_jsonl, args.gold, args.out)
        if args.command == "serve-workbench":
            serve_workbench(
                retrieval_index=args.retrieval_index,
                dataset_dir=_from_repo_root(args.dataset),
                host=args.host,
                port=args.port,
                model_name=_resolve_model(args.model),
                expansion_cache=args.expansion_cache,
                expansion_policy=args.expansion_policy,
                open_browser=not args.no_open,
            )
            return 0
    except DatasetValidationError as exc:
        for issue in exc.issues:
            print(issue, file=sys.stderr)
        return 1
    except BaselineNotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (
        ExpansionCacheMissError,
        ResponseCacheMissError,
        RetrievalValidationError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


def _inspect(dataset_dir: str, case_id: str | None) -> int:
    cases = load_cases(dataset_dir)
    validate_cases(cases)
    case = next(
        (item for item in cases if case_id is None or item["case_id"] == case_id),
        None,
    )
    if case is None:
        print(f"Case not found: {case_id}", file=sys.stderr)
        return 1
    print(json.dumps({key: value for key, value in case.items() if not key.startswith("_") and key != "observation_text"}, indent=2))
    print(f"\nObservation tokens are validated; content: {case['observation']['content_path']}")
    return 0


def _run(
    dataset_dir: str,
    baseline: str,
    output_rag_flow: str | None,
    retrieval_mode: str | None,
    retrieval_index: str | None,
    retrieval_limit: int,
    expansion_cache: str | None,
    expansion_policy: str | None,
    response_cache: str | None,
    response_policy: str | None,
    model_name: str | None,
    out: str,
    variants: list[str] | None = None,
) -> int:
    cases = load_cases(dataset_dir)
    validate_cases(cases)
    if variants:
        wanted = set(variants)
        cases = [case for case in cases if case["variant"] in wanted]
        if not cases:
            raise ValueError(
                "no cases match the requested --variant filter: "
                + ", ".join(sorted(wanted))
            )
    implementation = get_baseline(
        baseline,
        BaselineConfiguration(
            retrieval_limit=retrieval_limit,
            output_rag_flow=(
                OutputRagFlow(output_rag_flow)
                if output_rag_flow is not None
                else OutputRagFlow.TOOL_THEN_MODEL_OUTPUT
            ),
        ),
    )
    retrieval_resource: IndexRetriever | EntityExactRetriever | None = None
    if implementation.retrieval_resource is RetrievalResourceKind.MODE_RETRIEVER:
        assert retrieval_mode is not None
        assert retrieval_index is not None
        mode = RetrievalMode(retrieval_mode)
        expander = None
        if mode is RetrievalMode.ENHANCED:
            assert expansion_cache is not None
            expander = CachedOpenAIQueryExpander(
                expansion_cache,
                expansion_policy or ExpansionPolicy.FROZEN,
            )
        index = RetrievalIndex.load(retrieval_index)
        retrieval_resource = IndexRetriever(
            index,
            mode,
            OpenAIEmbedder(),
            expander=expander,
        )
    elif implementation.retrieval_resource is RetrievalResourceKind.ENTITY_EXACT:
        assert retrieval_index is not None
        retrieval_resource = EntityExactRetriever(RetrievalIndex.load(retrieval_index))
    model: OpenAIResponsesModel | CachingLanguageModel = OpenAIResponsesModel(
        _resolve_model(model_name)
    )
    if response_cache is not None:
        model = CachingLanguageModel(
            model,
            response_cache,
            ResponsePolicy(response_policy or ResponsePolicy.FROZEN),
        )
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            truncated = 0
            for case in cases:
                try:
                    row = implementation.run(
                        benchmark_input(case),
                        model,
                        retrieval_resource,
                    )
                except ModelTruncationError as error:
                    truncated += 1
                    row = _truncated_row(case, model, error)
                    print(
                        f"Case {case['case_id']} truncated "
                        f"(reason={error.reason}); recorded as abstention",
                        file=sys.stderr,
                    )
                row["baseline"] = baseline
                if output_rag_flow is not None:
                    row["output_rag_flow"] = output_rag_flow
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    _write_run_configuration(
        path,
        baseline=baseline,
        retrieval_resource=retrieval_resource,
        output_rag_flow=output_rag_flow,
        retrieval_limit=(
            retrieval_limit if implementation.requires_memory else None
        ),
        requested_model=model.model_name,
        response_cache_hash=(
            model.cache_hash if isinstance(model, CachingLanguageModel) else None
        ),
        variants=sorted(set(variants)) if variants else None,
    )
    print(f"Wrote {len(cases)} predictions to {path}")
    if truncated:
        print(
            f"{truncated} of {len(cases)} cases truncated before completion",
            file=sys.stderr,
        )
    return 0


def _truncated_row(
    case: dict[str, Any],
    model: OpenAIResponsesModel | CachingLanguageModel,
    error: ModelTruncationError,
) -> dict[str, Any]:
    """Build a prediction row for a case whose model response was truncated.

    An empty ``response_text`` with an empty trace scores as an abstention, so a
    truncated case is recorded rather than aborting the batch. ``truncated`` and
    ``truncation_reason`` mark the row for downstream inspection.
    """
    return {
        "case_id": case["case_id"],
        "response_text": "",
        "requested_model": model.model_name,
        "resolved_model": model.model_name,
        "provider_response_id": error.response_id,
        "usage": {},
        "truncated": True,
        "truncation_reason": error.reason,
        "trace": {
            "detected_cues": [],
            "retrieved_source_ids": [],
            "admitted_source_ids": [],
            "admitted_perturbations": {},
        },
    }


def _resolve_model(cli_model: str | None) -> str:
    return cli_model or os.environ.get("PARM_OPENAI_MODEL") or "gpt-5-mini"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65_535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def _write_run_configuration(
    results_path: Path,
    *,
    baseline: str,
    retrieval_resource: IndexRetriever | EntityExactRetriever | None,
    output_rag_flow: str | None,
    retrieval_limit: int | None,
    requested_model: str,
    response_cache_hash: str | None = None,
    variants: list[str] | None = None,
) -> None:
    index = (
        retrieval_resource.index if retrieval_resource is not None else None
    )
    mode = getattr(retrieval_resource, "mode", None)
    expander = getattr(retrieval_resource, "expander", None)
    payload = {
        "baseline": baseline,
        "variants": variants,
        "response_cache_hash": response_cache_hash,
        "output_rag_flow": output_rag_flow,
        "retrieval_mode": mode.value if mode is not None else None,
        "retrieval_condition_detail": (
            getattr(retrieval_resource, "retrieval_condition_detail", None)
            if mode is None
            else None
        ),
        "retrieval_index": (
            str(index.path) if index is not None else None
        ),
        "retrieval_manifest_hash": (
            index.manifest_hash if index is not None else None
        ),
        "corpus": (
            index.manifest["corpus_id"]
            if index is not None
            else None
        ),
        "gbrain_version": (
            index.manifest["gbrain_version"]
            if index is not None
            else None
        ),
        "chunker_version": (
            index.manifest["chunker_version"]
            if index is not None
            else None
        ),
        "embedding_model": (
            index.manifest["embedding_model"]
            if index is not None
            else None
        ),
        "retrieval_limit": retrieval_limit,
        "admission_policy": (
            "all_retrieved" if retrieval_resource is not None else None
        ),
        "perturbation_filtering": False if retrieval_resource is not None else None,
        "requested_model": requested_model,
        "constants": (
            {
                "candidate_depth": CANDIDATE_DEPTH,
                "rrf_k": RRF_K,
                "rrf_weight": RRF_WEIGHT,
                "cosine_weight": COSINE_WEIGHT,
                "graph_min_inbound": GRAPH_MIN_INBOUND,
                "graph_multiplier": GRAPH_MULTIPLIER,
            }
            if mode is not None
            else None
        ),
        "expansion_model": (
            EXPANSION_MODEL
            if mode is RetrievalMode.ENHANCED
            else None
        ),
        "expansion_prompt_version": (
            EXPANSION_PROMPT_VERSION
            if mode is RetrievalMode.ENHANCED
            else None
        ),
        "expansion_cache_hash": (
            expander.cache_hash
            if expander is not None
            else None
        ),
        "dependency_versions": (
            _dependency_versions() if retrieval_resource is not None else None
        ),
    }
    path = results_path.with_suffix(".config.json")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _validate_retrieval_args(args: argparse.Namespace) -> None:
    if getattr(args, "response_policy", None) and not getattr(
        args, "response_cache", None
    ):
        raise ValueError("--response-policy requires --response-cache")
    resource_kind = retrieval_resource_kind(args.baseline)
    uses_memory = resource_kind is not RetrievalResourceKind.NONE
    retrieval_arguments = {
        "--retrieval-mode": args.retrieval_mode,
        "--retrieval-index": args.retrieval_index,
        "--retrieval-limit": args.retrieval_limit,
        "--expansion-cache": args.expansion_cache,
        "--expansion-policy": args.expansion_policy,
    }
    if not uses_memory:
        supplied = [name for name, value in retrieval_arguments.items() if value]
        if args.output_rag_flow:
            supplied.append("--output-rag-flow")
        if supplied:
            raise ValueError(
                "no_memory rejects retrieval arguments: " + ", ".join(supplied)
            )
        return
    if args.baseline == "naive_output_rag":
        if not args.output_rag_flow:
            raise ValueError("naive_output_rag requires --output-rag-flow")
    elif args.output_rag_flow:
        raise ValueError(
            "--output-rag-flow is only valid for naive_output_rag"
        )
    if resource_kind is RetrievalResourceKind.ENTITY_EXACT:
        if not args.retrieval_index:
            raise ValueError("all_entity_output_rag requires --retrieval-index")
        supplied = []
        if args.retrieval_mode:
            supplied.append("--retrieval-mode")
        if args.expansion_cache:
            supplied.append("--expansion-cache")
        if args.expansion_policy:
            supplied.append("--expansion-policy")
        if supplied:
            raise ValueError(
                "all_entity_output_rag rejects retrieval-mode arguments: "
                + ", ".join(supplied)
            )
        return
    if not args.retrieval_mode or not args.retrieval_index:
        raise ValueError(
            "memory-using baselines require --retrieval-mode and "
            "--retrieval-index"
        )
    if args.retrieval_mode == RetrievalMode.ENHANCED.value:
        if not args.expansion_cache:
            raise ValueError("enhanced retrieval requires --expansion-cache")
    elif args.expansion_cache or args.expansion_policy:
        raise ValueError(
            "expansion cache arguments are only valid for enhanced retrieval"
        )


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for distribution in ("numpy", "openai"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unknown"
    return versions


def _from_repo_root(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _score(results_path: str, gold: str, out: str | None) -> int:
    cases = load_cases(gold)
    validate_cases(cases)
    predictions = _load_jsonl(Path(results_path))
    metrics = score_predictions(cases, predictions)
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote metrics to {path}")
    else:
        print(payload)
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
