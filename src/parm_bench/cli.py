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
    benchmark_input,
    get_baseline,
)
from .dataset import DatasetValidationError, load_cases, validate_cases
from .models import OpenAIResponsesModel
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
                args.model,
                args.out,
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
    model_name: str | None,
    out: str,
) -> int:
    cases = load_cases(dataset_dir)
    validate_cases(cases)
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
    retriever: IndexRetriever | None = None
    if implementation.requires_memory:
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
        retriever = IndexRetriever(
            index,
            mode,
            OpenAIEmbedder(),
            expander=expander,
        )
    model = OpenAIResponsesModel(_resolve_model(model_name))
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
            for case in cases:
                row = implementation.run(
                    benchmark_input(case),
                    model,
                    retriever,
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
        retriever=retriever,
        output_rag_flow=output_rag_flow,
        retrieval_limit=(
            retrieval_limit if implementation.requires_memory else None
        ),
        requested_model=model.model_name,
    )
    print(f"Wrote {len(cases)} predictions to {path}")
    return 0


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
    retriever: IndexRetriever | None,
    output_rag_flow: str | None,
    retrieval_limit: int | None,
    requested_model: str,
) -> None:
    payload = {
        "baseline": baseline,
        "output_rag_flow": output_rag_flow,
        "retrieval_mode": retriever.mode.value if retriever is not None else None,
        "retrieval_index": (
            str(retriever.index.path) if retriever is not None else None
        ),
        "retrieval_manifest_hash": (
            retriever.index.manifest_hash if retriever is not None else None
        ),
        "corpus": (
            retriever.index.manifest["corpus_id"]
            if retriever is not None
            else None
        ),
        "gbrain_version": (
            retriever.index.manifest["gbrain_version"]
            if retriever is not None
            else None
        ),
        "chunker_version": (
            retriever.index.manifest["chunker_version"]
            if retriever is not None
            else None
        ),
        "embedding_model": (
            retriever.index.manifest["embedding_model"]
            if retriever is not None
            else None
        ),
        "retrieval_limit": retrieval_limit,
        "admission_policy": (
            "all_retrieved" if retriever is not None else None
        ),
        "perturbation_filtering": False if retriever is not None else None,
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
            if retriever is not None
            else None
        ),
        "expansion_model": (
            EXPANSION_MODEL
            if retriever is not None
            and retriever.mode is RetrievalMode.ENHANCED
            else None
        ),
        "expansion_prompt_version": (
            EXPANSION_PROMPT_VERSION
            if retriever is not None
            and retriever.mode is RetrievalMode.ENHANCED
            else None
        ),
        "expansion_cache_hash": (
            retriever.expander.cache_hash
            if retriever is not None and retriever.expander is not None
            else None
        ),
        "dependency_versions": (
            _dependency_versions() if retriever is not None else None
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
    uses_memory = args.baseline != "no_memory"
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
