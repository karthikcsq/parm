from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import RateLimitError

from parm_bench.baselines import PARMBaseline, benchmark_input
from parm_bench.dataset import load_cases
from parm_bench.models import (
    CachingLanguageModel,
    OpenAIResponsesModel,
    ResponsePolicy,
)
from parm_bench.retrieval import (
    RetrievalHit,
    RetrievalIndex,
    RetrievalResult,
)
from parm_bench.scoring import score_predictions


class SingleResultRetriever:
    def __init__(
        self,
        result: RetrievalResult,
    ) -> None:
        self.result = result

    def retrieve_observation(
        self,
        prompt: str,
        observation_text: str,
        *,
        top_k: int,
        corpus_id: str | None = None,
    ) -> RetrievalResult:
        del prompt, observation_text, top_k, corpus_id
        return self.result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/benchmark_personamem_v0"),
    )
    parser.add_argument(
        "--retrieval-index",
        type=Path,
        default=Path("data/retrieval-indexes/personamem-v2-train-v0"),
    )
    parser.add_argument(
        "--judge-results",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "admission-judge-v2.jsonl"
        ),
    )
    parser.add_argument(
        "--response-cache",
        type=Path,
        default=Path(
            "data/response-caches/personamem-v0-admission-judge-v2"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "admission-judge-v2-predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "admission-judge-v2.metrics.json"
        ),
    )
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()

    load_dotenv()
    cases = [
        case
        for case in load_cases(args.dataset)
        if case["variant"] in {"positive", "cue-ablated"}
    ]
    judge_by_case = {
        row["case_id"]: row for row in _read_jsonl(args.judge_results)
    }
    index = RetrievalIndex.load(args.retrieval_index)
    page_by_id = {page.page_id: page for page in index.pages}
    chunks_by_page: dict[str, list[Any]] = defaultdict(list)
    for chunk in index.chunks:
        chunks_by_page[chunk.page_id].append(chunk)

    model = CachingLanguageModel(
        OpenAIResponsesModel(args.model),
        args.response_cache,
        ResponsePolicy.POPULATE,
    )
    baseline = PARMBaseline(retrieval_limit=1)
    predictions = []
    for position, case in enumerate(cases, start=1):
        judge = judge_by_case[case["case_id"]]
        result = _retrieval_result(
            case,
            judge,
            page_by_id=page_by_id,
            chunks_by_page=chunks_by_page,
        )
        row = _run_with_rate_limit_retry(
            baseline,
            case,
            model,
            SingleResultRetriever(result),
        )
        predictions.append(row)
        print(
            f"[{position:02d}/{len(cases)}] {case['case_id']}: "
            f"{row['response_text']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out, predictions)
    metrics = score_predictions(cases, predictions)
    args.metrics.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in metrics.items()
                if key != "rows"
                and not isinstance(value, dict)
            },
            sort_keys=True,
        )
    )


def _retrieval_result(
    case: dict[str, Any],
    judge: dict[str, Any],
    *,
    page_by_id: dict[str, Any],
    chunks_by_page: dict[str, list[Any]],
) -> RetrievalResult:
    trace = {
        "corpus_id": case["corpus_id"],
        "retrieval_condition_detail": "semantic_pair_judge_v2_replay",
        "evidence_projection_version": "raw_source_and_region_v1",
        "admission_policy": "contrastive_llm_judge_v2",
        "semantic_seeds": [],
        "entity_seeds": [],
        "regions": [],
        "judge": {
            "admit": judge["admit"],
            "confidence": judge["confidence"],
            "rationale": judge["rationale"],
            "response_id": judge["response_id"],
            "rubric_version": judge["rubric_version"],
        },
    }
    if not judge["admit"]:
        return RetrievalResult((), trace)

    page_id = judge["selected_page_id"]
    page = page_by_id[page_id]
    chunks = chunks_by_page[page_id]
    chunk = chunks[0]
    region_id = judge["selected_region_id"]
    region_text = judge["selected_region_text"]
    trace["regions"] = [
        {
            "region_id": region_id,
            "span": None,
            "text": region_text,
            "description": region_text,
            "rare_concepts": [],
        }
    ]
    hit = RetrievalHit(
        page_id=page.page_id,
        source_id=page.source_id,
        slug=page.slug,
        title=page.title,
        chunk_id=chunk.chunk_id,
        text="\n".join(item.text for item in chunks),
        score=float(judge["confidence"]),
        rank=1,
        perturbations=page.perturbations,
        diagnostics={
            "region_id": region_id,
            "admission_channel": "contrastive_llm_judge_v2",
        },
        corpus_id=page.corpus_id,
    )
    return RetrievalResult((hit,), trace)


def _run_with_rate_limit_retry(
    baseline: PARMBaseline,
    case: dict[str, Any],
    model: CachingLanguageModel,
    retriever: SingleResultRetriever,
) -> dict[str, Any]:
    for attempt in range(8):
        try:
            return baseline.run(
                benchmark_input(case),
                model,
                retriever,
            )
        except RateLimitError:
            if attempt == 7:
                raise
            time.sleep(min(20, 2 ** attempt))
    raise AssertionError("unreachable")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


if __name__ == "__main__":
    main()
