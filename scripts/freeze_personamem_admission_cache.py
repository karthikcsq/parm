from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parm_bench.dataset import load_cases
from parm_bench.retrieval import RetrievalIndex
from parm_bench.semantic_parm import (
    PARM_SEMANTIC_CANDIDATE_DEPTH,
    semantic_admission_cache_key,
    semantic_admission_cache_namespace,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/benchmark_personamem_mixed_v0"),
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
            "data/retrieval-experiments/personamem-mixed-v0/"
            "admission-judge-v2.jsonl"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-mixed-v0/"
            "admission-judge-v2-frozen-cache"
        ),
    )
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=PARM_SEMANTIC_CANDIDATE_DEPTH,
    )
    args = parser.parse_args()

    cases = {
        case["case_id"]: case
        for case in load_cases(args.dataset)
        if case["variant"] in {"positive", "cue-ablated"}
    }
    judge_rows = {
        row["case_id"]: row for row in _read_jsonl(args.judge_results)
    }
    if set(cases) != set(judge_rows):
        raise ValueError("judge result IDs do not match retrieval cases")
    index = RetrievalIndex.load(args.retrieval_index)
    namespace = semantic_admission_cache_namespace(
        index.manifest_hash,
        candidate_depth=args.candidate_depth,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for case_id, case in cases.items():
        row = judge_rows[case_id]
        key = semantic_admission_cache_key(
            case["prompt"],
            case["observation_text"],
            cache_namespace=namespace,
        )
        payload = {
            "admit": row["admit"],
            "candidate_id": row["candidate_id"],
            "confidence": row["confidence"],
            "rationale": row["rationale"],
            "selected_page_id": row["selected_page_id"],
            "selected_region_id": row["selected_region_id"],
            "selected_region_text": row["selected_region_text"],
            "resolved_model": row["resolved_model"],
            "response_id": row["response_id"],
        }
        (args.out / f"{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {len(cases)} frozen admission entries to {args.out}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
