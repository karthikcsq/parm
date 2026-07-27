from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from parm_bench.dataset import load_cases
from parm_bench.retrieval import RetrievalIndex
from parm_bench.semantic_parm import (
    PARM_SEMANTIC_JUDGE_INSTRUCTIONS,
    PARM_SEMANTIC_JUDGE_MODEL,
    PARM_SEMANTIC_JUDGE_RUBRIC,
    PARM_SEMANTIC_METHODS,
)


MODEL = PARM_SEMANTIC_JUDGE_MODEL
RUBRIC_VERSION = PARM_SEMANTIC_JUDGE_RUBRIC
METHODS = frozenset(PARM_SEMANTIC_METHODS)
INSTRUCTIONS = PARM_SEMANTIC_JUDGE_INSTRUCTIONS


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
        "--features",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "mutual-alignment-features.jsonl"
        ),
    )
    parser.add_argument(
        "--support-audit",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "source-support-audit.jsonl"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "admission-judge-v2-cache"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "admission-judge-v2.jsonl"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--candidate-depth", type=int, default=10)
    args = parser.parse_args()
    if args.candidate_depth < 1:
        raise ValueError("candidate depth must be at least 1")

    load_dotenv()
    cases = [
        case
        for case in load_cases(args.dataset)
        if case["variant"] in {"positive", "cue-ablated"}
    ]
    features_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(args.features):
        if (
            row["method"] in METHODS
            and row["global_rank"] <= args.candidate_depth
        ):
            features_by_case[row["case_id"]].append(row)
    support_by_base = {
        row["base_case_id"]: row["grade"]
        for row in _read_jsonl(args.support_audit)
    }
    index = RetrievalIndex.load(args.retrieval_index)
    page_text = _page_text_by_id(index)
    args.cache.mkdir(parents=True, exist_ok=True)

    work = [
        (
            position,
            case,
            _candidate_pairs(
                features_by_case[case["case_id"]],
                page_text,
            ),
        )
        for position, case in enumerate(cases, start=1)
    ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _judge_case,
                case,
                candidates,
                cache_dir=args.cache,
            ): (position, case, candidates)
            for position, case, candidates in work
        }
        for future in as_completed(futures):
            position, case, candidates = futures[future]
            result = future.result()
            candidate_by_id = {
                candidate["candidate_id"]: candidate
                for candidate in candidates
            }
            selected = candidate_by_id.get(result["candidate_id"])
            gold_page_id = (
                f"{case['corpus_id']}:"
                f"{case['memory']['gold_source_ids'][0]}"
            )
            support_key = case["base_case_id"].replace(
                "parm-personamem-mixed-",
                "parm-personamem-",
            )
            grade = support_by_base[support_key]
            expected_admission = (
                case["variant"] == "positive"
                and grade != "unsupported"
            )
            row = {
                "case_id": case["case_id"],
                "base_case_id": case["base_case_id"],
                "variant": case["variant"],
                "source_support_grade": grade,
                "expected_admission": expected_admission,
                "candidate_count": len(candidates),
                "admit": result["admit"],
                "candidate_id": result["candidate_id"],
                "confidence": result["confidence"],
                "rationale": result["rationale"],
                "selected_page_id": (
                    selected["page_id"] if selected else None
                ),
                "selected_region_id": (
                    selected["region_id"] if selected else None
                ),
                "selected_region_text": (
                    selected["region_text"] if selected else None
                ),
                "selected_is_gold_page": (
                    selected["page_id"] == gold_page_id
                    if selected
                    else False
                ),
                "selected_is_gold_region": (
                    selected["is_gold_region"] if selected else False
                ),
                "requested_model": MODEL,
                "resolved_model": result["resolved_model"],
                "response_id": result["response_id"],
                "rubric_version": RUBRIC_VERSION,
            }
            rows.append(row)
            print(
                f"[{position:02d}/{len(cases)}] {case['case_id']}: "
                f"{'admit' if result['admit'] else 'none'}"
            )

    rows.sort(key=lambda row: next(
        position
        for position, case in enumerate(cases)
        if case["case_id"] == row["case_id"]
    ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out, rows)
    true_positive = sum(
        row["admit"] and row["expected_admission"] for row in rows
    )
    false_positive = sum(
        row["admit"] and not row["expected_admission"] for row in rows
    )
    false_negative = sum(
        not row["admit"] and row["expected_admission"] for row in rows
    )
    strict_gold = sum(
        row["selected_is_gold_page"] and row["selected_is_gold_region"]
        for row in rows
    )
    print(
        json.dumps(
            {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "strict_gold_pair": strict_gold,
            },
            sort_keys=True,
        )
    )


def _candidate_pairs(
    feature_rows: list[dict[str, Any]],
    page_text: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        grouped[(row["region_id"], row["page_id"])].append(row)
    candidates = []
    for (region_id, page_id), rows in grouped.items():
        candidates.append(
            {
                "region_id": region_id,
                "region_text": rows[0]["region_text"],
                "page_id": page_id,
                "memory_text": page_text[page_id],
                "methods": sorted({row["method"] for row in rows}),
                "best_rank": min(row["global_rank"] for row in rows),
                "is_gold_region": any(
                    row["is_gold_region"] for row in rows
                ),
            }
        )
    candidates.sort(
        key=lambda candidate: (
            candidate["best_rank"],
            -len(candidate["methods"]),
            candidate["region_id"],
            candidate["page_id"],
        )
    )
    for position, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"C{position:02d}"
    return candidates


def _judge_case(
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    cache_dir: Path,
) -> dict[str, Any]:
    rendered_candidates = "\n\n".join(
        (
            f"[{candidate['candidate_id']}]\n"
            f"Visible passage:\n{candidate['region_text']}\n\n"
            f"User-history source:\n{candidate['memory_text']}"
        )
        for candidate in candidates
    )
    input_text = (
        f"User task:\n{case['prompt']}\n\n"
        f"Complete observation:\n{case['observation_text']}\n\n"
        f"Candidate pairs:\n{rendered_candidates}"
    )
    request = {
        "model": MODEL,
        "rubric_version": RUBRIC_VERSION,
        "input": input_text,
    }
    request_hash = hashlib.sha256(
        json.dumps(
            request, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{request_hash}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    for attempt in range(8):
        try:
            response = OpenAI().responses.create(
                model=MODEL,
                instructions=INSTRUCTIONS,
                input=input_text,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "parm_pair_admission",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "admit": {"type": "boolean"},
                                "candidate_id": {
                                    "type": ["string", "null"],
                                    "enum": [
                                        *[
                                            candidate["candidate_id"]
                                            for candidate in candidates
                                        ],
                                        None,
                                    ],
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "rationale": {"type": "string"},
                            },
                            "required": [
                                "admit",
                                "candidate_id",
                                "confidence",
                                "rationale",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                store=False,
            )
            break
        except RateLimitError:
            if attempt == 7:
                raise
            time.sleep(min(20, 2 ** attempt))
    else:
        raise AssertionError("unreachable")
    parsed = json.loads(response.output_text)
    if parsed["admit"] != (parsed["candidate_id"] is not None):
        raise ValueError("judge returned inconsistent admission fields")
    result = {
        **parsed,
        "resolved_model": response.model,
        "response_id": response.id,
    }
    cache_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _page_text_by_id(index: RetrievalIndex) -> dict[str, str]:
    chunks_by_page: dict[str, list[str]] = defaultdict(list)
    for chunk in index.chunks:
        chunks_by_page[chunk.page_id].append(chunk.text)
    return {
        page.page_id: "\n".join(chunks_by_page[page.page_id])
        for page in index.pages
    }


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
