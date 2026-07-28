"""Audit PersonaMem pilot memory claims against their raw source snippets.

Grading now runs through `parm_bench.evidence_gate`, which adds deterministic
pre-checks (unjustified frequency words, assistant-only evidence, topical
questions, missing lexical anchor) ahead of the LLM rubric. The rubric moved
from `personamem_source_support_v1` to `personamem_source_support_v2`, so the
cache key changed and the defaults point at v2 artifacts. The v1 cache and
report stay untouched and remain replayable by passing the old paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from parm_bench.evidence_gate import (
    EVIDENCE_GATE_MODEL,
    EVIDENCE_GATE_RUBRIC,
    CachedOpenAISupportJudge,
    EvidenceGateCachePolicy,
    grade_claim,
)

from personamem_v0_specs import SPECS


MODEL = EVIDENCE_GATE_MODEL
RUBRIC_VERSION = EVIDENCE_GATE_RUBRIC


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grade each PersonaMem pilot memory claim against its raw source "
            f"conversation using rubric {RUBRIC_VERSION}."
        ),
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("data/personamem-v2-train-v0/pilot_sources.jsonl"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "source-support-cache-v2"
        ),
    )
    parser.add_argument(
        "--policy",
        choices=[policy.value for policy in EvidenceGateCachePolicy],
        default=EvidenceGateCachePolicy.POPULATE.value,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "source-support-audit-v2.jsonl"
        ),
    )
    args = parser.parse_args()

    load_dotenv()
    sources = {
        row["source_row_id"]: row
        for row in _read_jsonl(args.sources)
    }
    judge = CachedOpenAISupportJudge(args.cache, args.policy)
    rows = []
    for position, spec in enumerate(SPECS, start=1):
        source_row_id = f"train_text:{spec['row_offset']}"
        source = sources[source_row_id]
        result = grade_claim(
            spec["memory_summary"],
            source["related_conversation_snippet"],
            judge=judge,
        )
        row = {
            "base_case_id": f"parm-personamem-{spec['slug']}",
            "source_row_id": source_row_id,
            "corpus_id": source["corpus_id"],
            "gold_source_id": source["gold_source_id"],
            "memory_claim": spec["memory_summary"],
            **result.to_dict(),
        }
        rows.append(row)
        detail = (
            f" ({', '.join(result.deterministic_rejections)})"
            if result.deterministic_rejections
            else ""
        )
        print(
            f"[{position:02d}/{len(SPECS)}] {spec['slug']}: "
            f"{result.grade}{detail}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    counts = {
        grade: sum(row["grade"] == grade for row in rows)
        for grade in ("explicit", "inferable", "unsupported")
    }
    print(json.dumps(counts, sort_keys=True))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
