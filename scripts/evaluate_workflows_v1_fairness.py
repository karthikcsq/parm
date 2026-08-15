"""Check that a workflow goal cannot reach its own gold memory.

The whole claim is that the memory becomes retrievable only after a tool
observation reveals the cue. If prompt-only retrieval finds a gold source from
the goal alone, the scenario is not testing late-cued retrieval no matter what
the run produces, and `input_rag` will beat everything for the wrong reason.

This has caught the pilot twice: once when the corpus was small enough that
top-five covered a quarter of it, and once when the goal was reworded into the
memory's own vocabulary.

Exits non-zero when any goal reaches a gold source.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

from parm_bench.retrieval import (
    IndexRetriever,
    OpenAIEmbedder,
    RetrievalIndex,
    RetrievalMode,
    RetrievalRequest,
)
from parm_bench.workflows import load_workflow_cases, validate_workflow_cases


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_v1"
TIER_INDEXES = {
    "tier-28": ROOT / "data" / "retrieval-indexes" / "workflow-eng-lead-v1",
    "tier-100": ROOT / "data" / "retrieval-indexes" / "workflow-eng-lead-v1-100",
}
TOP_K = 5
MODES = (RetrievalMode.DENSE, RetrievalMode.HYBRID)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="tier-100", choices=sorted(TIER_INDEXES))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET,
        help="workflow dataset directory (default: data/workflows_v1)",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="retrieval index directory; overrides the index selected by --tier",
    )
    arguments = parser.parse_args()
    tier = arguments.tier
    dataset = arguments.dataset
    index_path = arguments.index or TIER_INDEXES[tier]

    load_dotenv(ROOT / ".env", override=False)
    cases = load_workflow_cases(dataset)
    validate_workflow_cases(cases)
    index = RetrievalIndex.load(index_path)
    if arguments.dataset != DATASET or arguments.index:
        print(f"dataset: {dataset}")
        print(f"index: {index_path}")
    print(f"tier: {tier}, {len(index.pages)} records\n")
    embedder = OpenAIEmbedder()
    leaked = False
    for mode in MODES:
        retriever = IndexRetriever(index, mode, embedder)
        for case in cases:
            # The ceiling variant injects the memory into the goal on purpose.
            if case.variant == "memory-included":
                continue
            hits = retriever.retrieve(
                RetrievalRequest(
                    case.goal, top_k=TOP_K, corpus_id=case.corpus_id
                )
            ).hits
            slugs = [hit.slug for hit in hits]
            reached = [slug for slug in slugs if slug in set(case.gold_source_ids)]
            status = "LEAK" if reached else "ok"
            if reached:
                leaked = True
            print(f"{status:4} {mode.value:7} {case.variant:14} {slugs}")
    if leaked:
        print(
            "\nA goal reaches its own gold memory. Either the corpus needs more "
            "records that are closer to the goal, or the goal is written in the "
            "memory's vocabulary.",
            file=sys.stderr,
        )
        return 1
    print(f"\nNo goal reaches a gold source in the top {TOP_K}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
