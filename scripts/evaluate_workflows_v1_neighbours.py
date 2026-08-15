"""Report what sits closest to each gold memory in a corpus tier.

Growing a corpus only makes retrieval harder if the additions land near the
gold record. A corpus that doubles in size with unrelated text makes the gold
record *easier* to find, because it is the only thing on topic. This prints the
nearest neighbours of every gold source so that assumption is measured rather
than hoped for.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from parm_bench.retrieval import RetrievalIndex
from parm_bench.workflows import load_workflow_cases


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_v1"
TIER_INDEXES = {
    "tier-28": ROOT / "data" / "retrieval-indexes" / "workflow-eng-lead-v1",
    "tier-100": ROOT / "data" / "retrieval-indexes" / "workflow-eng-lead-v1-100",
}
DEPTH = 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="tier-100", choices=sorted(TIER_INDEXES))
    tier = parser.parse_args().tier

    load_dotenv(ROOT / ".env", override=False)
    index = RetrievalIndex.load(TIER_INDEXES[tier])
    gold = sorted(
        {
            source_id
            for case in load_workflow_cases(DATASET)
            for source_id in case.gold_source_ids
        }
    )

    slugs = [page.slug for page in index.pages]
    position = {slug: number for number, slug in enumerate(slugs)}
    vectors = np.asarray(index.embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms == 0, 1, norms)

    print(f"tier: {tier}, {len(slugs)} records, depth {DEPTH}\n")
    for source_id in gold:
        if source_id not in position:
            print(f"{source_id}: not in this tier")
            continue
        similarity = normalized @ normalized[position[source_id]]
        order = np.argsort(-similarity)
        print(f"{source_id}")
        for rank in order[1 : DEPTH + 1]:
            print(f"    {similarity[rank]:.3f}  {slugs[rank]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
