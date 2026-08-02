"""Refresh the workflow corpus tier declaration and check its integrity.

``tier-28`` is frozen: it is the corpus as it stood when the first-pass result
was measured, and this script never rewrites it. ``tier-100`` is regenerated
from disk so growing the corpus is a matter of adding a file, and the checks
below are what stop that from quietly breaking the comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from parm_bench.workflows.corpus_tiers import (
    TIERS_FILENAME,
    TIERS_SCHEMA_VERSION,
    load_tiers,
    tier_issues,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "workflows_v1" / "corpora" / "workflow-eng-lead-v1"
BASE_TIER = "tier-28"
FULL_TIER = "tier-100"


def main() -> int:
    path = CORPUS / TIERS_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = list(payload["tiers"][BASE_TIER]["source_ids"])

    on_disk = sorted(
        source.relative_to(CORPUS / "source").as_posix()[: -len(".md")]
        for source in (CORPUS / "source").rglob("*.md")
    )
    payload["schema_version"] = TIERS_SCHEMA_VERSION
    payload["tiers"][FULL_TIER]["source_ids"] = on_disk
    payload["tiers"][FULL_TIER]["size"] = len(on_disk)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    issues = tier_issues(CORPUS, base_tier=BASE_TIER, full_tier=FULL_TIER)
    if len(on_disk) != 100:
        issues.append(
            f"{FULL_TIER} holds {len(on_disk)} records; the milestone is exactly 100"
        )
    if list(load_tiers(CORPUS)[BASE_TIER]) != frozen:
        issues.append(f"{BASE_TIER} is frozen and must not be rewritten")
    if issues:
        for issue in issues:
            print(f"  {issue}", file=sys.stderr)
        return 1
    print(
        f"{BASE_TIER}: {len(frozen)} records, {FULL_TIER}: {len(on_disk)} records, "
        f"{len(set(on_disk) - set(frozen))} added"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
