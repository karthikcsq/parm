from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from parm_bench.workflows.corpus_tiers import (
    CorpusTierError,
    load_tiers,
    tier_issues,
    tier_manifest_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "workflows_v1" / "corpora" / "workflow-eng-lead-v1"


def _corpus(tmp: Path, tiers: dict, records: dict[str, str]) -> Path:
    source = tmp / "source"
    for source_id, text in records.items():
        path = source / f"{source_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (tmp / "corpus_tiers.json").write_text(
        json.dumps({"schema_version": 1, "tiers": tiers}), encoding="utf-8"
    )
    return tmp


class TrackedCorpusTest(unittest.TestCase):
    def test_tracked_corpus_declares_both_tiers(self) -> None:
        tiers = load_tiers(CORPUS)
        self.assertEqual(len(tiers["tier-28"]), 28)
        self.assertEqual(len(tiers["tier-100"]), 100)

    def test_tracked_corpus_passes_its_own_checks(self) -> None:
        self.assertEqual(
            tier_issues(CORPUS, base_tier="tier-28", full_tier="tier-100"), []
        )

    def test_base_tier_is_a_strict_subset(self) -> None:
        tiers = load_tiers(CORPUS)
        self.assertLess(set(tiers["tier-28"]), set(tiers["tier-100"]))

    def test_manifest_rows_cover_the_requested_tier_only(self) -> None:
        rows = tier_manifest_rows(CORPUS, "tier-28")
        self.assertEqual(len(rows), 28)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))


class TierValidationTest(unittest.TestCase):
    def test_a_record_on_disk_but_outside_the_full_tier_is_an_issue(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = _corpus(
                Path(name),
                {
                    "small": {"source_ids": ["notes/a"]},
                    "full": {"source_ids": ["notes/a", "notes/b"]},
                },
                {"notes/a": "one", "notes/b": "two", "notes/c": "three"},
            )
            issues = tier_issues(root, base_tier="small", full_tier="full")
        self.assertTrue(any("notes/c" in issue for issue in issues))

    def test_a_declared_record_that_is_missing_is_an_issue(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = _corpus(
                Path(name),
                {
                    "small": {"source_ids": ["notes/a"]},
                    "full": {"source_ids": ["notes/a", "notes/gone"]},
                },
                {"notes/a": "one"},
            )
            issues = tier_issues(root, base_tier="small", full_tier="full")
        self.assertTrue(any("notes/gone" in issue for issue in issues))

    def test_a_duplicated_body_is_an_issue(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = _corpus(
                Path(name),
                {
                    "small": {"source_ids": ["notes/a"]},
                    "full": {"source_ids": ["notes/a", "notes/copy"]},
                },
                {"notes/a": "same words here", "notes/copy": "same  words\nhere"},
            )
            issues = tier_issues(root, base_tier="small", full_tier="full")
        self.assertTrue(any("duplicates the body" in issue for issue in issues))

    def test_a_base_tier_that_is_not_a_subset_is_an_issue(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = _corpus(
                Path(name),
                {
                    "small": {"source_ids": ["notes/a", "notes/outside"]},
                    "full": {"source_ids": ["notes/a"]},
                },
                {"notes/a": "one", "notes/outside": "two"},
            )
            issues = tier_issues(root, base_tier="small", full_tier="full")
        self.assertTrue(any("strict subset" in issue for issue in issues))

    def test_a_declared_size_that_disagrees_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = _corpus(
                Path(name),
                {"small": {"size": 9, "source_ids": ["notes/a"]}},
                {"notes/a": "one"},
            )
            with self.assertRaises(CorpusTierError):
                load_tiers(root)


if __name__ == "__main__":
    unittest.main()
