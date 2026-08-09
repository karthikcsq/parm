from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_workflows_email_calendar_v1_index.py"
DATASET = ROOT / "data" / "workflows_email_calendar_v1"
V2_SCRIPT = ROOT / "scripts" / "build_workflows_email_calendar_v2_index.py"
V2_DATASET = ROOT / "data" / "workflows_email_calendar_v2"


def _load_builder(script: Path = SCRIPT):
    spec = importlib.util.spec_from_file_location("email_calendar_index_builder", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmailCalendarIndexBuilderTests(unittest.TestCase):
    def test_source_records_match_the_frozen_tier_and_case_perturbations(self) -> None:
        builder = _load_builder()

        records, manifest_rows = builder.source_records()

        tier = json.loads((DATASET / "corpus_tiers.json").read_text(encoding="utf-8"))[
            "tiers"
        ]["tier-24"]["source_ids"]
        self.assertEqual([record.source_id for record in records], sorted(tier))
        self.assertEqual([row["path"] for row in manifest_rows], [f"{source_id}.md" for source_id in sorted(tier)])
        self.assertEqual({record.corpus_id for record in records}, {builder.CORPUS_ID})
        self.assertEqual(
            {
                record.source_id
                for record in records
                if record.perturbations == ("stale-superseded",)
            },
            {
                "notes/atlas-legal-review-superseded",
                "notes/helix-nda-waiver-superseded",
                "notes/northstar-engineering-draft-superseded",
                "notes/orion-travel-autoapprove-superseded",
                "notes/trellis-blackout-draft-superseded",
                "notes/whiteboard-only-review-superseded",
            },
        )

    def test_source_manifest_is_canonical_and_hashed_before_index_write(self) -> None:
        builder = _load_builder()
        records, rows = builder.source_records()

        manifest = builder.source_manifest(rows)

        self.assertEqual(manifest["corpus_id"], builder.CORPUS_ID)
        self.assertEqual(manifest["dataset_revision"], builder.DATASET_REVISION)
        self.assertEqual(manifest["sources"], rows)
        self.assertEqual(len(records), 24)


class EmailCalendarV2IndexBuilderTests(unittest.TestCase):
    def test_source_manifest_covers_the_natural_pilot_and_frozen_index_loads(self) -> None:
        from parm_bench.retrieval import RetrievalIndex

        builder = _load_builder(V2_SCRIPT)
        records, rows = builder.source_records()
        expected_paths = sorted(
            path.relative_to(V2_DATASET / "corpora" / builder.CORPUS_ID / "source").as_posix()
            for path in (V2_DATASET / "corpora" / builder.CORPUS_ID / "source").rglob("*.md")
        )

        self.assertEqual(builder.CORPUS_ID, "ops-lead-email-calendar-v2")
        self.assertEqual([row["path"] for row in rows], expected_paths)
        self.assertEqual([record.source_id for record in records], [path[:-3] for path in expected_paths])
        self.assertEqual(builder.source_manifest(rows)["sources"], rows)
        index = RetrievalIndex.load(builder.OUTPUT)
        self.assertEqual(index.manifest["corpus_ids"], [builder.CORPUS_ID])
        self.assertEqual(len(index.pages), len(records))


if __name__ == "__main__":
    unittest.main()
