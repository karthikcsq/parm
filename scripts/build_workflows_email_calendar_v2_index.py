"""Freeze the natural Email+Calendar v2 pilot into a retrieval index.

The source inventory is canonicalized from the pilot corpus, byte-hashed in a
source manifest, and then embedded once into a non-overwriteable frozen index.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from parm_bench.corpus_index import write_corpus_retrieval_index
from parm_bench.retrieval import OpenAIEmbedder
from parm_bench.workflows.email_calendar_index import (
    source_manifest as _source_manifest,
)
from parm_bench.workflows.email_calendar_index import source_records as _source_records


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_email_calendar_v2"
CORPUS_ID = "ops-lead-email-calendar-v2"
OUTPUT = ROOT / "data" / "retrieval-indexes" / CORPUS_ID
DATASET_REVISION = "parmbench-workflows-email-calendar-v2"
SOURCE_MANIFEST = DATASET / "corpora" / CORPUS_ID / "source_manifest.json"


def source_records():
    return _source_records(
        dataset=DATASET,
        corpus_id=CORPUS_ID,
        dataset_revision=DATASET_REVISION,
    )


def source_manifest(rows: list[dict[str, str]]) -> dict[str, object]:
    return _source_manifest(
        corpus_id=CORPUS_ID, dataset_revision=DATASET_REVISION, rows=rows
    )


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    records, rows = source_records()
    manifest = source_manifest(rows)
    SOURCE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    index_manifest = write_corpus_retrieval_index(
        OUTPUT,
        records=records,
        embedder=OpenAIEmbedder(),
        source_manifest_hash=hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        dataset_revision=DATASET_REVISION,
    )
    print(
        f"wrote schema-v{index_manifest['schema_version']} natural pilot index with "
        f"{index_manifest['counts']['pages']} pages and "
        f"{index_manifest['counts']['sentences']} sentences to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
