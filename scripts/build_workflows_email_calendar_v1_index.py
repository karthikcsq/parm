"""Freeze the Email+Calendar workflow corpus into its retrieval index.

The dataset's cases, fixture observations, and all retrieval conditions share one
frozen source corpus.  This builder writes the canonical source manifest before
embedding it, so ``RetrievalIndex.load`` can validate every persisted artifact
and the source revision that produced it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from dotenv import load_dotenv

from parm_bench.corpus import NormalizedSourceRecord, SensitivityMetadata
from parm_bench.corpus_index import write_corpus_retrieval_index
from parm_bench.retrieval import OpenAIEmbedder
from parm_bench.workflows.corpus_tiers import load_tiers


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_email_calendar_v1"
CORPUS_ID = "ops-lead-email-calendar-v1"
CORPUS_ROOT = DATASET / "corpora" / CORPUS_ID / "source"
OUTPUT = ROOT / "data" / "retrieval-indexes" / CORPUS_ID
TIER = "tier-24"
DATASET_REVISION = "parmbench-workflows-email-calendar-v1"
SOURCE_MANIFEST = DATASET / "corpora" / CORPUS_ID / "source_manifest.json"


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
        f"wrote schema-v{index_manifest['schema_version']} {TIER} index with "
        f"{index_manifest['counts']['pages']} pages and "
        f"{index_manifest['counts']['sentences']} sentences to {OUTPUT}"
    )


def source_records() -> tuple[list[NormalizedSourceRecord], list[dict[str, str]]]:
    """Return canonical tier records and their byte-level source manifest rows."""

    perturbations = _perturbations_from_cases()
    members = load_tiers(DATASET / "corpora" / CORPUS_ID)[TIER]
    paths = [CORPUS_ROOT / f"{source_id}.md" for source_id in sorted(members)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"tier {TIER} names missing records: {missing}")

    records: list[NormalizedSourceRecord] = []
    rows: list[dict[str, str]] = []
    for path in paths:
        relative = path.relative_to(CORPUS_ROOT).as_posix()
        source_id = relative[: -len(".md")]
        text = path.read_text(encoding="utf-8")
        rows.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        records.append(
            NormalizedSourceRecord(
                corpus_id=CORPUS_ID,
                source_id=source_id,
                timestamp=_timestamp(text),
                title=_title(text, source_id),
                text=text,
                provenance={
                    "dataset": "parmbench-workflows-email-calendar-v1",
                    "corpus_id": CORPUS_ID,
                    "path": relative,
                },
                who="operations-lead",
                perturbations=perturbations.get(source_id, ()),
                sensitivity=SensitivityMetadata(False, "ordinary", ()),
            )
        )
    return records, rows


def source_manifest(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "corpus_id": CORPUS_ID,
        "dataset_revision": DATASET_REVISION,
        "sources": rows,
    }


def _perturbations_from_cases() -> dict[str, tuple[str, ...]]:
    """Derive labels from cases and reject inconsistent repeated declarations."""

    labels: dict[str, tuple[str, ...]] = {}
    for line in (DATASET / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        for section in ("memory", "distractors"):
            for entry in case[section]["sources"]:
                source_id = str(entry["source_id"])
                declared = tuple(entry.get("perturbations", ()))
                previous = labels.setdefault(source_id, declared)
                if previous != declared:
                    raise SystemExit(
                        f"cases disagree about perturbations for {source_id}: "
                        f"{previous or '()'} versus {declared or '()'}"
                    )
    return labels


def _title(text: str, source_id: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped.startswith("Subject: "):
            return stripped[len("Subject: ") :].strip()
    return source_id.rsplit("/", 1)[-1].replace("-", " ")


def _timestamp(text: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else "2026-01-01"


if __name__ == "__main__":
    main()
