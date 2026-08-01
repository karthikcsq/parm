"""Freeze the PARMBench Workflows v1 memory corpus into a retrieval index.

Every workflow retrieval condition reads this one index, so the comparison is
equal-information: policies differ in when they search and what they admit, not
in what memory exists. Perturbation labels are read back out of ``cases.jsonl``
so the index and the benchmark cannot disagree about which record is poison or
superseded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from parm_bench.corpus import NormalizedSourceRecord, SensitivityMetadata
from parm_bench.corpus_index import write_corpus_retrieval_index
from parm_bench.retrieval import OpenAIEmbedder


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_v1"
CORPUS_ID = "workflow-eng-lead-v1"
CORPUS_ROOT = DATASET / "corpora" / CORPUS_ID / "source"
OUTPUT = ROOT / "data" / "retrieval-indexes" / CORPUS_ID
DATASET_REVISION = "parmbench-workflows-v1-pilot"

SENSITIVE_SOURCES = {"notes/telemetry-privacy-hold"}


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    perturbations = _perturbations()
    paths = sorted(CORPUS_ROOT.rglob("*.md"))
    if not paths:
        raise SystemExit(f"no corpus sources under {CORPUS_ROOT}")
    records = []
    manifest_rows = []
    for path in paths:
        relative = path.relative_to(CORPUS_ROOT).as_posix()
        source_id = relative[: -len(".md")]
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_rows.append({"path": relative, "sha256": digest})
        flagged = source_id in SENSITIVE_SOURCES
        records.append(
            NormalizedSourceRecord(
                corpus_id=CORPUS_ID,
                source_id=source_id,
                timestamp=_timestamp(text),
                title=_title(text, source_id),
                text=text,
                provenance={
                    "dataset": "parmbench-workflows-v1",
                    "corpus_id": CORPUS_ID,
                    "path": relative,
                },
                who="workflow-eng-lead",
                perturbations=tuple(perturbations.get(source_id, ())),
                sensitivity=SensitivityMetadata(
                    flagged,
                    "contains a named enterprise customer" if flagged else "ordinary",
                    ("customer-name",) if flagged else (),
                ),
            )
        )

    source_manifest = {
        "corpus_id": CORPUS_ID,
        "dataset_revision": DATASET_REVISION,
        "sources": manifest_rows,
    }
    manifest_path = DATASET / "corpora" / CORPUS_ID / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = write_corpus_retrieval_index(
        OUTPUT,
        records=records,
        embedder=OpenAIEmbedder(),
        source_manifest_hash=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        dataset_revision=DATASET_REVISION,
    )
    print(
        f"wrote schema-v{manifest['schema_version']} index with "
        f"{manifest['counts']['pages']} pages and "
        f"{manifest['counts']['sentences']} sentences to {OUTPUT}"
    )


def _perturbations() -> dict[str, tuple[str, ...]]:
    labels: dict[str, tuple[str, ...]] = {}
    cases_path = DATASET / "cases.jsonl"
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        entries = list(case["memory"]["sources"]) + list(
            case["distractors"]["sources"]
        )
        for entry in entries:
            recorded = tuple(entry.get("perturbations", []))
            source_id = str(entry["source_id"])
            if source_id in labels and labels[source_id] != recorded:
                raise SystemExit(
                    f"cases disagree about perturbations for {source_id}"
                )
            labels[source_id] = recorded
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
    import re

    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else "2026-01-01"


if __name__ == "__main__":
    main()
