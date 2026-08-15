"""Freeze the PARMBench Workflows v1 memory corpus into a retrieval index.

Every workflow retrieval condition reads this one index, so the comparison is
equal-information: policies differ in when they search and what they admit, not
in what memory exists. Perturbation labels are read back out of ``cases.jsonl``
so the index and the benchmark cannot disagree about which record is poison or
superseded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from parm_bench.corpus import NormalizedSourceRecord, SensitivityMetadata
from parm_bench.corpus_index import write_corpus_retrieval_index
from parm_bench.retrieval import OpenAIEmbedder
from parm_bench.workflows.corpus_tiers import load_tiers


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_v1"
CORPUS_ID = "workflow-eng-lead-v1"
CORPUS_ROOT = DATASET / "corpora" / CORPUS_ID / "source"
DATASET_REVISION = "parmbench-workflows-v1-pilot"

SENSITIVE_SOURCES = {"notes/telemetry-privacy-hold"}

# tier-28 keeps the original index path and manifest filename so the frozen
# artifact behind the first-pass result stays byte-identical and its recorded
# source_manifest_hash keeps resolving. Later tiers get their own directory.
TIER_OUTPUTS = {
    "tier-28": (CORPUS_ID, "source_manifest.json"),
    "tier-100": (f"{CORPUS_ID}-100", "source_manifest.tier-100.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="tier-100", choices=sorted(TIER_OUTPUTS))
    arguments = parser.parse_args()
    tier = arguments.tier
    index_name, manifest_filename = TIER_OUTPUTS[tier]
    output = ROOT / "data" / "retrieval-indexes" / index_name

    load_dotenv(ROOT / ".env", override=False)
    perturbations = _perturbations()
    members = load_tiers(DATASET / "corpora" / CORPUS_ID)[tier]
    paths = [CORPUS_ROOT / f"{source_id}.md" for source_id in sorted(members)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"tier {tier} names missing records: {missing}")
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
    manifest_path = DATASET / "corpora" / CORPUS_ID / manifest_filename
    manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = write_corpus_retrieval_index(
        output,
        records=records,
        embedder=OpenAIEmbedder(),
        source_manifest_hash=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        dataset_revision=DATASET_REVISION,
    )
    print(
        f"wrote schema-v{manifest['schema_version']} {tier} index with "
        f"{manifest['counts']['pages']} pages and "
        f"{manifest['counts']['sentences']} sentences to {output}"
    )


def _perturbations() -> dict[str, tuple[str, ...]]:
    """Resolve perturbation labels from the corpus, cross-checked against cases.

    The corpus file is the source of truth, because a record added above the
    first scale tier has no case to be declared in. Anything a case *does*
    declare must still agree, so the index and the benchmark cannot drift apart
    about which record is poison or superseded.
    """

    corpus_path = DATASET / "corpora" / CORPUS_ID / "perturbations.json"
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise SystemExit(f"{corpus_path}: unsupported perturbation schema")
    labels: dict[str, tuple[str, ...]] = {
        str(source_id): tuple(values)
        for source_id, values in payload["perturbations"].items()
    }

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
            declared = labels.get(source_id, ())
            if recorded != declared:
                raise SystemExit(
                    f"case and corpus disagree about perturbations for "
                    f"{source_id}: case says {recorded or '()'}, corpus says "
                    f"{declared or '()'}"
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
    import re

    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else "2026-01-01"


if __name__ == "__main__":
    main()
