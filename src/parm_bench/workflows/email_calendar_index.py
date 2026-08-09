"""Canonical source records for frozen Email+Calendar retrieval indexes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from parm_bench.corpus import NormalizedSourceRecord, SensitivityMetadata


def source_records(
    *,
    dataset: Path,
    corpus_id: str,
    dataset_revision: str,
    who: str = "operations-lead",
    source_ids: tuple[str, ...] | None = None,
    provenance_dataset: str | None = None,
) -> tuple[list[NormalizedSourceRecord], list[dict[str, str]]]:
    """Return sorted, byte-hashed Email+Calendar corpus records.

    When ``source_ids`` is omitted, every markdown source in the corpus is part
    of the frozen pilot. A caller may supply a stable tier membership instead.
    """

    corpus_root = dataset / "corpora" / corpus_id / "source"
    provenance_dataset = provenance_dataset or dataset_revision
    perturbations = perturbations_from_cases(dataset)
    selected = (
        sorted(source_ids)
        if source_ids is not None
        else sorted(path.relative_to(corpus_root).with_suffix("").as_posix() for path in corpus_root.rglob("*.md"))
    )
    paths = [corpus_root / f"{source_id}.md" for source_id in selected]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"corpus names missing records: {missing}")

    records: list[NormalizedSourceRecord] = []
    rows: list[dict[str, str]] = []
    for path in paths:
        relative = path.relative_to(corpus_root).as_posix()
        source_id = relative[: -len(".md")]
        text = path.read_text(encoding="utf-8")
        rows.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        records.append(
            NormalizedSourceRecord(
                corpus_id=corpus_id,
                source_id=source_id,
                timestamp=timestamp(text),
                title=title(text, source_id),
                text=text,
                provenance={
                    "dataset": provenance_dataset,
                    "corpus_id": corpus_id,
                    "path": relative,
                },
                who=who,
                perturbations=perturbations.get(source_id, ()),
                sensitivity=SensitivityMetadata(False, "ordinary", ()),
            )
        )
    return records, rows


def source_manifest(
    *, corpus_id: str, dataset_revision: str, rows: list[dict[str, str]]
) -> dict[str, object]:
    return {"corpus_id": corpus_id, "dataset_revision": dataset_revision, "sources": rows}


def perturbations_from_cases(dataset: Path) -> dict[str, tuple[str, ...]]:
    """Derive case-declared perturbations and reject inconsistent labels."""

    labels: dict[str, tuple[str, ...]] = {}
    for line in (dataset / "cases.jsonl").read_text(encoding="utf-8").splitlines():
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


def title(text: str, source_id: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped.startswith("Subject: "):
            return stripped[len("Subject: ") :].strip()
    return source_id.rsplit("/", 1)[-1].replace("-", " ")


def timestamp(text: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else "2026-01-01"
