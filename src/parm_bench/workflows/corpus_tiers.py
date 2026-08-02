"""Corpus scale tiers for the workflow memory corpora.

A scale tier names an exact set of source records so the same scenario can be
run against a small and a large history and the two runs stay comparable. The
tiers are declared, not inferred from whatever happens to be on disk: the first
tier is a frozen list, and growing the corpus must not silently redefine the
point the earlier result was measured at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TIERS_FILENAME = "corpus_tiers.json"
TIERS_SCHEMA_VERSION = 1


class CorpusTierError(ValueError):
    pass


def load_tiers(corpus_root: str | Path) -> dict[str, tuple[str, ...]]:
    """Read the declared tiers for a corpus directory.

    ``corpus_root`` is the directory holding ``source/``, not ``source/``
    itself, because the tier declaration describes the corpus rather than one
    of its file trees.
    """

    path = Path(corpus_root) / TIERS_FILENAME
    if not path.is_file():
        raise CorpusTierError(f"corpus tier declaration not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusTierError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorpusTierError(f"{path}: expected an object")
    if payload.get("schema_version") != TIERS_SCHEMA_VERSION:
        raise CorpusTierError(f"{path}: unsupported tier schema")
    tiers = payload.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        raise CorpusTierError(f"{path}: tiers must be a non-empty object")
    resolved: dict[str, tuple[str, ...]] = {}
    for name, entry in tiers.items():
        if not isinstance(entry, dict):
            raise CorpusTierError(f"{path}: tier {name!r} must be an object")
        members = entry.get("source_ids")
        if not isinstance(members, list) or not members:
            raise CorpusTierError(
                f"{path}: tier {name!r} needs a non-empty source_ids list"
            )
        if any(not isinstance(value, str) or not value.strip() for value in members):
            raise CorpusTierError(f"{path}: tier {name!r} has a blank source_id")
        if len(set(members)) != len(members):
            raise CorpusTierError(f"{path}: tier {name!r} repeats a source_id")
        expected = entry.get("size")
        if isinstance(expected, int) and expected != len(members):
            raise CorpusTierError(
                f"{path}: tier {name!r} declares size {expected} but lists "
                f"{len(members)} records"
            )
        resolved[str(name)] = tuple(members)
    return resolved


def tier_issues(
    corpus_root: str | Path,
    *,
    base_tier: str,
    full_tier: str,
) -> list[str]:
    """Check a corpus against its declared tiers.

    Enforces the four properties the scaling comparison depends on: every
    declared record exists, the small tier is a strict subset of the large one,
    the large tier accounts for every file on disk, and no two records are the
    same document under different names.
    """

    root = Path(corpus_root)
    source_root = root / "source"
    issues: list[str] = []
    try:
        tiers = load_tiers(root)
    except CorpusTierError as exc:
        return [str(exc)]

    for name in (base_tier, full_tier):
        if name not in tiers:
            issues.append(f"corpus does not declare tier {name!r}")
    if issues:
        return issues

    base = set(tiers[base_tier])
    full = set(tiers[full_tier])
    if not base < full:
        issues.append(
            f"tier {base_tier!r} must be a strict subset of tier {full_tier!r}"
        )

    on_disk = {
        path.relative_to(source_root).as_posix()[: -len(".md")]
        for path in source_root.rglob("*.md")
    }
    for source_id in sorted(full - on_disk):
        issues.append(f"tier {full_tier!r} names a missing record {source_id!r}")
    for source_id in sorted(on_disk - full):
        issues.append(
            f"record {source_id!r} is on disk but not in tier {full_tier!r}"
        )

    bodies: dict[str, str] = {}
    for source_id in sorted(full & on_disk):
        text = (source_root / f"{source_id}.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        if normalized in bodies:
            issues.append(
                f"record {source_id!r} duplicates the body of "
                f"{bodies[normalized]!r}; a memory that says nothing new is "
                "padding, not history"
            )
        else:
            bodies[normalized] = source_id
    return issues


def tier_manifest_rows(
    corpus_root: str | Path, tier: str
) -> list[dict[str, Any]]:
    """Return ``{path, sha256}`` rows for one tier, in stable source order."""

    import hashlib

    root = Path(corpus_root)
    source_root = root / "source"
    members = load_tiers(root)[tier]
    rows = []
    for source_id in sorted(members):
        path = source_root / f"{source_id}.md"
        rows.append(
            {
                "path": f"{source_id}.md",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows
