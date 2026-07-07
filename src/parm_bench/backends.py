from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .amara import load_manifest


@dataclass(frozen=True)
class MemoryHit:
    source_id: str
    text: str
    score: float
    perturbations: tuple[str, ...] = ()


class MemoryBackend(Protocol):
    def search(self, query: str, limit: int = 5) -> list[MemoryHit]: ...


class GBrainBackend:
    """Adapter for the pinned repo-local GBrain CLI."""

    corpus_id = "amara-life-v1"
    embedding_model = "openai:text-embedding-3-small"

    def __init__(
        self,
        command: list[str] | None = None,
        cwd: str | Path | None = None,
        provenance_source: str | Path = "data/amara-life-v1/source",
        gbrain_home: str | Path | None = None,
        repo_root: str | Path | None = None,
    ):
        self.repo_root = Path(
            repo_root or Path(__file__).resolve().parents[2]
        ).resolve()
        self.command = command or [
            "bun",
            "run",
            "src/cli.ts",
            "search",
        ]
        self.cwd = _resolve_from_root(
            cwd
            or os.environ.get("PARM_GBRAIN_CWD")
            or ".gbrain-local/runtime/gbrain",
            self.repo_root,
        )
        self.gbrain_home = _resolve_from_root(
            gbrain_home
            or os.environ.get("GBRAIN_HOME")
            or ".gbrain-local/home",
            self.repo_root,
        )
        source = _resolve_from_root(provenance_source, self.repo_root)
        self.provenance = load_manifest(source) if source.exists() else {}

    def search(self, query: str, limit: int = 5) -> list[MemoryHit]:
        if len(query) > 8_000:
            merged: dict[str, MemoryHit] = {}
            for start in range(0, len(query), 6_000):
                for hit in self._search_once(query[start : start + 6_000], limit):
                    previous = merged.get(hit.source_id)
                    if previous is None or hit.score > previous.score:
                        merged[hit.source_id] = hit
            return sorted(
                merged.values(), key=lambda hit: (-hit.score, hit.source_id)
            )[:limit]
        return self._search_once(query, limit)

    def _search_once(self, query: str, limit: int) -> list[MemoryHit]:
        result = subprocess.run(
            [*self.command, query, "--limit", str(limit)],
            cwd=self.cwd,
            env={
                **os.environ,
                "GBRAIN_HOME": str(self.gbrain_home),
            },
            text=True,
            capture_output=True,
            check=True,
        )
        hits: list[MemoryHit] = []
        current: MemoryHit | None = None
        for line in result.stdout.splitlines():
            match = re.match(r"^\[([0-9.]+)\]\s+(\S+)\s+--\s*(.*)$", line)
            if match:
                source_id = _canonical_source_id(match.group(2))
                current = MemoryHit(
                    source_id,
                    match.group(3),
                    float(match.group(1)),
                    tuple(
                        self.provenance.get(source_id, {}).get(
                            "perturbations", []
                        )
                    ),
                )
                hits.append(current)
            elif current is not None and line.strip():
                hits[-1] = MemoryHit(
                    current.source_id,
                    f"{hits[-1].text}\n{line.strip()}",
                    current.score,
                    current.perturbations,
                )
        return hits[:limit]


def _canonical_source_id(value: str) -> str:
    if value.startswith("notes/"):
        return "note/" + value.removeprefix("notes/")
    if value.startswith("meetings/"):
        return "meeting/" + value.removeprefix("meetings/")
    return value


def _resolve_from_root(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()
