from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .amara import load_manifest
from .retrieval import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL


def export_gbrain_index(
    output: str | Path,
    *,
    runtime: str | Path,
    gbrain_home: str | Path,
    corpus_id: str,
    chunker_version: str,
    provenance_source: str | Path | None = None,
) -> dict[str, Any]:
    runtime_path = Path(runtime).resolve()
    script = Path(__file__).resolve().parents[2] / "scripts" / "dump_gbrain_retrieval.ts"
    completed = subprocess.run(
        ["bun", str(script), "--runtime", str(runtime_path)],
        cwd=runtime_path,
        env={**os.environ, "GBRAIN_HOME": str(Path(gbrain_home).resolve())},
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    snapshot = json.loads(completed.stdout)
    version = json.loads(
        (runtime_path / "package.json").read_text(encoding="utf-8")
    ).get("version")
    return write_retrieval_index(
        output,
        pages=snapshot["pages"],
        chunks=snapshot["chunks"],
        links=snapshot["links"],
        corpus_id=corpus_id,
        gbrain_version=str(version),
        chunker_version=chunker_version,
        configured_model=snapshot["config"].get("embedding_model"),
        configured_dimensions=snapshot["config"].get("embedding_dimensions"),
        provenance_source=provenance_source,
    )


def write_retrieval_index(
    output: str | Path,
    *,
    pages: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    links: Sequence[dict[str, Any]],
    corpus_id: str,
    gbrain_version: str,
    chunker_version: str,
    configured_model: str | None = EMBEDDING_MODEL,
    configured_dimensions: int | None = EMBEDDING_DIMENSIONS,
    provenance_source: str | Path | None = None,
) -> dict[str, Any]:
    model = _canonical_model(configured_model)
    if model != EMBEDDING_MODEL:
        raise ValueError(f"GBrain uses unsupported embedding model: {configured_model}")
    if configured_dimensions != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"GBrain uses unsupported dimensions: {configured_dimensions}"
        )
    perturbations = (
        load_manifest(provenance_source) if provenance_source is not None else {}
    )
    ordered_pages = sorted(
        pages, key=lambda page: (str(page["source_id"]), str(page["slug"]))
    )
    stable_page_ids = {
        int(page["id"]): f"{page['source_id']}:{page['slug']}"
        for page in ordered_pages
    }
    page_rows = [
        {
            "page_id": stable_page_ids[int(page["id"])],
            "source_id": str(page["source_id"]),
            "slug": str(page["slug"]),
            "title": str(page["title"]),
            "perturbations": list(
                perturbations.get(str(page["slug"]), {}).get("perturbations", [])
            ),
        }
        for page in ordered_pages
    ]
    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (
            stable_page_ids[int(chunk["page_id"])],
            int(chunk["chunk_index"]),
            int(chunk["id"]),
        ),
    )
    chunk_rows: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    for chunk in ordered_chunks:
        page_id = stable_page_ids.get(int(chunk["page_id"]))
        if page_id is None:
            raise ValueError(f"chunk references missing page: {chunk['page_id']}")
        if _canonical_model(chunk.get("model")) != EMBEDDING_MODEL:
            raise ValueError(f"chunk {chunk['id']} uses an unexpected model")
        vector = [float(value) for value in chunk["embedding"]]
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"chunk {chunk['id']} has an unexpected vector width")
        chunk_rows.append(
            {
                "chunk_id": f"{page_id}:{int(chunk['chunk_index'])}",
                "page_id": page_id,
                "chunk_index": int(chunk["chunk_index"]),
                "text": str(chunk["chunk_text"]),
            }
        )
        vectors.append(vector)
    link_rows = []
    for link in links:
        source = stable_page_ids.get(int(link["from_page_id"]))
        target = stable_page_ids.get(int(link["to_page_id"]))
        if source is None or target is None:
            raise ValueError("link references a missing page")
        link_rows.append(
            {
                "source_page_id": source,
                "target_page_id": target,
                "link_type": str(link.get("link_type") or ""),
                "provenance": link.get("link_source"),
            }
        )
    link_rows.sort(
        key=lambda link: (
            link["source_page_id"],
            link["target_page_id"],
            link["link_type"],
            link["provenance"] or "",
        )
    )

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        root = Path(temporary)
        _write_jsonl(root / "pages.jsonl", page_rows)
        _write_jsonl(root / "chunks.jsonl", chunk_rows)
        np.save(
            root / "embeddings.npy",
            np.asarray(vectors, dtype=np.float32).reshape(
                len(vectors), EMBEDDING_DIMENSIONS
            ),
            allow_pickle=False,
        )
        _write_jsonl(root / "links.jsonl", link_rows)
        artifact_names = (
            "pages.jsonl",
            "chunks.jsonl",
            "embeddings.npy",
            "links.jsonl",
        )
        content_hash = hashlib.sha256()
        for name in artifact_names:
            content_hash.update((root / name).read_bytes())
        manifest = {
            "schema_version": 1,
            "corpus_id": corpus_id,
            "content_hash": content_hash.hexdigest(),
            "gbrain_version": gbrain_version,
            "chunker_version": chunker_version,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "counts": {
                "pages": len(page_rows),
                "chunks": len(chunk_rows),
                "links": len(link_rows),
                "vectors": len(vectors),
            },
            "artifact_hashes": {
                name: _sha256(root / name) for name in artifact_names
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(
                f"refusing to replace existing retrieval index: {destination}"
            )
        Path(temporary).replace(destination)
    return manifest


def _canonical_model(value: Any) -> str:
    model = str(value or "")
    for prefix in ("llama-server:", "sentence-transformers:"):
        if model.startswith(prefix):
            model = model.removeprefix(prefix)
    return model


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
