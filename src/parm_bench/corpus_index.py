from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .corpus import NormalizedSourceRecord, validate_normalized_records
from .retrieval import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    TextEmbedder,
    segment_memory_sentences,
)


CORPUS_INDEX_SCHEMA_VERSION = 3
CORPUS_INDEX_BUILDER_VERSION = "normalized_corpus_index_v1"


def write_corpus_retrieval_index(
    output: str | Path,
    *,
    records: Sequence[NormalizedSourceRecord],
    embedder: TextEmbedder,
    source_manifest_hash: str,
    dataset_revision: str,
) -> dict[str, Any]:
    validate_normalized_records(records)
    if embedder.model_name != EMBEDDING_MODEL:
        raise ValueError("corpus index embedder model does not match PARM")
    if embedder.dimensions != EMBEDDING_DIMENSIONS:
        raise ValueError("corpus index embedder dimensions do not match PARM")
    if len(source_manifest_hash) != 64:
        raise ValueError("source_manifest_hash must be a sha256 digest")
    if not dataset_revision.strip():
        raise ValueError("dataset_revision must be non-empty")

    ordered = sorted(records, key=lambda record: (record.corpus_id, record.source_id))
    page_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    chunk_texts: list[str] = []
    sentence_texts: list[str] = []
    for record in ordered:
        page_id = f"{record.corpus_id}:{record.source_id}"
        page_rows.append(
            {
                "corpus_id": record.corpus_id,
                "page_id": page_id,
                "source_id": "personamem-v2",
                "slug": record.source_id,
                "title": record.title,
                "perturbations": list(record.perturbations),
            }
        )
        chunk_id = f"{page_id}:0"
        chunk_rows.append(
            {
                "corpus_id": record.corpus_id,
                "chunk_id": chunk_id,
                "page_id": page_id,
                "chunk_index": 0,
                "text": record.text,
            }
        )
        chunk_texts.append(record.text)
        sentences = segment_memory_sentences(record.text)
        if not sentences:
            sentences = (record.text,)
        for sentence_index, sentence in enumerate(sentences):
            sentence_rows.append(
                {
                    "corpus_id": record.corpus_id,
                    "sentence_id": f"{chunk_id}:sentence:{sentence_index}",
                    "chunk_id": chunk_id,
                    "page_id": page_id,
                    "sentence_index": sentence_index,
                    "text": sentence,
                }
            )
            sentence_texts.append(sentence)

    embeddings = embedder.embed(chunk_texts)
    sentence_embeddings = embedder.embed(sentence_texts)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to replace existing retrieval index: {destination}"
        )
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        root = Path(temporary)
        _write_jsonl(root / "pages.jsonl", page_rows)
        _write_jsonl(root / "chunks.jsonl", chunk_rows)
        np.save(
            root / "embeddings.npy",
            np.asarray(embeddings, dtype=np.float32),
            allow_pickle=False,
        )
        _write_jsonl(root / "links.jsonl", [])
        _write_jsonl(root / "sentences.jsonl", sentence_rows)
        np.save(
            root / "sentence_embeddings.npy",
            np.asarray(sentence_embeddings, dtype=np.float32),
            allow_pickle=False,
        )
        artifact_names = (
            "pages.jsonl",
            "chunks.jsonl",
            "embeddings.npy",
            "links.jsonl",
            "sentences.jsonl",
            "sentence_embeddings.npy",
        )
        content_hash = hashlib.sha256()
        for name in artifact_names:
            content_hash.update((root / name).read_bytes())
        manifest = {
            "schema_version": CORPUS_INDEX_SCHEMA_VERSION,
            "builder_version": CORPUS_INDEX_BUILDER_VERSION,
            "corpus_ids": sorted({record.corpus_id for record in ordered}),
            "source_manifest_hash": source_manifest_hash,
            "dataset_revision": dataset_revision,
            "content_hash": content_hash.hexdigest(),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "counts": {
                "pages": len(page_rows),
                "chunks": len(chunk_rows),
                "links": 0,
                "vectors": len(chunk_rows),
                "sentences": len(sentence_rows),
                "sentence_vectors": len(sentence_rows),
            },
            "artifact_hashes": {
                name: _sha256(root / name) for name in artifact_names
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(temporary).replace(destination)
    return manifest


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
