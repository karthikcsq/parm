from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from parm_bench.retrieval import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OpenAIEmbedder,
    RetrievalIndex,
    segment_memory_sentences,
)


CORE_ARTIFACTS = (
    "pages.jsonl",
    "chunks.jsonl",
    "embeddings.npy",
    "links.jsonl",
)
SENTENCE_ARTIFACTS = ("sentences.jsonl", "sentence_embeddings.npy")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Add deterministic sentence vectors to a frozen retrieval index."
    )
    parser.add_argument("index")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    root = Path(args.index)
    index = RetrievalIndex.load(root)
    if index.manifest["schema_version"] != 1:
        parser.error("index must use schema version 1 before enrichment")

    rows: list[dict[str, object]] = []
    texts: list[str] = []
    for chunk in index.chunks:
        for position, text in enumerate(segment_memory_sentences(chunk.text)):
            rows.append(
                {
                    "sentence_id": f"{chunk.chunk_id}:sentence:{position}",
                    "chunk_id": chunk.chunk_id,
                    "page_id": chunk.page_id,
                    "sentence_index": position,
                    "text": text,
                }
            )
            texts.append(text)

    embedder = OpenAIEmbedder()
    batches = [
        embedder.embed(texts[start : start + args.batch_size])
        for start in range(0, len(texts), args.batch_size)
    ]
    embeddings = np.vstack(batches).astype(np.float32, copy=False)
    if embeddings.shape != (len(rows), EMBEDDING_DIMENSIONS):
        raise ValueError(f"unexpected sentence embedding shape: {embeddings.shape}")

    sentences_path = root / "sentences.jsonl"
    sentences_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    np.save(root / "sentence_embeddings.npy", embeddings, allow_pickle=False)

    artifact_names = CORE_ARTIFACTS + SENTENCE_ARTIFACTS
    content_hash = hashlib.sha256()
    for name in artifact_names:
        content_hash.update((root / name).read_bytes())
    manifest = dict(index.manifest)
    manifest.update(
        {
            "schema_version": 2,
            "content_hash": content_hash.hexdigest(),
            "sentence_embedding_model": EMBEDDING_MODEL,
            "sentence_segmenter_version": "paragraph-punctuation-v1",
            "counts": {
                **index.manifest["counts"],
                "sentences": len(rows),
                "sentence_vectors": len(rows),
            },
            "artifact_hashes": {
                name: _sha256(root / name) for name in artifact_names
            },
        }
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "index": str(root.resolve()),
                "sentences": len(rows),
                "schema_version": 2,
                "content_hash": manifest["content_hash"],
            },
            indent=2,
        )
    )
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
