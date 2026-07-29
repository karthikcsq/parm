from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from parm_bench.corpus import NormalizedSourceRecord
from parm_bench.corpus_index import write_corpus_retrieval_index
from parm_bench.retrieval import (
    EMBEDDING_API_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "personamem-v2-train-v1"
RECORDS_PATH = SOURCE_ROOT / "records.jsonl"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "source_manifest.json"
OUTPUT = ROOT / "data" / "retrieval-indexes" / "personamem-v2-train-v1"
DATASET_REVISION = "b7b42b78917157afed063527a1c959e98f6109f2"

# Transient-error retry policy for the live embedding calls. This wraps the
# package's OpenAIEmbedder behavior (same model/batch constants, same request
# shape) with batch-level retry + exponential backoff, kept local to this
# script rather than modifying parm_bench.retrieval.
_RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    json.JSONDecodeError,
)
_MAX_ATTEMPTS = 6
_INITIAL_BACKOFF_SECONDS = 2.0
_BACKOFF_MULTIPLIER = 2.0


class RetryingOpenAIEmbedder:
    model_name = EMBEDDING_MODEL
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, client: Any | None = None):
        self._client = client if client is not None else OpenAI()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        batches: list[np.ndarray] = []
        total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
        for batch_index, start in enumerate(
            range(0, len(texts), EMBEDDING_BATCH_SIZE), start=1
        ):
            batch = list(texts[start : start + EMBEDDING_BATCH_SIZE])
            batches.append(self._embed_batch_with_retry(batch))
            if batch_index % 25 == 0 or batch_index == total_batches:
                print(
                    f"  embedded batch {batch_index}/{total_batches} "
                    f"({start + len(batch)}/{len(texts)} texts)",
                    flush=True,
                )
        result = np.concatenate(batches)
        if result.shape != (len(texts), self.dimensions):
            raise ValueError(
                f"embedder returned {result.shape}, expected "
                f"({len(texts)}, {self.dimensions})"
            )
        return result

    def _embed_batch_with_retry(self, batch: list[str]) -> np.ndarray:
        backoff = _INITIAL_BACKOFF_SECONDS
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.embeddings.create(
                    model=EMBEDDING_API_MODEL,
                    input=batch,
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                batch_result = np.asarray(
                    [item.embedding for item in ordered], dtype=np.float32
                )
                if batch_result.shape != (len(batch), self.dimensions):
                    raise ValueError(
                        f"embedder returned {batch_result.shape}, expected "
                        f"({len(batch)}, {self.dimensions})"
                    )
                return batch_result
            except _RETRYABLE_EXCEPTIONS as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise
                print(
                    f"  transient embedding error on attempt {attempt}/"
                    f"{_MAX_ATTEMPTS}: {exc!r}; retrying in {backoff:.1f}s",
                    flush=True,
                )
                time.sleep(backoff)
                backoff *= _BACKOFF_MULTIPLIER
        raise RuntimeError("unreachable")  # pragma: no cover


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    records = [
        NormalizedSourceRecord.from_dict(json.loads(line))
        for line in RECORDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"loaded {len(records)} normalized records", flush=True)
    manifest = write_corpus_retrieval_index(
        OUTPUT,
        records=records,
        embedder=RetryingOpenAIEmbedder(),
        source_manifest_hash=_sha256(SOURCE_MANIFEST_PATH),
        dataset_revision=DATASET_REVISION,
    )
    print(
        f"wrote schema-v{manifest['schema_version']} index with "
        f"{manifest['counts']['pages']} pages and "
        f"{manifest['counts']['sentences']} sentences"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
