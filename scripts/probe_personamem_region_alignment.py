from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from dotenv import load_dotenv

from parm_bench.dataset import load_cases
from parm_bench.retrieval import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OpenAIEmbedder,
    RetrievalIndex,
    SpacyCueConceptExtractor,
    _BM25Corpus,
    _parm_observation_regions,
)


SCHEMA_VERSION = 1


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=norms != 0,
    )


class EmbeddingCache:
    def __init__(self, path: Path, embedder: OpenAIEmbedder):
        self.path = path
        self.embedder = embedder
        self._entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{path}: unsupported cache schema")
            if payload.get("embedding_model") != EMBEDDING_MODEL:
                raise ValueError(f"{path}: embedding model mismatch")
            if payload.get("embedding_dimensions") != EMBEDDING_DIMENSIONS:
                raise ValueError(f"{path}: embedding dimensions mismatch")
            self._entries = dict(payload.get("entries", {}))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
        ]
        missing_by_key = {
            key: text
            for key, text in zip(keys, texts)
            if key not in self._entries
        }
        if missing_by_key:
            missing_keys = list(missing_by_key)
            missing_texts = [missing_by_key[key] for key in missing_keys]
            vectors = self.embedder.embed(missing_texts)
            for key, text, vector in zip(
                missing_keys, missing_texts, vectors
            ):
                self._entries[key] = {
                    "text": text,
                    "embedding": vector.tolist(),
                }
            self._write()
        return np.asarray(
            [self._entries[key]["embedding"] for key in keys],
            dtype=np.float32,
        )

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "entries": {
                key: self._entries[key] for key in sorted(self._entries)
            },
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _page_dense_scores(
    scoped_index: RetrievalIndex,
    query_vectors: np.ndarray,
    region_texts: Sequence[str],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, tuple[str, ...]],
]:
    assert scoped_index.sentence_embeddings is not None
    sentence_matrix = _normalize_rows(scoped_index.sentence_embeddings)
    chunk_matrix = _normalize_rows(scoped_index.embeddings)
    normalized_queries = _normalize_rows(query_vectors)

    sentence_positions: dict[str, list[int]] = defaultdict(list)
    for position, sentence in enumerate(scoped_index.sentences):
        sentence_positions[sentence.page_id].append(position)
    chunk_positions: dict[str, list[int]] = defaultdict(list)
    for position, chunk in enumerate(scoped_index.chunks):
        chunk_positions[chunk.page_id].append(position)

    page_ids = tuple(page.page_id for page in scoped_index.pages)
    page_texts = {
        page_id: " ".join(
            scoped_index.chunks[position].text
            for position in chunk_positions[page_id]
        )
        for page_id in page_ids
    }
    bm25 = _BM25Corpus(page_texts)
    bm25_scores = np.asarray(
        [
            [scores.get(page_id, 0.0) for page_id in page_ids]
            for scores in (bm25.scores(text) for text in region_texts)
        ],
        dtype=np.float32,
    )
    bm25_max = float(np.max(bm25_scores))
    normalized_bm25 = (
        bm25_scores / bm25_max
        if bm25_max > 0
        else np.zeros_like(bm25_scores)
    )
    sentence_cosines = normalized_queries @ sentence_matrix.T
    chunk_cosines = normalized_queries @ chunk_matrix.T
    score_rows = []
    for page_id in page_ids:
        per_sentence = sentence_cosines[:, sentence_positions[page_id]]
        ordered_sentences = np.sort(per_sentence, axis=1)
        top_count = min(3, ordered_sentences.shape[1])
        sentence_max = ordered_sentences[:, -1]
        sentence_top_three = np.mean(
            ordered_sentences[:, -top_count:], axis=1
        )
        chunk_max = np.max(
            chunk_cosines[:, chunk_positions[page_id]], axis=1
        )
        score_rows.append(
            np.stack(
                (
                    sentence_max,
                    sentence_top_three,
                    chunk_max,
                    0.6 * sentence_max
                    + 0.2 * sentence_top_three
                    + 0.2 * chunk_max,
                ),
                axis=1,
            )
        )
    scores = np.stack(score_rows, axis=1)
    blend = scores[:, :, 3]
    return (
        {
            "sentence_max": scores[:, :, 0],
            "sentence_top_three": scores[:, :, 1],
            "chunk_max": scores[:, :, 2],
            "blend": blend,
            "bm25": normalized_bm25,
            "hybrid_25": 0.75 * blend + 0.25 * normalized_bm25,
            "hybrid_50": 0.50 * blend + 0.50 * normalized_bm25,
        },
        {"page_ids": page_ids},
    )


def _rank_feature_rows(
    *,
    case: dict[str, Any],
    regions: list[dict[str, Any]],
    rare_concepts: Sequence[Sequence[str]],
    score_matrices: dict[str, np.ndarray],
    page_ids: Sequence[str],
) -> list[dict[str, Any]]:
    gold_page_id = (
        f"{case['corpus_id']}:{case['memory']['gold_source_ids'][0]}"
    )
    gold_region_label = case["decisions"]["memory_conditioned"]["choice"]
    rows: list[dict[str, Any]] = []
    for method, matrix in score_matrices.items():
        flat_order = np.argsort(matrix, axis=None)[::-1]
        for global_rank, flat_position in enumerate(flat_order, start=1):
            region_position, page_position = np.unravel_index(
                flat_position, matrix.shape
            )
            region_scores = matrix[region_position]
            page_scores = matrix[:, page_position]
            page_order = np.argsort(region_scores)[::-1]
            region_order = np.argsort(page_scores)[::-1]
            page_rank = int(np.where(page_order == page_position)[0][0]) + 1
            region_rank = int(np.where(region_order == region_position)[0][0]) + 1
            rows.append(
                {
                    "case_id": case["case_id"],
                    "base_case_id": case["base_case_id"],
                    "corpus_id": case["corpus_id"],
                    "variant": case["variant"],
                    "method": method,
                    "global_rank": global_rank,
                    "region_id": regions[region_position]["region_id"],
                    "region_text": regions[region_position]["text"],
                    "rare_concepts": list(rare_concepts[region_position]),
                    "page_id": page_ids[page_position],
                    "is_gold_page": page_ids[page_position] == gold_page_id,
                    "is_gold_region": gold_region_label.casefold()
                    in regions[region_position]["text"].casefold(),
                    "score": float(matrix[region_position, page_position]),
                    "page_rank_for_region": page_rank,
                    "region_rank_for_page": region_rank,
                    "mutual_top_one": page_rank == 1 and region_rank == 1,
                    "page_margin": float(
                        matrix[region_position, page_order[0]]
                        - matrix[region_position, page_order[1]]
                    ),
                    "region_margin": float(
                        matrix[region_order[0], page_position]
                        - (
                            matrix[region_order[1], page_position]
                            if len(region_order) > 1
                            else 0.0
                        )
                    ),
                }
            )
            if global_rank >= 10:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/benchmark_personamem_v0")
    )
    parser.add_argument(
        "--retrieval-index",
        type=Path,
        default=Path("data/retrieval-indexes/personamem-v2-train-v0"),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "region-embeddings.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/retrieval-experiments/personamem-v0/"
            "mutual-alignment-features.jsonl"
        ),
    )
    args = parser.parse_args()

    load_dotenv()
    cases = [
        case
        for case in load_cases(args.dataset)
        if case["variant"] in {"positive", "cue-ablated"}
    ]
    index = RetrievalIndex.load(args.retrieval_index)
    concept_extractor = SpacyCueConceptExtractor()
    cache = EmbeddingCache(args.embedding_cache, OpenAIEmbedder())

    prepared_cases = []
    all_descriptions = []
    for case in cases:
        all_regions = _parm_observation_regions(case["observation_text"])
        all_concepts = concept_extractor.rare_region_concepts(
            [region["description"] for region in all_regions]
        )
        kept = [
            (region, concepts)
            for region, concepts in zip(all_regions, all_concepts)
            if concepts
        ]
        regions = [region for region, _ in kept]
        concepts = [values for _, values in kept]
        prepared_cases.append((case, regions, concepts))
        all_descriptions.extend(
            region["description"] for region in regions
        )
    cache.embed(all_descriptions)

    output_rows: list[dict[str, Any]] = []
    for position, (case, regions, concepts) in enumerate(
        prepared_cases, start=1
    ):
        query_vectors = cache.embed(
            [region["description"] for region in regions]
        )
        scoped_index = index.scoped(case["corpus_id"])
        score_matrices, metadata = _page_dense_scores(
            scoped_index,
            query_vectors,
            [region["description"] for region in regions],
        )
        output_rows.extend(
            _rank_feature_rows(
                case=case,
                regions=regions,
                rare_concepts=concepts,
                score_matrices=score_matrices,
                page_ids=metadata["page_ids"],
            )
        )
        print(
            f"[{position:02d}/{len(cases)}] {case['case_id']}: "
            f"{len(regions)} distinctive regions"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    print(f"Wrote {len(output_rows)} feature rows to {args.out}")


if __name__ == "__main__":
    main()
