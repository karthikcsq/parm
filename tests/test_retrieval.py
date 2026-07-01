from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from parm_bench.retrieval import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    CachedOpenAIQueryExpander,
    ExpansionCacheMissError,
    ExpansionPolicy,
    IndexRetriever,
    RetrievalIndex,
    RetrievalMode,
    RetrievalRequest,
    RetrievalValidationError,
    _bm25_rank,
    _rrf,
)
from parm_bench.retrieval_export import write_retrieval_index


def vector(x: float, y: float = 0.0) -> list[float]:
    result = [0.0] * EMBEDDING_DIMENSIONS
    result[0] = x
    result[1] = y
    return result


class FakeEmbedder:
    model_name = EMBEDDING_MODEL
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class FakeExpander:
    model = "gpt-5-mini"
    prompt_version = "retrieval-expansion-v1"
    cache_hash = "cache-hash"

    def expand(self, query: str) -> tuple[str, str, str]:
        return ("alpha alternate", "alpha memory", "alpha personal")


def build_index(
    root: Path,
    *,
    pages: list[dict[str, object]] | None = None,
    chunks: list[dict[str, object]] | None = None,
    links: list[dict[str, object]] | None = None,
) -> RetrievalIndex:
    pages = pages or [
        {"id": 1, "source_id": "source", "slug": "p1", "title": "First"},
        {"id": 2, "source_id": "source", "slug": "p2", "title": "Second"},
        {"id": 3, "source_id": "source", "slug": "p3", "title": "Third"},
        {"id": 4, "source_id": "source", "slug": "p4", "title": "Fourth"},
    ]
    chunks = chunks or [
        {
            "id": 1,
            "page_id": 1,
            "chunk_index": 0,
            "chunk_text": "alpha weak",
            "embedding": vector(0.2, 0.98),
            "model": EMBEDDING_MODEL,
        },
        {
            "id": 2,
            "page_id": 1,
            "chunk_index": 1,
            "chunk_text": "alpha strongest",
            "embedding": vector(1.0),
            "model": EMBEDDING_MODEL,
        },
        {
            "id": 3,
            "page_id": 2,
            "chunk_index": 0,
            "chunk_text": "alpha second",
            "embedding": vector(0.99, 0.1),
            "model": EMBEDDING_MODEL,
        },
        {
            "id": 4,
            "page_id": 3,
            "chunk_index": 0,
            "chunk_text": "other",
            "embedding": vector(0.0, 1.0),
            "model": EMBEDDING_MODEL,
        },
        {
            "id": 5,
            "page_id": 4,
            "chunk_index": 0,
            "chunk_text": "other",
            "embedding": vector(-1.0),
            "model": EMBEDDING_MODEL,
        },
    ]
    write_retrieval_index(
        root,
        pages=pages,
        chunks=chunks,
        links=links or [],
        corpus_id="fixture",
        gbrain_version="test",
        chunker_version="test",
    )
    return RetrievalIndex.load(root)


class RetrievalIndexTests(unittest.TestCase):
    def test_loader_and_dense_best_chunk_page_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = build_index(Path(tmp) / "index")
            retriever = IndexRetriever(
                index, RetrievalMode.DENSE, FakeEmbedder({"alpha": vector(1.0)})
            )
            result = retriever.retrieve(RetrievalRequest("alpha", top_k=3))

        self.assertEqual(result.hits[0].page_id, "source:p1")
        self.assertEqual(result.hits[0].chunk_id, "source:p1:1")
        self.assertEqual(
            len({hit.page_id for hit in result.hits}), len(result.hits)
        )
        self.assertEqual(
            result.trace["candidate_lists"]["dense:original"][0], "source:p1"
        )

    def test_loader_rejects_hash_mismatch_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "index"
            build_index(root)
            with (root / "pages.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(
                RetrievalValidationError, "artifact hash mismatch"
            ):
                RetrievalIndex.load(root)

            root2 = Path(tmp) / "duplicates"
            build_index(root2)
            rows = (root2 / "pages.jsonl").read_text(encoding="utf-8")
            first = rows.splitlines()[0]
            (root2 / "pages.jsonl").write_text(
                rows + first + "\n", encoding="utf-8"
            )
            manifest = json.loads(
                (root2 / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["artifact_hashes"]["pages.jsonl"] = hashlib.sha256(
                (root2 / "pages.jsonl").read_bytes()
            ).hexdigest()
            manifest["counts"]["pages"] += 1
            content = hashlib.sha256()
            for name in (
                "pages.jsonl",
                "chunks.jsonl",
                "embeddings.npy",
                "links.jsonl",
            ):
                content.update((root2 / name).read_bytes())
            manifest["content_hash"] = content.hexdigest()
            (root2 / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RetrievalValidationError, "duplicate page IDs"
            ):
                RetrievalIndex.load(root2)

    def test_deterministic_page_id_tie_break(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = build_index(Path(tmp) / "index")
            retriever = IndexRetriever(
                index, "dense", FakeEmbedder({"vertical": vector(0.0, 1.0)})
            )
            result = retriever.retrieve(RetrievalRequest("vertical", top_k=2))
        self.assertEqual(
            [hit.page_id for hit in result.hits],
            ["source:p3", "source:p1"],
        )


class RankingTests(unittest.TestCase):
    def test_bm25_order_and_one_based_rrf(self) -> None:
        ranking = _bm25_rank(
            "orange climate",
            {
                "p1": "orange orange climate",
                "p2": "orange",
                "p3": "unrelated",
            },
            3,
        )
        self.assertEqual(ranking[0], "p1")
        scores = _rrf([["p1", "p2"], ["p2", "p1"]])
        expected = 1 / 61 + 1 / 62
        self.assertAlmostEqual(scores["p1"], expected)
        self.assertAlmostEqual(scores["p2"], expected)

    def test_hybrid_records_rrf_normalization_and_blend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = build_index(Path(tmp) / "index")
            retriever = IndexRetriever(
                index, "hybrid", FakeEmbedder({"alpha": vector(1.0)})
            )
            result = retriever.retrieve(RetrievalRequest("alpha", top_k=2))
        diagnostic = result.hits[0].diagnostics
        self.assertIsNotNone(diagnostic["raw_rrf"])
        self.assertAlmostEqual(diagnostic["normalized_rrf"], 1.0)
        self.assertAlmostEqual(
            diagnostic["final_score"],
            0.70 + 0.30 * diagnostic["original_query_cosine"],
        )

    def test_enhanced_title_candidate_and_graph_candidate_containment(self) -> None:
        pages = [
            {"id": 1, "source_id": "s", "slug": "p1", "title": "Ordinary"},
            {
                "id": 2,
                "source_id": "s",
                "slug": "p2",
                "title": "alpha alternate alpha memory alpha personal",
            },
            {"id": 3, "source_id": "s", "slug": "p3", "title": "Third"},
            {"id": 4, "source_id": "s", "slug": "p4", "title": "Fourth"},
            {"id": 5, "source_id": "s", "slug": "graph-only", "title": "Graph"},
        ]
        chunks = [
            {
                "id": position,
                "page_id": position,
                "chunk_index": 0,
                "chunk_text": "alpha" if position == 1 else "unrelated",
                "embedding": vector(1.0 - position * 0.01, position * 0.01),
                "model": EMBEDDING_MODEL,
            }
            for position in range(1, 6)
        ]
        links = [
            {
                "from_page_id": 3,
                "to_page_id": 2,
                "link_type": "mention",
                "link_source": "test",
            },
            {
                "from_page_id": 4,
                "to_page_id": 2,
                "link_type": "mention",
                "link_source": "test",
            },
            {
                "from_page_id": 1,
                "to_page_id": 5,
                "link_type": "mention",
                "link_source": "test",
            },
        ]
        query_vectors = {
            "alpha": vector(1.0),
            "alpha alternate": vector(1.0),
            "alpha memory": vector(1.0),
            "alpha personal": vector(1.0),
        }
        with tempfile.TemporaryDirectory() as tmp:
            index = build_index(
                Path(tmp) / "index", pages=pages, chunks=chunks, links=links
            )
            result = IndexRetriever(
                index,
                "enhanced",
                FakeEmbedder(query_vectors),
                expander=FakeExpander(),
            ).retrieve(RetrievalRequest("alpha", top_k=5))

        title_candidates = result.trace["candidate_lists"][
            "title_bm25:expansion_1"
        ]
        self.assertEqual(title_candidates[0], "s:p2")
        self.assertNotIn(
            "s:p2",
            result.trace["candidate_lists"]["body_bm25:expansion_1"],
        )
        p2 = next(hit for hit in result.hits if hit.page_id == "s:p2")
        p1 = next(hit for hit in result.hits if hit.page_id == "s:p1")
        self.assertLess(
            p2.diagnostics["pre_graph_score"],
            p1.diagnostics["pre_graph_score"],
        )
        self.assertEqual(result.hits[0].page_id, "s:p2")
        self.assertEqual(p2.diagnostics["graph_inbound_count"], 2)
        self.assertEqual(p2.diagnostics["graph_multiplier"], 1.05)
        candidate_union = {
            page_id
            for ranking in result.trace["candidate_lists"].values()
            for page_id in ranking
        }
        self.assertTrue(
            {hit.page_id for hit in result.hits}.issubset(candidate_union)
        )


class ExpansionCacheTests(unittest.TestCase):
    def test_populate_then_frozen_replay(self) -> None:
        response = SimpleNamespace(
            output_text=json.dumps(
                {"alternatives": ["query one", "query two", "query three"]}
            )
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_: response)
        )
        with tempfile.TemporaryDirectory() as tmp:
            populated = CachedOpenAIQueryExpander(
                tmp, ExpansionPolicy.POPULATE, client=client
            )
            expected = populated.expand("Original query")
            self.assertIsNotNone(populated.cache_hash)
            frozen = CachedOpenAIQueryExpander(tmp, ExpansionPolicy.FROZEN)
            self.assertEqual(frozen.expand("Original query"), expected)
            self.assertIsNotNone(frozen.cache_hash)

    def test_frozen_miss_and_duplicate_output_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frozen = CachedOpenAIQueryExpander(tmp, "frozen")
            with self.assertRaises(ExpansionCacheMissError):
                frozen.expand("missing")
            response = SimpleNamespace(
                output_text=json.dumps(
                    {"alternatives": ["same", "same", "different"]}
                )
            )
            client = SimpleNamespace(
                responses=SimpleNamespace(create=lambda **_: response)
            )
            populate = CachedOpenAIQueryExpander(tmp, "populate", client=client)
            with self.assertRaises(RetrievalValidationError):
                populate.expand("original")


if __name__ == "__main__":
    unittest.main()
