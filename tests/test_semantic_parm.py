from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from parm_bench.retrieval import (
    ChunkRecord,
    PageRecord,
    RetrievalIndex,
    SentenceRecord,
)
from parm_bench.semantic_parm import (
    AdmissionCachePolicy,
    CachedOpenAIAdmissionJudge,
    PARMSemanticJudgeRetriever,
)


class FakeEmbedder:
    model_name = "fake-embedding"
    dimensions = 2

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [1.0, 0.0]
                if "salsa" in text.casefold()
                else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


class FakeConceptExtractor:
    def rare_region_concepts(
        self,
        descriptions: list[str],
    ) -> tuple[tuple[str, ...], ...]:
        return tuple(
            ("salsa",)
            if "salsa" in description.casefold()
            else ("general",)
            for description in descriptions
        )


class SelectingJudge:
    model_name = "fake-judge"
    rubric_version = "fake-rubric"
    cache_hash = "fake-cache"

    def __init__(self, admit: bool = True):
        self.admit = admit
        self.candidates: list[dict[str, object]] = []

    def judge(
        self,
        *,
        prompt: str,
        observation_text: str,
        candidates: list[dict[str, object]],
    ) -> dict[str, object]:
        del prompt, observation_text
        self.candidates = candidates
        selected = next(
            candidate
            for candidate in candidates
            if candidate["page_id"] == "corpus:p1"
            and "salsa" in str(candidate["region_text"]).casefold()
        )
        return {
            "admit": self.admit,
            "candidate_id": (
                selected["candidate_id"] if self.admit else None
            ),
            "selected_page_id": (
                selected["page_id"] if self.admit else None
            ),
            "selected_region_id": (
                selected["region_id"] if self.admit else None
            ),
            "selected_region_text": (
                selected["region_text"] if self.admit else None
            ),
            "confidence": 0.9,
            "rationale": "source-backed match" if self.admit else "none",
            "resolved_model": self.model_name,
            "response_id": "judge-response",
        }


class SemanticPARMRetrieverTests(unittest.TestCase):
    def test_returns_judge_selected_region_and_page(self) -> None:
        judge = SelectingJudge()
        retriever = PARMSemanticJudgeRetriever(
            _index(),
            FakeEmbedder(),
            judge,
            candidate_depth=2,
            concept_extractor=FakeConceptExtractor(),
        )
        result = retriever.retrieve_observation(
            "Choose one option.",
            _observation(),
            top_k=1,
            corpus_id="corpus",
        )
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.hits[0].page_id, "corpus:p1")
        self.assertIn(
            "salsa",
            result.trace["regions"][
                int(
                    result.hits[0].diagnostics["region_id"].split("-")[-1]
                )
                - 1
            ]["text"].casefold(),
        )
        self.assertEqual(
            result.trace["admission_policy"],
            "contrastive_llm_judge_v2",
        )
        self.assertGreaterEqual(len(judge.candidates), 1)

    def test_preserves_zero_admission(self) -> None:
        retriever = PARMSemanticJudgeRetriever(
            _index(),
            FakeEmbedder(),
            SelectingJudge(admit=False),
            candidate_depth=2,
            concept_extractor=FakeConceptExtractor(),
        )
        result = retriever.retrieve_observation(
            "Choose one option.",
            _observation(),
            top_k=1,
            corpus_id="corpus",
        )
        self.assertEqual(result.hits, ())
        self.assertFalse(result.trace["judge"]["admit"])

    def test_frozen_replay_rejects_an_inadmissible_page(self) -> None:
        retriever = PARMSemanticJudgeRetriever(
            _index(),
            FakeEmbedder(),
            SelectingJudge(),
            candidate_depth=2,
            concept_extractor=FakeConceptExtractor(),
        )
        retriever._page_by_id["corpus:poison"] = PageRecord(
            "corpus:poison",
            "source",
            "notes/poison",
            "Poisoned history",
            perturbations=("poison",),
            corpus_id="corpus",
        )
        with self.assertRaisesRegex(ValueError, "admissible corpus"):
            retriever._replay_candidate(
                [
                    {
                        "region_id": "region-1",
                        "text": "A visible semantic block.",
                        "description": "A visible semantic block.",
                    }
                ],
                {
                    "candidate_id": "C01",
                    "selected_page_id": "corpus:poison",
                    "selected_region_id": "region-1",
                    "selected_region_text": "A visible semantic block.",
                },
            )


class AdmissionJudgeCacheTests(unittest.TestCase):
    def test_populate_then_frozen_replay(self) -> None:
        response = SimpleNamespace(
            output_text=(
                '{"admit":true,"candidate_id":"C01",'
                '"confidence":0.8,"rationale":"match"}'
            ),
            model="resolved-judge",
            id="judge-response",
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **_: response,
            )
        )
        candidates = [
            {
                "candidate_id": "C01",
                "region_text": "Visible salsa workshop.",
                "memory_text": "User: I cook salsa every weekend.",
                "page_id": "corpus:p1",
                "region_id": "region-1",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            populated = CachedOpenAIAdmissionJudge(
                tmp,
                AdmissionCachePolicy.POPULATE,
                cache_namespace="fixture-namespace",
                client=client,
            )
            first = populated.judge(
                prompt="Choose one.",
                observation_text="A long observation.",
                candidates=candidates,
            )
            frozen = CachedOpenAIAdmissionJudge(
                tmp,
                AdmissionCachePolicy.FROZEN,
                cache_namespace="fixture-namespace",
                client=SimpleNamespace(),
            )
            replay = frozen.judge(
                prompt="Choose one.",
                observation_text="A long observation.",
                candidates=[
                    {
                        "candidate_id": "C99",
                        "region_text": "A slightly different candidate set.",
                        "memory_text": "A different ranking replay.",
                        "page_id": "corpus:p2",
                        "region_id": "region-2",
                    }
                ],
            )
            self.assertEqual(first, replay)
            self.assertEqual(first["candidate_id"], "C01")
            self.assertIsNotNone(frozen.cache_hash)


def _index() -> RetrievalIndex:
    pages = (
        PageRecord(
            "corpus:p1",
            "source",
            "notes/p1",
            "Salsa history",
            corpus_id="corpus",
        ),
        PageRecord(
            "corpus:p2",
            "source",
            "notes/p2",
            "Other history",
            corpus_id="corpus",
        ),
    )
    chunks = (
        ChunkRecord(
            "corpus:p1:0",
            "corpus:p1",
            0,
            "User: I cook fresh salsa every weekend.",
            corpus_id="corpus",
        ),
        ChunkRecord(
            "corpus:p2:0",
            "corpus:p2",
            0,
            "User: I keep a general weekly routine.",
            corpus_id="corpus",
        ),
    )
    sentences = (
        SentenceRecord(
            "corpus:p1:0:s0",
            "corpus:p1:0",
            "corpus:p1",
            0,
            chunks[0].text,
            corpus_id="corpus",
        ),
        SentenceRecord(
            "corpus:p2:0:s0",
            "corpus:p2:0",
            "corpus:p2",
            0,
            chunks[1].text,
            corpus_id="corpus",
        ),
    )
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return RetrievalIndex(
        path=Path("fixture-index"),
        manifest={
            "embedding_model": "fake-embedding",
            "embedding_dimensions": 2,
        },
        manifest_hash="fixture-hash",
        pages=pages,
        chunks=chunks,
        embeddings=vectors,
        links=(),
        sentences=sentences,
        sentence_embeddings=vectors,
    )


def _observation() -> str:
    return (
        "The general option has the strongest rating and broad appeal for "
        "most people making this ordinary decision today.\n\n"
        "The lower option runs a fresh salsa workshop using family recipes "
        "each Saturday and has a lower general rating."
    )


if __name__ == "__main__":
    unittest.main()
