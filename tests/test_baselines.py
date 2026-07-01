from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path

from parm_bench.baselines import (
    INPUT_RAG_INSTRUCTIONS,
    InputRagBaseline,
    NoMemoryBaseline,
    benchmark_input,
)
from parm_bench.dataset import load_cases
from parm_bench.models import FINAL_ANSWER_INSTRUCTIONS, ModelResponse
from parm_bench.retrieval import (
    RetrievalHit,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "benchmark_v1"


class RecordingModel:
    model_name = "test-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def generate(
        self,
        *,
        prompt: str,
        observation_kind: str,
        observation_text: str,
        instructions: str = FINAL_ANSWER_INSTRUCTIONS,
        memory_context: str | None = None,
    ) -> ModelResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "observation_kind": observation_kind,
                "observation_text": observation_text,
                "instructions": instructions,
                "memory_context": memory_context,
            }
        )
        return ModelResponse(
            text="Visible Choice",
            response_id="resp_test",
            resolved_model="test-model-2026-01-01",
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )


class NoMemoryBaselineTests(unittest.TestCase):
    def test_benchmark_input_excludes_hidden_gold_fields(self) -> None:
        case = load_cases(DATASET)[0]
        public = benchmark_input(case)
        self.assertEqual(
            set(asdict(public)),
            {"case_id", "prompt", "observation_kind", "observation_text"},
        )
        self.assertNotIn(case["memory"]["text"], public.observation_text)

    def test_no_memory_uses_only_public_input_and_emits_empty_trace(self) -> None:
        case = benchmark_input(load_cases(DATASET)[0])
        model = RecordingModel()
        row = NoMemoryBaseline().run(case, model, None)

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            model.calls[0],
            {
                "prompt": case.prompt,
                "observation_kind": case.observation_kind,
                "observation_text": case.observation_text,
                "instructions": FINAL_ANSWER_INSTRUCTIONS,
                "memory_context": None,
            },
        )
        self.assertEqual(row["response_text"], "Visible Choice")
        self.assertEqual(row["requested_model"], "test-model")
        self.assertEqual(row["resolved_model"], "test-model-2026-01-01")
        self.assertEqual(
            row["trace"],
            {
                "detected_cues": [],
                "retrieved_source_ids": [],
                "admitted_source_ids": [],
                "admitted_perturbations": {},
            },
        )

    def test_no_memory_rejects_a_memory_backend(self) -> None:
        case = benchmark_input(load_cases(DATASET)[0])
        with self.assertRaisesRegex(ValueError, "must not receive"):
            NoMemoryBaseline().run(case, RecordingModel(), object())


class RecordingRetriever:
    mode = RetrievalMode.DENSE

    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.calls.append(request)
        return RetrievalResult(
            tuple(self.hits[: request.top_k]),
            {
                "retrieval_mode": "dense",
                "original_query": request.query,
                "expansion_queries": [],
                "candidate_lists": {"dense:original": ["page/one", "page/two"]},
                "returned_pages": [],
            },
        )


def hit(
    page_id: str,
    slug: str,
    text: str,
    score: float,
    perturbations: tuple[str, ...] = (),
) -> RetrievalHit:
    return RetrievalHit(
        page_id=page_id,
        source_id="fixture-source",
        slug=slug,
        title=page_id,
        chunk_id=f"{page_id}:0",
        text=text,
        score=score,
        rank=1,
        perturbations=perturbations,
    )


class InputRagBaselineTests(unittest.TestCase):
    def test_input_rag_queries_only_prompt_and_admits_all_hits(self) -> None:
        case = benchmark_input(load_cases(DATASET)[0])
        model = RecordingModel()
        retriever = RecordingRetriever(
            [
                hit("page/one", "note/one", "First memory", 0.9),
                hit(
                    "page/two",
                    "slack/two",
                    "Second memory",
                    0.8,
                    ("poison",),
                ),
            ]
        )

        row = InputRagBaseline(retrieval_limit=2).run(case, model, retriever)

        self.assertEqual(
            retriever.calls, [RetrievalRequest(case.prompt, top_k=2)]
        )
        self.assertEqual(model.calls[0]["instructions"], INPUT_RAG_INSTRUCTIONS)
        self.assertEqual(
            model.calls[0]["memory_context"],
            "1. First memory\n\n2. Second memory",
        )
        self.assertNotIn("note/one", model.calls[0]["memory_context"])
        self.assertEqual(row["trace"]["retrieval_mode"], "dense")
        self.assertEqual(
            row["trace"]["retrieved_page_ids"], ["page/one", "page/two"]
        )
        self.assertEqual(
            row["trace"]["admitted_source_ids"], ["note/one", "slack/two"]
        )
        self.assertEqual(
            row["trace"]["admitted_perturbations"], {"slack/two": ["poison"]}
        )

    def test_input_rag_requires_retriever(self) -> None:
        case = benchmark_input(load_cases(DATASET)[0])
        with self.assertRaisesRegex(ValueError, "requires a retriever"):
            InputRagBaseline().run(case, RecordingModel(), None)

    def test_input_rag_without_hits_uses_observation_only_instructions(self) -> None:
        case = benchmark_input(load_cases(DATASET)[0])
        model = RecordingModel()

        InputRagBaseline().run(case, model, RecordingRetriever([]))

        self.assertEqual(
            model.calls[0]["instructions"],
            FINAL_ANSWER_INSTRUCTIONS,
        )
        self.assertIsNone(model.calls[0]["memory_context"])

    def test_input_rag_rejects_non_positive_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            InputRagBaseline(retrieval_limit=0)


if __name__ == "__main__":
    unittest.main()
