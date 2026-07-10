from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path

from parm_bench.baselines import (
    AllEntityOutputRagBaseline,
    INPUT_RAG_INSTRUCTIONS,
    InputRagBaseline,
    NaiveOutputRagBaseline,
    NoMemoryBaseline,
    OutputRagFlow,
    benchmark_input,
)
from parm_bench.dataset import load_cases
from parm_bench.models import FINAL_ANSWER_INSTRUCTIONS, ModelResponse
from parm_bench.retrieval import (
    EntityRetrievalResult,
    EntitySeed,
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
        self.calls: list[dict[str, object]] = []
        self.responses: list[str] = []

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
        text = (
            self.responses.pop(0)
            if self.responses
            else f"Visible Choice {len(self.calls)}"
        )
        return ModelResponse(
            text=text,
            response_id=f"resp_{len(self.calls)}",
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
        self.assertEqual(row["response_text"], "Visible Choice 1")
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


class RecordingEntityRetriever:
    def __init__(
        self,
        seeds: tuple[EntitySeed, ...],
        hits: tuple[RetrievalHit, ...],
    ) -> None:
        self.seeds = seeds
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def retrieve_entities(
        self,
        observation_text: str,
        *,
        top_k: int,
    ) -> EntityRetrievalResult:
        self.calls.append((observation_text, top_k))
        return EntityRetrievalResult(
            self.seeds,
            self.hits,
            {
                "retrieval_condition_detail": "entity_exact_match",
                "entity_seeds": [seed.__dict__ for seed in self.seeds],
                "per_seed_retrievals": [],
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


class NaiveOutputRagBaselineTests(unittest.TestCase):
    def test_tool_output_only_retrieves_observation_then_final_model_pass(
        self,
    ) -> None:
        case = benchmark_input(load_cases(DATASET)[0])
        model = RecordingModel()
        retriever = RecordingRetriever(
            [hit("page/one", "note/one", "Tool memory", 0.9)]
        )

        row = NaiveOutputRagBaseline(
            retrieval_limit=2,
            output_rag_flow=OutputRagFlow.TOOL_OUTPUT_ONLY,
        ).run(case, model, retriever)

        self.assertEqual(
            retriever.calls, [RetrievalRequest(case.observation_text, top_k=2)]
        )
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["memory_context"], "1. Tool memory")
        self.assertEqual(model.calls[0]["instructions"], INPUT_RAG_INSTRUCTIONS)
        self.assertEqual(row["response_text"], "Visible Choice 1")
        self.assertEqual(row["trace"]["output_rag_flow"], "tool_output_only")
        self.assertEqual(row["trace"]["retrieval_queries"], [case.observation_text])
        self.assertEqual(row["trace"]["model_passes"], [])

    def test_model_output_only_retrieves_first_model_response_then_final(
        self,
    ) -> None:
        case = benchmark_input(load_cases(DATASET)[0])
        model = RecordingModel()
        model.responses = ["Intermediate choice", "Final choice"]
        retriever = RecordingRetriever(
            [hit("page/one", "note/one", "Model memory", 0.9)]
        )

        row = NaiveOutputRagBaseline(
            retrieval_limit=1,
            output_rag_flow=OutputRagFlow.MODEL_OUTPUT_ONLY,
        ).run(case, model, retriever)

        self.assertEqual(
            retriever.calls, [RetrievalRequest("Intermediate choice", top_k=1)]
        )
        self.assertEqual(len(model.calls), 2)
        self.assertIsNone(model.calls[0]["memory_context"])
        self.assertEqual(model.calls[1]["memory_context"], "1. Model memory")
        self.assertEqual(row["response_text"], "Final choice")
        self.assertEqual(
            row["trace"]["model_passes"][0]["response_text"],
            "Intermediate choice",
        )
        self.assertEqual(
            row["trace"]["retrieval_queries"], ["Intermediate choice"]
        )

    def test_tool_then_model_output_runs_two_retrieval_phases_and_dedupes(
        self,
    ) -> None:
        class PhaseRetriever(RecordingRetriever):
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                self.calls.append(request)
                if len(self.calls) == 1:
                    hits = [
                        hit("page/tool", "note/shared", "Tool memory", 0.9),
                    ]
                else:
                    hits = [
                        hit("page/model", "note/model", "Model memory", 0.8),
                        hit("page/dupe", "note/shared", "Duplicate", 0.7),
                    ]
                return RetrievalResult(
                    tuple(hits[: request.top_k]),
                    {
                        "retrieval_mode": "dense",
                        "original_query": request.query,
                        "expansion_queries": [],
                        "candidate_lists": {},
                        "returned_pages": [],
                    },
                )

        case = benchmark_input(load_cases(DATASET)[0])
        model = RecordingModel()
        model.responses = ["Intermediate with memory", "Final choice"]
        retriever = PhaseRetriever([])

        row = NaiveOutputRagBaseline(
            retrieval_limit=3,
            output_rag_flow=OutputRagFlow.TOOL_THEN_MODEL_OUTPUT,
        ).run(case, model, retriever)

        self.assertEqual(
            retriever.calls,
            [
                RetrievalRequest(case.observation_text, top_k=3),
                RetrievalRequest("Intermediate with memory", top_k=3),
            ],
        )
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]["memory_context"], "1. Tool memory")
        self.assertEqual(
            model.calls[1]["memory_context"],
            "1. Tool memory\n\n2. Model memory",
        )
        self.assertEqual(
            row["trace"]["retrieved_source_ids"],
            ["note/shared", "note/model", "note/shared"],
        )
        self.assertEqual(
            row["trace"]["admitted_source_ids"],
            ["note/shared", "note/model"],
        )

    def test_naive_output_rag_requires_retriever_and_positive_limit(self) -> None:
        case = benchmark_input(load_cases(DATASET)[0])
        with self.assertRaisesRegex(ValueError, "requires a retriever"):
            NaiveOutputRagBaseline().run(case, RecordingModel(), None)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            NaiveOutputRagBaseline(retrieval_limit=0)


class AllEntityOutputRagBaselineTests(unittest.TestCase):
    def test_retrieves_entities_from_observation_and_admits_flat_union(self) -> None:
        case = benchmark_input(load_cases(DATASET)[0])
        model = RecordingModel()
        seeds = (
            EntitySeed("entity-1", "Alpha Project", "alpha project", "gazetteer", 0, 13),
            EntitySeed("entity-2", "Beta Team", "beta team", "noun_phrase", 20, 29),
        )
        retriever = RecordingEntityRetriever(
            seeds,
            (
                hit("page/b", "note/b", "Beta memory", 0.7),
                hit("page/a", "note/a", "Alpha memory", 0.9),
                hit("page/a", "note/a", "Duplicate alpha", 0.6),
            ),
        )

        row = AllEntityOutputRagBaseline(retrieval_limit=3).run(
            case,
            model,
            retriever,
        )

        self.assertEqual(retriever.calls, [(case.observation_text, 3)])
        self.assertEqual(model.calls[0]["instructions"], INPUT_RAG_INSTRUCTIONS)
        self.assertEqual(
            model.calls[0]["memory_context"],
            "1. Alpha memory\n\n2. Beta memory",
        )
        self.assertEqual(
            row["trace"]["detected_cues"][0]["surface"],
            "Alpha Project",
        )
        self.assertEqual(
            row["trace"]["retrieved_source_ids"],
            ["note/b", "note/a", "note/a"],
        )
        self.assertEqual(row["trace"]["admitted_source_ids"], ["note/a", "note/b"])
        self.assertEqual(
            row["trace"]["retrieval_condition_detail"],
            "entity_exact_match",
        )

    def test_empty_entities_runs_without_memory_context(self) -> None:
        case = benchmark_input(load_cases(DATASET)[0])
        model = RecordingModel()

        row = AllEntityOutputRagBaseline().run(
            case,
            model,
            RecordingEntityRetriever((), ()),
        )

        self.assertEqual(model.calls[0]["instructions"], FINAL_ANSWER_INSTRUCTIONS)
        self.assertIsNone(model.calls[0]["memory_context"])
        self.assertEqual(row["trace"]["detected_cues"], [])
        self.assertEqual(row["trace"]["admitted_source_ids"], [])

    def test_requires_entity_retriever_and_positive_limit(self) -> None:
        case = benchmark_input(load_cases(DATASET)[0])
        with self.assertRaisesRegex(ValueError, "requires an entity retriever"):
            AllEntityOutputRagBaseline().run(case, RecordingModel(), None)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            AllEntityOutputRagBaseline(retrieval_limit=0)


if __name__ == "__main__":
    unittest.main()
