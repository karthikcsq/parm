from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from parm_bench.baselines import INPUT_RAG_INSTRUCTIONS
from parm_bench.models import ModelResponse
from parm_bench.retrieval import (
    EMBEDDING_MODEL,
    EntityRetrievalResult,
    EntitySeed,
    RetrievalHit,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
)
from parm_bench.workbench import (
    RetrievalCondition,
    WORKBENCH_INSTRUCTIONS,
    WorkbenchRequestError,
    WorkbenchRunError,
    WorkbenchService,
    _case_label,
    _evaluate_case,
    create_workbench_server,
)


class FakeModel:
    calls: list[dict[str, object]] = []

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, **kwargs: object) -> ModelResponse:
        self.__class__.calls.append(dict(kwargs))
        return ModelResponse(
            text="Workbench answer",
            response_id="resp_workbench",
            resolved_model=f"{self.model_name}-resolved",
            usage={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        )


class FakeRetriever:
    def __init__(self, mode: RetrievalMode):
        self.mode = mode
        self.calls: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.calls.append(request)
        hit = RetrievalHit(
            page_id="fixture:note/one",
            source_id="fixture",
            slug="note/one",
            title="One useful memory",
            chunk_id="fixture:note/one:0",
            text="Remember the quiet option.",
            score=0.91,
            rank=1,
            diagnostics={
                "original_query_cosine": 0.88,
                "normalized_rrf": 1.0,
                "graph_multiplier": 1.0,
                "graph_inbound_count": 0,
            },
        )
        return RetrievalResult(
            hits=(hit,),
            trace={
                "retrieval_mode": self.mode.value,
                "original_query": request.query,
                "candidate_lists": {
                    f"{self.mode.value}:original": [hit.page_id]
                },
                "returned_pages": [{"page_id": hit.page_id}],
            },
        )


class FailingRetriever:
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        raise RuntimeError("network down")


class FakeEntityRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve_entities(
        self,
        observation_text: str,
        *,
        top_k: int,
    ) -> EntityRetrievalResult:
        self.calls.append((observation_text, top_k))
        seed = EntitySeed(
            "entity-1",
            "Garden Table",
            "garden table",
            "gazetteer",
            10,
            22,
        )
        hit = RetrievalHit(
            page_id="fixture:note/entity",
            source_id="fixture",
            slug="note/one",
            title="Entity memory",
            chunk_id="fixture:note/entity:0",
            text="Entity-selected memory.",
            score=1.4,
            rank=1,
            diagnostics={
                "entity_seed_id": seed.seed_id,
                "entity_surface": seed.surface,
                "entity_source": seed.source,
            },
        )
        return EntityRetrievalResult(
            (seed,),
            (hit,),
            {
                "retrieval_condition_detail": "entity_exact_match",
                "entity_seeds": [seed.__dict__],
                "per_seed_retrievals": [],
            },
        )


class FailingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, **kwargs: object) -> ModelResponse:
        raise RuntimeError("model unavailable")


def fake_index() -> SimpleNamespace:
    return SimpleNamespace(
        path=Path("fixture-index").resolve(),
        manifest_hash="abc123manifest",
        manifest={
            "corpus_id": "fixture",
            "counts": {"pages": 3, "chunks": 4},
            "embedding_model": EMBEDDING_MODEL,
        },
    )


def fake_case() -> dict[str, object]:
    return {
        "case_id": "parm-amara-lunch-search-positive",
        "base_case_id": "parm-amara-lunch-search",
        "variant": "positive",
        "prompt": "Choose exactly one lunch option.",
        "observation": {"kind": "tool_result"},
        "observation_text": "Result 1. Garden Table",
        "decisions": {
            "output_only": {"choice": "Garden Table"},
            "memory_conditioned": {"choice": "Workbench answer"},
        },
        "memory": {"gold_source_ids": ["note/one"]},
    }


class WorkbenchServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeModel.calls.clear()
        self.retrievers: list[FakeRetriever] = []
        self.entity_retrievers: list[FakeEntityRetriever] = []

        def factory(mode: RetrievalMode) -> FakeRetriever:
            retriever = FakeRetriever(mode)
            self.retrievers.append(retriever)
            return retriever

        def entity_factory() -> FakeEntityRetriever:
            retriever = FakeEntityRetriever()
            self.entity_retrievers.append(retriever)
            return retriever

        self.service = WorkbenchService(
            fake_index(),
            SimpleNamespace(),
            default_model="test-model",
            model_factory=FakeModel,
            retriever_factory=factory,
            entity_retriever_factory=entity_factory,
            cases=[fake_case()],
        )

    def test_input_rag_returns_model_output_hits_and_trace(self) -> None:
        result = self.service.run(
            {
                "prompt": "Choose a plan",
                "condition": "input_rag",
                "retrieval_mode": "hybrid",
                "top_k": 3,
            }
        )

        self.assertEqual(
            self.retrievers[0].calls,
            [RetrievalRequest("Choose a plan", top_k=3)],
        )
        self.assertEqual(result["condition"], "input_rag")
        self.assertEqual(result["retrieval_mode"], "hybrid")
        self.assertEqual(result["model"]["text"], "Workbench answer")
        self.assertEqual(result["response_text"], "Workbench answer")
        self.assertEqual(result["retrieval"]["hits"][0]["slug"], "note/one")
        self.assertEqual(
            FakeModel.calls[0]["memory_context"],
            "1. Remember the quiet option.",
        )
        self.assertEqual(
            FakeModel.calls[0]["instructions"], WORKBENCH_INSTRUCTIONS
        )

    def test_no_memory_skips_retrieval_and_sends_only_prompt(self) -> None:
        result = self.service.run(
            {
                "prompt": "Explain this",
                "condition": "no_memory",
                "retrieval_mode": "enhanced",
                "top_k": 5,
            }
        )

        self.assertEqual(self.retrievers, [])
        self.assertIsNone(result["retrieval_mode"])
        self.assertEqual(result["retrieval"]["hits"], [])
        self.assertIsNone(FakeModel.calls[0]["memory_context"])
        self.assertEqual(FakeModel.calls[0]["observation_text"], "")

    def test_naive_output_rag_tool_output_flow_uses_case_observation(self) -> None:
        result = self.service.run(
            {
                "case_id": "parm-amara-lunch-search-positive",
                "condition": "naive_output_rag",
                "output_rag_flow": "tool_output_only",
                "retrieval_mode": "dense",
                "top_k": 2,
            }
        )

        self.assertEqual(
            self.retrievers[0].calls,
            [RetrievalRequest("Result 1. Garden Table", top_k=2)],
        )
        self.assertEqual(result["condition"], "naive_output_rag")
        self.assertEqual(result["output_rag_flow"], "tool_output_only")
        self.assertEqual(result["retrieval_mode"], "dense")
        self.assertEqual(result["retrieval"]["hits"][0]["rank"], 1)
        self.assertEqual(result["retrieval"]["hits"][0]["slug"], "note/one")
        self.assertEqual(
            result["retrieval"]["trace"]["output_rag_flow"],
            "tool_output_only",
        )
        self.assertEqual(
            FakeModel.calls[0]["memory_context"],
            "1. Remember the quiet option.",
        )

    def test_naive_output_rag_model_output_flow_allows_custom_prompt(self) -> None:
        result = self.service.run(
            {
                "prompt": "Choose a plan",
                "condition": "naive_output_rag",
                "output_rag_flow": "model_output_only",
                "retrieval_mode": "hybrid",
                "top_k": 1,
            }
        )

        self.assertEqual(len(FakeModel.calls), 2)
        self.assertEqual(
            self.retrievers[0].calls,
            [RetrievalRequest("Workbench answer", top_k=1)],
        )
        self.assertEqual(result["output_rag_flow"], "model_output_only")
        self.assertEqual(result["retrieval_mode"], "hybrid")

    def test_naive_output_rag_requires_case_for_tool_output_flows(self) -> None:
        with self.assertRaisesRegex(
            WorkbenchRequestError, "Choose a benchmark case"
        ):
            self.service.run(
                {
                    "prompt": "Custom text has no observation.",
                    "condition": "naive_output_rag",
                    "output_rag_flow": "tool_then_model_output",
                    "retrieval_mode": "dense",
                }
            )

    def test_all_entity_output_rag_uses_case_observation_without_mode(self) -> None:
        result = self.service.run(
            {
                "case_id": "parm-amara-lunch-search-positive",
                "condition": "all_entity_output_rag",
                "retrieval_mode": "dense",
                "top_k": 4,
            }
        )

        self.assertEqual(self.retrievers, [])
        self.assertEqual(
            self.entity_retrievers[0].calls,
            [("Result 1. Garden Table", 4)],
        )
        self.assertEqual(result["condition"], "all_entity_output_rag")
        self.assertIsNone(result["retrieval_mode"])
        self.assertEqual(
            result["retrieval"]["trace"]["retrieval_condition_detail"],
            "entity_exact_match",
        )
        self.assertEqual(
            result["retrieval"]["hits"][0]["diagnostics"]["entity_surface"],
            "Garden Table",
        )
        self.assertEqual(
            FakeModel.calls[0]["memory_context"],
            "1. Entity-selected memory.",
        )

    def test_all_entity_output_rag_requires_case(self) -> None:
        with self.assertRaisesRegex(WorkbenchRequestError, "Choose a benchmark case"):
            self.service.run(
                {
                    "prompt": "Custom text",
                    "condition": "all_entity_output_rag",
                }
            )

    def test_case_selection_uses_canonical_prompt_and_observation(self) -> None:
        result = self.service.run(
            {
                "case_id": "parm-amara-lunch-search-positive",
                "prompt": "Client-side text must not override the case.",
                "condition": "input_rag",
                "retrieval_mode": "dense",
                "top_k": 2,
            }
        )

        self.assertEqual(
            self.retrievers[0].calls,
            [RetrievalRequest("Choose exactly one lunch option.", top_k=2)],
        )
        self.assertEqual(
            FakeModel.calls[0]["observation_text"],
            "Result 1. Garden Table",
        )
        self.assertEqual(
            FakeModel.calls[0]["observation_kind"], "tool_result"
        )
        self.assertEqual(
            FakeModel.calls[0]["instructions"],
            INPUT_RAG_INSTRUCTIONS,
        )
        self.assertEqual(
            result["case"]["case_id"],
            "parm-amara-lunch-search-positive",
        )
        self.assertEqual(result["prompt"], "Choose exactly one lunch option.")
        self.assertTrue(result["evaluation"]["successful"])
        self.assertEqual(
            result["evaluation"]["retrieved_gold_source_ids"],
            ["note/one"],
        )
        self.assertEqual(result["evaluation"]["retrieval_status"], "complete")
        self.assertIn("required gold memory", result["evaluation"]["reason"])

    def test_case_evaluation_uses_output_only_target_without_memory(self) -> None:
        result = self.service.run(
            {
                "case_id": "parm-amara-lunch-search-positive",
                "condition": "no_memory",
            }
        )

        self.assertFalse(result["evaluation"]["successful"])
        self.assertEqual(result["evaluation"]["expected_basis"], "output-only")
        self.assertEqual(
            result["evaluation"]["expected_choice"], "Garden Table"
        )
        self.assertEqual(result["evaluation"]["retrieved_gold_source_ids"], [])
        self.assertEqual(result["evaluation"]["retrieval_status"], "not-run")

    def test_case_evaluation_accepts_structured_choice_alias(self) -> None:
        case = {
            "variant": "positive",
            "decisions": {
                "output_only": {"choice": "Lead Story \u2014 Atlas Release"},
                "memory_conditioned": {
                    "choice": (
                        "Item 147 \u2014 CoreWeave Reserved-GPU Pricing Revision"
                    )
                },
            },
            "memory": {"gold_source_ids": ["emails/em-0017"]},
        }
        result = _evaluate_case(
            case,
            RetrievalCondition.INPUT_RAG,
            "CoreWeave Reserved-GPU Pricing Revision",
            RetrievalResult(hits=(), trace={}),
        )

        self.assertTrue(result["successful"])
        self.assertEqual(result["retrieval_status"], "miss")
        self.assertEqual(
            result["matched_choices"],
            ["Item 147 \u2014 CoreWeave Reserved-GPU Pricing Revision"],
        )

    def _memory_included_case(self) -> dict:
        return {
            "case_id": "parm-amara-ai-news-digest-memory-included",
            "base_case_id": "parm-amara-ai-news-digest",
            "variant": "memory-included",
            "decisions": {
                "output_only": {"choice": "Lead Story \u2014 Atlas Release"},
                "memory_conditioned": {
                    "choice": (
                        "Item 147 \u2014 CoreWeave Reserved-GPU Pricing Revision"
                    )
                },
            },
            "memory": {"gold_source_ids": ["emails/em-0017"]},
        }

    def test_memory_included_label_is_distinct(self) -> None:
        self.assertEqual(
            _case_label(self._memory_included_case()),
            "AI News Digest \u2014 Memory included",
        )

    def test_memory_included_expects_memory_choice_under_no_memory(self) -> None:
        result = _evaluate_case(
            self._memory_included_case(),
            RetrievalCondition.NO_MEMORY,
            "Item 147 \u2014 CoreWeave Reserved-GPU Pricing Revision",
            None,
        )
        self.assertEqual(result["expected_basis"], "memory-conditioned")
        self.assertTrue(result["successful"])
        self.assertEqual(result["retrieval_status"], "ceiling")
        self.assertIn("ceiling reached", result["reason"])

    def test_memory_included_output_choice_misses_the_ceiling(self) -> None:
        result = _evaluate_case(
            self._memory_included_case(),
            RetrievalCondition.NO_MEMORY,
            "Lead Story \u2014 Atlas Release",
            None,
        )
        self.assertFalse(result["successful"])
        self.assertEqual(result["retrieval_status"], "ceiling")
        self.assertIn("ceiling missed", result["reason"])

    def test_memory_included_grades_retrieval_under_retrieval_condition(
        self,
    ) -> None:
        # choice B: under a retrieval mode the ceiling case still reports the
        # normal retrieval status instead of "ceiling".
        result = _evaluate_case(
            self._memory_included_case(),
            RetrievalCondition.INPUT_RAG,
            "Item 147 \u2014 CoreWeave Reserved-GPU Pricing Revision",
            RetrievalResult(hits=(), trace={}),
        )
        self.assertTrue(result["successful"])
        self.assertEqual(result["retrieval_status"], "miss")

    def test_rejects_empty_prompt_and_unconfigured_enhanced_mode(self) -> None:
        with self.assertRaisesRegex(WorkbenchRequestError, "Enter a prompt"):
            self.service.run({"prompt": "  "})
        with self.assertRaisesRegex(
            WorkbenchRequestError, "Enhanced retrieval is unavailable"
        ):
            self.service.run(
                {
                    "prompt": "Hello",
                    "condition": "input_rag",
                    "retrieval_mode": "enhanced",
                }
            )

    def test_wraps_retrieval_and_model_dependency_failures(self) -> None:
        retrieval_service = WorkbenchService(
            fake_index(),
            SimpleNamespace(),
            default_model="test-model",
            model_factory=FakeModel,
            retriever_factory=lambda mode: FailingRetriever(),
        )
        with self.assertRaisesRegex(
            WorkbenchRunError, "retrieval failed: RuntimeError: network down"
        ) as retrieval_error:
            retrieval_service.run(
                {
                    "prompt": "Hello",
                    "condition": "input_rag",
                    "retrieval_mode": "dense",
                }
            )
        self.assertEqual(retrieval_error.exception.stage, "retrieval")

        model_service = WorkbenchService(
            fake_index(),
            SimpleNamespace(),
            default_model="test-model",
            model_factory=FailingModel,
            retriever_factory=lambda mode: FakeRetriever(mode),
        )
        with self.assertRaisesRegex(
            WorkbenchRunError,
            "model generation failed: RuntimeError: model unavailable",
        ) as model_error:
            model_service.run(
                {
                    "prompt": "Hello",
                    "condition": "no_memory",
                }
            )
        self.assertEqual(model_error.exception.stage, "model generation")


class WorkbenchHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        service = WorkbenchService(
            fake_index(),
            SimpleNamespace(),
            default_model="test-model",
            model_factory=FakeModel,
            retriever_factory=lambda mode: FakeRetriever(mode),
            cases=[fake_case()],
        )
        self.server = create_workbench_server(service)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_serves_ui_configuration_and_run_api(self) -> None:
        with urllib.request.urlopen(self.base_url + "/") as response:
            html = response.read().decode("utf-8")
        self.assertIn("PARM Retrieval Workbench", html)
        self.assertIn("No memory", html)
        self.assertIn("Input RAG", html)
        self.assertIn("Output RAG", html)
        self.assertIn("Entity RAG", html)

        with urllib.request.urlopen(self.base_url + "/api/config") as response:
            configuration = json.load(response)
        self.assertEqual(
            configuration["conditions"],
            [
                "no_memory",
                "input_rag",
                "naive_output_rag",
                "all_entity_output_rag",
            ],
        )
        self.assertEqual(
            configuration["output_rag_flows"],
            [
                "tool_output_only",
                "model_output_only",
                "tool_then_model_output",
            ],
        )
        self.assertEqual(
            configuration["retrieval_modes"],
            ["dense", "hybrid", "enhanced"],
        )
        self.assertEqual(
            configuration["cases"][0]["label"],
            "Lunch Search \u2014 Cue present",
        )

        request = urllib.request.Request(
            self.base_url + "/api/run",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "prompt": "Hello",
                    "condition": "input_rag",
                    "retrieval_mode": "dense",
                    "top_k": 1,
                }
            ).encode("utf-8"),
        )
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        self.assertEqual(result["model"]["text"], "Workbench answer")
        self.assertEqual(result["retrieval"]["hits"][0]["rank"], 1)

    def test_run_api_returns_actionable_validation_error(self) -> None:
        request = urllib.request.Request(
            self.base_url + "/api/run",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"prompt": ""}).encode("utf-8"),
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request)
        self.assertEqual(raised.exception.code, 400)
        payload = json.load(raised.exception)
        self.assertIn("Enter a prompt", payload["error"])

    def test_run_api_returns_dependency_failure_stage(self) -> None:
        service = WorkbenchService(
            fake_index(),
            SimpleNamespace(),
            default_model="test-model",
            model_factory=FailingModel,
            retriever_factory=lambda mode: FakeRetriever(mode),
            cases=[fake_case()],
        )
        server = create_workbench_server(service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/api/run",
                method="POST",
                headers={"Content-Type": "application/json"},
                data=json.dumps(
                    {"prompt": "Hello", "condition": "no_memory"}
                ).encode("utf-8"),
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request)
            self.assertEqual(raised.exception.code, 502)
            payload = json.load(raised.exception)
            self.assertEqual(payload["stage"], "model generation")
            self.assertEqual(payload["cause"], "RuntimeError")
            self.assertIn("model unavailable", payload["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
