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
    RetrievalHit,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
)
from parm_bench.workbench import (
    RetrievalCondition,
    WORKBENCH_INSTRUCTIONS,
    WorkbenchRequestError,
    WorkbenchService,
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

        def factory(mode: RetrievalMode) -> FakeRetriever:
            retriever = FakeRetriever(mode)
            self.retrievers.append(retriever)
            return retriever

        self.service = WorkbenchService(
            fake_index(),
            SimpleNamespace(),
            default_model="test-model",
            model_factory=FakeModel,
            retriever_factory=factory,
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
                "output_only": {"choice": "Lead Story — Atlas Release"},
                "memory_conditioned": {
                    "choice": (
                        "Item 147 — CoreWeave Reserved-GPU Pricing Revision"
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
            ["Item 147 — CoreWeave Reserved-GPU Pricing Revision"],
        )

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

        with urllib.request.urlopen(self.base_url + "/api/config") as response:
            configuration = json.load(response)
        self.assertEqual(
            configuration["conditions"], ["no_memory", "input_rag"]
        )
        self.assertEqual(
            configuration["retrieval_modes"],
            ["dense", "hybrid", "enhanced"],
        )
        self.assertEqual(
            configuration["cases"][0]["label"],
            "Lunch Search — Cue present",
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


if __name__ == "__main__":
    unittest.main()
