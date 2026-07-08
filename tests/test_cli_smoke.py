from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parm_bench.baselines import available_baselines
from parm_bench.cli import _resolve_model, main
from parm_bench.models import (
    FINAL_ANSWER_INSTRUCTIONS,
    ModelResponse,
    ModelTruncationError,
)
from parm_bench.retrieval import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    RetrievalHit,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "benchmark_v1"


class FakeOpenAIModel:
    instances: list["FakeOpenAIModel"] = []

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calls: list[dict[str, object]] = []
        self.__class__.instances.append(self)

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
            text="Unscored fake response",
            response_id=f"resp_{len(self.calls)}",
            resolved_model=f"{self.model_name}-resolved",
            usage={"input_tokens": 100, "output_tokens": 4, "total_tokens": 104},
        )


class FakeIndex:
    path = Path("fixture-index").resolve()
    manifest_hash = "manifest-hash"
    manifest = {
        "corpus_id": "amara-life-v1",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "gbrain_version": "test",
        "chunker_version": "test",
    }


class FakeRetriever:
    instances: list["FakeRetriever"] = []

    def __init__(self, index: FakeIndex, mode: str, embedder: object, **kwargs: object):
        self.index = index
        self.mode = RetrievalMode(mode)
        self.expander = kwargs.get("expander")
        self.calls: list[RetrievalRequest] = []
        self.__class__.instances.append(self)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.calls.append(request)
        return RetrievalResult(
            (
                RetrievalHit(
                    page_id="page/retrieved",
                    source_id="fixture-source",
                    slug="note/retrieved",
                    title="Retrieved",
                    chunk_id="page/retrieved:0",
                    text="Retrieved memory",
                    score=0.9,
                    rank=1,
                ),
            ),
            {
                "retrieval_mode": self.mode.value,
                "original_query": request.query,
                "expansion_queries": [],
                "candidate_lists": {
                    "dense:original": ["page/retrieved"]
                },
                "returned_pages": [],
            },
        )


class CliSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeOpenAIModel.instances.clear()
        FakeRetriever.instances.clear()

    def test_validate_and_inspect(self) -> None:
        self.assertEqual(main(["validate", str(DATASET)]), 0)
        with patch("builtins.print") as output:
            self.assertEqual(
                main(
                    [
                        "inspect",
                        str(DATASET),
                        "--case",
                        "parm-amara-conference-agenda-positive",
                    ]
                ),
                0,
            )
        rendered = "\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("conference-agenda-positive", rendered)

    def test_no_memory_baseline_is_registered(self) -> None:
        self.assertEqual(
            set(available_baselines()),
            {"no_memory", "input_rag", "naive_output_rag"},
        )

    def test_run_refuses_unimplemented_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.jsonl"
            with patch("sys.stderr"):
                status = main(
                    [
                        "run",
                        str(DATASET),
                        "--baseline",
                        "not_real",
                        "--out",
                        str(result),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertFalse(result.exists())

    def test_no_memory_run_calls_model_once_for_all_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.jsonl"
            with (
                patch(
                    "parm_bench.cli.OpenAIResponsesModel",
                    FakeOpenAIModel,
                ),
                patch(
                    "parm_bench.cli.IndexRetriever",
                    side_effect=AssertionError("retriever must not be instantiated"),
                ),
            ):
                status = main(
                    [
                        "run",
                        str(DATASET),
                        "--baseline",
                        "no_memory",
                        "--model",
                        "chosen-model",
                        "--out",
                        str(result),
                    ]
                )

            self.assertEqual(status, 0)
            model = FakeOpenAIModel.instances[0]
            self.assertEqual(model.model_name, "chosen-model")
            self.assertEqual(len(model.calls), 15)
            rows = [
                json.loads(line)
                for line in result.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 15)
            self.assertTrue(all(row["baseline"] == "no_memory" for row in rows))
            self.assertTrue(
                all(row["trace"]["admitted_source_ids"] == [] for row in rows)
            )
            configuration = json.loads(
                result.with_suffix(".config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(configuration["baseline"], "no_memory")
            self.assertIsNone(configuration["retrieval_limit"])

    def test_variant_filter_restricts_the_run_and_records_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.jsonl"
            with (
                patch("parm_bench.cli.OpenAIResponsesModel", FakeOpenAIModel),
                patch(
                    "parm_bench.cli.IndexRetriever",
                    side_effect=AssertionError("retriever must not be instantiated"),
                ),
            ):
                status = main(
                    [
                        "run",
                        str(DATASET),
                        "--baseline",
                        "no_memory",
                        "--variant",
                        "memory-included",
                        "--out",
                        str(result),
                    ]
                )

            self.assertEqual(status, 0)
            rows = [
                json.loads(line)
                for line in result.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 5)
            self.assertTrue(
                all(
                    row["case_id"].endswith("-memory-included") for row in rows
                )
            )
            configuration = json.loads(
                result.with_suffix(".config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(configuration["variants"], ["memory-included"])

    def test_input_rag_run_uses_configured_limit_and_writes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.jsonl"
            with (
                patch(
                    "parm_bench.cli.OpenAIResponsesModel",
                    FakeOpenAIModel,
                ),
                patch(
                    "parm_bench.cli.RetrievalIndex.load",
                    return_value=FakeIndex(),
                ),
                patch(
                    "parm_bench.cli.OpenAIEmbedder",
                    return_value=object(),
                ),
                patch(
                    "parm_bench.cli.IndexRetriever",
                    FakeRetriever,
                ),
            ):
                status = main(
                    [
                        "run",
                        str(DATASET),
                        "--baseline",
                        "input_rag",
                        "--retrieval-mode",
                        "dense",
                        "--retrieval-index",
                        "fixture-index",
                        "--retrieval-limit",
                        "3",
                        "--model",
                        "chosen-model",
                        "--out",
                        str(result),
                    ]
                )

            self.assertEqual(status, 0)
            retriever = FakeRetriever.instances[0]
            self.assertEqual(len(retriever.calls), 15)
            self.assertTrue(all(call.top_k == 3 for call in retriever.calls))
            model = FakeOpenAIModel.instances[0]
            self.assertTrue(
                all(
                    call["memory_context"] == "1. Retrieved memory"
                    for call in model.calls
                )
            )
            rows = [
                json.loads(line)
                for line in result.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(
                all(
                    row["trace"]["admitted_source_ids"]
                    == ["note/retrieved"]
                    for row in rows
                )
            )
            configuration = json.loads(
                result.with_suffix(".config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(configuration["retrieval_mode"], "dense")
            self.assertEqual(
                configuration["retrieval_manifest_hash"], "manifest-hash"
            )
            self.assertEqual(configuration["retrieval_limit"], 3)
            self.assertEqual(configuration["admission_policy"], "all_retrieved")
            self.assertIsNone(configuration["output_rag_flow"])
            self.assertNotIn("memory_backend", configuration)

    def test_naive_output_rag_run_requires_and_records_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.jsonl"
            with (
                patch(
                    "parm_bench.cli.OpenAIResponsesModel",
                    FakeOpenAIModel,
                ),
                patch(
                    "parm_bench.cli.RetrievalIndex.load",
                    return_value=FakeIndex(),
                ),
                patch(
                    "parm_bench.cli.OpenAIEmbedder",
                    return_value=object(),
                ),
                patch(
                    "parm_bench.cli.IndexRetriever",
                    FakeRetriever,
                ),
            ):
                status = main(
                    [
                        "run",
                        str(DATASET),
                        "--baseline",
                        "naive_output_rag",
                        "--output-rag-flow",
                        "tool_output_only",
                        "--retrieval-mode",
                        "dense",
                        "--retrieval-index",
                        "fixture-index",
                        "--retrieval-limit",
                        "2",
                        "--out",
                        str(result),
                    ]
                )

            self.assertEqual(status, 0)
            retriever = FakeRetriever.instances[0]
            self.assertEqual(len(retriever.calls), 15)
            self.assertTrue(
                all(call.top_k == 2 for call in retriever.calls)
            )
            rows = [
                json.loads(line)
                for line in result.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(
                all(row["baseline"] == "naive_output_rag" for row in rows)
            )
            self.assertTrue(
                all(
                    row["output_rag_flow"] == "tool_output_only"
                    for row in rows
                )
            )
            self.assertTrue(
                all(
                    row["trace"]["output_rag_flow"] == "tool_output_only"
                    for row in rows
                )
            )
            configuration = json.loads(
                result.with_suffix(".config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                configuration["output_rag_flow"], "tool_output_only"
            )
            self.assertEqual(configuration["retrieval_mode"], "dense")

    def test_memory_run_requires_explicit_mode_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stderr"):
            status = main(
                [
                    "run",
                    str(DATASET),
                    "--baseline",
                    "input_rag",
                    "--out",
                    str(Path(tmp) / "result.jsonl"),
                ]
            )
        self.assertEqual(status, 2)

    def test_naive_output_rag_requires_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stderr"):
            status = main(
                [
                    "run",
                    str(DATASET),
                    "--baseline",
                    "naive_output_rag",
                    "--retrieval-mode",
                    "dense",
                    "--retrieval-index",
                    "fixture-index",
                    "--out",
                    str(Path(tmp) / "result.jsonl"),
                ]
            )
        self.assertEqual(status, 2)

    def test_output_rag_flow_is_rejected_for_other_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stderr"):
            status = main(
                [
                    "run",
                    str(DATASET),
                    "--baseline",
                    "input_rag",
                    "--output-rag-flow",
                    "tool_output_only",
                    "--retrieval-mode",
                    "dense",
                    "--retrieval-index",
                    "fixture-index",
                    "--out",
                    str(Path(tmp) / "result.jsonl"),
                ]
            )
        self.assertEqual(status, 2)

    def test_no_memory_rejects_retrieval_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("sys.stderr"):
            status = main(
                [
                    "run",
                    str(DATASET),
                    "--baseline",
                    "no_memory",
                    "--retrieval-mode",
                    "dense",
                    "--out",
                    str(Path(tmp) / "result.jsonl"),
                ]
            )
        self.assertEqual(status, 2)

    def test_cli_loads_repo_dotenv_without_overriding_environment(self) -> None:
        with patch("parm_bench.cli.load_dotenv") as load:
            self.assertEqual(main(["validate", str(DATASET)]), 0)
        load.assert_called_once()
        self.assertFalse(load.call_args.kwargs["override"])

    def test_serve_workbench_delegates_configuration(self) -> None:
        with patch("parm_bench.cli.serve_workbench") as serve:
            status = main(
                [
                    "serve-workbench",
                    "--retrieval-index",
                    "fixture-index",
                    "--port",
                    "9876",
                    "--model",
                    "chosen-model",
                    "--expansion-cache",
                    "fixture-cache",
                    "--expansion-policy",
                    "populate",
                    "--no-open",
                ]
            )

        self.assertEqual(status, 0)
        serve.assert_called_once_with(
            retrieval_index="fixture-index",
            dataset_dir=DATASET.resolve(),
            host="127.0.0.1",
            port=9876,
            model_name="chosen-model",
            expansion_cache="fixture-cache",
            expansion_policy="populate",
            open_browser=False,
        )

    def test_model_precedence(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_resolve_model(None), "gpt-5-mini")
        with patch.dict(os.environ, {"PARM_OPENAI_MODEL": "env-model"}):
            self.assertEqual(_resolve_model(None), "env-model")
            self.assertEqual(_resolve_model("cli-model"), "cli-model")

    def test_failed_request_does_not_publish_partial_results(self) -> None:
        class FailingModel(FakeOpenAIModel):
            def generate(self, **kwargs: str) -> ModelResponse:
                if self.calls:
                    raise RuntimeError("API failure")
                return super().generate(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.jsonl"
            with patch("parm_bench.cli.OpenAIResponsesModel", FailingModel):
                with self.assertRaisesRegex(RuntimeError, "API failure"):
                    main(
                        [
                            "run",
                            str(DATASET),
                            "--baseline",
                            "no_memory",
                            "--out",
                            str(result),
                        ]
                    )
            self.assertFalse(result.exists())
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_truncated_case_is_recorded_and_run_completes(self) -> None:
        class TruncatingModel(FakeOpenAIModel):
            def generate(self, **kwargs: str) -> ModelResponse:
                response = super().generate(**kwargs)
                if len(self.calls) == 2:
                    raise ModelTruncationError(
                        "max_output_tokens", response.response_id
                    )
                return response

        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.jsonl"
            with (
                patch("parm_bench.cli.OpenAIResponsesModel", TruncatingModel),
                patch("sys.stderr"),
            ):
                status = main(
                    [
                        "run",
                        str(DATASET),
                        "--baseline",
                        "no_memory",
                        "--out",
                        str(result),
                    ]
                )

            self.assertEqual(status, 0)
            rows = [
                json.loads(line)
                for line in result.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 15)
            truncated = [row for row in rows if row.get("truncated")]
            self.assertEqual(len(truncated), 1)
            self.assertEqual(truncated[0]["response_text"], "")
            self.assertEqual(
                truncated[0]["truncation_reason"], "max_output_tokens"
            )
            self.assertEqual(truncated[0]["trace"]["admitted_source_ids"], [])
            self.assertEqual(truncated[0]["baseline"], "no_memory")

    def test_generate_command_is_removed(self) -> None:
        with self.assertRaises(SystemExit):
            main(["generate", str(DATASET)])


if __name__ == "__main__":
    unittest.main()
