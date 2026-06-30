from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parm_bench.baselines import available_baselines
from parm_bench.cli import _resolve_model, main
from parm_bench.models import ModelResponse


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "benchmark_v1"


class FakeOpenAIModel:
    instances: list["FakeOpenAIModel"] = []

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calls: list[dict[str, str]] = []
        self.__class__.instances.append(self)

    def generate(
        self,
        *,
        prompt: str,
        observation_kind: str,
        observation_text: str,
    ) -> ModelResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "observation_kind": observation_kind,
                "observation_text": observation_text,
            }
        )
        return ModelResponse(
            text="Unscored fake response",
            response_id=f"resp_{len(self.calls)}",
            resolved_model=f"{self.model_name}-resolved",
            usage={"input_tokens": 100, "output_tokens": 4, "total_tokens": 104},
        )


class CliSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeOpenAIModel.instances.clear()

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
        self.assertEqual(set(available_baselines()), {"no_memory"})

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

    def test_no_memory_run_calls_model_once_for_all_ten_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.jsonl"
            with (
                patch(
                    "parm_bench.cli.OpenAIResponsesModel",
                    FakeOpenAIModel,
                ),
                patch(
                    "parm_bench.cli.GBrainBackend",
                    side_effect=AssertionError("GBrain must not be instantiated"),
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
            self.assertEqual(len(model.calls), 10)
            rows = [
                json.loads(line)
                for line in result.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row["baseline"] == "no_memory" for row in rows))
            self.assertTrue(
                all(row["trace"]["admitted_source_ids"] == [] for row in rows)
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

    def test_generate_command_is_removed(self) -> None:
        with self.assertRaises(SystemExit):
            main(["generate", str(DATASET)])


if __name__ == "__main__":
    unittest.main()
