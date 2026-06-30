from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path

from parm_bench.baselines import (
    NoMemoryBaseline,
    benchmark_input,
)
from parm_bench.dataset import load_cases
from parm_bench.models import ModelResponse


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
    ) -> ModelResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "observation_kind": observation_kind,
                "observation_text": observation_text,
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


if __name__ == "__main__":
    unittest.main()
