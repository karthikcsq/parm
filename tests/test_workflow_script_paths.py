from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkflowScriptPathTests(unittest.TestCase):
    def test_fairness_check_uses_explicit_dataset_and_index(self) -> None:
        fairness = _load_script("evaluate_workflows_v1_fairness.py")
        dataset = Path("email-calendar-dataset")
        index_path = Path("email-calendar-index")
        index = type("Index", (), {"pages": [object(), object()]})()
        stream = io.StringIO()

        with (
            patch.object(fairness, "load_dotenv"),
            patch.object(fairness, "load_workflow_cases", return_value=[]) as load_cases,
            patch.object(fairness, "validate_workflow_cases") as validate_cases,
            patch.object(fairness.RetrievalIndex, "load", return_value=index) as load_index,
            patch.object(fairness, "OpenAIEmbedder", return_value=object()),
            patch.object(fairness, "IndexRetriever"),
            patch("sys.argv", ["fairness", "--dataset", str(dataset), "--index", str(index_path)]),
            redirect_stdout(stream),
        ):
            self.assertEqual(fairness.main(), 0)

        load_cases.assert_called_once_with(dataset)
        validate_cases.assert_called_once_with([])
        load_index.assert_called_once_with(index_path)
        self.assertIn("email-calendar-index", stream.getvalue())

    def test_fairness_defaults_keep_the_v1_dataset_and_tier_output(self) -> None:
        fairness = _load_script("evaluate_workflows_v1_fairness.py")
        index = type("Index", (), {"pages": []})()
        stream = io.StringIO()

        with (
            patch.object(fairness, "load_dotenv"),
            patch.object(fairness, "load_workflow_cases", return_value=[]) as load_cases,
            patch.object(fairness, "validate_workflow_cases"),
            patch.object(fairness.RetrievalIndex, "load", return_value=index) as load_index,
            patch.object(fairness, "OpenAIEmbedder", return_value=object()),
            patch.object(fairness, "IndexRetriever"),
            patch("sys.argv", ["fairness"]),
            redirect_stdout(stream),
        ):
            self.assertEqual(fairness.main(), 0)

        load_cases.assert_called_once_with(fairness.DATASET)
        load_index.assert_called_once_with(fairness.TIER_INDEXES["tier-100"])
        self.assertEqual(
            stream.getvalue(),
            "tier: tier-100, 0 records\n\n\nNo goal reaches a gold source in the top 5.\n",
        )

    def test_summary_labels_an_explicit_dataset(self) -> None:
        summary = _load_script("summarize_workflows_v1_results.py")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tier = root / "tier-email-calendar"
            tier.mkdir()
            (tier / "no_memory.sample0.metrics.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "base_case_id": "email-contract-review",
                                "variant": "positive",
                                "decisive_success": True,
                                "gold_admitted_count": 0,
                                "spurious_admitted_count": 0,
                                "timely_gold_admission": False,
                                "injected_memory_tokens": 0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stream = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "summary",
                        "--results",
                        str(root),
                        "--tier",
                        "tier-email-calendar",
                        "--dataset",
                        "data/workflows_email_calendar_v1",
                    ],
                ),
                redirect_stdout(stream),
            ):
                self.assertEqual(summary.main(), 0)

        self.assertIn("Dataset: data/workflows_email_calendar_v1", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
