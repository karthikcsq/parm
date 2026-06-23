from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parm_bench.baselines import available_baselines
from parm_bench.cli import main


class CliSmokeTests(unittest.TestCase):
    def test_core_baselines_run_on_three_case_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            self.assertEqual(main(["generate", str(dataset_dir), "--count", "3"]), 0)
            self.assertEqual(main(["validate", str(dataset_dir)]), 0)

            for baseline in available_baselines():
                results_path = Path(tmp) / f"{baseline}.jsonl"
                metrics_path = Path(tmp) / f"{baseline}.metrics.json"
                self.assertEqual(
                    main(["run", str(dataset_dir), "--baseline", baseline, "--out", str(results_path)]),
                    0,
                )
                self.assertEqual(main(["score", str(results_path), "--gold", str(dataset_dir), "--out", str(metrics_path)]), 0)
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                self.assertEqual(metrics["case_count"], 3)

    def test_inspect_prints_requested_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            self.assertEqual(main(["generate", str(dataset_dir), "--count", "3"]), 0)

            with patch("builtins.print") as mock_print:
                self.assertEqual(main(["inspect", str(dataset_dir), "--case", "parm-v1-002"]), 0)

            output = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
            self.assertIn("Case: parm-v1-002", output)
            self.assertIn("Gold path:", output)
            self.assertIn("Expected suggestion:", output)

    def test_inspect_missing_case_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            self.assertEqual(main(["generate", str(dataset_dir), "--count", "3"]), 0)
            self.assertEqual(main(["inspect", str(dataset_dir), "--case", "missing"]), 1)


if __name__ == "__main__":
    unittest.main()
