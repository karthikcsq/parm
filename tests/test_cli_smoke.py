from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parm_bench.baselines import available_baselines
from parm_bench.cli import main


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "benchmark_v1"


class CliSmokeTests(unittest.TestCase):
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

    def test_no_baselines_are_registered(self) -> None:
        self.assertEqual(available_baselines(), {})

    def test_run_refuses_unimplemented_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.jsonl"
            with patch("sys.stderr"):
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
            self.assertEqual(status, 2)
            self.assertFalse(result.exists())

    def test_generate_command_is_removed(self) -> None:
        with self.assertRaises(SystemExit):
            main(["generate", str(DATASET)])


if __name__ == "__main__":
    unittest.main()
