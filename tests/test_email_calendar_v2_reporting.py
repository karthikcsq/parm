from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_email_calendar_v2_results.py"


def load_module():
    spec = importlib.util.spec_from_file_location("email_calendar_v2_results", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load reporting script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmailCalendarV2ReportingTest(unittest.TestCase):
    def test_rates_report_each_triplet_and_cue_and_memory_lifts(self) -> None:
        module = load_module()
        runs = {
            "no_memory": [
                {"rows": [
                    {"base_case_id": "atlas", "variant": "positive", "decisive_success": True},
                    {"base_case_id": "atlas", "variant": "cue-ablated", "decisive_success": False},
                    {"base_case_id": "atlas", "variant": "memory-included", "decisive_success": True},
                ]},
                {"rows": [
                    {"base_case_id": "atlas", "variant": "positive", "decisive_success": False},
                    {"base_case_id": "atlas", "variant": "cue-ablated", "decisive_success": False},
                    {"base_case_id": "atlas", "variant": "memory-included", "decisive_success": True},
                ]},
            ],
            "parm": [
                {"rows": [
                    {"base_case_id": "atlas", "variant": "positive", "decisive_success": True},
                    {"base_case_id": "atlas", "variant": "cue-ablated", "decisive_success": False},
                    {"base_case_id": "atlas", "variant": "memory-included", "decisive_success": True},
                ]},
                {"rows": [
                    {"base_case_id": "atlas", "variant": "positive", "decisive_success": True},
                    {"base_case_id": "atlas", "variant": "cue-ablated", "decisive_success": True},
                    {"base_case_id": "atlas", "variant": "memory-included", "decisive_success": True},
                ]},
            ],
        }

        report = module.build_report(runs, baseline="no_memory", parm="parm")

        no_memory = report["per_triplet"]["atlas"]["no_memory"]
        self.assertEqual(no_memory["positive"], {"successes": 1, "samples": 2, "rate": 0.5})
        self.assertEqual(no_memory["control"], {"successes": 0, "samples": 2, "rate": 0.0})
        self.assertEqual(no_memory["oracle"], {"successes": 2, "samples": 2, "rate": 1.0})
        self.assertEqual(no_memory["cue_triggered_lift"], 0.5)
        self.assertEqual(report["aggregate"]["parm_minus_no_memory_positive"], 0.5)
        self.assertIn("fresh independent samples", report["interpretation"])
        self.assertIn("preregistered threshold", report["interpretation"])


if __name__ == "__main__":
    unittest.main()
