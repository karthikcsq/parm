from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from parm_bench.workflows import get_environment
from parm_bench.workflows.verify import evaluate_assertions


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_email_calendar_v2"
BUILDER = ROOT / "scripts" / "build_workflows_email_calendar_v2_cases.py"


def cases() -> dict[str, dict]:
    return {
        row["case_id"]: row
        for row in (
            json.loads(line)
            for line in (DATASET / "cases.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    }


def environment(case: dict):
    fixture = json.loads(
        (DATASET / case["environment"]["fixture_path"]).read_text(encoding="utf-8")
    )
    return get_environment("email_calendar_fixture", fixture)


def score(case: dict, calls: list[tuple[str, dict]]) -> list[dict]:
    env = environment(case)
    for tool, arguments in calls:
        result = env.invoke(tool, arguments)
        if not result.ok:
            raise AssertionError(result.text)
    return evaluate_assertions(
        case["assertions"], state=env.state(), mutations=env.mutations(), trajectory=env.trajectory
    )


class EmailCalendarV2SemanticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run = subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, text=True, capture_output=True)
        if run.returncode:
            raise AssertionError(run.stderr)

    def test_atlas_positive_scores_target_bound_legal_review_state(self) -> None:
        case = cases()["parm-email-calendar-v2-atlas-renewal-sla-legal-review-positive"]
        thread = case["target"]["thread_id"]

        rows = score(case, [
            ("set_thread_hold", {"thread_id": thread, "hold_kind": "review", "target_id": "contact-legal"}),
        ])

        self.assertTrue(all(row["passed"] for row in rows), rows)
        self.assertEqual([row["kind"] for row in rows], ["thread_hold_state"])

    def test_atlas_control_scores_not_taking_legal_review_state_without_canned_action(self) -> None:
        case = cases()["parm-email-calendar-v2-atlas-renewal-sla-legal-review-cue-ablated"]

        rows = score(case, [])

        self.assertTrue(all(row["passed"] for row in rows), rows)
        self.assertEqual([row["kind"] for row in rows], ["thread_not_on_hold"])

    def test_trellis_positive_scores_target_bound_change_window_exception_state(self) -> None:
        case = cases()["parm-email-calendar-v2-trellis-production-migration-blackout-positive"]
        event = case["target"]["event_id"]

        rows = score(case, [
            ("mark_event_pending_exception", {"event_id": event, "exception_type": "change_window"}),
        ])

        self.assertTrue(all(row["passed"] for row in rows), rows)
        self.assertEqual([row["kind"] for row in rows], ["event_pending_exception"])

    def test_trellis_control_scores_not_taking_change_window_exception_without_canned_action(self) -> None:
        case = cases()["parm-email-calendar-v2-trellis-production-migration-blackout-cue-ablated"]

        rows = score(case, [])

        self.assertTrue(all(row["passed"] for row in rows), rows)
        self.assertEqual([row["kind"] for row in rows], ["event_not_pending_exception"])

    def test_orion_positive_scores_target_bound_incident_manager_approval_state(self) -> None:
        case = cases()["parm-email-calendar-v2-orion-incident-travel-approval-positive"]
        thread = case["target"]["thread_id"]

        rows = score(case, [
            ("set_thread_hold", {"thread_id": thread, "hold_kind": "approval", "target_id": "contact-incident"}),
        ])

        self.assertTrue(all(row["passed"] for row in rows), rows)
        self.assertEqual([row["kind"] for row in rows], ["thread_hold_state"])

    def test_orion_control_scores_not_taking_incident_manager_approval_without_canned_action(self) -> None:
        case = cases()["parm-email-calendar-v2-orion-incident-travel-approval-cue-ablated"]

        rows = score(case, [])

        self.assertTrue(all(row["passed"] for row in rows), rows)
        self.assertEqual([row["kind"] for row in rows], ["thread_not_on_hold"])


if __name__ == "__main__":
    unittest.main()
