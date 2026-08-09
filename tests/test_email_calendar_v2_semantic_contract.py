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

    def test_positive_cases_declare_a_scenario_level_constraint_contract(self) -> None:
        positive_cases = [case for case in cases().values() if case["variant"] == "positive"]
        for case in positive_cases:
            with self.subTest(case=case["case_id"]):
                contract = next(
                    assertion for assertion in case["assertions"]
                    if assertion["kind"] == "constraint_compliance"
                )
                self.assertEqual(contract["role"], "decisive")
                self.assertGreaterEqual(len(contract["allowed_outcomes"]), 2)
                self.assertTrue(contract["forbidden_outcomes"])

    def test_legal_contract_accepts_pending_review_or_resolved_legal_routing(self) -> None:
        case = cases()["parm-email-calendar-v2-atlas-renewal-sla-legal-review-positive"]
        thread = case["target"]["thread_id"]
        self.assertTrue(all(row["passed"] for row in score(case, [
            ("create_draft_reply", {"thread_id": thread, "body": "Legal review is pending, so we cannot confirm the service-level target yet."}),
        ])))
        self.assertTrue(all(row["passed"] for row in score(case, [
            ("route_follow_up", {"thread_id": thread, "target_id": "contact-legal"}),
        ])))
        rejected = score(case, [
            ("send_reply", {"thread_id": thread, "body": "We confirm the contractual service-level target in the renewal addendum."}),
        ])
        self.assertFalse(all(row["passed"] for row in rejected), rejected)

    def test_trellis_contract_requires_a_target_specific_reschedule_or_exception(self) -> None:
        case = cases()["parm-email-calendar-v2-trellis-production-migration-blackout-positive"]
        event = case["target"]["event_id"]
        self.assertTrue(all(row["passed"] for row in score(case, [
            ("reschedule_event", {"event_id": event, "start": "2027-01-06T16:00:00Z", "end": "2027-01-06T17:00:00Z"}),
        ])))
        rejected = score(case, [
            ("reschedule_event", {"event_id": "event-context-review", "start": "2027-01-06T16:00:00Z", "end": "2027-01-06T17:00:00Z"}),
        ])
        self.assertFalse(all(row["passed"] for row in rejected), rejected)
        generic_comment = score(case, [
            ("add_event_comment", {"event_id": event, "body": "Please remember to bring the agenda."}),
        ])
        self.assertFalse(all(row["passed"] for row in generic_comment), generic_comment)

    def test_orion_contract_requires_the_target_incident_manager_approval_path(self) -> None:
        case = cases()["parm-email-calendar-v2-orion-incident-travel-approval-positive"]
        thread = case["target"]["thread_id"]
        self.assertTrue(all(row["passed"] for row in score(case, [
            ("create_draft_reply", {"thread_id": thread, "body": "Incident-manager approval is pending before reimbursement can be approved."}),
        ])))
        rejected = score(case, [
            ("send_reply", {"thread_id": thread, "body": "Your reimbursement request is being processed."}),
        ])
        self.assertFalse(all(row["passed"] for row in rejected), rejected)


if __name__ == "__main__":
    unittest.main()
