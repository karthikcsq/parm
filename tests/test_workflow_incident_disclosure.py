from __future__ import annotations

import unittest
from pathlib import Path

from parm_bench.workflows import load_workflow_cases
from parm_bench.workflows.verify import evaluate_assertions


DATASET = Path(__file__).resolve().parents[1] / "data" / "workflows_v1"
SCENARIO = "parm-workflow-github-incident-disclosure"


def _cases() -> dict[str, object]:
    return {
        case.variant: case
        for case in load_workflow_cases(DATASET)
        if case.base_case_id == SCENARIO
    }


class IncidentDisclosureScenarioTest(unittest.TestCase):
    def test_scenario_is_a_symmetric_triplet_with_a_late_customer_draft_cue(self) -> None:
        cases = _cases()
        self.assertEqual(set(cases), {"positive", "cue-ablated", "memory-included"})
        positive = cases["positive"]
        control = cases["cue-ablated"]
        included = cases["memory-included"]
        self.assertEqual(positive.goal, control.goal)
        self.assertEqual(positive.data["environment"]["fixture_path"], included.data["environment"]["fixture_path"])
        self.assertEqual(positive.data["cue"]["text"], "customer-facing draft")
        self.assertNotIn("customer-facing draft", positive.goal.casefold())

    def test_customer_draft_cue_requires_routing_wording_to_aditi(self) -> None:
        cases = _cases()
        positive = cases["positive"]
        environment = positive.build_environment()
        environment.invoke(
            "add_issue_comment",
            {
                "issue_number": 81,
                "body": "Routing the customer-facing draft to Aditi for wording.",
            },
        )
        results = {
            row["id"]: row
            for row in evaluate_assertions(
                positive.data["assertions"],
                state=environment.state(),
                mutations=environment.mutations(),
                trajectory=environment.trajectory,
            )
        }
        self.assertTrue(results["issue81_aditi_owns_customer_wording"]["passed"])

    def test_internal_draft_control_rejects_customer_comms_routing(self) -> None:
        cases = _cases()
        control = cases["cue-ablated"]
        environment = control.build_environment()
        environment.invoke(
            "add_issue_comment",
            {
                "issue_number": 81,
                "body": "Routing the customer-facing draft to Aditi for wording.",
            },
        )
        results = {
            row["id"]: row
            for row in evaluate_assertions(
                control.data["assertions"],
                state=environment.state(),
                mutations=environment.mutations(),
                trajectory=environment.trajectory,
            )
        }
        self.assertFalse(results["issue81_no_customer_comms_routing"]["passed"])


if __name__ == "__main__":
    unittest.main()
