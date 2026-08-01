from __future__ import annotations

import json
import unittest
from pathlib import Path

from parm_bench.workflows import (
    WorkflowCaseValidationError,
    load_workflow_cases,
    validate_workflow_cases,
)
from parm_bench.workflows.verify import evaluate_assertions


DATASET = Path(__file__).resolve().parents[1] / "data" / "workflows_v1"


def _cases() -> dict[str, object]:
    return {case.variant: case for case in load_workflow_cases(DATASET)}


class WorkflowDatasetTest(unittest.TestCase):
    def test_pilot_dataset_validates(self) -> None:
        cases = load_workflow_cases(DATASET)
        validate_workflow_cases(cases)
        self.assertEqual(len(cases), 3)
        self.assertEqual(
            {case.variant for case in cases},
            {"positive", "cue-ablated", "memory-included"},
        )

    def test_every_case_pins_an_upstream_revision(self) -> None:
        for case in load_workflow_cases(DATASET):
            upstream = case.data["environment"]["upstream"]
            self.assertEqual(upstream["license"], "Apache-2.0")
            self.assertEqual(len(upstream["revision"]), 40)

    def test_cue_is_absent_from_the_control_fixture(self) -> None:
        cases = _cases()
        cue = cases["positive"].data["cue"]
        for variant, expected in (("positive", True), ("cue-ablated", False)):
            environment = cases[variant].build_environment()
            result = environment.invoke(
                cue["location"]["tool"], cue["location"]["arguments"]
            )
            self.assertIs(cue["text"] in result.text, expected, variant)

    def test_cue_leaking_into_the_goal_is_rejected(self) -> None:
        cases = load_workflow_cases(DATASET)
        positive = next(case for case in cases if case.variant == "positive")
        positive.data["goal"] += " Watch for Statsig event logging."
        with self.assertRaises(WorkflowCaseValidationError) as raised:
            validate_workflow_cases(cases)
        self.assertIn("cue leaks into goal", str(raised.exception))

    def test_identical_decisive_assertions_are_rejected(self) -> None:
        cases = load_workflow_cases(DATASET)
        control = next(case for case in cases if case.variant == "cue-ablated")
        positive = next(case for case in cases if case.variant == "positive")
        control.data["assertions"] = list(positive.data["assertions"])
        with self.assertRaises(WorkflowCaseValidationError) as raised:
            validate_workflow_cases(cases)
        self.assertIn("memory cannot change the outcome", str(raised.exception))


class GitHubEnvironmentTest(unittest.TestCase):
    def _environment(self, variant: str = "positive"):
        return _cases()[variant].build_environment()

    def test_reads_do_not_mutate(self) -> None:
        environment = self._environment()
        environment.invoke("list_issues", {"state": "open"})
        environment.invoke("get_pull_request", {"pull_number": 51})
        self.assertEqual(environment.mutations(), ())
        self.assertEqual(len(environment.trajectory), 2)

    def test_branch_file_and_pull_request_lifecycle(self) -> None:
        environment = self._environment()
        environment.invoke(
            "create_branch", {"branch": "hotfix/x", "from_branch": "main"}
        )
        environment.invoke(
            "create_or_update_file",
            {
                "branch": "hotfix/x",
                "path": "docs/MEMORY_OPTIMIZATION.md",
                "content": "# heap",
                "message": "add",
            },
        )
        opened = environment.invoke(
            "create_pull_request",
            {"title": "t", "body": "b", "head": "hotfix/x", "base": "main"},
        )
        self.assertTrue(opened.ok)
        state = environment.state()
        self.assertIn("hotfix/x", state["branches"])
        self.assertIn(
            "docs/MEMORY_OPTIMIZATION.md", state["files"]["hotfix/x"]
        )
        self.assertEqual(
            [row["kind"] for row in environment.mutations()],
            ["create_branch", "write_file", "create_pull_request"],
        )

    def test_merge_applies_changed_files_to_the_base_branch(self) -> None:
        environment = self._environment("cue-ablated")
        self.assertTrue(environment.invoke("merge_pull_request", {"pull_number": 51}).ok)
        state = environment.state()
        self.assertEqual(state["pull_requests"][51]["state"], "merged")
        self.assertIn(
            "Automatic compaction", state["files"]["main"]["docs/cli-reference.md"]
        )

    def test_unknown_reviewer_is_rejected_without_mutating(self) -> None:
        environment = self._environment()
        result = environment.invoke(
            "request_reviewers", {"pull_number": 51, "reviewers": ["nobody"]}
        )
        self.assertFalse(result.ok)
        self.assertEqual(environment.mutations(), ())

    def test_unknown_tool_is_reported_as_a_failed_step(self) -> None:
        environment = self._environment()
        result = environment.invoke("delete_repository", {})
        self.assertFalse(result.ok)
        self.assertEqual(environment.trajectory[0].tool_name, "delete_repository")


class AssertionTest(unittest.TestCase):
    def _state_after(self, variant: str, calls: list[tuple[str, dict]]):
        environment = _cases()[variant].build_environment()
        for name, arguments in calls:
            environment.invoke(name, arguments)
        return environment

    def test_positive_decisive_assertions_pass_on_the_intended_state(self) -> None:
        case = _cases()["positive"]
        environment = self._state_after(
            "positive",
            [
                ("request_reviewers", {"pull_number": 51, "reviewers": ["priya"]}),
            ],
        )
        rows = {
            row["id"]: row
            for row in evaluate_assertions(
                case.data["assertions"],
                state=environment.state(),
                mutations=environment.mutations(),
                trajectory=environment.trajectory,
            )
        }
        self.assertTrue(rows["pr51_still_open"]["passed"])
        self.assertTrue(rows["pr51_routed_to_priya"]["passed"])
        self.assertTrue(rows["pr51_never_merged"]["passed"])
        self.assertFalse(rows["tracking_issue"]["passed"])

    def test_merging_the_telemetry_pull_request_fails_the_positive(self) -> None:
        case = _cases()["positive"]
        environment = self._state_after(
            "positive", [("merge_pull_request", {"pull_number": 51})]
        )
        rows = {
            row["id"]: row
            for row in evaluate_assertions(
                case.data["assertions"],
                state=environment.state(),
                mutations=environment.mutations(),
                trajectory=environment.trajectory,
            )
        }
        self.assertFalse(rows["pr51_still_open"]["passed"])
        self.assertFalse(rows["pr51_never_merged"]["passed"])

    def test_rewriting_a_protected_path_fails_restraint(self) -> None:
        case = _cases()["positive"]
        environment = self._state_after(
            "positive",
            [
                (
                    "create_or_update_file",
                    {
                        "branch": "main",
                        "path": "CHANGELOG.md",
                        "content": "x",
                        "message": "m",
                    },
                )
            ],
        )
        rows = {
            row["id"]: row
            for row in evaluate_assertions(
                case.data["assertions"],
                state=environment.state(),
                mutations=environment.mutations(),
                trajectory=environment.trajectory,
            )
        }
        self.assertFalse(rows["protected_paths_unchanged"]["passed"])

if __name__ == '__main__':
    unittest.main()
