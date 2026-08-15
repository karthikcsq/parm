from __future__ import annotations

import json
import unittest
from pathlib import Path

from parm_bench.workflows import WorkflowCase, load_workflow_cases, validate_workflow_cases
from parm_bench.workflows.runner import assert_gold_reachable
from parm_bench.workflows.corpus_tiers import load_tiers
from parm_bench.workflows.verify import evaluate_assertions


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_v1"
CORPUS = DATASET / "corpora" / "workflow-eng-lead-v1"


def _cases() -> dict[str, dict[str, WorkflowCase]]:
    grouped: dict[str, dict[str, WorkflowCase]] = {}
    for case in load_workflow_cases(DATASET):
        grouped.setdefault(case.base_case_id, {})[case.variant] = case
    return grouped


class _FakeIndex:
    def __init__(self, slugs):
        self.path = Path("fake-index")
        self.pages = [type("Page", (), {"slug": slug})() for slug in slugs]


class ScenarioSetTest(unittest.TestCase):
    def test_every_scenario_is_a_complete_triplet(self) -> None:
        for base_case_id, variants in _cases().items():
            with self.subTest(base_case_id):
                self.assertEqual(
                    set(variants),
                    {"positive", "cue-ablated", "memory-included"},
                )

    def test_all_scenarios_validate(self) -> None:
        validate_workflow_cases(load_workflow_cases(DATASET))

    def test_rollback_canary_triplet_keeps_its_late_cue_and_symmetric_fixture(self) -> None:
        variants = _cases()["parm-workflow-github-canary-rollback"]
        positive = variants["positive"]
        ablated = variants["cue-ablated"]
        included = variants["memory-included"]
        self.assertEqual(
            positive.data["cue"]["location"],
            {"tool": "get_pull_request", "arguments": {"pull_number": 67}},
        )
        self.assertTrue(positive.data["cue"]["present"])
        self.assertFalse(ablated.data["cue"]["present"])
        self.assertEqual(
            positive.data["environment"]["fixture_path"], included.data["environment"]["fixture_path"]
        )
        self.assertNotEqual(
            positive.data["environment"]["fixture_path"], ablated.data["environment"]["fixture_path"]
        )
        self.assertEqual(positive.data["memory"]["gold_source_ids"], ["notes/customer-constraints-ashcroft"])
        self.assertIn("ashcroft-university-sso-beta", positive.data["cue"]["text"])
        self.assertNotIn(positive.data["cue"]["text"].casefold(), positive.goal.casefold())
        self.assertEqual(
            ablated.data["decisive_action"]["matchers"],
            positive.data["decisive_action"]["matchers"],
        )

    def test_rollback_canary_fixture_is_ordinary_until_one_target_applicability_fact(self) -> None:
        variants = _cases()["parm-workflow-github-canary-rollback"]
        positive = variants["positive"]
        control = variants["cue-ablated"]
        opening = positive.goal.casefold()
        for gold_or_cue_term in (
            "ashcroft",
            "university",
            "sso",
            "canary",
            "rollout",
            "august",
            "semester",
            "january",
            "version bump",
        ):
            self.assertNotIn(gold_or_cue_term, opening)

        positive_fixture = json.loads(
            (DATASET / positive.data["environment"]["fixture_path"]).read_text(
                encoding="utf-8"
            )
        )
        control_fixture = json.loads(
            (DATASET / control.data["environment"]["fixture_path"]).read_text(
                encoding="utf-8"
            )
        )
        positive_body = positive_fixture["pull_requests"][0]["body"]
        control_body = control_fixture["pull_requests"][0]["body"]
        self.assertIn("ashcroft-university-sso-beta", positive_body)
        self.assertIn("shared-internal-sso-beta", control_body)
        self.assertEqual(
            positive_body.replace(
                "ashcroft-university-sso-beta", "<target>"
            ),
            control_body.replace("shared-internal-sso-beta", "<target>"),
        )
        positive_fixture["pull_requests"][0]["body"] = "<target>"
        control_fixture["pull_requests"][0]["body"] = "<target>"
        self.assertEqual(positive_fixture, control_fixture)

    def test_rollback_canary_assertions_distinguish_hold_from_shared_rollout(self) -> None:
        variants = _cases()["parm-workflow-github-canary-rollback"]
        expected_calls = {
            "positive": [
                ("add_issue_comment", {"issue_number": 67, "body": "Hold Ashcroft's canary until the semester window."}),
                ("merge_pull_request", {"pull_number": 68}),
            ],
            "cue-ablated": [
                ("merge_pull_request", {"pull_number": 67}),
                ("merge_pull_request", {"pull_number": 68}),
            ],
        }
        for variant, calls in expected_calls.items():
            with self.subTest(variant=variant):
                environment = variants[variant].build_environment()
                for tool_name, arguments in calls:
                    self.assertTrue(environment.invoke(tool_name, arguments).ok)
                rows = evaluate_assertions(
                    variants[variant].data["assertions"],
                    state=environment.state(),
                    mutations=environment.mutations(),
                    trajectory=environment.trajectory,
                )
                self.assertTrue(all(row["passed"] for row in rows), rows)

    def test_every_gold_record_lives_in_the_declared_tier(self) -> None:
        tiers = load_tiers(CORPUS)
        for case in load_workflow_cases(DATASET):
            with self.subTest(case.case_id):
                self.assertIn(case.corpus_tier, tiers)
                members = set(tiers[case.corpus_tier])
                self.assertTrue(set(case.gold_source_ids) <= members)

    def test_a_cue_is_absent_from_its_own_goal(self) -> None:
        for case in load_workflow_cases(DATASET):
            with self.subTest(case.case_id):
                cue = str(case.data["cue"]["text"]).casefold()
                self.assertNotIn(cue, case.goal.casefold())

    def test_authz_codeowner_triplet_changes_only_after_pr_inspection(self) -> None:
        variants = _cases()["parm-workflow-github-authz-codeowner"]
        positive = variants["positive"]
        control = variants["cue-ablated"]
        ceiling = variants["memory-included"]

        self.assertEqual(positive.corpus_tier, "tier-100")
        self.assertEqual(positive.gold_source_ids, ("notes/code-owners-policy",))
        self.assertEqual(
            positive.data["cue"]["location"],
            {"tool": "get_pull_request", "arguments": {"pull_number": 83}},
        )
        positive_environment = positive.build_environment()
        control_environment = control.build_environment()
        self.assertNotIn("src/auth/", positive_environment.invoke(
            "list_pull_requests", {"state": "open"}
        ).text)
        self.assertIn("src/auth/", positive_environment.invoke(
            "get_pull_request", {"pull_number": 83}
        ).text)
        self.assertNotIn("src/auth/", control_environment.invoke(
            "get_pull_request", {"pull_number": 83}
        ).text)
        self.assertEqual(
            positive.data["environment"]["fixture_path"],
            ceiling.data["environment"]["fixture_path"],
        )


class GoldReachabilityTest(unittest.TestCase):
    def test_an_index_missing_gold_is_refused(self) -> None:
        cases = [
            case
            for case in load_workflow_cases(DATASET)
            if case.corpus_tier == "tier-100"
        ]
        self.assertTrue(cases)
        index = _FakeIndex(["notes/unrelated"])
        with self.assertRaises(ValueError) as caught:
            assert_gold_reachable(cases, type("R", (), {"index": index})())
        self.assertIn("missing its gold", str(caught.exception))

    def test_a_complete_index_is_accepted(self) -> None:
        cases = load_workflow_cases(DATASET)
        slugs = {
            source_id for case in cases for source_id in case.gold_source_ids
        }
        assert_gold_reachable(cases, type("R", (), {"index": _FakeIndex(slugs)})())

    def test_a_policy_without_an_index_is_ignored(self) -> None:
        assert_gold_reachable(load_workflow_cases(DATASET), None)


class NewAssertionKindTest(unittest.TestCase):
    def _evaluate(self, assertion, state):
        return evaluate_assertions(
            [assertion], state=state, mutations=[], trajectory=[]
        )[0]

    def test_issue_state_reads_the_issue(self) -> None:
        state = {"issues": {72: {"state": "open", "comments": []}}, "pull_requests": {}}
        row = self._evaluate(
            {"id": "x", "role": "decisive", "kind": "issue_state",
             "issue_number": 72, "expected": "open"},
            state,
        )
        self.assertTrue(row["passed"])

    def test_issues_closed_requires_every_number(self) -> None:
        state = {
            "issues": {
                70: {"state": "closed", "comments": []},
                71: {"state": "open", "comments": []},
            },
            "pull_requests": {},
        }
        row = self._evaluate(
            {"id": "x", "role": "workflow", "kind": "issues_closed",
             "issue_numbers": [70, 71]},
            state,
        )
        self.assertFalse(row["passed"])
        self.assertIn("#71", row["detail"])

    def test_pull_requests_merged_requires_every_number(self) -> None:
        state = {
            "issues": {},
            "pull_requests": {
                64: {"state": "merged", "comments": []},
                65: {"state": "merged", "comments": []},
            },
        }
        row = self._evaluate(
            {"id": "x", "role": "workflow", "kind": "pull_requests_merged",
             "pull_numbers": [64, 65]},
            state,
        )
        self.assertTrue(row["passed"])

    def test_comment_contains_any_accepts_one_keyword(self) -> None:
        state = {
            "issues": {},
            "pull_requests": {
                66: {"state": "open", "comments": [{"body": "holding until January"}]}
            },
        }
        row = self._evaluate(
            {"id": "x", "role": "decisive", "kind": "comment_contains",
             "number": 66, "keywords": ["freeze", "january"], "match": "any"},
            state,
        )
        self.assertTrue(row["passed"])

    def test_comment_contains_defaults_to_requiring_all(self) -> None:
        state = {
            "issues": {},
            "pull_requests": {
                66: {"state": "open", "comments": [{"body": "holding until January"}]}
            },
        }
        row = self._evaluate(
            {"id": "x", "role": "decisive", "kind": "comment_contains",
             "number": 66, "keywords": ["freeze", "january"]},
            state,
        )
        self.assertFalse(row["passed"])


if __name__ == "__main__":
    unittest.main()
