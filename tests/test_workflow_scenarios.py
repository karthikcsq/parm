from __future__ import annotations

import unittest
from pathlib import Path

from parm_bench.workflows import load_workflow_cases, validate_workflow_cases
from parm_bench.workflows.runner import assert_gold_reachable
from parm_bench.workflows.corpus_tiers import load_tiers
from parm_bench.workflows.verify import evaluate_assertions


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_v1"
CORPUS = DATASET / "corpora" / "workflow-eng-lead-v1"


def _cases() -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
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
