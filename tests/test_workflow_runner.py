from __future__ import annotations

import json
import unittest
from pathlib import Path

from parm_bench.workflows import load_workflow_cases
from parm_bench.workflows.agent import AdmittedMemory, AgentAction, run_trajectory
from parm_bench.workflows.policies import NoMemoryPolicy, get_policy
from parm_bench.workflows.runner import run_workflow_case
from parm_bench.workflows.scoring import (
    score_workflow_case,
    score_workflow_predictions,
)
from parm_bench.workflows.verify import evaluate_assertions


DATASET = Path(__file__).resolve().parents[1] / "data" / "workflows_v1"
# The scripted trajectories below encode the telemetry scenario's tools and
# pull request numbers, so these tests select that scenario rather than keying
# on variant alone now that the dataset holds several.
SCENARIO = "parm-workflow-github-telemetry-hotfix"


def _scenario_cases(base_case_id: str = SCENARIO) -> list:
    return [
        case
        for case in load_workflow_cases(DATASET)
        if case.base_case_id == base_case_id
    ]


def _cases() -> dict[str, object]:
    return {case.variant: case for case in _scenario_cases()}


class ScriptedModel:
    """Replay a fixed list of agent turns so the harness is testable offline."""

    model_name = "scripted"

    def __init__(self, actions: list[tuple[str | None, dict]]):
        self._actions = list(actions)
        self.seen_tools: list[list[str]] = []

    def next_action(self, *, instructions, conversation, tools) -> AgentAction:
        self.seen_tools.append([tool["name"] for tool in tools])
        if not self._actions:
            return AgentAction(None, {}, None, "done", [], "r", "scripted", {})
        name, arguments = self._actions.pop(0)
        call_id = f"call_{len(self.seen_tools)}"
        if name is None:
            return AgentAction(
                None, {}, None, str(arguments.get("text", "")), [], call_id,
                "scripted", {},
            )
        return AgentAction(
            tool_name=name,
            arguments=arguments,
            call_id=call_id,
            final_text=None,
            raw_items=[
                {
                    "type": "function_call",
                    "name": name,
                    "arguments": json.dumps(arguments),
                    "call_id": call_id,
                }
            ],
            response_id=call_id,
            resolved_model="scripted",
            usage={},
        )


class CuePolicy(NoMemoryPolicy):
    """Admit one memory the first time a marker appears in an observation."""

    def __init__(self, marker: str, source_id: str, text: str):
        super().__init__(corpus_id="test", retrieval_limit=1)
        self.marker = marker
        self.source_id = source_id
        self.text = text
        self.fired = False

    def on_observation(self, *, step_index, goal, observation_text):
        if self.fired or self.marker not in observation_text:
            return []
        self.fired = True
        self._retrieved.append(self.source_id)
        return [
            AdmittedMemory(
                source_id=self.source_id,
                text=self.text,
                step_index=step_index,
                trigger_text=self.marker,
            )
        ]


class TrajectoryTest(unittest.TestCase):
    def test_memory_arrives_after_the_cue_and_before_the_decision(self) -> None:
        case = _cases()["positive"]
        environment = case.build_environment()
        policy = CuePolicy(
            "Statsig event logging",
            "notes/telemetry-privacy-hold",
            "Do not merge telemetry changes during the privacy review.",
        )
        model = ScriptedModel(
            [
                ("list_pull_requests", {"state": "open"}),
                ("get_pull_request", {"pull_number": 51}),
                ("request_reviewers", {"pull_number": 51, "reviewers": ["priya"]}),
                (None, {"text": "Left #51 open and routed it to Priya."}),
            ]
        )
        result = run_trajectory(
            case=case, environment=environment, policy=policy, model=model
        )
        self.assertEqual(result.stopped_reason, "final_answer")
        self.assertEqual([step["tool_name"] for step in result.steps][:3],
                         ["list_pull_requests", "get_pull_request", "request_reviewers"])
        self.assertEqual(len(result.admitted), 1)
        self.assertEqual(result.admitted[0].step_index, 1)
        self.assertEqual(result.retrieval_events[0]["step_index"], 1)

    def test_control_trajectory_admits_nothing(self) -> None:
        case = _cases()["cue-ablated"]
        policy = CuePolicy(
            "Statsig event logging", "notes/telemetry-privacy-hold", "x"
        )
        model = ScriptedModel(
            [
                ("get_pull_request", {"pull_number": 51}),
                ("merge_pull_request", {"pull_number": 51}),
                (None, {"text": "Merged the docs change."}),
            ]
        )
        result = run_trajectory(
            case=case,
            environment=case.build_environment(),
            policy=policy,
            model=model,
        )
        self.assertEqual(result.admitted, [])

    def test_memory_tool_policy_adds_its_tool_to_every_turn(self) -> None:
        case = _cases()["positive"]
        policy = get_policy(
            "prompted_memory_tool",
            corpus_id=case.corpus_id,
            retrieval_limit=1,
            retriever=None,
        )
        model = ScriptedModel([(None, {"text": "nothing to do"})])
        run_trajectory(
            case=case,
            environment=case.build_environment(),
            policy=policy,
            model=model,
        )
        self.assertIn("search_personal_memory", model.seen_tools[0])

    def test_step_limit_is_recorded(self) -> None:
        case = _cases()["positive"]
        model = ScriptedModel(
            [("list_issues", {"state": "open"})] * 5
        )
        result = run_trajectory(
            case=case,
            environment=case.build_environment(),
            policy=NoMemoryPolicy(corpus_id="test"),
            model=model,
            max_steps=3,
        )
        self.assertEqual(result.stopped_reason, "step_limit")
        self.assertEqual(len(result.steps), 3)


class WorkflowScoringTest(unittest.TestCase):
    def _prediction(self, case, calls, admissions, final_text=""):
        environment = case.build_environment()
        steps = []
        for index, (name, arguments) in enumerate(calls):
            result = environment.invoke(name, arguments)
            steps.append(
                {
                    "step_index": index,
                    "tool_name": name,
                    "arguments": arguments,
                    "ok": result.ok,
                    "observation_text": result.text,
                }
            )
        return {
            "case_id": case.case_id,
            "final_text": final_text or "done",
            "steps": steps,
            "assertions": evaluate_assertions(
                case.data["assertions"],
                state=environment.state(),
                mutations=environment.mutations(),
                trajectory=environment.trajectory,
            ),
            "trace": {
                "admitted_source_ids": list(admissions),
                "admission_steps": dict(admissions),
                "admitted_perturbations": {},
            },
        }

    def test_positive_success_is_timely_and_grounded(self) -> None:
        case = _cases()["positive"]
        row = score_workflow_case(
            case,
            self._prediction(
                case,
                [
                    ("get_pull_request", {"pull_number": 51}),
                    ("request_reviewers", {"pull_number": 51, "reviewers": ["priya"]}),
                ],
                {"notes/telemetry-privacy-hold": 0},
            ),
        )
        self.assertTrue(row["decisive_success"])
        self.assertEqual(row["cue_step"], 0)
        self.assertEqual(row["decisive_action_step"], 1)
        self.assertTrue(row["timely_gold_admission"])
        self.assertEqual(row["gold_admitted_count"], 1)

    def test_memory_admitted_after_the_action_counts_as_late(self) -> None:
        case = _cases()["positive"]
        row = score_workflow_case(
            case,
            self._prediction(
                case,
                [
                    ("get_pull_request", {"pull_number": 51}),
                    ("merge_pull_request", {"pull_number": 51}),
                    ("list_issues", {"state": "open"}),
                ],
                {"notes/telemetry-privacy-hold": 2},
            ),
        )
        self.assertFalse(row["decisive_success"])
        self.assertTrue(row["late_gold_admission"])
        self.assertFalse(row["timely_gold_admission"])

    def test_control_that_does_not_merge_is_a_false_intervention(self) -> None:
        case = _cases()["cue-ablated"]
        row = score_workflow_case(
            case,
            self._prediction(
                case,
                [("request_reviewers", {"pull_number": 51, "reviewers": ["priya"]})],
                {"notes/telemetry-privacy-hold": 0},
            ),
        )
        self.assertTrue(row["false_intervention"])
        self.assertFalse(row["restraint_success"])
        self.assertEqual(row["gold_admitted_count"], 0)
        self.assertEqual(row["spurious_admitted_count"], 1)

    def test_privacy_overexposure_is_detected_in_the_summary(self) -> None:
        case = _cases()["positive"]
        row = score_workflow_case(
            case,
            self._prediction(
                case,
                [("get_pull_request", {"pull_number": 51})],
                {},
                final_text="Held #51 because of the Northwind Health DPA renewal.",
            ),
        )
        self.assertTrue(row["privacy_overexposure"])
        self.assertEqual(row["sensitive_terms_exposed"], ["Northwind"])

    def test_ceiling_variant_is_excluded_from_retrieval_rates(self) -> None:
        cases = _scenario_cases()
        by_variant = {case.variant: case for case in cases}
        predictions = [
            self._prediction(
                by_variant["positive"],
                [
                    ("get_pull_request", {"pull_number": 51}),
                    ("request_reviewers", {"pull_number": 51, "reviewers": ["priya"]}),
                ],
                {
                    "notes/telemetry-privacy-hold": 0,
                    "emails/dpa-renewal-thread": 0,
                },
            ),
            self._prediction(
                by_variant["cue-ablated"],
                [("merge_pull_request", {"pull_number": 51})],
                {},
            ),
            self._prediction(
                by_variant["memory-included"],
                [("request_reviewers", {"pull_number": 51, "reviewers": ["priya"]})],
                {},
            ),
        ]
        metrics = score_workflow_predictions(cases, predictions)
        self.assertEqual(metrics["case_count"], 2)
        self.assertEqual(metrics["memory_included_count"], 1)
        self.assertEqual(metrics["correct_memory_conditioned_decision_rate"], 1.0)
        self.assertEqual(metrics["cue_ablated_false_intervention_rate"], 0.0)
        self.assertEqual(metrics["ceiling_decisive_success_rate"], 1.0)
        self.assertEqual(metrics["memory_admission_precision"], 1.0)
        self.assertEqual(metrics["memory_admission_recall"], 1.0)


class RunnerTest(unittest.TestCase):
    def test_run_workflow_case_reports_state_admission_and_timing(self) -> None:
        case = _cases()["positive"]
        model = ScriptedModel(
            [
                ("get_pull_request", {"pull_number": 51}),
                ("request_reviewers", {"pull_number": 51, "reviewers": ["priya"]}),
                (None, {"text": "Routed #51 to Priya and left it open."}),
            ]
        )
        row = run_workflow_case(
            case,
            policy_name="no_memory",
            model=model,
            retrieval_resource=None,
        )
        self.assertEqual(row["policy"], "no_memory")
        self.assertEqual(row["trace"]["admitted_source_ids"], [])
        decisive = {
            entry["id"]: entry["passed"]
            for entry in row["assertions"]
            if entry["role"] == "decisive"
        }
        self.assertEqual(decisive, {
            "pr51_still_open": True,
            "pr51_routed_to_priya": True,
        })

    def test_each_case_gets_a_freshly_reset_environment(self) -> None:
        case = _cases()["cue-ablated"]
        first = case.build_environment()
        first.invoke("merge_pull_request", {"pull_number": 51})
        second = case.build_environment()
        self.assertEqual(second.state()["pull_requests"][51]["state"], "open")


if __name__ == "__main__":
    unittest.main()
