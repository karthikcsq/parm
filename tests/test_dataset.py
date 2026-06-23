from __future__ import annotations

import copy
import unittest

from parm_bench.dataset import DatasetValidationError, validate_cases
from parm_bench.synthetic import generate_cases


class DatasetValidationTests(unittest.TestCase):
    def test_generated_cases_are_valid(self) -> None:
        cases = generate_cases(50)
        validate_cases(cases)
        self.assertEqual(len(cases), 50)

    def test_generated_cases_cover_multiple_domains(self) -> None:
        cases = generate_cases(50)
        families = {case["goal_family"] for case in cases}
        self.assertGreaterEqual(len(families), 8)
        self.assertIn("travel_planning", families)
        self.assertIn("general_travel", families)
        self.assertIn("cooking_planning", families)
        self.assertIn("health_admin", families)
        self.assertIn("home_ops", families)

    def test_user_goals_do_not_prompt_memory_search(self) -> None:
        cases = generate_cases(50)
        forbidden_phrases = [
            "flag ",
            "hidden",
            "latent",
            "personal context",
            "personal memory",
            "memory",
            "note what",
            "surface hidden",
        ]
        for case in cases:
            user_goal = case["user_goal"].lower()
            with self.subTest(case_id=case["case_id"]):
                for phrase in forbidden_phrases:
                    self.assertNotIn(phrase, user_goal)

    def test_trigger_cue_appears_after_initial_prompt(self) -> None:
        cases = generate_cases(50)
        for case in cases:
            trigger_label = case["trigger_entity"]["label"].lower()
            user_goal = case["user_goal"].lower()
            assistant_output = case["assistant_output"].lower()
            tool_summary = case["tool_response"]["summary"].lower()

            with self.subTest(case_id=case["case_id"]):
                self.assertNotIn(trigger_label, user_goal)
                if case["trigger_source"] == "generated_output":
                    self.assertIn(trigger_label, assistant_output)
                else:
                    self.assertIn(trigger_label, tool_summary)

    def test_context_requirement_controls_tool_use(self) -> None:
        cases = generate_cases(50)
        requirements = {case["context_requirement"] for case in cases}
        self.assertIn("tool", requirements)
        self.assertIn("general_knowledge", requirements)

        for case in cases:
            tool_response = case["tool_response"]
            with self.subTest(case_id=case["case_id"]):
                if case["context_requirement"] == "tool":
                    self.assertEqual(case["trigger_source"], "tool_response")
                    self.assertEqual(tool_response["source"], "synthetic_local_context")
                    self.assertGreaterEqual(tool_response["summary"].count("- "), 3)
                    self.assertIn("Best objective fit:", tool_response["summary"])
                else:
                    self.assertEqual(case["trigger_source"], "generated_output")
                    self.assertEqual(tool_response["source"], "not_required")
                    self.assertIn("general model knowledge", tool_response["summary"])

    def test_cases_encode_memory_decision_effect_without_generic_gold_answer(self) -> None:
        cases = generate_cases(50)
        for case in cases:
            expected = case["expected"]
            with self.subTest(case_id=case["case_id"]):
                self.assertNotIn("generic_decision", expected)
                self.assertNotIn("memory_augmented_delta", expected)
                self.assertIn("decision_effect", expected)
                self.assertEqual(expected["decision_effect"]["item_node_id"], "n_item")

    def test_missing_required_field_fails(self) -> None:
        case = generate_cases(1)[0]
        del case["expected"]
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases([case])
        self.assertIn("missing required field 'expected'", str(context.exception))

    def test_duplicate_case_id_fails(self) -> None:
        cases = generate_cases(2)
        cases[1]["case_id"] = cases[0]["case_id"]
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(cases)
        self.assertIn("duplicate case_id", str(context.exception))

    def test_invalid_graph_reference_fails(self) -> None:
        case = generate_cases(1)[0]
        case["graph"]["edges"][0]["target"] = "missing"
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases([case])
        self.assertIn("references unknown nodes", str(context.exception))

    def test_broken_gold_path_fails(self) -> None:
        case = copy.deepcopy(generate_cases(1)[0])
        case["expected"]["gold_path"] = ["n_trigger", "n_distractor", "n_memory"]
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases([case])
        self.assertIn("gold_path has no valid edge", str(context.exception))


if __name__ == "__main__":
    unittest.main()
