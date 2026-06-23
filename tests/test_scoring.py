from __future__ import annotations

import unittest

from parm_bench.baselines import no_memory, parm_oracle_monitor
from parm_bench.scoring import score_predictions
from parm_bench.synthetic import generate_cases


class ScoringTests(unittest.TestCase):
    def test_oracle_prediction_scores_correct(self) -> None:
        case = generate_cases(1)[0]
        prediction = parm_oracle_monitor(case)
        metrics = score_predictions([case], [prediction])
        self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_oracle_scores_all_generated_cases_as_recoverable(self) -> None:
        cases = generate_cases(50)
        predictions = [parm_oracle_monitor(case) for case in cases]
        metrics = score_predictions(cases, predictions)
        self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["useless_intervention_rate"], 0.0)

    def test_empty_prediction_scores_as_miss_without_useless_intervention(self) -> None:
        case = generate_cases(1)[0]
        prediction = no_memory(case)
        metrics = score_predictions([case], [prediction])
        self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 0.0)
        self.assertEqual(metrics["total_suggestions"], 0)
        self.assertEqual(metrics["useless_intervention_rate"], 0.0)
        self.assertEqual(metrics["abstention_rate"], 1.0)

    def test_wrong_path_is_not_correct(self) -> None:
        case = generate_cases(1)[0]
        prediction = parm_oracle_monitor(case)
        prediction["suggestions"][0]["path"] = ["n_trigger", "n_distractor"]
        metrics = score_predictions([case], [prediction])
        self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 0.0)
        self.assertEqual(metrics["path_correct_rate"], 0.0)
        self.assertEqual(metrics["rows"][0]["first_failure"], "invalid_or_stale_edge_used")

    def test_current_warning_oracle_phrasing_scores_correct(self) -> None:
        for index in (2, 8, 12):
            case = generate_cases(index + 1)[index]
            prediction = parm_oracle_monitor(case)
            metrics = score_predictions([case], [prediction])
            self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 1.0, case["case_id"])
            self.assertEqual(metrics["decision_effect_found_rate"], 1.0, case["case_id"])

    def test_vague_text_without_action_keywords_is_not_correct(self) -> None:
        case = generate_cases(3)[2]
        prediction = parm_oracle_monitor(case)
        prediction["suggestions"][0]["text"] = (
            "This option has some relevant personal context and may be worth considering carefully."
        )
        metrics = score_predictions([case], [prediction])
        self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 0.0)
        self.assertIn("actionability_miss", metrics["rows"][0]["failed_stages"])

    def test_wrong_affected_item_is_not_correct(self) -> None:
        case = generate_cases(1)[0]
        prediction = parm_oracle_monitor(case)
        prediction["suggestions"][0]["text"] = prediction["suggestions"][0]["text"].replace(
            "Acme Robotics 100",
            "Brightpath Labs 200",
        )
        metrics = score_predictions([case], [prediction])
        self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 0.0)
        self.assertIn("wrong_affected_item", metrics["rows"][0]["failed_stages"])

    def test_extra_incorrect_suggestion_counts_as_useless_intervention(self) -> None:
        case = generate_cases(1)[0]
        prediction = parm_oracle_monitor(case)
        prediction["suggestions"].append({"text": "Also remember an unrelated note."})
        metrics = score_predictions([case], [prediction])
        self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 1.0)
        self.assertEqual(metrics["total_suggestions"], 2)
        self.assertEqual(metrics["useless_intervention_rate"], 0.5)
        self.assertEqual(metrics["rows"][0]["first_failure"], "overtrigger_noise")

    def test_natural_response_text_does_not_require_path(self) -> None:
        case = generate_cases(4)[3]
        prediction = {
            "case_id": case["case_id"],
            "response_text": (
                f"{case['expected']['suggestion']} "
                f"{case['graph']['nodes'][3]['label']}"
            ),
        }
        metrics = score_predictions([case], [prediction])
        self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 1.0)
        self.assertIsNone(metrics["path_correct_rate"])
        self.assertIsNone(metrics["rows"][0]["path_correct"])


if __name__ == "__main__":
    unittest.main()
