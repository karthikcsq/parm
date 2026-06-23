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

    def test_empty_prediction_scores_as_miss_without_useless_intervention(self) -> None:
        case = generate_cases(1)[0]
        prediction = no_memory(case)
        metrics = score_predictions([case], [prediction])
        self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 0.0)
        self.assertEqual(metrics["total_suggestions"], 0)
        self.assertEqual(metrics["useless_intervention_rate"], 0.0)

    def test_wrong_path_is_not_correct(self) -> None:
        case = generate_cases(1)[0]
        prediction = parm_oracle_monitor(case)
        prediction["suggestions"][0]["path"] = ["n_trigger", "n_distractor"]
        metrics = score_predictions([case], [prediction])
        self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 0.0)
        self.assertEqual(metrics["path_correct_rate"], 0.0)

    def test_warning_case_requires_decision_reconsideration(self) -> None:
        case = generate_cases(4)[3]
        prediction = parm_oracle_monitor(case)
        prediction["suggestions"][0]["text"] = (
            "Flag a warning for Riverfront Hotel 103: Dr. Rivera's note about West Loop "
            "suggests changing timing or transit plans."
        )
        metrics = score_predictions([case], [prediction])
        self.assertEqual(metrics["correct_surfaced_suggestion_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
