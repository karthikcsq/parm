from __future__ import annotations

import unittest
from pathlib import Path

from parm_bench.dataset import load_cases
from parm_bench.scoring import score_predictions


ROOT = Path(__file__).resolve().parents[1]
CASES = load_cases(ROOT / "data" / "benchmark_v1")


def prediction(
    case: dict,
    selected_choice: str | None,
    admitted_source_ids: list[str] | None = None,
    admitted_perturbations: dict[str, list[str]] | None = None,
    response_text: str = "",
) -> dict:
    if selected_choice is not None and not response_text:
        response_text = f"My choice is {selected_choice}."
    return {
        "case_id": case["case_id"],
        "response_text": response_text,
        "trace": {
            "admitted_source_ids": admitted_source_ids or [],
            "admitted_perturbations": admitted_perturbations or {},
        },
    }


class ScoringTests(unittest.TestCase):
    def test_hand_authored_gold_predictions_score_as_beneficial(self) -> None:
        predictions = []
        for case in CASES:
            selected = case["decisions"]["memory_conditioned"]["choice"]
            admitted = (
                case["memory"]["gold_source_ids"]
                if case["variant"] == "positive"
                else []
            )
            predictions.append(prediction(case, selected, admitted))
        metrics = score_predictions(CASES, predictions)
        self.assertEqual(metrics["beneficial_decision_change_rate"], 1.0)
        self.assertEqual(metrics["cue_ablated_false_intervention_rate"], 0.0)
        self.assertEqual(metrics["memory_admission_precision"], 1.0)

    def test_output_only_predictions_do_not_claim_benefit(self) -> None:
        predictions = [
            prediction(case, case["decisions"]["output_only"]["choice"])
            for case in CASES
        ]
        metrics = score_predictions(CASES, predictions)
        self.assertEqual(metrics["beneficial_decision_change_rate"], 0.0)
        self.assertEqual(metrics["cue_ablated_false_intervention_rate"], 0.0)

    def test_poison_and_spurious_admission_are_counted(self) -> None:
        case = next(case for case in CASES if case["variant"] == "positive")
        poison_id = case["distractors"]["sources"][2]["source_id"]
        row = prediction(
            case,
            case["decisions"]["memory_conditioned"]["choice"],
            [*case["memory"]["gold_source_ids"], poison_id],
            {poison_id: ["poison"]},
        )
        metrics = score_predictions([case], [row])
        self.assertEqual(metrics["poison_admission_rate"], 1.0)
        self.assertGreater(metrics["spurious_memory_admission_rate"], 0.0)

    def test_tactful_response_does_not_need_to_repeat_memory(self) -> None:
        case = next(case for case in CASES if case["variant"] == "positive")
        row = prediction(
            case,
            case["decisions"]["memory_conditioned"]["choice"],
            case["memory"]["gold_source_ids"],
        )
        metrics = score_predictions([case], [row])
        self.assertEqual(metrics["correct_memory_conditioned_decision_rate"], 1.0)
        self.assertEqual(metrics["privacy_overexposure_rate"], 0.0)

    def test_sensitive_detail_is_flagged(self) -> None:
        case = next(
            case
            for case in CASES
            if case["base_case_id"].endswith("lunch-search")
            and case["variant"] == "positive"
        )
        row = prediction(
            case,
            case["decisions"]["memory_conditioned"]["choice"],
            response_text="You ate there four days in one week.",
        )
        metrics = score_predictions([case], [row])
        self.assertEqual(metrics["privacy_overexposure_rate"], 1.0)

    def test_choice_is_scored_from_natural_language_response(self) -> None:
        case = next(case for case in CASES if case["variant"] == "positive")
        choice = case["decisions"]["memory_conditioned"]["choice"]
        row = prediction(case, None, response_text=f"I'd pick: {choice}.")
        metrics = score_predictions([case], [row])
        self.assertEqual(metrics["correct_memory_conditioned_decision_rate"], 1.0)

    def test_naming_both_choices_is_not_a_single_final_choice(self) -> None:
        case = next(case for case in CASES if case["variant"] == "positive")
        output_choice = case["decisions"]["output_only"]["choice"]
        memory_choice = case["decisions"]["memory_conditioned"]["choice"]
        row = prediction(
            case,
            None,
            response_text=f"Either {output_choice} or {memory_choice}.",
        )
        metrics = score_predictions([case], [row])
        self.assertEqual(metrics["correct_memory_conditioned_decision_rate"], 0.0)
        self.assertFalse(metrics["rows"][0]["choice_identifiable"])


if __name__ == "__main__":
    unittest.main()
