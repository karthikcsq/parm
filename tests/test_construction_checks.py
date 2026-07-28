from __future__ import annotations

import copy
import unittest
from pathlib import Path

from parm_bench.construction_checks import (
    MIN_SCENARIOS_FOR_CHECKS,
    construction_issues,
    judge_leakage_violations,
)
from parm_bench.dataset import load_cases


ROOT = Path(__file__).resolve().parents[1]
PERSONAMEM_MIXED_DATASET = ROOT / "data" / "benchmark_personamem_mixed_v0"

SCENARIO_COUNT = 12
BODY_LINES = 40
OUTPUT_ROW = 2
MEMORY_ROW = 37
FIRST_CUE_ROW = 5
LAST_CUE_ROW = 35


def _line(scenario: int, row: int) -> str:
    # Every word carries the scenario index so no six-gram can be shared by two
    # scenarios unless a test deliberately plants one.
    return " ".join(f"t{scenario}x{row}y{word}" for word in range(8))


def _cue_text(scenario: int) -> str:
    return " ".join(f"cue{scenario}affordance{word}" for word in range(8))


def _observation(
    scenario: int,
    *,
    cue_fraction: float,
    memory_first: bool = False,
    opening: str | None = None,
    extra_lines: tuple[str, ...] = (),
) -> str:
    body = [_line(scenario, row) for row in range(BODY_LINES)]
    output_row, memory_row = (
        (MEMORY_ROW, OUTPUT_ROW) if memory_first else (OUTPUT_ROW, MEMORY_ROW)
    )
    body[output_row] = f"Ordinary Option {scenario} " + _line(scenario, 100)
    body[memory_row] = f"Personal Option {scenario} " + _line(scenario, 101)
    cue_row = FIRST_CUE_ROW + round(cue_fraction * (LAST_CUE_ROW - FIRST_CUE_ROW))
    body[cue_row] = _cue_text(scenario) + " " + _line(scenario, 102)
    head = opening if opening is not None else f"header {scenario} " + _line(scenario, 200)
    return "\n".join([head, *body, *extra_lines])


def _case(scenario: int, observation: str) -> dict:
    return {
        "base_case_id": f"scenario-{scenario:02d}",
        "case_id": f"scenario-{scenario:02d}-positive",
        "variant": "positive",
        "observation_text": observation,
        "cue": {"present": True, "text": _cue_text(scenario)},
        "decisions": {
            "answer_type": "natural_language_choice",
            "output_only": {"choice": f"Ordinary Option {scenario}"},
            "memory_conditioned": {"choice": f"Personal Option {scenario}"},
        },
    }


def varied_cases(count: int = SCENARIO_COUNT) -> list[dict]:
    """Build a batch with no construction signature the detectors can find."""

    cases = []
    for scenario in range(count):
        cases.append(
            _case(
                scenario,
                _observation(
                    scenario,
                    cue_fraction=scenario / max(1, count - 1),
                    memory_first=scenario % 2 == 1,
                ),
            )
        )
    return cases


def _messages(cases: list[dict]) -> list[str]:
    return [message for _, message in construction_issues(cases)]


class ConstructionCheckTests(unittest.TestCase):
    def test_varied_batch_reports_no_issues(self) -> None:
        self.assertEqual(construction_issues(varied_cases()), [])

    def test_small_batch_is_not_judged(self) -> None:
        cases = varied_cases(MIN_SCENARIOS_FOR_CHECKS - 1)
        for case in cases:
            case["observation_text"] += "\nrated 9.8 against 8.8 overall"
        self.assertEqual(construction_issues(cases), [])

    def test_repeated_numeric_pair_is_flagged(self) -> None:
        cases = varied_cases()
        for index, case in enumerate(cases):
            case["observation_text"] += (
                f"\nscore {index} t{index}x300y0 9.8 t{index}x300y1 8.8"
            )
        self.assertTrue(
            any("repeated numeric pair" in message for message in _messages(cases))
        )

    def test_distinct_numeric_pairs_are_not_flagged(self) -> None:
        cases = varied_cases()
        for index, case in enumerate(cases):
            case["observation_text"] += (
                f"\nscore t{index}x300y0 {index}.1 t{index}x300y1 {index}.2"
            )
        self.assertFalse(
            any("repeated numeric pair" in message for message in _messages(cases))
        )

    def test_shared_phrase_is_flagged(self) -> None:
        cases = varied_cases()
        for case in cases:
            case["observation_text"] += (
                "\nreviewers repeatedly called it polished and dependable"
            )
        self.assertTrue(
            any("6-grams recur" in message for message in _messages(cases))
        )

    def test_phrase_below_the_share_threshold_is_not_flagged(self) -> None:
        cases = varied_cases()
        cases[0]["observation_text"] += (
            "\nreviewers repeatedly called it polished and dependable"
        )
        self.assertFalse(
            any("6-grams recur" in message for message in _messages(cases))
        )

    def test_banned_phrase_in_two_scenarios_is_flagged(self) -> None:
        cases = varied_cases()
        for index, case in enumerate(cases[:2]):
            case["observation_text"] += (
                f"\n{_line(index, 400)} narrower choice {_line(index, 401)}"
            )
        issues = construction_issues(cases)
        flagged = [
            case_id
            for case_id, message in issues
            if "narrower choice" in message
        ]
        self.assertEqual(flagged, ["scenario-00", "scenario-01"])

    def test_banned_phrase_in_one_scenario_is_allowed(self) -> None:
        cases = varied_cases()
        cases[0]["observation_text"] += "\nthe author called it a narrower choice"
        self.assertFalse(
            any("narrower choice" in message for message in _messages(cases))
        )

    def test_fixed_cue_position_is_flagged(self) -> None:
        cases = [
            _case(scenario, _observation(scenario, cue_fraction=0.5))
            for scenario in range(SCENARIO_COUNT)
        ]
        self.assertTrue(
            any("cue position varies" in message for message in _messages(cases))
        )

    def test_spread_cue_position_is_not_flagged(self) -> None:
        self.assertFalse(
            any("cue position varies" in message for message in _messages(varied_cases()))
        )

    def test_invariant_option_role_is_flagged(self) -> None:
        cases = [
            _case(
                scenario,
                _observation(
                    scenario,
                    cue_fraction=scenario / (SCENARIO_COUNT - 1),
                    memory_first=False,
                ),
            )
            for scenario in range(SCENARIO_COUNT)
        ]
        self.assertTrue(
            any(
                "memory-conditioned choice appears after" in message
                for message in _messages(cases)
            )
        )

    def test_alternating_option_role_is_not_flagged(self) -> None:
        self.assertFalse(
            any(
                "memory-conditioned choice appears" in message
                for message in _messages(varied_cases())
            )
        )

    def test_reused_opening_line_is_flagged(self) -> None:
        cases = varied_cases()
        for scenario in range(4):
            cases[scenario]["observation_text"] = _observation(
                scenario,
                cue_fraction=scenario / (SCENARIO_COUNT - 1),
                memory_first=scenario % 2 == 1,
                opening="# working research notebook",
            )
        self.assertTrue(
            any("opening lines are shared" in message for message in _messages(cases))
        )

    def test_three_shared_opening_lines_are_allowed(self) -> None:
        cases = varied_cases()
        for scenario in range(3):
            cases[scenario]["observation_text"] = _observation(
                scenario,
                cue_fraction=scenario / (SCENARIO_COUNT - 1),
                memory_first=scenario % 2 == 1,
                opening="# working research notebook",
            )
        self.assertFalse(
            any("opening lines are shared" in message for message in _messages(cases))
        )

    def test_triplet_variants_count_once_per_base_scenario(self) -> None:
        cases = varied_cases()
        triplets = []
        for case in cases:
            for variant in ("positive", "cue-ablated", "memory-included"):
                clone = copy.deepcopy(case)
                clone["variant"] = variant
                clone["case_id"] = f"{case['base_case_id']}-{variant}"
                triplets.append(clone)
        self.assertEqual(construction_issues(triplets), [])


class JudgeLeakageTests(unittest.TestCase):
    def test_clean_instructions_report_no_violations(self) -> None:
        self.assertEqual(
            judge_leakage_violations(
                "Admit a memory only when the user's own words support it and "
                "the visible passage makes it useful to this task."
            ),
            [],
        )

    def test_rank_language_is_reported(self) -> None:
        self.assertEqual(
            judge_leakage_violations(
                "Admit it when recalling the fact gives a reason to choose a "
                "LOWER-RANKED viable candidate over the ordinary evidence "
                "winner instead of the already-strongest general option, and "
                "never reveal the Target Rank."
            ),
            [
                "lower-ranked",
                "ordinary evidence winner",
                "already-strongest",
                "target rank",
            ],
        )


class LegacyMixedDatasetRegressionTests(unittest.TestCase):
    def test_legacy_mixed_dataset_trips_the_detectors(self) -> None:
        # The legacy slice is development-only precisely because it repeats one
        # 9.8/8.8 rating pair and one envelope template pool across all thirty
        # scenarios. Validation profile gates are skipped on purpose here: the
        # detectors are called directly against the loaded cases.
        cases = load_cases(PERSONAMEM_MIXED_DATASET)
        messages = _messages(cases)
        self.assertTrue(
            any(
                "repeated numeric pair" in message and "8.8/9.8" in message
                for message in messages
            ),
            messages,
        )
        self.assertTrue(
            any("6-grams recur" in message for message in messages), messages
        )


if __name__ == "__main__":
    unittest.main()
