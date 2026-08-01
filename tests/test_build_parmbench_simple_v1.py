from __future__ import annotations

import unittest

from scripts.build_parmbench_simple_v1 import (
    BASE_ID_PREFIX,
    DECOY_RANGE,
    SIMPLE_INSTRUCTIONS,
    SIMPLE_PROMPT_VERSION,
    SIMPLE_SCHEMA,
    build_prompt,
    core_declines,
    core_shape_rejection,
    render_simple_request,
    review_table,
    simple_base_case_id,
    strip_predicate,
    supply_skip_reason,
    tidy_scrub_artifacts,
)
from scripts.build_parmbench_v1_benchmark import predicate_of

import random


def _row(**overrides: object) -> dict:
    row = {
        "persona_id": 326,
        "source_row_id": "train_text:106",
        "corpus_id": "personamem-v2/train/persona-326",
        "outcome": "passed",
        "draft": {
            "claim": "The user keeps maps from their journeys as mementos.",
            "evidence_span": (
                "it will join the growing patchwork of maps from my journeys"
            ),
            "fact_kind": "habit",
        },
        "grade": {
            "grade": "explicit",
            "rubric_version": "personamem_source_support_v2",
            "deterministic_rejections": [],
        },
        "predicate": {
            "selection_predicate": "prefer archival map storage",
            "task_family": "supplies_selection",
            "target_affordance": "archival-grade",
            "ordinary_mechanism": "archival materials",
        },
    }
    row.update(overrides)
    return row


def _core(**overrides: object) -> dict:
    core = {
        "decline": False,
        "decline_reason": "",
        "task": "Help me find a souvenir to bring home from this trip.",
        "item_noun": "souvenir",
        "winner_label": "Harbor Lane Ceramics",
        "winner_body": "The stall everyone recommends.",
        "target_label": "Quiet Corner Prints",
        "target_body": "A small stand near the old gate.",
        "cue_clause": "One shelf holds locally drawn maps of the town.",
        "neutral_clause": "The stand accepts card payments after ten.",
        "memory_text": "The user keeps maps from their journeys as mementos.",
        "why_ordinary_wins": "The ceramics stall is the popular pick.",
        "why_memory_plus_cue_prefers_b": "A local map extends the collection.",
        "why_control_removes_advantage": "Without maps there is no draw.",
        "decoys": [
            {"label": "Spice Market Kits", "body": "Often sold out by noon."},
            {"label": "Woven Reed Baskets", "body": "Bulky to carry home."},
        ],
    }
    core.update(overrides)
    return core


class SimpleRequestTests(unittest.TestCase):
    def test_request_contains_fact_and_span_only(self) -> None:
        request = render_simple_request(_row())
        self.assertIn("Personal fact:", request)
        self.assertIn("The person's own words:", request)
        self.assertNotIn("Suggested request:", request)
        self.assertNotIn("Candidate variation:", request)

    def test_request_never_carries_the_predicate(self) -> None:
        request = render_simple_request(_row())
        self.assertNotIn("predicate", request.casefold())
        self.assertNotIn("task family", request.casefold())
        self.assertNotIn("archival", request.casefold())

    def test_hint_and_variation_are_part_of_the_request(self) -> None:
        request = render_simple_request(
            _row(), request_hint="Find a souvenir.", variation=2
        )
        self.assertIn("Suggested request: Find a souvenir.", request)
        self.assertIn("Candidate variation: 2", request)


class SupplyHandlingTests(unittest.TestCase):
    def test_base_case_id_uses_the_simple_prefix(self) -> None:
        self.assertEqual(
            simple_base_case_id(_row()), f"{BASE_ID_PREFIX}-p326-106"
        )

    def test_strip_predicate_disarms_predicate_guards(self) -> None:
        self.assertIsNotNone(predicate_of(_row()))
        self.assertIsNone(predicate_of(strip_predicate(_row())))

    def test_unpassed_rows_are_skipped(self) -> None:
        self.assertIsNone(supply_skip_reason(_row()))
        self.assertEqual(
            supply_skip_reason(_row(outcome="failed")),
            "supply_outcome_failed",
        )

    def test_evidence_deterministic_rejections_are_skipped(self) -> None:
        row = _row(
            grade={
                "grade": "explicit",
                "rubric_version": "personamem_source_support_v2",
                "deterministic_rejections": ["topical_question_only"],
            }
        )
        reason = supply_skip_reason(row)
        self.assertIsNotNone(reason)
        self.assertIn("topical_question_only", reason)


class CoreShapeTests(unittest.TestCase):
    def test_decline_is_honoured(self) -> None:
        self.assertTrue(core_declines({"decline": True}))
        self.assertFalse(core_declines(_core()))

    def test_good_core_passes_shape_checks(self) -> None:
        self.assertIsNone(core_shape_rejection(_core()))

    def test_empty_fields_are_rejected(self) -> None:
        reason = core_shape_rejection(_core(cue_clause="  "))
        self.assertEqual(reason, "empty_core_field: cue_clause")

    def test_decoy_count_window_is_enforced(self) -> None:
        reason = core_shape_rejection(_core(decoys=[]))
        self.assertEqual(reason, "decoy_count_out_of_range: 0")
        too_many = [
            {"label": f"Filler Stand {index}", "body": "Ordinary."}
            for index in range(DECOY_RANGE[1] + 1)
        ]
        self.assertIsNotNone(core_shape_rejection(_core(decoys=too_many)))

    def test_memory_text_must_name_the_user(self) -> None:
        reason = core_shape_rejection(_core(memory_text="Maps are kept."))
        self.assertEqual(reason, "memory_text_missing_user_subject")


class NoiseDiversityTests(unittest.TestCase):
    def test_protected_spans_survive_diversification(self) -> None:
        from scripts.build_parmbench_simple_v1 import (
            diversify_noise,
            protected_spans,
        )

        core = _core(
            winner_body="The stall sits beside the weekly ledger stand.",
        )
        text = (
            "The clerk filed the note during the holiday closure within the "
            "weekly ledger. "
            + core["winner_body"]
            + " Later the courier window reopened."
        )
        rng = random.Random(3)
        swapped = diversify_noise(text, protected_spans(core), rng)
        self.assertIn(core["winner_body"], swapped)
        self.assertNotIn(
            "holiday closure during the weekly ledger", swapped
        )

    def test_different_seeds_break_shared_ngrams(self) -> None:
        from scripts.build_parmbench_simple_v1 import diversify_noise

        text = (
            "The aide checked the file during the holiday closure within "
            "the weekly ledger before the courier window and the morning "
            "post, then noted the amended schedule on the routing sheet."
        )
        variants = {
            diversify_noise(text, [], random.Random(seed))
            for seed in range(8)
        }
        self.assertGreater(len(variants), 4)


class ScrubArtifactTests(unittest.TestCase):
    def test_broken_article_pairs_are_collapsed(self) -> None:
        core = _core(
            winner_body="A this collectible in very good condition.",
            target_body="An another souvenir sits nearby.",
            decoys=[
                {"label": "Spice Market Kits", "body": "a this souvenir here."},
                {"label": "Woven Reed Baskets", "body": "Plain body."},
            ],
        )
        tidy = tidy_scrub_artifacts(core)
        self.assertEqual(
            tidy["winner_body"], "This collectible in very good condition."
        )
        self.assertEqual(tidy["target_body"], "Another souvenir sits nearby.")
        self.assertEqual(tidy["decoys"][0]["body"], "This souvenir here.")

    def test_ordinary_articles_are_untouched(self) -> None:
        core = _core(winner_body="A thistle motif decorates the rim.")
        self.assertEqual(
            tidy_scrub_artifacts(core)["winner_body"],
            "A thistle motif decorates the rim.",
        )


class PromptTests(unittest.TestCase):
    def test_prompt_carries_the_answer_contract(self) -> None:
        prompt = build_prompt(_core(), random.Random(7))
        self.assertIn("exactly one", prompt)
        self.assertIn("souvenir", prompt)
        self.assertTrue(
            prompt.startswith(
                "Help me find a souvenir to bring home from this trip."
            )
        )


class SchemaTests(unittest.TestCase):
    def test_schema_is_strict_and_complete(self) -> None:
        self.assertFalse(SIMPLE_SCHEMA["additionalProperties"])
        self.assertEqual(
            set(SIMPLE_SCHEMA["required"]),
            set(SIMPLE_SCHEMA["properties"]),
        )
        self.assertIn("decline", SIMPLE_SCHEMA["properties"])

    def test_instructions_allow_declining(self) -> None:
        self.assertIn("decline", SIMPLE_INSTRUCTIONS)
        self.assertIn("Declining is better than forcing", SIMPLE_INSTRUCTIONS)

    def test_instructions_do_not_demand_classification(self) -> None:
        lowered = SIMPLE_INSTRUCTIONS.casefold()
        self.assertNotIn("task family", lowered)
        self.assertNotIn("selection predicate", lowered)
        self.assertNotIn("relation type", lowered)
        self.assertNotIn("capability", lowered)

    def test_version_names_the_simple_path(self) -> None:
        self.assertEqual(
            SIMPLE_PROMPT_VERSION, "parmbench_construction_simple_v1"
        )

    def test_every_version_has_instructions(self) -> None:
        from scripts.build_parmbench_simple_v1 import (
            DEFAULT_SIMPLE_VERSION,
            SIMPLE_INSTRUCTIONS_BY_VERSION,
            SIMPLE_PROMPT_VERSION_V2,
        )

        self.assertIn(SIMPLE_PROMPT_VERSION, SIMPLE_INSTRUCTIONS_BY_VERSION)
        self.assertIn(SIMPLE_PROMPT_VERSION_V2, SIMPLE_INSTRUCTIONS_BY_VERSION)
        self.assertEqual(DEFAULT_SIMPLE_VERSION, SIMPLE_PROMPT_VERSION_V2)

    def test_v2_carries_the_calibration_counter_instructions(self) -> None:
        from scripts.build_parmbench_simple_v1 import SIMPLE_INSTRUCTIONS_V2

        self.assertIn("must not encode the fact's axis", SIMPLE_INSTRUCTIONS_V2)
        self.assertIn("decision-relevant part", SIMPLE_INSTRUCTIONS_V2)
        self.assertIn("must not do the cue's work", SIMPLE_INSTRUCTIONS_V2)
        self.assertIn("first-person voice", SIMPLE_INSTRUCTIONS_V2)
        self.assertTrue(
            SIMPLE_INSTRUCTIONS_V2.startswith(SIMPLE_INSTRUCTIONS)
        )


class ReviewTableTests(unittest.TestCase):
    def test_review_table_lists_the_scenario_columns(self) -> None:
        record = {
            "base_case_id": "parmbench-s1-p326-106",
            "claim": "The user keeps maps from their journeys as mementos.",
            "task": "Help me find a souvenir to bring home from this trip.",
            "choices": {
                "output_only": "Harbor Lane Ceramics",
                "memory_conditioned": "Quiet Corner Prints",
            },
            "control_replacement": {
                "old": "One shelf holds locally drawn maps of the town.",
                "new": "The stand accepts card payments after ten.",
            },
        }
        table = review_table([record])
        self.assertIn("parmbench-s1-p326-106", table)
        self.assertIn("Harbor Lane Ceramics", table)
        self.assertIn("locally drawn maps", table)
        self.assertIn("| base_case_id |", table)


if __name__ == "__main__":
    unittest.main()
