from __future__ import annotations

import random
import unittest

from scripts.build_parmbench_v1_benchmark import (
    CAUSAL_CHAIN_FIELDS,
    CONSTRUCTION_PROMPT_VERSION,
    CONSTRUCTION_SCHEMA,
    capability_for,
    causal_chain_record,
    causal_chain_rejection,
    normalise_core,
    prompt_claim_overlap,
    render_construction_request,
    scenario_rejection,
    source_turns,
)
from scripts.draft_parmbench_v1_claims import span_rejection
from scripts.parmbench_v1_envelopes import (
    ACTION_OBJECTS,
    ACTION_VERBS,
    CONNECTORS,
    DETAIL_LEADS,
    DETAIL_TAILS,
    ENVELOPES,
    PLACE_NOUNS,
    QUALIFIER_NOUNS,
    ROLE_ADJECTIVES,
    ROLE_NOUNS,
    balanced_assignment,
)


def _chain(**overrides: object) -> dict:
    chain = {
        "why_ordinary_wins": (
            "The morning slot is the only one that fits the stated diary."
        ),
        "why_cue_neutral_without_memory": (
            "A step-free entrance reads as a routine building detail."
        ),
        "why_memory_plus_cue_prefers_b": (
            "Somebody walking with a wheeled frame cannot use the stairs the "
            "other rooms are reached by."
        ),
        "assumptions_required": [],
        "why_control_removes_advantage": (
            "With the stair sentence in place the room offers nothing the "
            "other rooms do not."
        ),
    }
    chain.update(overrides)
    return chain


def _core(**overrides: object) -> dict:
    core = {
        "task": "Pick a slot for the weekly session.",
        "item_noun": "slot",
        "winner_label": "Harbour Morning",
        "winner_body": "Runs at a time that fits the stated schedule.",
        "target_label": "Riverside Evening",
        "target_body": "A quiet room on the upper floor.",
        "cue_clause": "The room keeps a step-free entrance from the lane.",
        "neutral_clause": "The room is reached by a short flight of stairs.",
        "memory_text": "The user uses a wheeled frame outdoors.",
        "decoys": [
            {"label": "Northgate Late", "body": "Close, but the room is upstairs."},
            {"label": "Selkirk Midday", "body": "Fits the diary but is fully booked."},
            {"label": "Clover Weekend", "body": "Step-free but only on alternate weeks."},
        ],
    }
    core.update(overrides)
    return core


class NormaliseCoreTests(unittest.TestCase):
    def test_cue_repeated_in_the_target_body_is_lifted_out(self) -> None:
        cue = "The room keeps a step-free entrance from the lane."
        core = _core(target_body=f"A quiet room on the upper floor. {cue}")
        normalised = normalise_core(core)
        self.assertNotIn(cue, normalised["target_body"])
        self.assertEqual(normalised["cue_clause"], cue)

    def test_cue_repeated_in_a_decoy_body_is_lifted_out(self) -> None:
        cue = "The room keeps a step-free entrance from the lane."
        core = _core()
        core["decoys"][0]["body"] = f"Close, but noisy. {cue}"
        normalised = normalise_core(core)
        self.assertNotIn(cue, normalised["decoys"][0]["body"])

    def test_normalising_does_not_mutate_the_cached_result(self) -> None:
        core = _core(target_body="A quiet room. The room keeps a step-free entrance from the lane.")
        normalise_core(core)
        self.assertIn("step-free entrance", core["target_body"])


class ScenarioRejectionTests(unittest.TestCase):
    def _document(self, core: dict) -> str:
        entries = [
            f"{core['winner_label']}: {core['winner_body']}",
            f"{core['target_label']}: {core['target_body']} {core['cue_clause']}",
        ]
        entries.extend(
            f"{decoy['label']}: {decoy['body']}" for decoy in core["decoys"]
        )
        return "log extract\n\n" + "\n\n".join(entries) + "\n"

    def _claim_row(self) -> dict:
        return {
            "draft": {
                "evidence_span": "I use a wheeled frame when I go out.",
                "claim": "The user walks with a wheeled frame outdoors.",
            }
        }

    def test_a_well_formed_scenario_is_accepted(self) -> None:
        core = normalise_core(_core())
        prompt = "Pick a slot. Name exactly one slot from the material below."
        self.assertIsNone(
            scenario_rejection(self._document(core), core, prompt, self._claim_row())
        )

    def test_a_repeated_cue_is_rejected(self) -> None:
        core = normalise_core(_core())
        text = self._document(core) + core["cue_clause"] + "\n"
        prompt = "Pick a slot. Name exactly one slot from the material below."
        self.assertEqual(
            scenario_rejection(text, core, prompt, self._claim_row()),
            "cue_not_unique_in_observation",
        )

    def test_a_label_that_contains_another_label_is_rejected(self) -> None:
        core = normalise_core(_core(target_label="Harbour Morning Annexe"))
        prompt = "Pick a slot. Name exactly one slot from the material below."
        self.assertIn(
            scenario_rejection(self._document(core), core, prompt, self._claim_row()),
            {"label_is_substring_of_another", "label_not_unique_in_observation"},
        )

    def test_a_prompt_without_the_answer_contract_is_rejected(self) -> None:
        core = normalise_core(_core())
        self.assertEqual(
            scenario_rejection(
                self._document(core), core, "Pick a slot.", self._claim_row()
            ),
            "prompt_missing_answer_contract",
        )

    def test_an_evidence_span_in_the_observation_is_rejected(self) -> None:
        core = normalise_core(_core())
        row = self._claim_row()
        text = self._document(core) + row["draft"]["evidence_span"] + "\n"
        prompt = "Pick a slot. Name exactly one slot from the material below."
        self.assertEqual(
            scenario_rejection(text, core, prompt, row),
            "evidence_span_leaks_into_observation",
        )


class PromptOpacityTests(unittest.TestCase):
    def test_short_tokens_count_towards_the_overlap(self) -> None:
        self.assertEqual(
            prompt_claim_overlap(
                "Pick a supplier for gpu racks.",
                "The user tunes gpu training jobs.",
            ),
            ("gpu",),
        )

    def test_an_unrelated_prompt_shares_nothing(self) -> None:
        self.assertEqual(
            prompt_claim_overlap(
                "Choose a venue for the rehearsal.",
                "The user visits the harbour library.",
            ),
            (),
        )

    def test_a_prompt_echoing_the_claim_is_rejected(self) -> None:
        core = normalise_core(_core())
        prompt = (
            "Find a wheeled frame friendly outdoors walking route. "
            "Name exactly one slot from the material below."
        )
        row = {
            "draft": {
                "evidence_span": "I use a wheeled frame when I go out.",
                "claim": "The user walks with a wheeled frame outdoors.",
            }
        }
        entries = [
            f"{core['winner_label']}: {core['winner_body']}",
            f"{core['target_label']}: {core['target_body']} {core['cue_clause']}",
        ]
        entries.extend(f"{d['label']}: {d['body']}" for d in core["decoys"])
        text = "log extract\n\n" + "\n\n".join(entries) + "\n"
        self.assertEqual(
            scenario_rejection(text, core, prompt, row),
            "prompt_shares_too_much_wording_with_the_claim",
        )


class RepairRequestTests(unittest.TestCase):
    spec = {
        "claim": "The user walks with a wheeled frame outdoors.",
        "evidence_span": "I use a wheeled frame when I go out.",
        "domain": "choosing a volunteering slot",
        "mechanism": "schedule fit",
        "ratings_allowed": False,
        "overlap_mode": "paraphrase_only",
        "decoy_count": 4,
        "register": "log extract",
    }

    def test_an_unrepaired_scenario_renders_the_original_request(self) -> None:
        rendered = render_construction_request({**self.spec, "repair_attempt": 0})
        self.assertNotIn("Regeneration attempt", rendered)
        self.assertEqual(
            rendered, render_construction_request(dict(self.spec))
        )

    def test_a_repair_changes_the_request_per_attempt(self) -> None:
        first = render_construction_request(
            {**self.spec, "repair_attempt": 1, "repair_failed_variants": ()}
        )
        second = render_construction_request(
            {**self.spec, "repair_attempt": 2, "repair_failed_variants": ()}
        )
        self.assertIn("Regeneration attempt: 1", first)
        self.assertIn("Regeneration attempt: 2", second)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, render_construction_request(dict(self.spec)))

    def test_the_failing_variant_is_named_in_the_repair_request(self) -> None:
        rendered = render_construction_request(
            {
                **self.spec,
                "repair_attempt": 1,
                "repair_failed_variants": ("memory-included",),
            }
        )
        self.assertIn("was not decisive for this task", rendered)
        self.assertNotIn("already picked the", rendered)


class CausalChainTests(unittest.TestCase):
    def _document(self, core: dict) -> str:
        entries = [
            f"{core['winner_label']}: {core['winner_body']}",
            f"{core['target_label']}: {core['target_body']} {core['cue_clause']}",
        ]
        entries.extend(
            f"{decoy['label']}: {decoy['body']}" for decoy in core["decoys"]
        )
        return "log extract\n\n" + "\n\n".join(entries) + "\n"

    def _claim_row(self) -> dict:
        return {
            "draft": {
                "evidence_span": "I use a wheeled frame when I go out.",
                "claim": "The user walks with a wheeled frame outdoors.",
            }
        }

    def test_the_schema_requires_every_causal_chain_field(self) -> None:
        for field in CAUSAL_CHAIN_FIELDS + ("assumptions_required",):
            self.assertIn(field, CONSTRUCTION_SCHEMA["properties"])
            self.assertIn(field, CONSTRUCTION_SCHEMA["required"])

    def test_the_prompt_version_is_bumped_for_the_causal_chain(self) -> None:
        self.assertEqual(CONSTRUCTION_PROMPT_VERSION, "parmbench_construction_v4")

    def test_a_complete_chain_without_assumptions_is_accepted(self) -> None:
        self.assertIsNone(causal_chain_rejection(_core(**_chain())))

    def test_a_declared_assumption_rejects_the_core(self) -> None:
        core = _core(
            **_chain(
                assumptions_required=[
                    "that the user attends this session in person"
                ]
            )
        )
        self.assertEqual(
            causal_chain_rejection(core),
            "construction_declares_required_assumptions",
        )

    def test_a_missing_chain_field_rejects_the_core(self) -> None:
        core = _core(**_chain(why_ordinary_wins="  "))
        self.assertEqual(causal_chain_rejection(core), "missing_causal_chain")

    def test_a_core_without_a_chain_at_all_rejects(self) -> None:
        self.assertEqual(causal_chain_rejection(_core()), "missing_causal_chain")

    def test_the_chain_is_recorded_with_its_assumptions(self) -> None:
        record = causal_chain_record(_core(**_chain(assumptions_required=["a", " "])))
        self.assertEqual(record["assumptions_required"], ["a"])
        for field in CAUSAL_CHAIN_FIELDS:
            self.assertTrue(record[field])

    def test_a_chain_sentence_in_the_observation_is_rejected(self) -> None:
        chain = _chain()
        core = normalise_core(_core(**chain))
        text = self._document(core) + chain["why_ordinary_wins"] + "\n"
        prompt = "Pick a slot. Name exactly one slot from the material below."
        self.assertEqual(
            scenario_rejection(text, core, prompt, self._claim_row()),
            "causal_chain_leaks_into_observation",
        )

    def test_a_chain_sentence_in_the_prompt_is_rejected(self) -> None:
        chain = _chain()
        core = normalise_core(_core(**chain))
        prompt = (
            "Pick a slot. Name exactly one slot from the material below. "
            + str(chain["why_control_removes_advantage"])
        )
        self.assertEqual(
            scenario_rejection(
                self._document(core), core, prompt, self._claim_row()
            ),
            "causal_chain_leaks_into_prompt",
        )

    def test_a_scenario_that_keeps_the_chain_private_is_accepted(self) -> None:
        core = normalise_core(_core(**_chain()))
        prompt = "Pick a slot. Name exactly one slot from the material below."
        self.assertIsNone(
            scenario_rejection(self._document(core), core, prompt, self._claim_row())
        )


class SourceTurnTests(unittest.TestCase):
    def test_a_stored_record_splits_back_into_roles(self) -> None:
        turns = source_turns(
            "User: Could you polish this?\n\nAssistant: Here is a tidier "
            "version.\n\nUser: Add a line about the console."
        )
        self.assertEqual(
            [turn["role"] for turn in turns], ["user", "assistant", "user"]
        )
        self.assertEqual(turns[2]["content"], "Add a line about the console.")

    def test_an_untagged_record_is_kept_whole(self) -> None:
        turns = source_turns("a plain note with no role prefixes")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["role"], "source record")


class CapabilityTests(unittest.TestCase):
    def test_an_avoidance_claim_is_an_exclusion(self) -> None:
        self.assertEqual(
            capability_for("The user skips syrupy desserts.", "habit", False, "share_wording"),
            "negative_preference_exclusion",
        )

    def test_a_commitment_claim_is_a_schedule_case(self) -> None:
        self.assertEqual(
            capability_for("The user sits on the Tuesday panel.", "commitment", False, "share_wording"),
            "schedule_commitment",
        )

    def test_a_named_relation_with_a_hop_is_relational(self) -> None:
        self.assertEqual(
            capability_for("The user's sister runs a shop in Leeds.", "relationship", True, "paraphrase_only"),
            "one_hop_relational",
        )

    def test_a_recovery_constraint_is_no_longer_an_exclusion(self) -> None:
        # The audit's knee-surgery case: an accessibility need rules nothing
        # out, so the old constraint shortcut mislabelled it.
        self.assertNotEqual(
            capability_for(
                "The user was recovering from minor knee surgery.",
                "constraint",
                True,
                "paraphrase_only",
            ),
            "negative_preference_exclusion",
        )

    def test_a_memory_category_overrides_the_older_fact_kind(self) -> None:
        self.assertEqual(
            capability_for(
                "The user keeps a standing Thursday shift at the depot.",
                "taste",
                False,
                "share_wording",
                memory_category="concrete_schedule",
            ),
            "schedule_commitment",
        )

    def test_wording_relationship_splits_plain_facts(self) -> None:
        self.assertEqual(
            capability_for("The user plays the cello.", "taste", False, "share_wording"),
            "direct_lexical_fact",
        )
        self.assertEqual(
            capability_for("The user plays the cello.", "taste", False, "paraphrase_only"),
            "paraphrased_semantic_fact",
        )


class SpanRejectionTests(unittest.TestCase):
    snippet = (
        {"role": "user", "content": "I walk to the harbour every payday and buy fish."},
        {"role": "assistant", "content": "That sounds like a nice routine."},
    )

    def test_a_verbatim_user_span_is_accepted(self) -> None:
        self.assertIsNone(
            span_rejection("I walk to the harbour every payday", self.snippet)
        )

    def test_a_span_only_the_assistant_wrote_is_rejected(self) -> None:
        self.assertEqual(
            span_rejection("That sounds like a nice routine.", self.snippet),
            "span_not_verbatim_in_user_turn",
        )

    def test_a_span_with_a_line_break_is_rejected(self) -> None:
        self.assertEqual(
            span_rejection("I walk to the harbour\nevery payday", self.snippet),
            "span_has_line_break",
        )

    def test_a_short_span_is_rejected(self) -> None:
        self.assertEqual(
            span_rejection("I walk", self.snippet), "span_length_out_of_range"
        )


class EnvelopeVocabularyTests(unittest.TestCase):
    """Guard the entropy argument behind the filler generator.

    Every slot value stays at four words or fewer and every connector is a
    single word, so a six-word run of filler always spans at least two
    independently drawn slots.
    """

    def test_slot_values_are_short(self) -> None:
        for pool in (
            ROLE_ADJECTIVES,
            ROLE_NOUNS,
            ACTION_VERBS,
            ACTION_OBJECTS,
            PLACE_NOUNS,
            QUALIFIER_NOUNS,
            DETAIL_LEADS,
            DETAIL_TAILS,
        ):
            for value in pool:
                self.assertLessEqual(len(value.split()), 4, value)

    def test_connectors_are_single_words(self) -> None:
        for connector in CONNECTORS:
            self.assertLessEqual(len(connector.split()), 1, connector)

    def test_at_least_twelve_envelope_styles(self) -> None:
        self.assertGreaterEqual(len(ENVELOPES), 12)
        self.assertEqual(len({e.name for e in ENVELOPES}), len(ENVELOPES))
        for envelope in ENVELOPES:
            self.assertGreaterEqual(len(envelope.openings), 3)

    def test_balanced_assignment_spreads_options_evenly(self) -> None:
        assignment = balanced_assignment(("a", "b", "c"), 10, random.Random(3))
        self.assertEqual(len(assignment), 10)
        counts = {option: assignment.count(option) for option in ("a", "b", "c")}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
