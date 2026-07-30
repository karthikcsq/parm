from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

from parm_bench.decision_validity import (
    build_scenario,
    capability_conflicts_with_lexical_target,
    control_residual_advantage,
)
from scripts.build_parmbench_v1_benchmark import (
    CAUSAL_CHAIN_FIELDS,
    CONSTRUCTION_PROMPT_VERSION,
    CONSTRUCTION_PROMPT_VERSION_V5,
    CONSTRUCTION_SCHEMA,
    DOMAINS,
    NEUTRAL_LENGTH_BAND,
    NO_PREDICATE_REASON,
    PREDICATE_ASSUMPTIONS_REASON,
    TASK_FAMILY_SURFACES,
    build_case_rows,
    build_specs,
    capability_for,
    case_sensitive_terms,
    causal_chain_record,
    causal_chain_rejection,
    control_negates_cue,
    decoy_annotates_axis,
    neutral_length_ratio,
    normalise_core,
    predicate_of,
    predicate_skip_reason,
    prompt_claim_overlap,
    prompt_states_mechanism_as_instruction,
    render_construction_request,
    resolve_task_family,
    scenario_rejection,
    source_turns,
    task_surfaces_for,
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


ANCHOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "parmbench-v1-supply"
    / "selection_predicate_anchors.json"
)


def _anchors() -> dict[str, dict]:
    payload = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))
    return {anchor["anchor_id"]: anchor for anchor in payload["anchors"]}


# Hand-authored realisations of the three approved anchor scenarios. The three
# predicate fields the anchors file does not carry - target affordance,
# ordinary mechanism, control affordance - are written here against that
# anchor's own task_shape, ordinary_winner_shape, target_shape, cue_rule, and
# control_rule. Everything else is read from the file.
ANCHOR_REALISATIONS: dict[str, dict] = {
    "anchor-dessert-menu": {
        "target_affordance": (
            "a dessert of plain cut strawberries with nothing poured over them"
        ),
        "ordinary_mechanism": (
            "chef recommendation and long-standing signature status"
        ),
        "control_affordance": "an ordinary serving note about the china",
        "relational_hop": False,
        "prompt": (
            "A friend has sent over the dessert card from tonight's supper "
            "and wants one dish picked for the table. Name exactly one "
            "dessert from the material below."
        ),
        "core": {
            "task": (
                "A friend has sent over the dessert card from tonight's "
                "supper and wants one dish picked for the table."
            ),
            "item_noun": "dessert",
            "winner_label": "Kitchen Signature Jamun",
            "winner_body": (
                "The kitchen has served this jamun since it opened and the "
                "pastry chef puts it forward first at every supper. Each "
                "piece arrives soaked in warm rose syrup."
            ),
            "target_label": "Amber Lane Plate",
            "target_body": (
                "It is the newest item on the card and the kitchen has not "
                "put it forward to anyone yet. Everything on it is prepared "
                "in the pastry room at the back."
            ),
            "cue_clause": (
                "Each plate is a handful of strawberries cut that morning "
                "and set down with nothing poured over them."
            ),
            "neutral_clause": (
                "Each plate is carried out from the pastry room on the same "
                "white china as the rest of the card."
            ),
            "memory_text": (
                "The user skips syrupy desserts and chooses a bowl of fresh "
                "fruit instead."
            ),
            "decoys": [
                {
                    "label": "Cardamom Milk Ice",
                    "body": (
                        "This one has been on the card a fortnight and the "
                        "kitchen puts it forward only when the jamun runs "
                        "out."
                    ),
                },
                {
                    "label": "Chocolate Fig Tart",
                    "body": (
                        "The pastry chef added this last month and it is the "
                        "one dish the kitchen sends back to be reworked most "
                        "often."
                    ),
                },
                {
                    "label": "Burnt Honey Custard",
                    "body": (
                        "It came onto the card this week and the kitchen has "
                        "yet to decide whether to keep it."
                    ),
                },
            ],
            "why_ordinary_wins": (
                "The jamun is the dish the kitchen stands behind and puts in "
                "front of every table."
            ),
            "why_cue_neutral_without_memory": (
                "A plate of cut strawberries reads as a light option and "
                "nothing more."
            ),
            "why_memory_plus_cue_prefers_b": (
                "Somebody who skips syrup-soaked sweets cannot take the "
                "jamun, and the plate is the one dish that suits them."
            ),
            "assumptions_required": [],
            "why_control_removes_advantage": (
                "With the china sentence in place the plate is an unremarked "
                "new item with nothing on it for this person."
            ),
        },
    },
    "anchor-reading-group": {
        "target_affordance": (
            "a study of how established faith communities revise their "
            "teaching inside religiously mixed democracies"
        ),
        "ordinary_mechanism": (
            "how warmly a title was received and how much a table can argue "
            "about it"
        ),
        "control_affordance": "a plain note about the edition and its printing",
        "relational_hop": False,
        "prompt": (
            "Someone has sent over the shortlist their college reading group "
            "drew up for next month and wants one title picked. Reply with "
            "exactly one book name taken from the material below."
        ),
        "core": {
            "task": (
                "Someone has sent over the shortlist their college reading "
                "group drew up for next month and wants one title picked."
            ),
            "item_noun": "book",
            "winner_label": "The Glass Orbit",
            "winner_body": (
                "It arrived to warm notices last spring and the convenor "
                "says it splits a table within ten minutes. Copies sit on "
                "the college library shelf already."
            ),
            "target_label": "Borrowed Ground",
            "target_body": (
                "A compact volume from a university press that few outside "
                "its field have picked up. The shop can order it in by the "
                "end of the week."
            ),
            "cue_clause": (
                "Its middle chapters follow what established faith "
                "communities change in their teaching after moving into "
                "religiously mixed democracies."
            ),
            "neutral_clause": (
                "Its middle chapters were reset for a second printing that "
                "the same press issued three years ago."
            ),
            "memory_text": (
                "The user is drafting a piece on how theological traditions "
                "adjust in politically diverse societies."
            ),
            "decoys": [
                {
                    "label": "Salt And Ledger",
                    "body": (
                        "A port-town history the convenor liked and nobody "
                        "else finished last winter."
                    ),
                },
                {
                    "label": "Harbour Town Letters",
                    "body": (
                        "A memoir the college shop has in stock, though two "
                        "members already know it from school."
                    ),
                },
                {
                    "label": "Nine Winters Down",
                    "body": (
                        "A thriller that arrived to warm notices but gives a "
                        "table very little to disagree over."
                    ),
                },
            ],
            "why_ordinary_wins": (
                "A group choosing for one evening takes the title everybody "
                "has heard of and can argue about."
            ),
            "why_cue_neutral_without_memory": (
                "A line about the middle chapters reads as an ordinary "
                "summary of what the volume covers."
            ),
            "why_memory_plus_cue_prefers_b": (
                "The volume feeds the piece this person is writing, so a "
                "month spent on it is a month of their own work."
            ),
            "assumptions_required": [],
            "why_control_removes_advantage": (
                "A note about the printing leaves the volume with nothing "
                "that bears on the piece."
            ),
        },
    },
    "anchor-vintage-store": {
        "target_affordance": (
            "a grey cartridge cut for the slot a Nintendo Entertainment "
            "System deck takes"
        ),
        "ordinary_mechanism": (
            "how rare the piece is and what condition the shelf card claims"
        ),
        "control_affordance": "a plain note about how the lot was stored",
        "relational_hop": True,
        "prompt": (
            "A friend has walked around a vintage-items store and typed up "
            "what is on the shelves; they want one thing called out as worth "
            "a look. State exactly one item from the list below and no other "
            "text."
        ),
        "core": {
            "task": (
                "A friend has walked around a vintage-items store and typed "
                "up what is on the shelves; they want one thing called out "
                "as worth a look."
            ),
            "item_noun": "item",
            "winner_label": "Brass Diving Lamp",
            "winner_body": (
                "The store has had two of these in twenty years and this one "
                "still carries its original glass. The shelf card puts the "
                "condition close to unused."
            ),
            "target_label": "Sandpiper Box Lot",
            "target_body": (
                "The lot came in from a house clearance last month and the "
                "store has yet to sort it. It sits under the counter with "
                "the rest of that delivery."
            ),
            "cue_clause": (
                "The box holds three grey cartridges cut for the slot a "
                "Nintendo Entertainment System deck takes."
            ),
            "neutral_clause": (
                "The box holds three folded tablecloths the previous seller "
                "kept wrapped in tissue."
            ),
            "memory_text": "The user owns an old NES console.",
            "decoys": [
                {
                    "label": "Enamel Tin Sign",
                    "body": (
                        "Common enough that the store has three of them, and "
                        "one corner is bent back."
                    ),
                },
                {
                    "label": "Folding Camp Stool",
                    "body": (
                        "Sound frame, though the canvas was replaced at some "
                        "point and the store admits it."
                    ),
                },
                {
                    "label": "Chrome Desk Fan",
                    "body": (
                        "Rare in this finish, but the motor has been apart "
                        "and nobody has run it since."
                    ),
                },
            ],
            "why_ordinary_wins": (
                "Two in twenty years and original glass is the strongest "
                "ordinary evidence on the shelves."
            ),
            "why_cue_neutral_without_memory": (
                "An unsorted box of old cartridges is a shelf detail for "
                "anybody who cannot run them."
            ),
            "why_memory_plus_cue_prefers_b": (
                "Somebody with that console at home can use the contents of "
                "the box, which nothing else on the shelves offers."
            ),
            "assumptions_required": [],
            "why_control_removes_advantage": (
                "Folded tablecloths leave the lot an unsorted clearance box "
                "with nothing usable in it."
            ),
        },
    },
}


def _predicate_for(anchor: dict, realisation: dict) -> dict:
    """The mapper's documented dict, filled from the frozen anchor."""

    return {
        "selection_predicate": anchor["selection_predicate"],
        "task_family": anchor["task_family"],
        "target_affordance": realisation["target_affordance"],
        "ordinary_mechanism": realisation["ordinary_mechanism"],
        "control_affordance": realisation["control_affordance"],
        "relation_type": anchor["relation_type"],
        "material_assumptions": [],
        "sensitive": anchor["sensitive"],
        "sensitive_terms": list(anchor["sensitive_terms"]),
    }


def _render_document(core: dict) -> str:
    entries = [
        f"{core['winner_label']}: {core['winner_body']}",
        f"{core['target_label']}: {core['target_body']} {core['cue_clause']}",
    ]
    entries.extend(f"{decoy['label']}: {decoy['body']}" for decoy in core["decoys"])
    return "store listing\n\n" + "\n\n".join(entries) + "\n"


class AnchorRealisationTests(unittest.TestCase):
    """The three approved anchors must survive the v5 construction checks.

    A generator version that cannot produce these three examples is not fit to
    generate anything else, so the shapes frozen in
    `selection_predicate_anchors.json` are exercised end to end against the
    deterministic rejections and the relevance backstops.
    """

    def setUp(self) -> None:
        self.anchors = _anchors()

    def test_every_anchor_family_is_in_the_surface_registry(self) -> None:
        for anchor_id, anchor in self.anchors.items():
            with self.subTest(anchor=anchor_id):
                resolved, known = resolve_task_family(anchor["task_family"])
                self.assertTrue(known, anchor["task_family"])
                self.assertIn(resolved, TASK_FAMILY_SURFACES)
                self.assertGreaterEqual(
                    len(task_surfaces_for(anchor["task_family"])), 3
                )

    def test_each_anchor_realisation_is_accepted(self) -> None:
        for anchor_id, anchor in self.anchors.items():
            realisation = ANCHOR_REALISATIONS[anchor_id]
            predicate = _predicate_for(anchor, realisation)
            core = normalise_core(realisation["core"])
            prompt = realisation["prompt"]
            row = {
                "draft": {
                    "claim": anchor["fact"],
                    "evidence_span": anchor["evidence_span"],
                },
                "predicate": predicate,
            }
            with self.subTest(anchor=anchor_id):
                self.assertIsNone(causal_chain_rejection(core))
                self.assertIsNone(
                    scenario_rejection(_render_document(core), core, prompt, row)
                )
                self.assertEqual(decoy_annotates_axis(core, predicate, prompt), ())
                self.assertEqual(control_negates_cue(core), ())
                self.assertEqual(
                    prompt_states_mechanism_as_instruction(
                        prompt, predicate["ordinary_mechanism"]
                    ),
                    (),
                )
                ratio = neutral_length_ratio(core)
                self.assertGreaterEqual(ratio, NEUTRAL_LENGTH_BAND[0])
                self.assertLessEqual(ratio, NEUTRAL_LENGTH_BAND[1])

    def test_each_anchor_realisation_passes_the_relevance_backstops(self) -> None:
        for anchor_id, anchor in self.anchors.items():
            realisation = ANCHOR_REALISATIONS[anchor_id]
            predicate = _predicate_for(anchor, realisation)
            core = normalise_core(realisation["core"])
            capability = capability_for(
                anchor["fact"],
                "habit",
                realisation["relational_hop"],
                "paraphrase_only",
                cue_text=str(core["cue_clause"]),
                evidence_span=anchor["evidence_span"],
            )
            scenario = build_scenario(
                claim=anchor["fact"],
                evidence_span=anchor["evidence_span"],
                task_prompt=realisation["prompt"],
                core=core,
                ordinary_mechanism=predicate["ordinary_mechanism"],
                capability_label=capability,
                sensitive_terms=predicate["sensitive_terms"],
            )
            with self.subTest(anchor=anchor_id):
                self.assertEqual(control_residual_advantage(scenario), ())
                self.assertFalse(
                    capability_conflicts_with_lexical_target(
                        capability,
                        anchor["fact"],
                        anchor["evidence_span"],
                        str(core["target_label"]),
                    )
                )


def _predicate_row(**overrides: object) -> dict:
    predicate = {
        "selection_predicate": (
            "Prefer a room reachable without stairs."
        ),
        "task_family": "room_route_or_seating_selection",
        "target_affordance": "a step-free entrance from the lane",
        "ordinary_mechanism": "schedule fit",
        "control_affordance": "an ordinary note about the room's furniture",
        "relation_type": "accessibility_need",
        "material_assumptions": [],
        "sensitive": False,
        "sensitive_terms": [],
    }
    predicate.update(overrides)
    return {
        "draft": {
            "evidence_span": "I use a wheeled frame when I go out.",
            "claim": "The user walks with a wheeled frame outdoors.",
        },
        "predicate": predicate,
    }


def _predicate_core(**overrides: object) -> dict:
    """A v5-shaped core: decoys compete on ordinary grounds alone."""

    core = _core(**_chain())
    core["item_noun"] = "room"
    core["decoys"] = [
        {
            "label": "Northgate Late",
            "body": "Close by, but another group already holds that hour.",
        },
        {
            "label": "Selkirk Midday",
            "body": "Fits the diary but is fully booked.",
        },
        {
            "label": "Clover Weekend",
            "body": "Runs at the right hour only on alternate weeks.",
        },
    ]
    core.update(overrides)
    return core


class PredicateGuardTests(unittest.TestCase):
    """The three fixture-level gaps the v3 pilot left open."""

    prompt = "Pick a room. Name exactly one room from the material below."

    def test_a_clean_predicate_scenario_is_accepted(self) -> None:
        core = normalise_core(_predicate_core())
        self.assertIsNone(
            scenario_rejection(
                _render_document(core), core, self.prompt, _predicate_row()
            )
        )

    def test_a_decoy_annotating_the_axis_is_rejected(self) -> None:
        core = _predicate_core()
        core["decoys"][0]["body"] = (
            "Close by, but there is no step-free entrance of any kind."
        )
        core = normalise_core(core)
        self.assertEqual(
            scenario_rejection(
                _render_document(core), core, self.prompt, _predicate_row()
            ),
            "decoy_annotates_personalisation_axis",
        )

    def test_the_axis_guard_names_the_repeated_words(self) -> None:
        row = _predicate_row()
        core = _predicate_core()
        core["decoys"][1]["body"] = "Fits the diary but the entrance is narrow."
        self.assertIn(
            "entrance",
            decoy_annotates_axis(
                normalise_core(core), row["predicate"], self.prompt
            ),
        )

    def test_a_decoy_on_ordinary_grounds_is_accepted(self) -> None:
        row = _predicate_row()
        core = _predicate_core()
        core["decoys"][0]["body"] = "Runs at the same hour but is fully booked."
        self.assertEqual(
            decoy_annotates_axis(
                normalise_core(core), row["predicate"], self.prompt
            ),
            (),
        )

    def test_an_antonym_control_is_rejected(self) -> None:
        core = normalise_core(
            _predicate_core(
                neutral_clause=(
                    "The room keeps no step-free entrance from the lane."
                ),
            )
        )
        self.assertEqual(
            scenario_rejection(
                _render_document(core), core, self.prompt, _predicate_row()
            ),
            "control_negates_the_cue",
        )

    def test_a_control_that_denies_nothing_is_accepted(self) -> None:
        core = normalise_core(_predicate_core())
        self.assertEqual(control_negates_cue(core), ())

    def test_a_negated_control_about_something_else_is_accepted(self) -> None:
        core = normalise_core(
            _predicate_core(
                neutral_clause=(
                    "Bookings close at noon and no refunds are issued after "
                    "that."
                ),
            )
        )
        self.assertEqual(control_negates_cue(core), ())

    def test_a_prompt_stating_the_mechanism_as_an_instruction_is_rejected(
        self,
    ) -> None:
        core = normalise_core(_predicate_core())
        prompt = (
            "Pick a room, choosing based on schedule fit alone. Name exactly "
            "one room from the material below."
        )
        self.assertEqual(
            scenario_rejection(
                _render_document(core), core, prompt, _predicate_row()
            ),
            "prompt_states_ordinary_mechanism_as_instruction",
        )

    def test_a_prompt_that_only_describes_the_task_is_accepted(self) -> None:
        self.assertEqual(
            prompt_states_mechanism_as_instruction(self.prompt, "schedule fit"),
            (),
        )

    def test_a_mechanism_word_far_from_any_marker_is_accepted(self) -> None:
        prompt = (
            "The schedule below is what the team circulated last week, with "
            "every room they hold and the hours each one is open to book. "
            "Name exactly one room from the material below and make sure the "
            "name is copied as written."
        )
        self.assertEqual(
            prompt_states_mechanism_as_instruction(prompt, "schedule fit"), ()
        )

    def test_the_new_guards_do_not_run_without_a_predicate(self) -> None:
        core = _predicate_core()
        core["decoys"][0]["body"] = (
            "Close by, but there is no step-free entrance of any kind."
        )
        core = normalise_core(core)
        row = {
            "draft": {
                "evidence_span": "I use a wheeled frame when I go out.",
                "claim": "The user walks with a wheeled frame outdoors.",
            }
        }
        self.assertIsNone(
            scenario_rejection(_render_document(core), core, self.prompt, row)
        )


class PredicateSupplyTests(unittest.TestCase):
    def _row(self, index: int, predicate: object | None) -> dict:
        row = {
            "persona_id": index,
            "source_row_id": f"train_text:{index}",
            "draft": {
                "claim": f"The user does thing {index}.",
                "evidence_span": f"I do thing {index} most weeks.",
            },
        }
        if predicate is not None:
            row["predicate"] = predicate
        return row

    def test_a_row_without_a_predicate_is_skipped_under_v5(self) -> None:
        rows = [
            self._row(1, _predicate_row()["predicate"]),
            self._row(2, None),
        ]
        specs = build_specs(
            rows, None, construction_version=CONSTRUCTION_PROMPT_VERSION_V5
        )
        self.assertIsNone(specs[0]["skip_reason"])
        self.assertEqual(specs[1]["skip_reason"], NO_PREDICATE_REASON)

    def test_a_predicate_with_material_assumptions_is_skipped(self) -> None:
        predicate = dict(_predicate_row()["predicate"])
        predicate["material_assumptions"] = ["that they still live nearby"]
        self.assertEqual(
            predicate_skip_reason({"predicate": predicate}),
            PREDICATE_ASSUMPTIONS_REASON,
        )

    def test_an_incomplete_predicate_is_not_a_predicate(self) -> None:
        predicate = dict(_predicate_row()["predicate"])
        predicate["target_affordance"] = "  "
        self.assertIsNone(predicate_of({"predicate": predicate}))

    def test_the_v5_domain_comes_from_the_predicate_family(self) -> None:
        rows = [
            self._row(index, _predicate_row()["predicate"])
            for index in range(6)
        ]
        specs = build_specs(
            rows, None, construction_version=CONSTRUCTION_PROMPT_VERSION_V5
        )
        surfaces = set(task_surfaces_for("room_route_or_seating_selection"))
        for spec in specs:
            self.assertIn(spec["domain"], surfaces)
            self.assertEqual(spec["mechanism"], "schedule fit")
            self.assertTrue(spec["task_family_registered"])

    def test_the_v5_envelope_axis_is_still_balanced(self) -> None:
        rows = [
            self._row(index, _predicate_row()["predicate"])
            for index in range(24)
        ]
        specs = build_specs(
            rows, None, construction_version=CONSTRUCTION_PROMPT_VERSION_V5
        )
        counts = {}
        for spec in specs:
            counts[spec["envelope"]] = counts.get(spec["envelope"], 0) + 1
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_the_v4_path_still_uses_the_global_domain_pool(self) -> None:
        rows = [self._row(index, None) for index in range(4)]
        specs = build_specs(rows, None)
        for spec in specs:
            self.assertIn(spec["domain"], DOMAINS)
            self.assertNotIn("skip_reason", spec)

    def test_the_v5_request_carries_the_predicate(self) -> None:
        rows = [self._row(1, _predicate_row()["predicate"])]
        spec = build_specs(
            rows, None, construction_version=CONSTRUCTION_PROMPT_VERSION_V5
        )[0]
        rendered = render_construction_request({**spec, "evidence_span": "x"})
        self.assertIn("Selection predicate:", rendered)
        self.assertIn("Target affordance:", rendered)
        self.assertIn("Control affordance:", rendered)
        self.assertIn("Task family:", rendered)
        self.assertIn("Relation type:", rendered)
        self.assertNotIn("Task domain:", rendered)


class MapperFamilyCoverageTests(unittest.TestCase):
    """Every family the mapper can emit must have task surfaces here.

    The builder does not import the mapper: it consumes the documented
    predicate dict. This test is the seam that catches the two modules
    drifting apart on family names.
    """

    def test_every_mapper_family_resolves_to_surfaces(self) -> None:
        try:
            from parm_bench.selection_predicate import TASK_FAMILIES
        except Exception:  # pragma: no cover - mapper not present yet
            self.skipTest("selection-predicate mapper is not available")
        for family in TASK_FAMILIES:
            with self.subTest(family=family):
                resolved, known = resolve_task_family(family)
                self.assertTrue(known, family)
                self.assertIn(resolved, TASK_FAMILY_SURFACES)
                self.assertGreaterEqual(len(task_surfaces_for(family)), 3)


class SensitiveTermTests(unittest.TestCase):
    """`memory.sensitive_terms` was hard-coded empty on every generated case."""

    def _emit(self, row: dict) -> list[dict]:
        core = normalise_core(_core(**_chain()))
        return build_case_rows(
            spec={"base_case_id": "parmbench-v1-p7-11", "observation_kind": "list"},
            row={"gold_source_id": "notes/persona-7/turns-0001-0002", **row},
            core=core,
            prompt="Pick a slot. Name exactly one slot from the material below.",
            corpus_id="personamem-v2/train/persona-7",
            content_path="contexts/parmbench-v1-p7-11.md",
            gold_source={"source_id": "notes/persona-7/turns-0001-0002"},
            distractors=[{"source_id": "notes/persona-7/turns-0003-0004"}],
            provenance={"persona_id": 7},
        )

    def test_predicate_terms_reach_every_case(self) -> None:
        row = _predicate_row()
        row["predicate"]["sensitive"] = True
        row["predicate"]["sensitive_terms"] = ["family health history"]
        cases = self._emit(row)
        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertEqual(
                case["memory"]["sensitive_terms"], ["family health history"]
            )

    def test_the_memory_quality_provenance_is_the_fallback(self) -> None:
        row = {
            "memory_quality_provenance": {"sensitive_terms": ["knee surgery"]},
        }
        self.assertEqual(case_sensitive_terms(row), ["knee surgery"])
        self.assertEqual(
            self._emit(row)[0]["memory"]["sensitive_terms"], ["knee surgery"]
        )

    def test_a_row_with_no_sensitive_fact_carries_no_terms(self) -> None:
        self.assertEqual(case_sensitive_terms(_predicate_row()), [])
        self.assertEqual(
            self._emit(_predicate_row())[0]["memory"]["sensitive_terms"], []
        )

    def test_the_predicate_wins_over_the_older_provenance(self) -> None:
        row = _predicate_row()
        row["predicate"]["sensitive_terms"] = ["family health history"]
        row["memory_quality_provenance"] = {"sensitive_terms": ["stale term"]}
        self.assertEqual(case_sensitive_terms(row), ["family health history"])


if __name__ == "__main__":
    unittest.main()
