from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from parm_bench.decision_validity import (
    ACCESSIBILITY_NEED,
    ACTIVE_COMMITMENT,
    CONCRETE_SCHEDULE,
    CONSTRAINT,
    DIRECT_RELATION,
    EXCLUSION,
    ONE_HOP_RELATION,
    OWNED_ITEM,
    PREFERENCE,
    STABLE_RELATIONSHIP,
    CachedOpenAIDecisionValidityJudge,
    DecisionScenario,
    DecisionValidityCacheMissError,
    DecisionValidityCachePolicy,
    audit_scenario,
    build_scenario,
    capability_for_fact,
    decision_validity_cache_key,
    named_entities,
    passes_decision_validity,
    render_decision_validity_request,
    wording_relation,
)
from parm_bench.relevance_taxonomy import (
    CONTROL_DOES_NOT_REMOVE_ADVANTAGE,
    CUE_ALONE_DETERMINES_CHOICE,
    INCORRECT_CAPABILITY_LABEL,
    INVENTED_RELATIONSHIP_OR_PERMISSION,
    MEDICAL_OR_SENSITIVE_OVERREACH,
)


ACCEPTING_VERDICT: dict[str, Any] = {
    "ordinary_winner_valid": True,
    "cue_specificity": "concrete_affordance",
    "memory_affordance_entailment": "entailed",
    "assumptions_required": [],
    "cue_alone_sufficient": False,
    "control_valid": True,
    "sensitivity": "none",
    "capability_label_valid": True,
    "suggested_capability": "one_hop_relational",
    "rejection_reasons": [],
    "rationale": "The fact and the affordance settle the choice on their own.",
    "accept": True,
}


def verdict(**overrides: Any) -> dict[str, Any]:
    payload = json.loads(json.dumps(ACCEPTING_VERDICT))
    payload.update(overrides)
    return payload


class StubJudge:
    """Auditor stand-in that returns a fixed verdict without a model call."""

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        model_name: str = "stub-auditor",
        rubric_version: str = "parmbench_decision_validity_v1",
    ) -> None:
        self.payload = payload if payload is not None else verdict()
        self.model_name = model_name
        self.rubric_version = rubric_version
        self.seen: list[DecisionScenario] = []

    def judge(self, *, scenario: DecisionScenario) -> dict[str, Any]:
        self.seen.append(scenario)
        return json.loads(json.dumps(self.payload))


class RecordingClient:
    """Minimal stand-in for the OpenAI client used by the cached auditor."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._output_text = json.dumps(
            payload if payload is not None else verdict(), ensure_ascii=False
        )
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=self._output_text,
            model="resolved-auditor",
            id="validity-response",
        )


class ExplodingClient:
    def __init__(self) -> None:
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        raise AssertionError("the auditor must not be called")


def scenario(**overrides: Any) -> DecisionScenario:
    """A scenario shaped like the audit's approved NES installer case."""

    fields: dict[str, Any] = {
        "claim": "The user owns an old NES console.",
        "evidence_span": (
            "pulled the old NES out of the closet and played through my "
            "favorite side scroller."
        ),
        "task_prompt": (
            "Choose a home AV installer to update the living-room television "
            "and wall connections in a single visit. Name exactly one "
            "installer from the material below."
        ),
        "option_a_label": "Same-Day HDMI Swap",
        "option_a_body": (
            "This installer guarantees completion in a single appointment and "
            "carries the wall plate and finish materials on the van. The area "
            "is restored before they leave and no follow-up booking is needed."
        ),
        "option_b_label": "Legacy-Connection Specialist",
        "option_b_body": (
            "This installer books for the next business day and routes cable "
            "carefully for a neat finish. They install and retain accessible "
            "analog video and audio connections on a recessed plate behind "
            "the television."
        ),
        "cue_text": (
            "They install and retain accessible analog video and audio "
            "connections on a recessed plate behind the television."
        ),
        "neutral_replacement": (
            "They bring a small cable organizer and label each lead during "
            "the job."
        ),
        "ordinary_mechanism": "schedule fit",
        "capability_label": "one_hop_relational",
        "evidence_turns": (
            {
                "role": "user",
                "content": (
                    "Could you polish this post? I pulled the old NES out of "
                    "the closet and played through my favorite side scroller."
                ),
            },
            {"role": "assistant", "content": "Here is a tidier version."},
        ),
        "sensitive_terms": (),
    }
    fields.update(overrides)
    return DecisionScenario(**fields)


class AcceptPathTests(unittest.TestCase):
    def test_a_clean_verdict_is_accepted(self) -> None:
        judge = StubJudge()
        result = audit_scenario(scenario(), judge=judge)
        self.assertTrue(result.accept)
        self.assertTrue(passes_decision_validity(result))
        self.assertEqual(result.unmet_requirements, ())
        self.assertEqual(result.rejection_reasons, ())
        self.assertEqual(result.model, "stub-auditor")
        self.assertEqual(
            result.rubric_version, "parmbench_decision_validity_v1"
        )

    def test_a_handled_sensitive_fact_is_accepted(self) -> None:
        judge = StubJudge(verdict(sensitivity="sensitive_handled"))
        self.assertTrue(audit_scenario(scenario(), judge=judge).accept)

    def test_the_auditor_reads_the_full_option_bodies(self) -> None:
        judge = StubJudge()
        audit_scenario(scenario(), judge=judge)
        rendered = render_decision_validity_request(judge.seen[0])
        self.assertIn("guarantees completion in a single appointment", rendered)
        self.assertIn("routes cable", rendered)
        self.assertIn("Here is a tidier version.", rendered)
        self.assertIn("small cable organizer", rendered)

    def test_the_control_arm_is_rendered_for_the_auditor(self) -> None:
        rendered = render_decision_validity_request(scenario())
        self.assertIn("Option B as it reads in the control arm", rendered)
        control_arm = rendered.split(
            "Option B as it reads in the control arm:"
        )[1]
        self.assertIn("small cable organizer", control_arm)
        self.assertNotIn("analog video", control_arm.split("Declared")[0])


class FieldRejectionTests(unittest.TestCase):
    """Each structured field must be able to reject on its own."""

    def _reject(self, **overrides: Any) -> Any:
        return audit_scenario(scenario(), judge=StubJudge(verdict(**overrides)))

    def test_a_broken_ordinary_winner_rejects(self) -> None:
        result = self._reject(ordinary_winner_valid=False)
        self.assertFalse(result.accept)
        self.assertIn("ordinary_winner_valid", result.unmet_requirements)

    def test_a_generic_cue_rejects(self) -> None:
        result = self._reject(cue_specificity="generic_evaluative")
        self.assertFalse(result.accept)
        self.assertIn("cue_specificity", result.unmet_requirements)

    def test_a_wording_echo_cue_rejects(self) -> None:
        result = self._reject(cue_specificity="wording_echo")
        self.assertFalse(result.accept)
        self.assertIn("cue_specificity", result.unmet_requirements)

    def test_a_stretched_entailment_rejects(self) -> None:
        result = self._reject(memory_affordance_entailment="plausible_stretch")
        self.assertFalse(result.accept)
        self.assertIn(
            "memory_affordance_entailment", result.unmet_requirements
        )

    def test_an_invented_entailment_rejects(self) -> None:
        result = self._reject(memory_affordance_entailment="invented")
        self.assertFalse(result.accept)
        self.assertIn(
            "memory_affordance_entailment", result.unmet_requirements
        )

    def test_any_required_assumption_rejects(self) -> None:
        result = self._reject(
            assumptions_required=["that the user still owns the console"]
        )
        self.assertFalse(result.accept)
        self.assertIn("assumptions_required", result.unmet_requirements)
        self.assertEqual(
            result.assumptions_required,
            ("that the user still owns the console",),
        )

    def test_blank_assumptions_do_not_reject(self) -> None:
        result = self._reject(assumptions_required=["", "   "])
        self.assertTrue(result.accept)
        self.assertEqual(result.assumptions_required, ())

    def test_a_self_sufficient_cue_rejects(self) -> None:
        result = self._reject(cue_alone_sufficient=True)
        self.assertFalse(result.accept)
        self.assertIn("cue_alone_sufficient", result.unmet_requirements)
        self.assertIn(CUE_ALONE_DETERMINES_CHOICE, result.rejection_reasons)

    def test_a_leaking_control_rejects(self) -> None:
        result = self._reject(control_valid=False)
        self.assertFalse(result.accept)
        self.assertIn("control_valid", result.unmet_requirements)
        self.assertIn(
            CONTROL_DOES_NOT_REMOVE_ADVANTAGE, result.rejection_reasons
        )

    def test_an_unlabelled_sensitive_fact_rejects(self) -> None:
        result = self._reject(sensitivity="sensitive_unlabeled")
        self.assertFalse(result.accept)
        self.assertIn("sensitivity", result.unmet_requirements)
        self.assertIn(MEDICAL_OR_SENSITIVE_OVERREACH, result.rejection_reasons)

    def test_a_wrong_capability_label_rejects(self) -> None:
        result = self._reject(
            capability_label_valid=False,
            suggested_capability="paraphrased_semantic_fact",
        )
        self.assertFalse(result.accept)
        self.assertIn("capability_label_valid", result.unmet_requirements)
        self.assertIn(INCORRECT_CAPABILITY_LABEL, result.rejection_reasons)
        self.assertEqual(
            result.suggested_capability, "paraphrased_semantic_fact"
        )

    def test_any_named_reason_rejects(self) -> None:
        result = self._reject(
            rejection_reasons=[INVENTED_RELATIONSHIP_OR_PERMISSION]
        )
        self.assertFalse(result.accept)
        self.assertIn("rejection_reasons", result.unmet_requirements)

    def test_the_auditor_can_refuse_a_scenario_its_fields_allow(self) -> None:
        result = self._reject(accept=False)
        self.assertFalse(result.accept)
        self.assertEqual(result.unmet_requirements, ("judge_accept",))

    def test_an_unknown_enum_value_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            audit_scenario(
                scenario(), judge=StubJudge(verdict(sensitivity="fine"))
            )

    def test_an_unknown_rejection_reason_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            audit_scenario(
                scenario(),
                judge=StubJudge(verdict(rejection_reasons=["looks_odd"])),
            )


class AuditRegressionTests(unittest.TestCase):
    """Fixtures taken from the v1 relevance calibration audit."""

    def test_vaccination_booklet_parcel_desk_is_rejected(self) -> None:
        # parmbench-v1-p113-223: possession of a childhood booklet, and the
        # absence of photo ID, are both invented by the justification.
        fixture = scenario(
            claim="The user got jabs as a kid.",
            evidence_span=(
                "Why do they still tell me to get some jabs again years after "
                "I got them as a kid and even for work trips?"
            ),
            task_prompt=(
                "Choose how to receive a parcel that requires identity "
                "verification at handover. Name exactly one collection point "
                "from the material below."
            ),
            option_a_label="Post Office Counter",
            option_a_body=(
                "The counter checks government-issued photo identification "
                "and matches the name on the parcel, which straightforwardly "
                "satisfies the stated requirement."
            ),
            option_b_label="Health Centre Reception",
            option_b_body=(
                "Reception holds parcels behind the desk during opening "
                "hours. Reception accepts older vaccination booklets or paper "
                "immunisation records as valid proof for parcel collection."
            ),
            cue_text=(
                "Reception accepts older vaccination booklets or paper "
                "immunisation records as valid proof for parcel collection."
            ),
            neutral_replacement=(
                "Bring any relevant paper documentation to reception during "
                "opening hours."
            ),
            capability_label="paraphrased_semantic_fact",
        )
        result = audit_scenario(
            fixture,
            judge=StubJudge(
                verdict(
                    memory_affordance_entailment="invented",
                    assumptions_required=[
                        "that the user still holds a childhood vaccination "
                        "booklet",
                        "that the user lacks or avoids photo identification",
                    ],
                    sensitivity="sensitive_unlabeled",
                    rejection_reasons=[MEDICAL_OR_SENSITIVE_OVERREACH],
                    accept=False,
                )
            ),
        )
        self.assertFalse(result.accept)
        self.assertIn("assumptions_required", result.unmet_requirements)
        self.assertEqual(len(result.assumptions_required), 2)

    def test_barangay_officer_signing_is_rejected(self) -> None:
        # parmbench-v1-p172-153: one invitation to a celebration cannot
        # license a standing signing relationship.
        fixture = scenario(
            claim=(
                "The user was invited to join a barangay celebration in Datu "
                "Odin Sinsuat."
            ),
            evidence_span=(
                "Last weekend after our coastal cleanup in Datu Odin Sinsuat, "
                "the barangay invited me to join their celebration."
            ),
            task_prompt=(
                "Compare local contractors for a routine household repair. "
                "Name exactly one contractor from the material below."
            ),
            option_a_label="Fixed-Price Bidder",
            option_a_body=(
                "The lowest all-in price sits in a short written contract "
                "itemising materials, labour and debris removal, with payment "
                "after the final walkthrough."
            ),
            option_b_label="Community Signoff",
            option_b_body=(
                "This contractor quotes a higher hourly rate and works to a "
                "looser schedule. They will accept a named local community "
                "officer or neighbour to sign the completion certificate on "
                "behalf of the homeowner."
            ),
            cue_text=(
                "They will accept a named local community officer or "
                "neighbour to sign the completion certificate on behalf of "
                "the homeowner."
            ),
            neutral_replacement=(
                "They send photo documentation of the finished work with the "
                "final invoice."
            ),
            capability_label="one_hop_relational",
        )
        result = audit_scenario(
            fixture,
            judge=StubJudge(
                verdict(
                    memory_affordance_entailment="invented",
                    assumptions_required=[
                        "that the user lives in that barangay",
                        "that an officer there would accept signing authority",
                        "that the user is absent at final sign-off",
                    ],
                    capability_label_valid=False,
                    suggested_capability="paraphrased_semantic_fact",
                    rejection_reasons=[INVENTED_RELATIONSHIP_OR_PERMISSION],
                    accept=False,
                )
            ),
        )
        self.assertFalse(result.accept)
        self.assertIn(
            INVENTED_RELATIONSHIP_OR_PERMISSION, result.rejection_reasons
        )
        self.assertIn(INCORRECT_CAPABILITY_LABEL, result.rejection_reasons)

    def test_a_control_that_keeps_the_advantage_is_rejected(self) -> None:
        # parmbench-v1-p553-44: the advantage survives in non-cue text, so
        # the auditor has to be reading both full bodies.
        fixture = scenario(
            claim="The user has been curating particularly rare vintages.",
            evidence_span=(
                "I've been curating some particularly rare vintages for an "
                "intimate industry gathering next week"
            ),
            task_prompt=(
                "Compare the shortlisted flats and decide which to schedule a "
                "viewing for. Name exactly one flat from the material below."
            ),
            option_a_label="Same-Day Viewing",
            option_a_body=(
                "Two slots are open today and tomorrow and the agent will "
                "hold the flat on a short deposit. The building's management "
                "states they do not permit on-site storage of alcohol or "
                "other temperature-sensitive goods."
            ),
            option_b_label="Locked Climate Cellar",
            option_b_body=(
                "Viewings run by appointment with weekday flexibility next "
                "week, and there is an insulated storage area off the "
                "kitchen. The flat includes a locked, climate-controlled room "
                "sized for storing rare vintages."
            ),
            cue_text=(
                "The flat includes a locked, climate-controlled room sized "
                "for storing rare vintages."
            ),
            neutral_replacement=(
                "The flat includes an extra storage closet off the hallway."
            ),
            capability_label="direct_lexical_fact",
        )
        judge = StubJudge(
            verdict(
                control_valid=False,
                rejection_reasons=[CONTROL_DOES_NOT_REMOVE_ADVANTAGE],
                accept=False,
            )
        )
        result = audit_scenario(fixture, judge=judge)
        self.assertFalse(result.accept)
        self.assertEqual(result.unmet_requirements[0], "control_valid")
        rendered = render_decision_validity_request(judge.seen[0])
        self.assertIn("do not permit on-site storage of alcohol", rendered)
        self.assertIn("insulated storage area off the kitchen", rendered)

    def test_a_wording_echo_cue_is_rejected(self) -> None:
        # parmbench-v1-p102-182: the cue repeats the person's own phrasing, so
        # string matching finds it without any inference.
        fixture = scenario(
            claim="The user performs in a small room with just friends.",
            evidence_span=(
                "Why does my singing sometimes feel more relaxed and even "
                "better when I'm performing in a small room with just friends"
            ),
            option_b_label="Low Riser Alcove",
            option_b_body=(
                "The alcove sits off the main corridor. It is sized and "
                "arranged for small room setups and provides close sightlines "
                "suited to friends."
            ),
            cue_text=(
                "It is sized and arranged for small room setups and provides "
                "close sightlines suited to friends."
            ),
            capability_label="relationship_named_entity",
        )
        result = audit_scenario(
            fixture,
            judge=StubJudge(
                verdict(
                    cue_specificity="wording_echo",
                    capability_label_valid=False,
                    suggested_capability="direct_lexical_fact",
                    accept=False,
                )
            ),
        )
        self.assertFalse(result.accept)
        self.assertIn("cue_specificity", result.unmet_requirements)
        self.assertEqual(result.cue_specificity, "wording_echo")

    def test_an_unlabelled_medical_fact_is_rejected(self) -> None:
        # parmbench-v1-p350-39: a surgery does the deciding while the case
        # declares no sensitive terms at all.
        fixture = scenario(
            claim="The user was recovering from minor knee surgery.",
            evidence_span=(
                "After a few weeks of lighter activity while recovering from "
                "minor knee surgery, I've been back on my feet and walking "
                "the campus again."
            ),
            option_b_label="Ground-Floor Studio",
            option_b_body=(
                "The studio holds a modest technical write-up. It has "
                "step-free access from the main campus walkways and is a "
                "short walking route across campus."
            ),
            cue_text=(
                "It has step-free access from the main campus walkways and is "
                "a short walking route across campus."
            ),
            capability_label="negative_preference_exclusion",
            sensitive_terms=(),
        )
        result = audit_scenario(
            fixture,
            judge=StubJudge(
                verdict(
                    sensitivity="sensitive_unlabeled",
                    capability_label_valid=False,
                    suggested_capability="one_hop_relational",
                    accept=False,
                )
            ),
        )
        self.assertFalse(result.accept)
        self.assertIn("sensitivity", result.unmet_requirements)
        self.assertIn(MEDICAL_OR_SENSITIVE_OVERREACH, result.rejection_reasons)
        self.assertIn("(none declared)", render_decision_validity_request(fixture))

    def test_the_nes_installer_fixture_is_accepted(self) -> None:
        result = audit_scenario(scenario(), judge=StubJudge())
        self.assertTrue(result.accept)

    def test_the_dietary_exclusion_fixture_is_accepted(self) -> None:
        fixture = scenario(
            claim=(
                "The user skips syrupy desserts and chooses a small bowl of "
                "fresh fruit."
            ),
            evidence_span=(
                "But I've learned to skip the syrupy desserts, not just for "
                "the sake of staying sharp, but because I've seen what "
                "unchecked habits can do in my own family."
            ),
            task_prompt=(
                "Pick a caterer for a small family gathering, comparing "
                "quotes and substitution policies. Name exactly one caterer "
                "from the material below."
            ),
            option_a_label="Budget Home Services",
            option_a_body=(
                "The lowest overall quote covers staffing, delivery, setup "
                "and cleanup. The kitchen makes no custom dessert "
                "substitutions."
            ),
            option_b_label="Green Kitchen Collective",
            option_b_body=(
                "The quote sits a little above the cheapest option and covers "
                "the same staffing. They can prepare modest individual plates "
                "featuring plain seasonal produce with no added sweeteners "
                "and will omit sweet pastries on request."
            ),
            cue_text=(
                "They can prepare modest individual plates featuring plain "
                "seasonal produce with no added sweeteners and will omit "
                "sweet pastries on request."
            ),
            neutral_replacement=(
                "They coordinate delivery and include servers, setup and "
                "cleanup in the estimate."
            ),
            ordinary_mechanism="cost",
            capability_label="negative_preference_exclusion",
        )
        result = audit_scenario(
            fixture,
            judge=StubJudge(
                verdict(suggested_capability="negative_preference_exclusion")
            ),
        )
        self.assertTrue(result.accept)


class CapabilityMappingTests(unittest.TestCase):
    def test_a_writing_project_is_not_a_schedule_commitment(self) -> None:
        # parmbench-v1-p28-251: no time or date appears anywhere in the fact.
        capability = capability_for_fact(
            claim=(
                "The user is drafting a piece on how theological traditions "
                "adjust when entering politically diverse societies."
            ),
            memory_category=ACTIVE_COMMITMENT,
            causal_relation=DIRECT_RELATION,
            evidence_span=(
                "I am drafting a piece on how theological traditions adjust "
                "when entering politically diverse societies"
            ),
            cue_text=(
                "They keep a curated selection of titles on faith communities "
                "and their relations with government institutions."
            ),
        )
        self.assertNotEqual(capability, "schedule_commitment")
        self.assertEqual(capability, "paraphrased_semantic_fact")

    def test_a_dated_commitment_is_a_schedule_commitment(self) -> None:
        self.assertEqual(
            capability_for_fact(
                claim="The user sits on the Tuesday review panel.",
                memory_category=ACTIVE_COMMITMENT,
            ),
            "schedule_commitment",
        )
        self.assertEqual(
            capability_for_fact(
                claim="The user holds a standing slot for practice.",
                memory_category=CONCRETE_SCHEDULE,
            ),
            "schedule_commitment",
        )

    def test_an_accessibility_need_is_not_an_exclusion(self) -> None:
        # parmbench-v1-p350-39: a mobility need rules nothing out; it needs
        # something.
        capability = capability_for_fact(
            claim="The user was recovering from minor knee surgery.",
            memory_category=ACCESSIBILITY_NEED,
            causal_relation=ONE_HOP_RELATION,
            evidence_span=(
                "After a few weeks of lighter activity while recovering from "
                "minor knee surgery"
            ),
            cue_text="It has step-free access from the main campus walkways.",
        )
        self.assertNotEqual(capability, "negative_preference_exclusion")
        self.assertEqual(capability, "one_hop_relational")

    def test_a_constraint_without_exclusion_wording_is_not_an_exclusion(
        self,
    ) -> None:
        self.assertNotEqual(
            capability_for_fact(
                claim="The user was recovering from minor knee surgery.",
                memory_category=CONSTRAINT,
                causal_relation=ONE_HOP_RELATION,
            ),
            "negative_preference_exclusion",
        )

    def test_a_stated_exclusion_keeps_its_label(self) -> None:
        self.assertEqual(
            capability_for_fact(
                claim="The user skips syrupy desserts.",
                memory_category=EXCLUSION,
            ),
            "negative_preference_exclusion",
        )
        self.assertEqual(
            capability_for_fact(
                claim="The user avoids late evening classes.",
                memory_category=PREFERENCE,
            ),
            "negative_preference_exclusion",
        )

    def test_no_named_entity_means_no_named_entity_label(self) -> None:
        capability = capability_for_fact(
            claim="The user's neighbour helps with the garden.",
            memory_category=STABLE_RELATIONSHIP,
            causal_relation=DIRECT_RELATION,
            evidence_span="my neighbour helps me with the garden",
            cue_text="The plot is tended under a shared rota of residents.",
        )
        self.assertNotEqual(capability, "relationship_named_entity")
        self.assertEqual(capability, "paraphrased_semantic_fact")

    def test_a_named_relation_keeps_the_named_entity_label(self) -> None:
        self.assertEqual(
            capability_for_fact(
                claim="The user's sister runs a shop in Leeds.",
                memory_category=STABLE_RELATIONSHIP,
                causal_relation=DIRECT_RELATION,
                overlap_mode="paraphrase_only",
            ),
            "relationship_named_entity",
        )

    def test_an_owned_item_behind_an_inference_is_relational(self) -> None:
        self.assertEqual(
            capability_for_fact(
                claim="The user owns an old NES console.",
                memory_category=OWNED_ITEM,
                causal_relation=ONE_HOP_RELATION,
                evidence_span="pulled the old NES out of the closet",
                cue_text=(
                    "They retain accessible analog video and audio "
                    "connections."
                ),
            ),
            "one_hop_relational",
        )

    def test_an_owned_item_named_in_the_cue_is_lexical(self) -> None:
        self.assertEqual(
            capability_for_fact(
                claim="The user owns a treadle sewing machine.",
                memory_category=OWNED_ITEM,
                causal_relation=DIRECT_RELATION,
                evidence_span="I still use my treadle sewing machine",
                cue_text="The studio keeps space for a treadle machine.",
            ),
            "direct_lexical_fact",
        )

    def test_an_unknown_category_falls_back_to_the_wording_split(self) -> None:
        self.assertEqual(
            capability_for_fact(
                claim="The user plays the cello.",
                memory_category="mystery",
                overlap_mode="share_wording",
            ),
            "direct_lexical_fact",
        )
        self.assertEqual(
            capability_for_fact(
                claim="The user plays the cello.",
                memory_category="mystery",
                overlap_mode="paraphrase_only",
            ),
            "paraphrased_semantic_fact",
        )

    def test_wording_relation_reads_the_span_against_the_cue(self) -> None:
        self.assertEqual(
            wording_relation(
                "I've learned to skip the syrupy desserts",
                "plain seasonal produce with no added sweeteners",
            ),
            "paraphrase_only",
        )
        self.assertEqual(
            wording_relation(
                "I still use my treadle sewing machine",
                "space for a treadle machine",
            ),
            "share_wording",
        )

    def test_named_entities_ignore_the_leading_word(self) -> None:
        self.assertEqual(
            named_entities("The user's sister runs a shop in Leeds."),
            ("Leeds",),
        )
        self.assertEqual(named_entities("The user keeps a small allotment."), ())


class ScenarioFromCoreTests(unittest.TestCase):
    def test_the_target_body_carries_the_cue_as_the_reader_sees_it(
        self,
    ) -> None:
        built = build_scenario(
            claim="The user owns an old NES console.",
            evidence_span="pulled the old NES out of the closet",
            task_prompt="Pick an installer. Name exactly one installer.",
            core={
                "winner_label": "Same-Day Swap",
                "winner_body": "One visit, tidy finish.",
                "target_label": "Legacy Specialist",
                "target_body": "Books for the next business day.",
                "cue_clause": "They retain analog connections on the plate.",
                "neutral_clause": "They label each lead during the job.",
            },
            ordinary_mechanism="schedule fit",
            capability_label="one_hop_relational",
        )
        self.assertEqual(
            built.option_b_body,
            "Books for the next business day. They retain analog connections "
            "on the plate.",
        )
        self.assertEqual(built.option_a_body, "One visit, tidy finish.")


class AuditorCacheTests(unittest.TestCase):
    def test_cache_key_is_stable_and_input_sensitive(self) -> None:
        baseline = decision_validity_cache_key(scenario())
        self.assertEqual(baseline, decision_validity_cache_key(scenario()))
        self.assertNotEqual(
            baseline,
            decision_validity_cache_key(
                scenario(option_a_body="A different ordinary case.")
            ),
        )
        self.assertNotEqual(
            baseline,
            decision_validity_cache_key(
                scenario(capability_label="direct_lexical_fact")
            ),
        )
        self.assertNotEqual(
            baseline,
            decision_validity_cache_key(scenario(), model="other-model"),
        )
        self.assertNotEqual(
            baseline,
            decision_validity_cache_key(
                scenario(), rubric_version="parmbench_decision_validity_v0"
            ),
        )

    def test_populate_then_frozen_replay(self) -> None:
        client = RecordingClient()
        fixture = scenario()
        with tempfile.TemporaryDirectory() as tmp:
            first = audit_scenario(fixture, cache_dir=tmp, client=client)
            request_hash = decision_validity_cache_key(fixture)
            self.assertTrue((Path(tmp) / f"{request_hash}.json").exists())

            frozen = CachedOpenAIDecisionValidityJudge(
                tmp,
                DecisionValidityCachePolicy.FROZEN,
                client=ExplodingClient(),
            )
            replay = audit_scenario(fixture, judge=frozen)
            self.assertEqual(first, replay)
            self.assertEqual(len(client.calls), 1)
            self.assertIsNotNone(frozen.cache_hash)
            self.assertTrue(replay.accept)
            self.assertEqual(replay.resolved_model, "resolved-auditor")
            self.assertEqual(replay.response_id, "validity-response")

    def test_frozen_cache_miss_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DecisionValidityCacheMissError):
                audit_scenario(
                    scenario(),
                    cache_dir=tmp,
                    policy="frozen",
                    client=ExplodingClient(),
                )

    def test_a_judgeless_call_needs_a_cache_directory(self) -> None:
        with self.assertRaises(ValueError):
            audit_scenario(scenario())

    def test_the_cache_entry_records_the_rubric_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_scenario(scenario(), cache_dir=tmp, client=RecordingClient())
            entries = list(Path(tmp).glob("*.json"))
            self.assertEqual(len(entries), 1)
            payload = json.loads(entries[0].read_text(encoding="utf-8"))
            self.assertEqual(
                payload["rubric_version"], "parmbench_decision_validity_v1"
            )
            self.assertEqual(payload["model"], "gpt-5-mini")
            self.assertEqual(payload["resolved_model"], "resolved-auditor")


if __name__ == "__main__":
    unittest.main()
