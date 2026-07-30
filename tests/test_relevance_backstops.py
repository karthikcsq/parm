import unittest

from parm_bench.decision_validity import (
    DecisionScenario,
    build_scenario,
    capability_conflicts_with_lexical_target,
    control_residual_advantage,
)


def _scenario(**overrides) -> DecisionScenario:
    core = {
        "winner_label": "Studio Report Co",
        "winner_body": (
            "Publishes weekly circulation summaries and holds the largest "
            "verified distribution list in the region."
        ),
        "target_label": overrides.pop("target_label", "Harbour Print Works"),
        "target_body": overrides.pop(
            "target_body",
            "A smaller operation with a steady output of community notices.",
        ),
        "cue_clause": overrides.pop(
            "cue_clause",
            "Its reading panel reviews submissions in eleven languages.",
        ),
        "neutral_clause": overrides.pop(
            "neutral_clause",
            "Its office closes for an hour at midday on Fridays.",
        ),
        "decoys": [],
    }
    return build_scenario(
        claim=overrides.pop(
            "claim",
            "The user is conducting research at Columbia University "
            "examining how multilingual online spaces frame and spread "
            "contested information.",
        ),
        evidence_span=overrides.pop(
            "evidence_span",
            "In my current research at Columbia University, I'm examining "
            "how multilingual online spaces frame and spread contested "
            "information.",
        ),
        task_prompt="Pick exactly one supplier from the material below.",
        core=core,
        ordinary_mechanism="quality of the write-up",
        capability_label=overrides.pop("capability_label", "one_hop_relational"),
        **overrides,
    )


class ControlResidualAdvantageTests(unittest.TestCase):
    def test_pilot_p92_label_residue_is_flagged(self):
        # Regression for the v3 pilot: the "multilingual" claim was answered
        # by an option literally named "Multilingual Data Pack", and the
        # judge still returned control_valid on it.
        scenario = _scenario(target_label="Multilingual Data Pack")
        residual = control_residual_advantage(scenario)
        self.assertIn("multilingual", residual)

    def test_residue_in_non_cue_body_is_flagged(self):
        scenario = _scenario(
            target_body=(
                "The desk keeps a standing archive of multilingual "
                "circulars from previous seasons."
            )
        )
        self.assertIn("multilingual", control_residual_advantage(scenario))

    def test_residue_only_inside_the_cue_is_clean(self):
        # The cue may carry the affordance; ablation removes it, so a claim
        # word confined to the cue sentence is not residual.
        scenario = _scenario(
            cue_clause=(
                "Its reading panel reviews multilingual submissions from "
                "eleven regions."
            )
        )
        self.assertEqual(control_residual_advantage(scenario), ())

    def test_words_shared_with_the_winner_are_not_distinctive(self):
        scenario = _scenario(
            target_body="Research summaries appear in its monthly bulletin."
        )
        # "research" appears in the claim, but if the winner also carries
        # it, the word cannot be a personalized advantage.
        winner_shared = _scenario(
            target_body="Research summaries appear in its monthly bulletin."
        )
        object.__setattr__(
            winner_shared,
            "option_a_body",
            winner_shared.option_a_body + " Research digests ship quarterly.",
        )
        self.assertIn("research", control_residual_advantage(scenario))
        self.assertNotIn(
            "research", control_residual_advantage(winner_shared)
        )


class CapabilityConflictTests(unittest.TestCase):
    def test_pilot_p92_one_hop_with_lexical_label_conflicts(self):
        self.assertTrue(
            capability_conflicts_with_lexical_target(
                "one_hop_relational",
                "The user is conducting research at Columbia University "
                "examining how multilingual online spaces frame and spread "
                "contested information.",
                "In my current research at Columbia University, I'm "
                "examining how multilingual online spaces frame and spread "
                "contested information.",
                "Multilingual Data Pack",
            )
        )

    def test_one_hop_with_disjoint_label_is_fine(self):
        self.assertFalse(
            capability_conflicts_with_lexical_target(
                "one_hop_relational",
                "The user owns an NES console.",
                "I still have my old NES hooked up.",
                "Harbour AV Installations",
            )
        )

    def test_other_capabilities_are_not_checked(self):
        self.assertFalse(
            capability_conflicts_with_lexical_target(
                "direct_lexical_fact",
                "The user avoids dairy.",
                "I skip dairy entirely.",
                "Dairy-Free Kitchen",
            )
        )


if __name__ == "__main__":
    unittest.main()
