from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from parm_bench.relevance_taxonomy import (
    INCOMPATIBLE_TASK_FAMILY_ROUTING,
    MATERIAL_ASSUMPTION_REQUIRED,
    NO_SELECTION_PREDICATE,
    UNANCHORED_SELECTION_PREDICATE,
)
from parm_bench.selection_predicate import (
    NON_DURABLE_MEMORY,
    RELATION_TYPES,
    SELECTION_PREDICATE_RUBRIC,
    TASK_FAMILY_REGISTRY,
    UNMAPPABLE_MEMORY_CATEGORY,
    CachedOpenAISelectionPredicateJudge,
    SelectionPredicateCacheMissError,
    SelectionPredicateCachePolicy,
    allowed_task_families,
    deterministic_abstentions,
    is_legal_routing,
    map_selection_predicate,
    normalise_relation_type,
    passes_selection_predicate,
    render_selection_predicate_request,
    selection_predicate_cache_key,
)
from scripts.draft_parmbench_v1_claims import (
    predicate_row,
    selection_predicate_enabled,
    selection_predicate_rejection_reason,
)


ROOT = Path(__file__).resolve().parents[1]
ANCHORS_PATH = (
    ROOT / "data" / "parmbench-v1-supply" / "selection_predicate_anchors.json"
)

# The memory-quality labels each anchor would arrive with. The mapper reads
# them, so an anchor test that omitted them would exercise a different path.
ANCHOR_MEMORY_LABELS = {
    "anchor-dessert-menu": ("preference", "durable"),
    "anchor-reading-group": ("active_commitment", "currently_operative"),
    "anchor-vintage-store": ("owned_item", "durable"),
}


def load_anchors() -> list[dict[str, Any]]:
    payload = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    return list(payload["anchors"])


def _turns(*contents: str) -> list[dict[str, str]]:
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": text}
        for index, text in enumerate(contents)
    ]


class StubJudge:
    """Mapper stand-in that records what the stage asked it to map."""

    model_name = "gpt-5-mini"
    rubric_version = SELECTION_PREDICATE_RUBRIC

    def __init__(self, **verdict: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.verdict: dict[str, Any] = {
            "maps": True,
            "selection_predicate": "Prefer a plain fresh-fruit dessert.",
            "relation_type": "stable_preference",
            "task_family": "menu_or_catering_selection",
            "target_affordance": "the dessert is plain fresh fruit",
            "ordinary_mechanism": "the chef recommends the other dessert",
            "control_affordance": "the dessert is a warm spiced cake",
            "material_assumptions": [],
            "rationale": "stub",
        }
        self.verdict.update(verdict)

    def judge(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return dict(self.verdict)


class RecordingClient:
    """Minimal stand-in for the OpenAI client used by the cached mapper."""

    def __init__(self, **verdict: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        payload: dict[str, Any] = {
            "maps": True,
            "selection_predicate": (
                "Prefer vintage items usable with an NES."
            ),
            "relation_type": "compatibility",
            "task_family": "vintage_or_secondhand_finds",
            "target_affordance": "the listing states NES compatibility",
            "ordinary_mechanism": "the other item is the rarest on the shelf",
            "control_affordance": "the listing states the original box size",
            "material_assumptions": [],
            "rationale": "the user took their own NES out of the closet",
        }
        payload.update(verdict)
        self._output_text = json.dumps(payload, ensure_ascii=False)
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=self._output_text,
            model="resolved-mapper",
            id="selection-predicate-response",
        )


class ExplodingClient:
    def __init__(self) -> None:
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        raise AssertionError("the mapper must not be called")


class ExplodingJudge:
    model_name = "gpt-5-mini"
    rubric_version = SELECTION_PREDICATE_RUBRIC

    def judge(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("the mapper must not be called")


class RegistryTests(unittest.TestCase):
    def test_every_family_routes_only_known_relations(self) -> None:
        for family_id, family in TASK_FAMILY_REGISTRY.items():
            with self.subTest(family=family_id):
                self.assertEqual(family.family_id, family_id)
                self.assertTrue(family.relation_types)
                for relation in family.relation_types:
                    self.assertIn(relation, RELATION_TYPES)

    def test_every_relation_reaches_at_least_one_family(self) -> None:
        for relation in RELATION_TYPES:
            with self.subTest(relation=relation):
                self.assertTrue(allowed_task_families(relation))

    def test_routing_rejects_an_unknown_family(self) -> None:
        self.assertFalse(
            is_legal_routing("stable_preference", "hiring_selection")
        )

    def test_anchor_relation_alias_is_canonicalised(self) -> None:
        self.assertEqual(
            normalise_relation_type("owned_item_compatibility"),
            "compatibility",
        )
        with self.assertRaises(ValueError):
            normalise_relation_type("vibes")


class AnchorTests(unittest.TestCase):
    """The three user-approved anchors have to survive the whole stage."""

    def setUp(self) -> None:
        self.anchors = load_anchors()

    def map_anchor(self, anchor: Mapping[str, Any]) -> Any:
        category, durability = ANCHOR_MEMORY_LABELS[anchor["anchor_id"]]
        judge = StubJudge(
            selection_predicate=anchor["selection_predicate"],
            relation_type=anchor["relation_type"],
            task_family=anchor["task_family"],
            target_affordance=anchor["target_shape"],
            ordinary_mechanism=anchor["ordinary_winner_shape"],
            control_affordance=anchor["control_rule"],
            material_assumptions=[],
            rationale=anchor["span_caveat"],
        )
        result = map_selection_predicate(
            anchor["fact"],
            anchor["evidence_span"],
            _turns(anchor["evidence_span"], "Understood."),
            memory_category=category,
            durability=durability,
            sensitive=bool(anchor["sensitive"]),
            sensitive_terms=tuple(anchor["sensitive_terms"]),
            judge=judge,
        )
        return result, judge

    def test_three_anchors_are_loaded(self) -> None:
        self.assertEqual(len(self.anchors), 3)
        self.assertEqual(
            sorted(anchor["anchor_id"] for anchor in self.anchors),
            sorted(ANCHOR_MEMORY_LABELS),
        )

    def test_anchors_pass_every_deterministic_check(self) -> None:
        for anchor in self.anchors:
            with self.subTest(anchor=anchor["anchor_id"]):
                result, judge = self.map_anchor(anchor)
                self.assertEqual(len(judge.calls), 1)
                self.assertTrue(result.maps)
                self.assertEqual(result.rejection_reasons, ())
                self.assertTrue(passes_selection_predicate(result))
                self.assertEqual(result.task_family, anchor["task_family"])
                self.assertEqual(
                    result.relation_type,
                    normalise_relation_type(anchor["relation_type"]),
                )
                self.assertTrue(
                    is_legal_routing(result.relation_type, result.task_family)
                )
                self.assertEqual(result.material_assumptions, ())
                self.assertEqual(
                    result.selection_predicate, anchor["selection_predicate"]
                )

    def test_the_mapper_sees_the_fact_span_and_memory_labels(self) -> None:
        anchor = self.anchors[0]
        _, judge = self.map_anchor(anchor)
        call = judge.calls[0]
        self.assertEqual(call["claim"], anchor["fact"])
        self.assertEqual(call["evidence_span"], anchor["evidence_span"])
        self.assertEqual(call["memory_category"], "preference")
        self.assertEqual(call["durability"], "durable")
        self.assertTrue(call["sensitive"])
        rendered = render_selection_predicate_request(
            anchor["fact"],
            anchor["evidence_span"],
            _turns(anchor["evidence_span"]),
            memory_category="preference",
            durability="durable",
            sensitive=True,
            sensitive_terms=anchor["sensitive_terms"],
        )
        self.assertIn(anchor["fact"], rendered)
        self.assertIn(anchor["evidence_span"], rendered)
        self.assertIn("family health history", rendered)
        self.assertIn("menu_or_catering_selection", rendered)

    def test_anchor_sensitivity_survives_into_the_row(self) -> None:
        anchor = next(
            item for item in self.anchors if item["anchor_id"] == (
                "anchor-dessert-menu"
            )
        )
        result, _ = self.map_anchor(anchor)
        self.assertTrue(result.sensitive)
        self.assertEqual(result.sensitive_terms, ("family health history",))
        self.assertEqual(
            result.to_dict()["sensitive_terms"], ["family health history"]
        )
        row = predicate_row(result)
        self.assertEqual(
            sorted(row),
            sorted(
                (
                    "selection_predicate",
                    "task_family",
                    "target_affordance",
                    "ordinary_mechanism",
                    "control_affordance",
                    "relation_type",
                    "material_assumptions",
                    "sensitive",
                    "sensitive_terms",
                )
            ),
        )
        self.assertTrue(row["sensitive"])
        self.assertEqual(row["sensitive_terms"], ["family health history"])
        self.assertEqual(row["task_family"], anchor["task_family"])
        self.assertEqual(row["relation_type"], "stable_preference")

    def test_an_insensitive_anchor_carries_no_terms(self) -> None:
        anchor = next(
            item
            for item in self.anchors
            if item["anchor_id"] == "anchor-vintage-store"
        )
        result, _ = self.map_anchor(anchor)
        self.assertFalse(result.sensitive)
        self.assertEqual(result.sensitive_terms, ())
        self.assertEqual(predicate_row(result)["sensitive_terms"], [])


class PostCheckTests(unittest.TestCase):
    claim = "The user owns an old NES console."
    span = "pulled the old NES out of the closet and played a side scroller"

    def map_with(self, **verdict: Any) -> Any:
        return map_selection_predicate(
            self.claim,
            self.span,
            _turns(self.span, "That console needs a composite input."),
            memory_category="owned_item",
            durability="durable",
            judge=StubJudge(**verdict),
        )

    def test_incompatible_routing_is_rejected(self) -> None:
        result = self.map_with(
            selection_predicate="Prefer vintage items usable with an NES.",
            relation_type="stable_preference",
            task_family="compatible_accessory_or_part_selection",
            target_affordance="the part fits an NES",
        )
        self.assertTrue(result.maps)
        self.assertEqual(
            result.rejection_reasons, (INCOMPATIBLE_TASK_FAMILY_ROUTING,)
        )
        self.assertFalse(passes_selection_predicate(result))
        self.assertEqual(
            selection_predicate_rejection_reason(result),
            f"selection_predicate_{INCOMPATIBLE_TASK_FAMILY_ROUTING}",
        )

    def test_material_assumptions_reject_the_mapping(self) -> None:
        result = self.map_with(
            selection_predicate="Prefer vintage items usable with an NES.",
            relation_type="compatibility",
            task_family="vintage_or_secondhand_finds",
            target_affordance="the listing states NES compatibility",
            material_assumptions=["the user still has a working CRT"],
        )
        self.assertEqual(
            result.rejection_reasons, (MATERIAL_ASSUMPTION_REQUIRED,)
        )
        self.assertFalse(passes_selection_predicate(result))
        self.assertEqual(
            result.material_assumptions, ("the user still has a working CRT",)
        )

    def test_an_unanchored_predicate_without_an_affordance_is_rejected(
        self,
    ) -> None:
        result = self.map_with(
            selection_predicate="Prefer whatever feels nostalgic.",
            relation_type="compatibility",
            task_family="vintage_or_secondhand_finds",
            target_affordance="   ",
        )
        self.assertEqual(
            result.rejection_reasons, (UNANCHORED_SELECTION_PREDICATE,)
        )

    def test_a_named_affordance_saves_a_paraphrased_predicate(self) -> None:
        result = self.map_with(
            selection_predicate="Prefer whatever feels nostalgic.",
            relation_type="compatibility",
            task_family="vintage_or_secondhand_finds",
            target_affordance="the listing states NES compatibility",
        )
        self.assertEqual(result.rejection_reasons, ())
        self.assertTrue(passes_selection_predicate(result))

    def test_an_unknown_family_or_relation_is_refused(self) -> None:
        for verdict in (
            {"relation_type": "aesthetic_affinity"},
            {"task_family": "hiring_selection"},
        ):
            with self.subTest(verdict=verdict):
                with self.assertRaises(ValueError):
                    self.map_with(**verdict)


class AbstentionTests(unittest.TestCase):
    def test_a_broad_interest_abstention_is_respected(self) -> None:
        judge = StubJudge(
            maps=False,
            selection_predicate="",
            relation_type="none",
            task_family="none",
            target_affordance="",
            ordinary_mechanism="",
            control_affordance="",
            rationale="a broad interest ranks no ordinary option",
        )
        result = map_selection_predicate(
            "The user is interested in medieval history.",
            "I have always been interested in medieval history.",
            _turns("I have always been interested in medieval history."),
            memory_category="preference",
            durability="durable",
            judge=judge,
        )
        self.assertTrue(result.abstained)
        self.assertFalse(passes_selection_predicate(result))
        self.assertEqual(result.rejection_reasons, ())
        self.assertEqual(result.task_family, "")
        self.assertEqual(result.relation_type, "")
        self.assertEqual(
            selection_predicate_rejection_reason(result),
            NO_SELECTION_PREDICATE,
        )

    def test_a_spent_fact_never_reaches_the_mapper(self) -> None:
        result = map_selection_predicate(
            "The user drank a flat white this morning.",
            "I drank a flat white this morning before the standup.",
            _turns("I drank a flat white this morning before the standup."),
            memory_category="none",
            durability="ephemeral",
            judge=ExplodingJudge(),
        )
        self.assertTrue(result.abstained)
        self.assertEqual(
            result.deterministic_abstentions,
            (UNMAPPABLE_MEMORY_CATEGORY, NON_DURABLE_MEMORY),
        )
        self.assertFalse(passes_selection_predicate(result))

    def test_a_spent_fact_skips_the_client_and_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = map_selection_predicate(
                "The user drank a flat white this morning.",
                "I drank a flat white this morning.",
                _turns("I drank a flat white this morning."),
                memory_category="preference",
                durability="ephemeral",
                cache_dir=tmp,
                client=ExplodingClient(),
            )
            self.assertEqual(list(Path(tmp).glob("*.json")), [])
        self.assertEqual(
            result.deterministic_abstentions, (NON_DURABLE_MEMORY,)
        )

    def test_deterministic_abstentions_ignore_missing_labels(self) -> None:
        self.assertEqual(
            deterministic_abstentions(
                "The user owns an NES.", "I still have my old NES.", "", ""
            ),
            (),
        )

    def test_abstention_still_forwards_sensitivity(self) -> None:
        judge = StubJudge(maps=False, relation_type="none", task_family="none")
        result = map_selection_predicate(
            "The user is recovering from knee surgery.",
            "I had knee surgery in March and I still avoid stairs.",
            _turns("I had knee surgery in March and I still avoid stairs."),
            memory_category="accessibility_need",
            durability="durable",
            sensitive=True,
            sensitive_terms=("knee surgery",),
            judge=judge,
        )
        self.assertEqual(result.sensitive_terms, ("knee surgery",))
        self.assertTrue(result.to_dict()["sensitive"])


class CacheTests(unittest.TestCase):
    claim = "The user owns an old NES console."
    span = "pulled the old NES out of the closet"

    def turns(self) -> list[dict[str, str]]:
        return _turns(self.span, "That needs a composite input.")

    def test_cache_key_is_stable_and_input_sensitive(self) -> None:
        turns = self.turns()
        baseline = selection_predicate_cache_key(
            self.claim,
            self.span,
            turns,
            memory_category="owned_item",
            durability="durable",
        )
        self.assertEqual(
            baseline,
            selection_predicate_cache_key(
                self.claim,
                self.span,
                list(turns),
                memory_category="owned_item",
                durability="durable",
            ),
        )
        for other in (
            selection_predicate_cache_key(
                "The user owns a Mega Drive.", self.span, turns
            ),
            selection_predicate_cache_key(self.claim, "I own an NES.", turns),
            selection_predicate_cache_key(self.claim, self.span, turns[:1]),
            selection_predicate_cache_key(
                self.claim,
                self.span,
                turns,
                memory_category="owned_item",
                durability="currently_operative",
            ),
            selection_predicate_cache_key(
                self.claim, self.span, turns, model="other-model"
            ),
            selection_predicate_cache_key(
                self.claim,
                self.span,
                turns,
                rubric_version="parmbench_selection_predicate_v0",
            ),
        ):
            self.assertNotEqual(baseline, other)

    def test_populate_then_frozen_replay(self) -> None:
        turns = self.turns()
        client = RecordingClient()
        with tempfile.TemporaryDirectory() as tmp:
            first = map_selection_predicate(
                self.claim,
                self.span,
                turns,
                memory_category="owned_item",
                durability="durable",
                cache_dir=tmp,
                client=client,
            )
            request_hash = selection_predicate_cache_key(
                self.claim,
                self.span,
                turns,
                memory_category="owned_item",
                durability="durable",
            )
            self.assertTrue((Path(tmp) / f"{request_hash}.json").exists())

            frozen = CachedOpenAISelectionPredicateJudge(
                tmp,
                SelectionPredicateCachePolicy.FROZEN,
                client=ExplodingClient(),
            )
            replay = map_selection_predicate(
                self.claim,
                self.span,
                turns,
                memory_category="owned_item",
                durability="durable",
                judge=frozen,
            )
            self.assertEqual(first, replay)
            self.assertTrue(passes_selection_predicate(first))
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(first.resolved_model, "resolved-mapper")
            self.assertEqual(first.response_id, "selection-predicate-response")
            self.assertIsNotNone(frozen.cache_hash)

    def test_frozen_cache_miss_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SelectionPredicateCacheMissError):
                map_selection_predicate(
                    self.claim,
                    self.span,
                    self.turns(),
                    memory_category="owned_item",
                    durability="durable",
                    cache_dir=tmp,
                    policy="frozen",
                    client=ExplodingClient(),
                )

    def test_cached_mapper_is_given_the_versioned_rubric(self) -> None:
        client = RecordingClient()
        with tempfile.TemporaryDirectory() as tmp:
            map_selection_predicate(
                self.claim,
                self.span,
                self.turns(),
                memory_category="owned_item",
                durability="durable",
                cache_dir=tmp,
                client=client,
            )
            cached = json.loads(
                next(Path(tmp).glob("*.json")).read_text(encoding="utf-8")
            )
        self.assertEqual(cached["rubric_version"], SELECTION_PREDICATE_RUBRIC)
        self.assertEqual(cached["model"], "gpt-5-mini")
        self.assertEqual(client.calls[0]["model"], "gpt-5-mini")


class SupplyStageTests(unittest.TestCase):
    """How the drafting script decides to run the stage."""

    def test_the_stage_follows_the_memory_quality_gate(self) -> None:
        self.assertTrue(
            selection_predicate_enabled(None, memory_quality_gate=True)
        )
        self.assertFalse(
            selection_predicate_enabled(None, memory_quality_gate=False)
        )

    def test_an_explicit_flag_wins(self) -> None:
        self.assertFalse(
            selection_predicate_enabled(False, memory_quality_gate=True)
        )
        self.assertTrue(
            selection_predicate_enabled(True, memory_quality_gate=False)
        )


if __name__ == "__main__":
    unittest.main()
