from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from parm_bench.amara import normalize_amara
from parm_bench.dataset import (
    LISTING_PREFIXES,
    DatasetValidationError,
    load_cases,
    validate_cases,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "benchmark_v1"


class DatasetValidationTests(unittest.TestCase):
    def test_pilot_dataset_is_valid(self) -> None:
        cases = load_cases(DATASET)
        validate_cases(cases)
        self.assertEqual(len(cases), 10)
        self.assertEqual(len({case["base_case_id"] for case in cases}), 5)

    def test_every_case_has_positive_and_ablated_variant(self) -> None:
        cases = load_cases(DATASET)
        by_base: dict[str, set[str]] = {}
        for case in cases:
            by_base.setdefault(case["base_case_id"], set()).add(case["variant"])
        self.assertTrue(all(value == {"positive", "cue-ablated"} for value in by_base.values()))

    def test_prompts_do_not_leak_cues_or_memory_search(self) -> None:
        for case in load_cases(DATASET):
            prompt = case["prompt"].casefold()
            self.assertNotIn(case["cue"]["text"].casefold(), prompt)
            for phrase in ("search memory", "look through memory", "gbrain"):
                self.assertNotIn(phrase, prompt)

    def test_contexts_are_large_and_distractor_rich(self) -> None:
        for case in load_cases(DATASET):
            labels = [
                line.split(".", 1)[0].strip()
                for line in case["observation_text"].splitlines()
                if line.startswith(LISTING_PREFIXES)
            ]
            self.assertGreaterEqual(len(labels), 25)
            self.assertEqual(len(labels), len({label.casefold() for label in labels}))
            self.assertGreaterEqual(len(case["distractors"]["sources"]), 3)

    def test_each_case_requests_one_natural_language_choice(self) -> None:
        for case in load_cases(DATASET):
            self.assertIn("exactly one", case["prompt"].casefold())
            self.assertEqual(
                case["decisions"]["answer_type"], "natural_language_choice"
            )
            for condition in ("output_only", "memory_conditioned"):
                choice = case["decisions"][condition]["choice"]
                self.assertIsInstance(choice, str)
                self.assertEqual(
                    case["observation_text"].casefold().count(choice.casefold()), 1
                )

    def test_memory_is_readable_prose_not_an_opaque_answer_id(self) -> None:
        for case in load_cases(DATASET):
            memory_text = case["memory"]["text"]
            self.assertGreater(len(memory_text.split()), 6)
            self.assertNotIn("selected_item_id", memory_text)
            self.assertNotIn("session-g-147", memory_text)

    def test_memory_sources_expose_identifying_language_in_prose(self) -> None:
        identifying_terms = {
            "parm-amara-conference-agenda": "novamind",
            "parm-amara-ai-news-digest": "coreweave",
            "parm-amara-podcast-feed": "burnout",
            "parm-amara-vendor-report": "novatech labs",
            "parm-amara-lunch-search": "lunch at my desk",
        }
        corpus = ROOT / "data" / "amara-life-v1" / "source"
        for case in load_cases(DATASET):
            source_text = "\n".join(
                (corpus / source["path"]).read_text(encoding="utf-8")
                for source in case["memory"]["sources"]
            ).casefold()
            self.assertIn(identifying_terms[case["base_case_id"]], source_text)

    def test_gold_sources_are_real_and_never_poison(self) -> None:
        for case in load_cases(DATASET):
            for source in case["memory"]["sources"]:
                self.assertNotIn("poison", source.get("perturbations", []))

    def test_broken_source_hash_fails(self) -> None:
        cases = load_cases(DATASET)
        broken = copy.deepcopy(cases)
        broken[0]["memory"]["sources"][0]["sha256"] = "0" * 64
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(broken)
        self.assertIn("source hash mismatch", str(context.exception))

    def test_gold_source_ids_use_frozen_index_slug_convention(self) -> None:
        # Every per-file corpus source must use the same collection prefix the
        # frozen index derives from its directory (notes/, meetings/), so that
        # scoring's gold-admission intersection can ever match retrieved slugs.
        for case in load_cases(DATASET):
            for source in case["memory"]["sources"]:
                path = source["path"]
                if path.endswith(".md") and "/" in path:
                    self.assertEqual(
                        source["source_id"].split("/", 1)[0],
                        path.split("/", 1)[0],
                        f"{case['case_id']}: {source['source_id']} vs {path}",
                    )
            for gold in case["memory"]["gold_source_ids"]:
                self.assertIn(
                    gold,
                    {source["source_id"] for source in case["memory"]["sources"]},
                )

    def test_singular_source_prefix_fails_validation(self) -> None:
        cases = load_cases(DATASET)
        broken = copy.deepcopy(cases)
        case = next(
            c
            for c in broken
            if any(s["path"].endswith(".md") for s in c["memory"]["sources"])
        )
        source = next(
            s for s in case["memory"]["sources"] if s["path"].endswith(".md")
        )
        old_id = source["source_id"]
        singular = "note/" + old_id.split("/", 1)[1]
        source["source_id"] = singular
        case["memory"]["gold_source_ids"] = [
            singular if gid == old_id else gid
            for gid in case["memory"]["gold_source_ids"]
        ]
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(broken)
        self.assertIn("does not match corpus directory", str(context.exception))

    def test_positive_decision_must_change(self) -> None:
        cases = load_cases(DATASET)
        broken = copy.deepcopy(cases)
        positive = next(case for case in broken if case["variant"] == "positive")
        positive["decisions"]["memory_conditioned"] = positive["decisions"]["output_only"]
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(broken)
        self.assertIn("does not change decision", str(context.exception))

    def test_normalizer_preserves_poison_metadata(self) -> None:
        source = ROOT / "data" / "amara-life-v1" / "source"
        with tempfile.TemporaryDirectory() as tmp:
            count = normalize_amara(source, tmp)
            self.assertEqual(count, 600)
            poison = (Path(tmp) / "slack" / "sl-0178.md").read_text(encoding="utf-8")
            self.assertIn('"poison"', poison)
            self.assertIn("perturbation_fixture_id", poison)


if __name__ == "__main__":
    unittest.main()
