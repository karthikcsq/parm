from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import tiktoken

from parm_bench.amara import normalize_amara, sha256_file
from parm_bench.dataset import (
    LISTING_PREFIXES,
    TOKENIZER,
    DatasetValidationError,
    load_cases,
    validate_cases,
    validation_profile,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "benchmark_v1"
PERSONAMEM_DATASET = ROOT / "data" / "benchmark_personamem_v0"
PERSONAMEM_MIXED_DATASET = (
    ROOT / "data" / "benchmark_personamem_mixed_v0"
)

PARMBENCH_PERSONAS = (7, 8)
PARMBENCH_PROMPT = (
    "After reading the collected material, choose exactly one option you "
    "would move forward with. Reply using only its name exactly as written."
)
PARMBENCH_EVIDENCE = {
    7: "I keep a shared log of my weekly swimming sessions.",
    8: "I stopped drinking coffee after two in the afternoon.",
}
PARMBENCH_MEMORY = {
    7: "The user tracks weekly swimming sessions in a shared log.",
    8: "The user avoids caffeine in the second half of the day.",
}
PARMBENCH_CUE = {
    7: "The listed slot reserves a lane and writes each set into a shared log.",
    8: "The listed slot serves decaffeinated pours after midday.",
}
PARMBENCH_NEUTRAL = {
    7: "The listed slot runs at the same hour as the others on the sheet.",
    8: "The listed slot pours the same house blend as the others on the sheet.",
}


def _parmbench_filler(encoding, persona: int, target_tokens: int) -> list[str]:
    lines: list[str] = []
    total = 0
    row = 0
    while total < target_tokens:
        line = f"note {persona} row {row}: " + " ".join(
            f"detail{persona}x{row}y{word}" for word in range(10)
        )
        lines.append(line)
        total += len(encoding.encode(line))
        row += 1
    return lines


def _write_parmbench_fixture(root: Path) -> None:
    """Write a minimal retrieval-agnostic dataset that validates cleanly."""

    (root / "contexts").mkdir(parents=True, exist_ok=True)
    encoding = tiktoken.get_encoding(TOKENIZER)
    corpora = []
    cases = []
    for index, persona in enumerate(PARMBENCH_PERSONAS):
        corpus_id = f"parmbench/persona-{persona}"
        source_id = f"notes/persona-{persona}/turns-0001"
        source_path = f"notes/persona-{persona}/turns-0001.md"
        absolute_source = root / "corpus" / source_path
        absolute_source.parent.mkdir(parents=True, exist_ok=True)
        absolute_source.write_text(
            "User: A quick note from this week.\n"
            f"User: {PARMBENCH_EVIDENCE[persona]}\n",
            encoding="utf-8",
        )
        corpora.append(
            {
                "corpus_id": corpus_id,
                "source_root": "corpus",
                "source_id_prefix": f"notes/persona-{persona}/",
            }
        )

        filler = _parmbench_filler(encoding, persona, 6_400)
        body = list(filler)
        body.insert(3, f"Ordinary Option {persona} leads the sheet on schedule fit.")
        cue_row = 12 + index * 40
        body.insert(cue_row, PARMBENCH_CUE[persona])
        body.insert(cue_row + 1, f"Personal Option {persona} sits further down.")
        content_path = f"contexts/scenario-{persona}.md"
        (root / content_path).write_text(
            f"# collected material {persona}\n\n" + "\n".join(body) + "\n",
            encoding="utf-8",
        )

        output_choice = f"Ordinary Option {persona}"
        memory_choice = f"Personal Option {persona}"
        base_case_id = f"parmbench-fixture-{persona}"
        memory_sources = [
            {
                "source_id": source_id,
                "path": source_path,
                "perturbations": [],
                "sha256": sha256_file(root / "corpus" / source_path),
                "evidence_span": {"text": PARMBENCH_EVIDENCE[persona]},
            }
        ]
        distractors = {
            "sources": [
                {
                    "source_id": f"notes/persona-{persona}/turns-{row:04d}",
                    "text": f"User: an unrelated note {persona} number {row}.",
                    "perturbations": [],
                }
                for row in (2, 3, 4)
            ]
        }
        for variant in ("positive", "cue-ablated", "memory-included"):
            cue_present = variant != "cue-ablated"
            prompt = (
                f"{PARMBENCH_MEMORY[persona]} {PARMBENCH_PROMPT}"
                if variant == "memory-included"
                else PARMBENCH_PROMPT
            )
            cases.append(
                {
                    "case_id": f"{base_case_id}-{variant}",
                    "base_case_id": base_case_id,
                    "corpus_id": corpus_id,
                    "variant": variant,
                    "prompt": prompt,
                    "observation": {
                        "kind": "tool_result",
                        "content_path": content_path,
                        "replacements": (
                            []
                            if cue_present
                            else [
                                {
                                    "old": PARMBENCH_CUE[persona],
                                    "new": PARMBENCH_NEUTRAL[persona],
                                }
                            ]
                        ),
                    },
                    "cue": {
                        "present": cue_present,
                        "type": "logging_affordance",
                        "text": PARMBENCH_CUE[persona],
                    },
                    "memory": {
                        "corpus_id": corpus_id,
                        "text": PARMBENCH_MEMORY[persona],
                        "gold_source_ids": [source_id],
                        "sensitive_terms": [],
                        "sources": copy.deepcopy(memory_sources),
                    },
                    "decisions": {
                        "answer_type": "natural_language_choice",
                        "output_only": {"choice": output_choice},
                        "memory_conditioned": {
                            "choice": (
                                output_choice
                                if variant == "cue-ablated"
                                else memory_choice
                            )
                        },
                    },
                    "distractors": copy.deepcopy(distractors),
                    "provenance": {
                        "approved": True,
                        "persona_id": persona,
                        "evaluation_split": "development",
                        "case_builder_version": "parmbench_fixture_v1",
                    },
                }
            )
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "validation_profile": "parmbench_v1", "corpora": corpora},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (root / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")


class DatasetValidationTests(unittest.TestCase):
    def test_pilot_dataset_is_valid(self) -> None:
        cases = load_cases(DATASET)
        validate_cases(cases)
        self.assertEqual(len(cases), 54)
        self.assertEqual(len({case["base_case_id"] for case in cases}), 18)
        self.assertEqual({case["corpus_id"] for case in cases}, {"amara-life-v1"})

    def test_personamem_development_dataset_is_valid_and_persona_disjoint(
        self,
    ) -> None:
        cases = load_cases(PERSONAMEM_DATASET)
        validate_cases(cases)
        self.assertEqual(len(cases), 90)
        self.assertEqual(len({case["base_case_id"] for case in cases}), 30)
        self.assertEqual(len({case["corpus_id"] for case in cases}), 30)
        self.assertTrue(
            all(
                case["provenance"]["source_split"] == "train_text"
                for case in cases
            )
        )

    def test_personamem_mixed_dataset_is_valid_without_listing_rows(
        self,
    ) -> None:
        cases = load_cases(PERSONAMEM_MIXED_DATASET)
        validate_cases(cases)
        self.assertEqual(len(cases), 90)
        self.assertEqual(len({case["base_case_id"] for case in cases}), 30)
        self.assertEqual(len({case["corpus_id"] for case in cases}), 30)
        self.assertEqual(
            {
                case["provenance"]["envelope_style"]
                for case in cases
            },
            {
                "research_notebook",
                "forwarded_thread",
                "meeting_dump",
                "web_clippings",
                "working_draft",
                "mixed_markdown",
            },
        )
        self.assertTrue(
            all(
                sum(
                    line.startswith(LISTING_PREFIXES)
                    for line in case["observation_text"].splitlines()
                )
                < 10
                for case in cases
            )
        )
        legacy_positive = {
            case["base_case_id"]: case
            for case in load_cases(PERSONAMEM_DATASET)
            if case["variant"] == "positive"
        }
        mixed_by_base = {
            case["base_case_id"]: {
                variant["variant"]: variant
                for variant in cases
                if variant["base_case_id"] == case["base_case_id"]
            }
            for case in cases
        }
        for base_case_id, variants in mixed_by_base.items():
            positive = variants["positive"]
            control = variants["cue-ablated"]
            target_name = positive["decisions"]["memory_conditioned"]["choice"]
            legacy_base_case_id = base_case_id.replace(
                "parm-personamem-mixed-",
                "parm-personamem-",
            )
            legacy_target = legacy_positive[legacy_base_case_id][
                "decisions"
            ]["memory_conditioned"]["choice"]
            self.assertNotEqual(target_name, legacy_target)
            self.assertEqual(
                positive["observation_text"].count(target_name), 1
            )
            self.assertEqual(
                control["observation_text"].count(target_name), 1
            )
            self.assertIn(
                "does not describe a specialized",
                control["observation"]["replacements"][0]["new"],
            )

    def test_personamem_profile_rejects_hidden_preference_prompt_leakage(
        self,
    ) -> None:
        cases = load_cases(PERSONAMEM_DATASET)
        broken = copy.deepcopy(cases)
        positive = next(case for case in broken if case["variant"] == "positive")
        positive["prompt"] += " Enjoys historical dramas on TV."
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(broken)
        self.assertIn("preference or query leaks", str(context.exception))

    def test_every_base_case_has_all_three_variants(self) -> None:
        cases = load_cases(DATASET)
        by_base: dict[str, set[str]] = {}
        for case in cases:
            by_base.setdefault(case["base_case_id"], set()).add(case["variant"])
        self.assertTrue(
            all(
                value == {"positive", "cue-ablated", "memory-included"}
                for value in by_base.values()
            )
        )

    def test_pilot_and_expansion_splits_are_complete(self) -> None:
        cases = load_cases(DATASET)
        counts: dict[str, set[str]] = {}
        for case in cases:
            split = case["provenance"]["evaluation_split"]
            counts.setdefault(split, set()).add(case["base_case_id"])
        self.assertEqual(len(counts["pilot"]), 5)
        self.assertEqual(len(counts["expansion"]), 13)

    def test_memory_included_injects_memory_and_reuses_positive_observation(
        self,
    ) -> None:
        cases = {case["case_id"]: case for case in load_cases(DATASET)}
        for base in {case["base_case_id"] for case in cases.values()}:
            positive = cases[f"{base}-positive"]
            included = cases[f"{base}-memory-included"]
            # the memory fact is present in the prompt (the injected ceiling)
            self.assertIn(
                included["memory"]["text"].casefold(), included["prompt"].casefold()
            )
            # the observation is byte-identical to the positive variant
            self.assertEqual(
                included["observation_text"], positive["observation_text"]
            )
            # the decision the model should reach is the memory-conditioned one
            self.assertNotEqual(
                included["decisions"]["output_only"]["choice"],
                included["decisions"]["memory_conditioned"]["choice"],
            )
            self.assertTrue(included["cue"]["present"])

    def test_memory_included_without_injected_memory_fails(self) -> None:
        cases = load_cases(DATASET)
        broken = copy.deepcopy(cases)
        included = next(
            case for case in broken if case["variant"] == "memory-included"
        )
        included["prompt"] = "Choose exactly one option. Reply with its name."
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(broken)
        self.assertIn("must contain the injected memory", str(context.exception))

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

    def test_repaired_fairness_cases_keep_symmetric_output_leads(self) -> None:
        cases = {case["case_id"]: case for case in load_cases(DATASET)}
        expected = {
            "parm-amara-phone-feature-digest": (
                "Feature F-004 — Focus Stack",
                "Feature F-147 — Private Connection Check-In",
                ("personal call",),
            ),
            "parm-amara-weekend-events": (
                "Event E-004 — Saturday Investor Breakfast",
                "Event E-147 — Offline Saturday Field Workshop",
                ("phone-free", "work routines"),
            ),
            "parm-amara-human-factors-event": (
                "Session H-004 — Evidence-Based Override Design",
                "Session H-147 — Readable Handoffs Under Low Confidence",
                ("interpretable", "cognitive load", "automation confidence"),
            ),
        }
        for base_case_id, (
            output_lead,
            memory_target,
            control_forbidden_phrases,
        ) in expected.items():
            positive = cases[f"{base_case_id}-positive"]
            control = cases[f"{base_case_id}-cue-ablated"]
            included = cases[f"{base_case_id}-memory-included"]
            self.assertEqual(
                positive["decisions"]["output_only"]["choice"], output_lead
            )
            self.assertEqual(
                control["decisions"]["memory_conditioned"]["choice"], output_lead
            )
            self.assertEqual(
                included["decisions"]["memory_conditioned"]["choice"], memory_target
            )
            control_text = control["observation"]["replacements"][0]["new"].casefold()
            for phrase in control_forbidden_phrases:
                self.assertNotIn(phrase, control_text)

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
            "parm-amara-startup-expo": "marcus reid",
            "parm-amara-webinar-catalog": "vp of sales with utility sector experience",
            "parm-amara-robotics-market-map": "vela robotics",
            "parm-amara-growth-case-studies": "vespera dynamics",
            "parm-amara-venture-law-roundup": "participating preferred",
            "parm-amara-workflow-marketplace": "48-hour flag system",
            "parm-amara-phone-feature-digest": "called my mother",
            "parm-amara-documentary-catalog": "infrastructure engineer",
            "parm-amara-weekend-events": "three consecutive saturdays",
            "parm-amara-essay-digest": "founders pre-term sheet",
            "parm-amara-market-chart-pack": "compute costs",
            "parm-amara-human-factors-event": "interpretable decision logs",
            "parm-amara-austin-tech-roundup": "hannah liu",
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


class ParmbenchProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        _write_parmbench_fixture(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def cases(self) -> list[dict]:
        return copy.deepcopy(load_cases(self.root))

    def test_profile_drops_listing_rows_and_widens_the_token_band(self) -> None:
        profile = validation_profile("parmbench_v1")
        self.assertFalse(profile.require_listing_rows)
        self.assertEqual(profile.min_context_tokens, 6_000)
        self.assertEqual(profile.max_context_tokens, 16_000)
        self.assertFalse(profile.run_personamem_pilot_gate)

    def test_fixture_validates_without_listing_rows(self) -> None:
        cases = self.cases()
        validate_cases(cases)
        self.assertEqual(len(cases), 6)
        self.assertEqual(len({case["base_case_id"] for case in cases}), 2)
        for case in cases:
            self.assertFalse(
                any(
                    line.startswith(LISTING_PREFIXES)
                    for line in case["observation_text"].splitlines()
                )
            )

    def test_missing_evidence_span_fails(self) -> None:
        cases = self.cases()
        del cases[0]["memory"]["sources"][0]["evidence_span"]
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(cases)
        self.assertIn("is missing evidence_span", str(context.exception))

    def test_paraphrased_evidence_span_fails(self) -> None:
        cases = self.cases()
        cases[0]["memory"]["sources"][0]["evidence_span"]["text"] = (
            "The user swims every week and logs it."
        )
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(cases)
        self.assertIn("is not verbatim in", str(context.exception))

    def test_memory_conditioned_choice_in_prompt_fails(self) -> None:
        cases = self.cases()
        positive = next(case for case in cases if case["variant"] == "positive")
        positive["prompt"] += (
            " " + positive["decisions"]["memory_conditioned"]["choice"]
        )
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(cases)
        self.assertIn("memory-conditioned choice leaks into prompt", str(context.exception))

    def test_evidence_span_in_the_ceiling_prompt_fails(self) -> None:
        cases = self.cases()
        included = next(
            case for case in cases if case["variant"] == "memory-included"
        )
        included["prompt"] += (
            " " + included["memory"]["sources"][0]["evidence_span"]["text"]
        )
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(cases)
        self.assertIn("leaks into", str(context.exception))

    def test_observation_below_the_widened_token_floor_fails(self) -> None:
        cases = self.cases()
        cases[0]["observation_text"] = "too short to be a realistic observation"
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(cases)
        self.assertIn("expected 6000-16000", str(context.exception))

    def test_source_outside_the_declared_persona_prefix_fails(self) -> None:
        cases = self.cases()
        source = cases[0]["memory"]["sources"][0]
        source["source_id"] = "notes/persona-999/turns-0001"
        cases[0]["memory"]["gold_source_ids"] = [source["source_id"]]
        with self.assertRaises(DatasetValidationError) as context:
            validate_cases(cases)
        self.assertIn("is outside corpus prefix", str(context.exception))


if __name__ == "__main__":
    unittest.main()
