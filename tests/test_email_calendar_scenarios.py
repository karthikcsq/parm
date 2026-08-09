from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_email_calendar_v1"
CORPUS = DATASET / "corpora" / "ops-lead-email-calendar-v1"
SCRIPT = ROOT / "scripts" / "build_workflows_email_calendar_v1_cases.py"
EXPECTED_SLUGS = {
    "email-contract-commitment-review",
    "calendar-nda-attendee-gate",
    "email-incident-external-wording-owner",
    "calendar-customer-blackout-reschedule",
    "email-expense-exception-routing",
    "calendar-accessibility-meeting-format",
}
VARIANTS = {"positive", "cue-ablated", "memory-included"}


def _load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in (DATASET / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class EmailCalendarScenarioDataTest(unittest.TestCase):
    def test_builder_is_deterministic(self) -> None:
        first = subprocess.run(
            [sys.executable, str(SCRIPT)], cwd=ROOT, text=True, capture_output=True
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        before = (DATASET / "cases.jsonl").read_bytes()
        second = subprocess.run(
            [sys.executable, str(SCRIPT)], cwd=ROOT, text=True, capture_output=True
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(before, (DATASET / "cases.jsonl").read_bytes())

    def test_exact_inventory_and_triplet_symmetry(self) -> None:
        grouped: dict[str, dict[str, dict]] = {}
        for case in _load_cases():
            grouped.setdefault(case["base_case_id"], {})[case["variant"]] = case
        self.assertEqual(set(grouped), {f"parm-email-calendar-{slug}" for slug in EXPECTED_SLUGS})
        for base_id, variants in grouped.items():
            with self.subTest(base_id=base_id):
                self.assertEqual(set(variants), VARIANTS)
                positive = variants["positive"]
                ceiling = variants["memory-included"]
                control = variants["cue-ablated"]
                self.assertEqual(positive["environment"]["adapter"], "email_calendar_fixture")
                self.assertEqual(positive["environment"]["fixture_path"], ceiling["environment"]["fixture_path"])
                self.assertNotEqual(positive["environment"]["fixture_path"], control["environment"]["fixture_path"])
                self.assertTrue(positive["cue"]["present"])
                self.assertFalse(control["cue"]["present"])
                self.assertEqual(positive["memory"]["gold_source_ids"], ceiling["memory"]["gold_source_ids"])
                self.assertIn(positive["memory"]["text"], ceiling["goal"])
                self.assertNotIn(positive["memory"]["text"].casefold(), positive["goal"].casefold())
                self.assertNotIn(positive["cue"]["text"].casefold(), positive["goal"].casefold())
                self.assertGreaterEqual(len(positive["distractors"]["sources"]), 3)
                self.assertEqual(positive["provenance"]["constructed_by"], SCRIPT.relative_to(ROOT).as_posix())
                self.assertEqual(positive["provenance"]["persona_id"], "ops-lead-email-calendar")
                fixture = json.loads((DATASET / positive["environment"]["fixture_path"]).read_text(encoding="utf-8"))
                self.assertEqual(set(fixture), {"schema_version", "actor", "mailbox", "calendar", "contacts", "attachments"})
                self.assertIn("threads", fixture["mailbox"])
                self.assertIn("events", fixture["calendar"])

    def test_sources_are_hashed_non_poison_and_evidence_is_verbatim(self) -> None:
        for case in _load_cases():
            for source in case["memory"]["sources"]:
                with self.subTest(case=case["case_id"], source=source["source_id"]):
                    self.assertNotIn("poison", source.get("perturbations", []))
                    path = CORPUS / "source" / source["path"]
                    self.assertEqual(source["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                    self.assertIn(source["evidence_span"]["text"], path.read_text(encoding="utf-8"))
            for source in case["distractors"]["sources"]:
                path = CORPUS / "source" / source["path"]
                self.assertEqual(source["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_ablation_changes_only_the_declared_late_cue_fact_and_list_views_hide_it(self) -> None:
        for case in _load_cases():
            if case["variant"] != "positive":
                continue
            control = next(
                row for row in _load_cases()
                if row["base_case_id"] == case["base_case_id"] and row["variant"] == "cue-ablated"
            )
            positive_fixture = json.loads((DATASET / case["environment"]["fixture_path"]).read_text(encoding="utf-8"))
            control_fixture = json.loads((DATASET / control["environment"]["fixture_path"]).read_text(encoding="utf-8"))
            path = case["ablation"]["fixture_path"]
            self.assertEqual(path, control["ablation"]["fixture_path"])
            self.assertEqual(case["cue"]["text"], case["ablation"]["positive_value"])
            self.assertEqual(control["ablation"]["control_value"], case["ablation"]["control_value"])
            def list_projection(fixture: dict) -> dict:
                return {
                    "messages": [
                        {key: thread[key] for key in ("id", "sender", "subject", "labels", "date", "snippet") if key in thread}
                        for thread in fixture["mailbox"]["threads"]
                    ],
                    "events": [
                        {key: event[key] for key in ("id", "title", "start", "end", "organizer", "attendee_summary") if key in event}
                        for event in fixture["calendar"]["events"]
                    ],
                }
            self.assertNotIn(case["cue"]["text"], json.dumps(list_projection(positive_fixture), sort_keys=True))
            self.assertNotIn(case["cue"]["text"], json.dumps(list_projection(control_fixture), sort_keys=True))
            def replace_at(document: dict, dotted_path: str, value: str) -> dict:
                copied = json.loads(json.dumps(document))
                target = copied
                bits = dotted_path.split(".")
                for bit in bits[:-1]:
                    target = target[int(bit)] if bit.isdigit() else target[bit]
                target[bits[-1]] = value
                return copied
            normalized_positive = replace_at(positive_fixture, path, "<cue>")
            normalized_control = replace_at(control_fixture, path, "<cue>")
            self.assertEqual(normalized_positive, normalized_control)


if __name__ == "__main__":
    unittest.main()
