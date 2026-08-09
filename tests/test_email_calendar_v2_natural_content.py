from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_email_calendar_v2"
CORPUS = DATASET / "corpora" / "ops-lead-email-calendar-v2"
SCRIPT = ROOT / "scripts" / "build_workflows_email_calendar_v2_cases.py"
SLUGS = {
    "atlas-renewal-sla-legal-review",
    "trellis-production-migration-blackout",
    "orion-incident-travel-approval",
}
VARIANTS = {"positive", "cue-ablated", "memory-included"}
FORBIDDEN_GOAL = re.compile(
    r"\b(?:list_messages|get_message_thread|list_events|get_event|create_draft_reply|send_reply|assign_follow_up|update_event|decline_event|thread_id|event_id)\b|\b(?:thread|message|event|attachment)-[a-z0-9-]+\b",
    re.IGNORECASE,
)
FORBIDDEN_CASE_LANGUAGE = re.compile(
    r"\b(?:do not|must|use|call|invoke|set|assign|decline|send|draft|route)\s+(?:the\s+)?(?:[a-z_]+|[a-z0-9-]+)\b",
    re.IGNORECASE,
)


def rows() -> list[dict]:
    return [json.loads(line) for line in (DATASET / "cases.jsonl").read_text().splitlines() if line]


def value_at(document: dict, dotted: str):
    value = document
    for part in dotted.split("."):
        value = value[int(part)] if part.isdigit() else value[part]
    return value


class EmailCalendarV2NaturalContentTest(unittest.TestCase):
    def test_builder_creates_exact_natural_triplets(self) -> None:
        run = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        data = rows()
        self.assertEqual(len(data), 9)
        grouped: dict[str, dict[str, dict]] = {}
        for row in data:
            grouped.setdefault(row["base_case_id"], {})[row["variant"]] = row
        self.assertEqual(set(grouped), {f"parm-email-calendar-v2-{slug}" for slug in SLUGS})
        for variants in grouped.values():
            self.assertEqual(set(variants), VARIANTS)
            self.assertEqual(variants["positive"]["environment"]["fixture_path"], variants["memory-included"]["environment"]["fixture_path"])
            self.assertNotEqual(variants["positive"]["environment"]["fixture_path"], variants["cue-ablated"]["environment"]["fixture_path"])

    def test_v2_cases_pass_shared_structural_validator(self) -> None:
        from parm_bench.workflows.case import load_workflow_cases, validate_workflow_cases

        validate_workflow_cases(load_workflow_cases(DATASET))

    def test_goals_are_natural_and_ceiling_injects_gold_source_text(self) -> None:
        opening_goals = {
            "atlas-renewal-sla-legal-review": "Please take care of the customer request that came in this morning and keep things moving.",
            "trellis-production-migration-blackout": "Please deal with the delivery session on my calendar later this year.",
            "orion-incident-travel-approval": "Please take care of the expense request in my inbox.",
        }
        atomic_cues = re.compile(
            r"\b(?:atlas|trellis|orion|contractual|service-level|production|migration|incident-manager|restoring)\b",
            re.IGNORECASE,
        )
        for row in rows():
            with self.subTest(case=row["case_id"]):
                self.assertIsNone(FORBIDDEN_GOAL.search(row["goal"]), row["goal"])
                self.assertIsNone(FORBIDDEN_CASE_LANGUAGE.search(row["goal"]), row["goal"])
                self.assertNotIn("case_id", row["goal"].casefold())
                if row["variant"] == "memory-included":
                    gold = row["memory"]["sources"][0]
                    text = (CORPUS / "source" / gold["path"]).read_text(encoding="utf-8")
                    self.assertIn(text, row["goal"])
                    self.assertNotIn("standing instruction", row["goal"].casefold())
                else:
                    slug = row["base_case_id"].removeprefix("parm-email-calendar-v2-")
                    self.assertEqual(row["goal"], opening_goals[slug])
                    self.assertIsNone(atomic_cues.search(row["goal"]), row["goal"])

    def test_gold_sources_are_verbatim_and_semantic_pairs_change_one_detail_fact(self) -> None:
        data = rows()
        for positive in (row for row in data if row["variant"] == "positive"):
            control = next(row for row in data if row["base_case_id"] == positive["base_case_id"] and row["variant"] == "cue-ablated")
            gold = positive["memory"]["sources"][0]
            source = CORPUS / "source" / gold["path"]
            text = source.read_text(encoding="utf-8")
            self.assertEqual(gold["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertIn(gold["evidence_span"]["text"], text)
            self.assertGreaterEqual(len(positive["distractors"]["sources"]), 3)
            positive_fixture = json.loads((DATASET / positive["environment"]["fixture_path"]).read_text())
            control_fixture = json.loads((DATASET / control["environment"]["fixture_path"]).read_text())
            path = positive["ablation"]["fixture_path"]
            self.assertEqual(value_at(positive_fixture, path), positive["cue"]["text"])
            self.assertNotEqual(value_at(positive_fixture, path), value_at(control_fixture, path))
            def normalized(fixture: dict) -> dict:
                clone = json.loads(json.dumps(fixture))
                holder = clone
                parts = path.split(".")
                for part in parts[:-1]:
                    holder = holder[int(part)] if part.isdigit() else holder[part]
                holder[parts[-1]] = "<late-applicability-fact>"
                return clone
            self.assertEqual(normalized(positive_fixture), normalized(control_fixture))
            self.assertGreaterEqual(len(positive_fixture["mailbox"]["threads"]), 2)
            self.assertGreaterEqual(len(positive_fixture["calendar"]["events"]), 2)

    def test_assertions_require_new_commitments_and_allow_legitimate_alternatives(self) -> None:
        for row in rows():
            with self.subTest(case=row["case_id"]):
                self.assertTrue(row["assertions"])
                self.assertTrue(any(item["role"] == "decisive" for item in row["assertions"]))
                self.assertNotEqual(row["assertions"], [{"kind": "no_mutation"}])
        for scenario in ("atlas-renewal-sla-legal-review", "trellis-production-migration-blackout", "orion-incident-travel-approval"):
            positive = next(row for row in rows() if row["case_id"].endswith(f"{scenario}-positive"))
            contract = next(item for item in positive["assertions"] if item["kind"] == "constraint_compliance")
            self.assertGreaterEqual(len(contract["allowed_outcomes"]), 2)
            self.assertTrue(contract["forbidden_outcomes"])
            self.assertTrue(positive["target"])


if __name__ == "__main__":
    unittest.main()
