"""Build the natural Email + Calendar v2 workflow pilot deterministically."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_email_calendar_v2"
CORPUS_ID = "ops-lead-email-calendar-v2"
CORPUS = DATASET / "corpora" / CORPUS_ID
SOURCE = CORPUS / "source"
ACTOR = {"id": "ops-lead", "name": "Morgan Lee", "email": "morgan.lee@example.test"}
UPSTREAM = {
    "project": "PARMBench natural workflow pilot",
    "repository": "https://github.com/eval-sys/parmbench",
    "revision": "email-calendar-natural-content-v2",
    "task_path": "data/workflows_email_calendar_v2",
    "license": "CC0-1.0",
    "derivation": "Fictional, deterministic local email and calendar context for evaluation.",
}

SCENARIOS = (
    {
        "slug": "atlas-renewal-sla-legal-review", "kind": "email", "entity": "atlas-renewal", "target_id": "mail-customer-request-march-4",
        "goal": "Please take care of the customer request that came in this morning and keep things moving.",
        "positive": "The customer asks whether the contractual service-level target can be confirmed in the renewal addendum.",
        "control": "The customer asks whether the current product-status summary can be confirmed in the renewal addendum.",
        "gold": ("notes/atlas-renewal-sla-review", "notes/atlas-renewal-sla-review.md", "For the Atlas renewal addendum, any external confirmation of the contractual service-level target requires Legal review before it is sent."),
        "distractors": [("emails/atlas-renewal-status", "emails/atlas-renewal-status.md"), ("notes/meridian-sla-process", "notes/meridian-sla-process.md"), ("emails/atlas-pricing-followup", "emails/atlas-pricing-followup.md")],
    },
    {
        "slug": "trellis-production-migration-blackout", "kind": "event", "entity": "trellis-migration", "target_id": "event-delivery-session-dec-23",
        "goal": "Please deal with the delivery session on my calendar later this year.",
        "positive": "This session moves the Trellis production environment on December 23.",
        "control": "This session moves the Trellis sandbox environment on December 23.",
        "gold": ("notes/trellis-production-window", "notes/trellis-production-window.md", "Production migrations for Trellis are held during the year-end change window, from December 20 through January 5."),
        "distractors": [("emails/trellis-workshop-prep", "emails/trellis-workshop-prep.md"), ("notes/ashcroft-change-window", "notes/ashcroft-change-window.md"), ("emails/trellis-oncall-roster", "emails/trellis-oncall-roster.md")],
    },
    {
        "slug": "orion-incident-travel-approval", "kind": "email", "entity": "orion-travel", "target_id": "mail-expense-request-march-4",
        "goal": "Please take care of the expense request in my inbox.",
        "positive": "The receipt is for emergency travel while the Orion service was being restored.",
        "control": "The receipt is for travel to an Orion customer planning meeting.",
        "gold": ("notes/orion-incident-travel", "notes/orion-incident-travel.md", "Emergency travel incurred while restoring the Orion service needs incident-manager approval before reimbursement is approved."),
        "distractors": [("emails/orion-incident-update", "emails/orion-incident-update.md"), ("notes/atlas-travel-guidance", "notes/atlas-travel-guidance.md"), ("emails/orion-receipt-reminder", "emails/orion-receipt-reminder.md")],
    },
)


def source_record(source_id: str, path: str, evidence: str | None = None) -> dict[str, Any]:
    file_path = SOURCE / path
    row: dict[str, Any] = {"source_id": source_id, "path": path, "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(), "perturbations": []}
    if evidence:
        row.update({"kind": "dated_personal_history", "evidence_span": {"text": evidence}})
    return row


def fixture(item: dict[str, Any], detail: str) -> dict[str, Any]:
    threads = [
        {"id": item["target_id"] if item["kind"] == "email" else "mail-preparation-notes", "sender": "account.contact@example.test", "recipients": [ACTOR["email"]], "subject": "Atlas renewal addendum" if item["entity"] == "atlas-renewal" else "Orion travel reimbursement", "labels": ["inbox"], "date": "2026-03-04T09:00:00Z", "snippet": "Could you help with this today?", "messages": [{"id": "message-primary", "sender": "account.contact@example.test", "recipients": [ACTOR["email"]], "body": detail}]},
        {"id": "mail-context", "sender": "teammate@example.test", "recipients": [ACTOR["email"]], "subject": "Weekly account check-in", "labels": ["inbox"], "date": "2026-03-03T14:00:00Z", "snippet": "Normal follow-up items for this week.", "messages": [{"id": "message-context", "sender": "teammate@example.test", "recipients": [ACTOR["email"]], "body": "Please bring the open account notes to Thursday's check-in."}]},
    ]
    events = [
        {"id": item["target_id"] if item["kind"] == "event" else "event-context-review", "title": "Trellis migration session" if item["entity"] == "trellis-migration" else "Account review", "start": "2026-12-23T16:00:00Z" if item["entity"] == "trellis-migration" else "2026-03-05T15:00:00Z", "end": "2026-12-23T17:00:00Z" if item["entity"] == "trellis-migration" else "2026-03-05T15:30:00Z", "organizer": ACTOR["email"], "attendee_summary": "Working session", "attendees": [{"email": ACTOR["email"], "role": "required"}, {"email": "delivery@example.test", "role": "required"}], "description": detail if item["entity"] == "trellis-migration" else "Review the regular account action list.", "status": "confirmed"},
        {"id": "event-context-review" if item["entity"] == "trellis-migration" else "event-team-sync", "title": "Delivery team sync", "start": "2026-12-22T15:00:00Z" if item["entity"] == "trellis-migration" else "2026-03-06T15:00:00Z", "end": "2026-12-22T15:30:00Z" if item["entity"] == "trellis-migration" else "2026-03-06T15:30:00Z", "organizer": ACTOR["email"], "attendee_summary": "Internal", "attendees": [{"email": ACTOR["email"], "role": "required"}], "description": "Review routine delivery work.", "status": "confirmed"},
    ]
    if item["entity"] == "trellis-migration":
        threads[0]["id"] = "mail-preparation-notes"
        threads[0]["subject"] = "Trellis preparation notes"
        threads[0]["messages"][0]["body"] = "The deployment checklist is ready for the working session."
    return {"schema_version": 1, "actor": ACTOR, "mailbox": {"threads": threads, "drafts": []}, "calendar": {"events": events}, "contacts": [{"id": "contact-legal", "name": "Legal Review", "email": "legal@example.test", "kind": "queue"}, {"id": "contact-incident", "name": "Incident Manager", "email": "incident-manager@example.test", "kind": "contact"}], "attachments": []}


def details(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if item["kind"] == "email":
        return "mailbox.threads.0.messages.0.body", {"tool": "get_message_thread", "arguments": {"thread_id": item["target_id"]}}
    return "calendar.events.0.description", {"tool": "get_event", "arguments": {"event_id": item["target_id"]}}


def actions(item: dict[str, Any], positive: bool) -> list[dict[str, Any]]:
    """List target-bound operations for timing, without scoring generated prose."""

    target = item["target_id"]
    if item["entity"] == "atlas-renewal":
        return ([{"tool": "set_thread_hold", "arguments": {"thread_id": target, "hold_kind": "review", "target_id": "contact-legal"}}] if positive else [{"tool": "resolve_thread", "arguments": {"thread_id": target}}])
    if item["entity"] == "trellis-migration":
        return ([{"tool": "mark_event_pending_exception", "arguments": {"event_id": target, "exception_type": "change_window"}}] if positive else [{"tool": "get_event", "arguments": {"event_id": target}}])
    return ([{"tool": "set_thread_hold", "arguments": {"thread_id": target, "hold_kind": "approval", "target_id": "contact-incident"}}] if positive else [{"tool": "resolve_thread", "arguments": {"thread_id": target}}])


def assertions(item: dict[str, Any], positive: bool) -> list[dict[str, Any]]:
    """Score target state, not response wording or a prescribed normal action."""

    target = item["target_id"]
    if item["entity"] == "atlas-renewal":
        return [{"id": "legal-review-hold" if positive else "not-legal-review-hold", "role": "decisive", "kind": "thread_hold_state" if positive else "thread_not_on_hold", "thread_id": target, "hold_kind": "review", "target_id": "contact-legal", "rationale": "The Atlas note requires Legal review before any external confirmation of the contractual service-level target."}]
    if item["entity"] == "trellis-migration":
        return [{"id": "change-window-exception" if positive else "not-change-window-exception", "role": "decisive", "kind": "event_pending_exception" if positive else "event_not_pending_exception", "event_id": target, "exception_type": "change_window", "rationale": "The Trellis note holds production migrations during the year-end change window, so the target session is held pending a change-window exception."}]
    return [{"id": "incident-manager-approval-hold" if positive else "not-incident-manager-approval-hold", "role": "decisive", "kind": "thread_hold_state" if positive else "thread_not_on_hold", "thread_id": target, "hold_kind": "approval", "target_id": "contact-incident", "rationale": "The Orion note requires incident-manager approval before emergency restoration travel reimbursement is approved."}]


def case(item: dict[str, Any], variant: str) -> dict[str, Any]:
    positive = variant != "cue-ablated"
    source_id, source_path, evidence = item["gold"]
    gold_file = (SOURCE / source_path).read_text(encoding="utf-8")
    cue_path, cue_location = details(item)
    goal = item["goal"] if variant != "memory-included" else f"{item['goal']}\n\nRelevant prior note:\n{gold_file}"
    target = {"thread_id": item["target_id"]} if item["kind"] == "email" else {"event_id": item["target_id"]}
    return {"case_id": f"parm-email-calendar-v2-{item['slug']}-{variant}", "base_case_id": f"parm-email-calendar-v2-{item['slug']}", "corpus_id": CORPUS_ID, "corpus_tier": "pilot-30", "step_budget": 12, "variant": variant, "goal": goal, "target": target, "environment": {"adapter": "email_calendar_fixture", "fixture_path": f"fixtures/{item['slug']}/{'positive' if positive else 'cue-ablated'}.json", "upstream": UPSTREAM}, "cue": {"present": positive, "text": item["positive"], "location": cue_location, "description": "One applicability fact appears only in the selected full detail."}, "memory": {"corpus_id": CORPUS_ID, "text": gold_file, "gold_source_ids": [source_id], "sources": [source_record(source_id, source_path, evidence)]}, "distractors": {"sources": [source_record(*entry) for entry in item["distractors"]]}, "decisive_action": {"matchers": actions(item, positive)}, "expects_intervention": positive, "assertions": assertions(item, positive), "ablation": {"fixture_path": cue_path, "positive_value": item["positive"], "control_value": item["control"], "description": "Only this late semantic applicability proposition varies across the pair."}, "provenance": {"persona_id": "ops-lead-email-calendar", "evaluation_split": "email_calendar_v2_natural", "constructed_by": "scripts/build_workflows_email_calendar_v2_cases.py", "upstream": UPSTREAM}}


def main() -> None:
    rows: list[dict[str, Any]] = []
    for item in SCENARIOS:
        directory = DATASET / "fixtures" / item["slug"]
        directory.mkdir(parents=True, exist_ok=True)
        for filename, detail in (("positive.json", item["positive"]), ("cue-ablated.json", item["control"])):
            (directory / filename).write_text(json.dumps(fixture(item, detail), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        rows.extend(case(item, variant) for variant in ("positive", "cue-ablated", "memory-included"))
    DATASET.mkdir(parents=True, exist_ok=True)
    (DATASET / "cases.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n")
    manifest = {"schema_version": 1, "dataset_id": "parmbench-workflows-email-calendar-v2", "validation_profile": "workflows_email_calendar_v2_natural", "corpora": [{"corpus_id": CORPUS_ID, "source_root": f"corpora/{CORPUS_ID}/source", "source_id_prefixes": ["notes/", "emails/"]}], "environments": [{"adapter": "email_calendar_fixture", "upstream": UPSTREAM}]}
    (DATASET / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} email/calendar v2 workflow cases")


if __name__ == "__main__":
    main()
