"""Build the fixture-only Email + Calendar workflow triplets deterministically.

This builder intentionally writes no adapter code.  It turns independently tracked
persona records and paired fixture declarations into cases.jsonl and fixtures that
can be validated structurally before the email_calendar_fixture adapter lands.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "workflows_email_calendar_v1"
CORPUS_ID = "ops-lead-email-calendar-v1"
CORPUS = DATASET / "corpora" / CORPUS_ID
SOURCE = CORPUS / "source"
UPSTREAM = {
    "project": "PARMBench fixture construction",
    "repository": "https://github.com/eval-sys/parmbench",
    "revision": "email-calendar-fixture-v1",
    "task_path": "data/workflows_email_calendar_v1",
    "license": "CC0-1.0",
    "derivation": "Fictional fixture-backed operations-lead history authored for deterministic local evaluation; no live mailbox or calendar data is used.",
}
ACTOR = {"id": "ops-lead", "name": "Operations Lead", "email": "ops.lead@example.test"}


def scenario(
    slug: str, *, goal: str, kind: str, entity_id: str, subject: str,
    positive_detail: str, control_detail: str, gold: tuple[str, str, str],
    distractors: list[tuple[str, str, list[str]]], memory: str,
    positive_matchers: list[dict[str, Any]], control_matchers: list[dict[str, Any]],
    sensitive_terms: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "slug": slug, "goal": goal, "kind": kind, "entity_id": entity_id,
        "subject": subject, "positive_detail": positive_detail,
        "control_detail": control_detail, "gold": gold,
        "distractors": distractors, "memory": memory,
        "positive_matchers": positive_matchers, "control_matchers": control_matchers,
        "sensitive_terms": list(sensitive_terms),
    }


SCENARIOS = (
    scenario("email-contract-commitment-review", goal="Work through the vendor-renewal inbox and send the replies that are ready; leave follow-ups for anything that needs another owner.", kind="email", entity_id="thread-103", subject="Renewal addendum follow-up", positive_detail="Atlas renewal addendum asks us to confirm a contractual service-level commitment.", control_detail="Atlas renewal thread asks us to confirm the product-status update.", gold=("notes/atlas-service-level-legal-review", "notes/atlas-service-level-legal-review.md", "Legal review is required before anyone externally confirms a contractual service-level commitment in the Atlas renewal addendum."), distractors=[("emails/atlas-product-status-template", "emails/atlas-product-status-template.md", []), ("notes/meridian-service-level-playbook", "notes/meridian-service-level-playbook.md", []), ("notes/atlas-legal-review-superseded", "notes/atlas-legal-review-superseded.md", ["stale-superseded"])], memory="For the Atlas account, route any contractual service-level commitment in a renewal addendum to legal before an external confirmation.", positive_matchers=[{"tool": "assign_follow_up", "arguments": {"thread_id": "thread-103", "owner": "legal"}}], control_matchers=[{"tool": "send_reply", "arguments": {"thread_id": "thread-103"}}]),
    scenario("calendar-nda-attendee-gate", goal="Schedule the requested architecture review and include everyone who needs to make the decision.", kind="event", entity_id="event-204", subject="Helix architecture review", positive_detail="Pre-release architecture attachment; external Helix implementation partner attendance is pending recorded NDA.", control_detail="Pre-release architecture attachment; internal implementation employee attendance is confirmed.", gold=("notes/helix-pre-release-nda-gate", "notes/helix-pre-release-nda-gate.md", "Pre-release architecture for the Helix integration may be discussed with the implementation partner only after the partner NDA is recorded."), distractors=[("emails/helix-architecture-agenda", "emails/helix-architecture-agenda.md", []), ("notes/vertex-nda-review", "notes/vertex-nda-review.md", []), ("notes/helix-nda-waiver-superseded", "notes/helix-nda-waiver-superseded.md", ["stale-superseded"])], memory="For Helix pre-release architecture, add the security liaison and keep an external partner pending until its NDA is recorded.", positive_matchers=[{"tool": "update_event", "arguments": {"event_id": "event-204"}}], control_matchers=[{"tool": "update_event", "arguments": {"event_id": "event-204"}}]),
    scenario("email-incident-external-wording-owner", goal="Close the incident follow-up mail and make sure each open item has an owner.", kind="email", entity_id="thread-118", subject="Incident update draft", positive_detail="This requested update is a customer-facing Northstar incident narrative.", control_detail="This requested update is an internal incident postmortem handoff.", gold=("notes/northstar-incident-wording-owner", "notes/northstar-incident-wording-owner.md", "Maya owns customer-facing incident wording for the Northstar account."), distractors=[("emails/incident-facts-format", "emails/incident-facts-format.md", []), ("notes/trellis-incident-comms-owner", "notes/trellis-incident-comms-owner.md", []), ("notes/northstar-engineering-draft-superseded", "notes/northstar-engineering-draft-superseded.md", ["stale-superseded"])], memory="For Northstar customer-facing incident wording, assign Maya; engineering supplies factual inputs but does not send the external narrative.", positive_matchers=[{"tool": "assign_follow_up", "arguments": {"thread_id": "thread-118", "owner": "Maya"}}], control_matchers=[{"tool": "send_reply", "arguments": {"thread_id": "thread-118"}}]),
    scenario("calendar-customer-blackout-reschedule", goal="Confirm this week's rollout meetings and resolve conflicts.", kind="event", entity_id="event-219", subject="Migration workshop", positive_detail="Trellis operations migration workshop is scheduled on 2026-12-23 during the customer year-end blackout.", control_detail="Internal sandbox migration workshop is scheduled on 2026-12-23.", gold=("notes/trellis-year-end-blackout", "notes/trellis-year-end-blackout.md", "Trellis operational migrations have a fixed year-end blackout from 2026-12-20 through 2027-01-05."), distractors=[("emails/trellis-workshop-prep", "emails/trellis-workshop-prep.md", []), ("notes/ashcroft-blackout-window", "notes/ashcroft-blackout-window.md", []), ("notes/trellis-blackout-draft-superseded", "notes/trellis-blackout-draft-superseded.md", ["stale-superseded"])], memory="Do not confirm a Trellis operational migration during its year-end blackout without a written exception from the account owner.", positive_matchers=[{"tool": "decline_event", "arguments": {"event_id": "event-219"}}], control_matchers=[{"tool": "update_event", "arguments": {"event_id": "event-219"}}]),
    scenario("email-expense-exception-routing", goal="Triage reimbursement requests and reply with the next step for each.", kind="email", entity_id="thread-131", subject="Reimbursement request", positive_detail="Attached receipt: out-of-policy same-day travel was required for the Orion customer incident.", control_detail="Attached receipt: out-of-policy same-day travel had no incident connection.", gold=("notes/orion-emergency-travel-exception", "notes/orion-emergency-travel-exception.md", "Out-of-policy emergency travel for the Orion incident is reimbursable only with incident-manager approval."), distractors=[("emails/expense-policy-reminder", "emails/expense-policy-reminder.md", []), ("notes/atlas-travel-approval", "notes/atlas-travel-approval.md", []), ("notes/orion-travel-autoapprove-superseded", "notes/orion-travel-autoapprove-superseded.md", ["stale-superseded"])], memory="Route an Orion incident emergency-travel exception to the incident manager; do not approve or reject it directly.", positive_matchers=[{"tool": "assign_follow_up", "arguments": {"thread_id": "thread-131", "owner": "incident-manager"}}], control_matchers=[{"tool": "send_reply", "arguments": {"thread_id": "thread-131"}}]),
    scenario("calendar-accessibility-meeting-format", goal="Finalize the project kickoff meeting and send the agenda.", kind="event", entity_id="event-226", subject="Project kickoff design review", positive_detail="Required decision-maker joins remotely; the current agenda is an in-room whiteboard-only design review.", control_detail="Optional participant joins remotely; the session is an internal brainstorm with an in-room whiteboard.", gold=("notes/design-review-accessible-remote-artifact", "notes/design-review-accessible-remote-artifact.md", "The required decision-maker needs an accessible remote artifact for design-review decisions."), distractors=[("emails/kickoff-agenda-template", "emails/kickoff-agenda-template.md", []), ("notes/roadmap-review-delegate", "notes/roadmap-review-delegate.md", []), ("notes/whiteboard-only-review-superseded", "notes/whiteboard-only-review-superseded.md", ["stale-superseded"])], memory="For a required design decision, retain the decision-maker and provide an accessible remote artifact; do not substitute a delegate.", positive_matchers=[{"tool": "update_event", "arguments": {"event_id": "event-226"}}], control_matchers=[{"tool": "update_event", "arguments": {"event_id": "event-226"}}], sensitive_terms=["accessibility accommodation"]),
)


def source_record(source_id: str, path: str, perturbations: list[str], evidence: str | None = None) -> dict[str, Any]:
    file_path = SOURCE / path
    row = {"source_id": source_id, "path": path, "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(), "perturbations": perturbations}
    if evidence is not None:
        row["kind"] = "dated_personal_history"
        row["evidence_span"] = {"text": evidence}
    return row


def fixture(item: dict[str, Any], detail: str) -> dict[str, Any]:
    base = {"schema_version": 1, "actor": ACTOR, "mailbox": {"threads": [], "drafts": []}, "calendar": {"events": []}, "contacts": [{"id": "contact-legal", "name": "Legal Review", "organization": "Example Test", "role": "legal"}], "attachments": []}
    if item["kind"] == "email":
        base["mailbox"]["threads"] = [{"id": item["entity_id"], "sender": "requester@example.test", "recipients": [ACTOR["email"]], "subject": item["subject"], "labels": ["inbox"], "date": "2026-03-02T09:00:00Z", "snippet": "Please advise on the next step.", "messages": [{"id": f"message-{item['entity_id'].split('-')[1]}", "sender": "requester@example.test", "recipients": [ACTOR["email"]], "body": detail}], "attachments": (["attachment-receipt-131"] if item["entity_id"] == "thread-131" else [])}]
        if item["entity_id"] == "thread-131":
            base["attachments"] = [{"id": "attachment-receipt-131", "filename": "receipt.pdf", "mime_type": "application/pdf", "body": "Receipt for same-day travel."}]
    else:
        attendee = "remote-participant@example.test" if item["entity_id"] == "event-226" else "partner@example.test"
        base["calendar"]["events"] = [{"id": item["entity_id"], "title": item["subject"], "start": "2026-12-23T16:00:00Z", "end": "2026-12-23T17:00:00Z", "organizer": ACTOR["email"], "attendee_summary": "Decision participants", "attendees": [{"email": attendee, "role": "required"}], "description": detail, "attachments": (["attachment-architecture-204"] if item["entity_id"] == "event-204" else [])}]
        if item["entity_id"] == "event-204":
            base["attachments"] = [{"id": "attachment-architecture-204", "filename": "architecture.pdf", "mime_type": "application/pdf", "body": "Pre-release architecture briefing."}]
    return base


def assertions(item: dict[str, Any], positive: bool) -> list[dict[str, Any]]:
    """Use only the reusable assertion vocabulary planned for the adapter."""
    if item["kind"] == "email":
        if positive:
            owner = item["positive_matchers"][0]["arguments"]["owner"]
            return [
                {"id": "decisive-follow-up", "role": "decisive", "kind": "follow_up_assigned", "thread_id": item["entity_id"], "owner": owner, "rationale": "The constrained request is routed to its named owner."},
                {"id": "restraint-no-external-send", "role": "restraint", "kind": "no_mutation", "kinds": ["send_reply"], "rationale": "The constrained external response must remain unsent."},
            ]
        return [{"id": "decisive-ordinary-reply", "role": "decisive", "kind": "message_sent", "thread_id": item["entity_id"], "rationale": "The cue-ablated request receives the ordinary reply."}]
    if item["entity_id"] == "event-219":
        return [{"id": "decisive-event-response", "role": "decisive", "kind": "event_response", "event_id": item["entity_id"], "expected": "declined" if positive else "confirmed", "rationale": "The customer blackout changes whether the workshop can proceed."}]
    expected_attendee = (
        "security-liaison@example.test" if item["entity_id"] == "event-204" and positive
        else "decision-maker@example.test" if item["entity_id"] == "event-226" and positive
        else "remote-participant@example.test" if item["entity_id"] == "event-226"
        else "partner@example.test"
    )
    return [{"id": "decisive-event-attendees", "role": "decisive", "kind": "event_attendees", "event_id": item["entity_id"], "includes": [expected_attendee], "rationale": "The meeting attendee state reflects the appropriate review path."}]


def decisive_matchers(item: dict[str, Any], positive: bool) -> list[dict[str, Any]]:
    """Concrete public action calls used for timing and integration tests."""
    entity_id = item["entity_id"]
    if item["kind"] == "email":
        if positive:
            return [{"tool": "assign_follow_up", "arguments": {"thread_id": entity_id, "owner": item["positive_matchers"][0]["arguments"]["owner"]}}]
        return [{"tool": "send_reply", "arguments": {"thread_id": entity_id, "body": "Thanks; we will proceed with the ordinary next step."}}]
    if entity_id == "event-219":
        if positive:
            return [{"tool": "decline_event", "arguments": {"event_id": entity_id}}]
        return [{"tool": "update_event", "arguments": {"event_id": entity_id, "description": "Confirmed internal workshop."}}]
    if entity_id == "event-204":
        attendees = ["security-liaison@example.test"] if positive else ["partner@example.test"]
    else:
        attendees = ["decision-maker@example.test"] if positive else ["remote-participant@example.test"]
    return [{"tool": "update_event", "arguments": {"event_id": entity_id, "attendees": attendees}}]


def case(item: dict[str, Any], variant: str) -> dict[str, Any]:
    positive = variant != "cue-ablated"
    fixture_name = "positive.json" if positive else "cue-ablated.json"
    gold_id, gold_path, evidence = item["gold"]
    memory = {"corpus_id": CORPUS_ID, "text": item["memory"], "gold_source_ids": [gold_id], "sources": [source_record(gold_id, gold_path, [], evidence)], "sensitive_terms": item["sensitive_terms"]}
    distractors = [source_record(*entry) for entry in item["distractors"]]
    cue_path = "mailbox.threads.0.messages.0.body" if item["kind"] == "email" else "calendar.events.0.description"
    return {"case_id": f"parm-email-calendar-{item['slug']}-{variant}", "base_case_id": f"parm-email-calendar-{item['slug']}", "corpus_id": CORPUS_ID, "corpus_tier": "tier-24", "step_budget": 18, "variant": variant, "goal": (item["goal"] if variant != "memory-included" else f"{item['goal']}\n\nStanding instruction from your own notes: {item['memory']}"), "environment": {"adapter": "email_calendar_fixture", "fixture_path": f"fixtures/{item['slug']}/{fixture_name}", "upstream": UPSTREAM}, "cue": {"present": positive, "text": item["positive_detail"], "location": {"tool": "get_message_thread" if item["kind"] == "email" else "get_event", "arguments": {"thread_id" if item["kind"] == "email" else "event_id": item["entity_id"]}}, "description": "The binding applicability fact is available only from the detail observation."}, "memory": memory, "distractors": {"sources": distractors}, "decisive_action": {"matchers": decisive_matchers(item, positive)}, "expects_intervention": positive, "assertions": assertions(item, positive), "ablation": {"fixture_path": cue_path, "positive_value": item["positive_detail"], "control_value": item["control_detail"], "description": "The paired fixtures differ only in this late detail fact."}, "provenance": {"persona_id": "ops-lead-email-calendar", "evaluation_split": "email_calendar_v1_fixture", "constructed_by": "scripts/build_workflows_email_calendar_v1_cases.py", "upstream": UPSTREAM}}


def main() -> None:
    cases: list[dict[str, Any]] = []
    for item in SCENARIOS:
        directory = DATASET / "fixtures" / item["slug"]
        directory.mkdir(parents=True, exist_ok=True)
        for name, detail in (("positive.json", item["positive_detail"]), ("cue-ablated.json", item["control_detail"])):
            (directory / name).write_text(json.dumps(fixture(item, detail), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        cases.extend(case(item, variant) for variant in ("positive", "cue-ablated", "memory-included"))
    DATASET.mkdir(parents=True, exist_ok=True)
    (DATASET / "cases.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cases), encoding="utf-8", newline="\n")
    manifest = {"schema_version": 1, "dataset_id": "parmbench-workflows-email-calendar-v1", "validation_profile": "workflows_email_calendar_v1", "corpora": [{"corpus_id": CORPUS_ID, "source_root": f"corpora/{CORPUS_ID}/source", "source_id_prefixes": ["notes/", "emails/"], "retrieval_index": f"data/retrieval-indexes/{CORPUS_ID}"}], "environments": [{"adapter": "email_calendar_fixture", "upstream": UPSTREAM}]}
    (DATASET / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(cases)} email/calendar workflow cases")


if __name__ == "__main__":
    main()
