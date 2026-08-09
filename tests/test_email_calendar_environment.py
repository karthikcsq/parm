from __future__ import annotations

import copy
import unittest

from parm_bench.workflows import get_environment
from parm_bench.workflows.verify import evaluate_assertions


FIXTURE = {
    "schema_version": 1,
    "actor": {"id": "alex", "name": "Alex", "email": "alex@example.test"},
    "mailbox": {
        "threads": [{
            "id": "t-1", "sender": "pat@example.test", "recipients": ["alex@example.test"],
            "subject": "Launch review", "labels": ["inbox"], "date": "2026-08-10T09:00:00Z",
            "messages": [{"id": "m-1", "sender": "pat@example.test", "recipients": ["alex@example.test"], "body": "Can we confirm the launch review?"}],
        }],
        "drafts": [],
    },
    "calendar": {"events": [{
        "id": "e-1", "title": "Launch review", "start": "2026-08-11T10:00:00Z", "end": "2026-08-11T10:30:00Z",
        "attendees": [{"email": "alex@example.test"}, {"email": "pat@example.test"}],
        "description": "Review launch readiness.", "status": "confirmed", "comments": [],
    }]},
    "contacts": [
        {"id": "queue-legal", "name": "Legal review", "email": "legal@example.test", "kind": "queue"},
        {"id": "contact-maya", "name": "Maya", "email": "maya@example.test", "kind": "contact"},
    ],
    "attachments": [],
}


class EmailCalendarFixtureEnvironmentTest(unittest.TestCase):
    def _environment(self):
        return get_environment("email_calendar_fixture", copy.deepcopy(FIXTURE))

    def test_schema_exposes_canonical_email_and_calendar_action_surface(self) -> None:
        specs = {tool.name: tool for tool in self._environment().tools()}
        self.assertEqual(specs["assign_follow_up"].parameters["required"], ["thread_id", "owner"])
        self.assertIn("owner", specs["assign_follow_up"].parameters["properties"])
        self.assertNotIn("assignee", specs["assign_follow_up"].parameters["properties"])
        self.assertTrue(specs["send_reply"].mutating)

    def test_read_tools_render_nested_fixture_details_without_mutating(self) -> None:
        environment = self._environment()
        self.assertIn("Launch review", environment.invoke("list_messages", {"folder": "inbox"}).text)
        self.assertIn("Can we confirm", environment.invoke("get_message_thread", {"thread_id": "t-1"}).text)
        self.assertIn("Review launch readiness", environment.invoke("get_event", {"event_id": "e-1"}).text)
        self.assertEqual(environment.mutations(), ())

    def test_reset_restores_nested_fixture_without_aliasing_input(self) -> None:
        fixture = copy.deepcopy(FIXTURE)
        environment = get_environment("email_calendar_fixture", fixture)
        environment.invoke("send_reply", {"thread_id": "t-1", "body": "Confirmed."})
        fixture["mailbox"]["threads"][0]["messages"][0]["body"] = "externally modified"
        environment.reset()
        self.assertEqual(environment.state()["messages"]["m-1"]["body"], "Can we confirm the launch review?")
        self.assertEqual(environment.mutations(), ())

    def test_routes_follow_up_to_a_resolved_contact_or_queue(self) -> None:
        environment = self._environment()

        targets = environment.invoke("list_routing_targets", {})
        routed = environment.invoke(
            "route_follow_up", {"thread_id": "t-1", "target_id": "queue-legal"}
        )

        self.assertTrue(targets.ok, targets.text)
        self.assertIn("queue-legal", targets.text)
        self.assertTrue(routed.ok, routed.text)
        self.assertEqual(environment.state()["follow_ups"]["t-1"], {
            "target_id": "queue-legal",
            "target_name": "Legal review",
            "target_email": "legal@example.test",
            "target_kind": "queue",
        })
        self.assertEqual(environment.mutations()[-1]["kind"], "route_follow_up")
        self.assertFalse(
            environment.invoke(
                "route_follow_up", {"thread_id": "t-1", "target_id": "not-a-target"}
            ).ok
        )

    def test_canonical_mutations_preserve_audit_state(self) -> None:
        environment = self._environment()
        self.assertTrue(environment.invoke("send_reply", {"thread_id": "t-1", "body": "Confirmed."}).ok)
        self.assertTrue(environment.invoke("assign_follow_up", {"thread_id": "t-1", "owner": "maya"}).ok)
        state = environment.state()
        self.assertEqual(state["follow_ups"]["t-1"]["owner"], "maya")
        self.assertEqual([row["kind"] for row in environment.mutations()], ["send_reply", "assign_follow_up"])


class EmailCalendarAssertionTest(unittest.TestCase):
    def test_rescheduling_is_reversible_and_scores_time_with_approval(self) -> None:
        environment = get_environment("email_calendar_fixture", copy.deepcopy(FIXTURE))
        result = environment.invoke(
            "reschedule_event",
            {
                "event_id": "e-1",
                "start": "2026-08-12T10:00:00Z",
                "end": "2026-08-12T10:30:00Z",
                "approval_reference": "approval-42",
            },
        )

        self.assertTrue(result.ok, result.text)
        self.assertEqual(environment.state()["events"]["e-1"]["status"], "confirmed")
        rows = evaluate_assertions(
            [{
                "id": "rescheduled", "role": "decisive", "kind": "event_rescheduled",
                "event_id": "e-1", "start": "2026-08-12T10:00:00Z",
                "end": "2026-08-12T10:30:00Z", "approval_reference": "approval-42",
            }],
            state=environment.state(), mutations=environment.mutations(), trajectory=environment.trajectory,
        )
        self.assertTrue(rows[0]["passed"], rows)

    def test_draft_content_assertion_requires_a_non_confirming_handoff(self) -> None:
        environment = get_environment("email_calendar_fixture", copy.deepcopy(FIXTURE))
        self.assertTrue(environment.invoke(
            "create_draft_reply",
            {"thread_id": "t-1", "body": "We have sent this to legal for review and cannot confirm terms yet."},
        ).ok)

        rows = evaluate_assertions(
            [{
                "id": "legal-handoff", "role": "decisive", "kind": "thread_message_content",
                "thread_id": "t-1", "folder": "drafts", "includes": ["legal", "review"],
                "excludes": ["confirmed", "we confirm"],
            }],
            state=environment.state(), mutations=environment.mutations(), trajectory=environment.trajectory,
        )
        self.assertTrue(rows[0]["passed"], rows)

    def test_public_assertions_score_message_owner_response_and_attendees(self) -> None:
        environment = get_environment("email_calendar_fixture", copy.deepcopy(FIXTURE))
        for name, arguments in [
            ("send_reply", {"thread_id": "t-1", "body": "Confirmed with Pat"}),
            ("assign_follow_up", {"thread_id": "t-1", "owner": "maya"}),
            ("update_event", {"event_id": "e-1", "attendees": ["maya@example.test"]}),
            ("decline_event", {"event_id": "e-1"}),
        ]:
            self.assertTrue(environment.invoke(name, arguments).ok)
        assertions = [
            {"id": "sent", "role": "decisive", "kind": "message_sent", "thread_id": "t-1", "body_keywords": ["confirmed"]},
            {"id": "owner", "role": "decisive", "kind": "follow_up_assigned", "thread_id": "t-1", "owner": "maya"},
            {"id": "response", "role": "decisive", "kind": "event_response", "event_id": "e-1", "expected": "declined"},
            {"id": "attendees", "role": "decisive", "kind": "event_attendees", "event_id": "e-1", "includes": ["maya@example.test"], "excludes": ["pat@example.test"]},
        ]
        rows = evaluate_assertions(assertions, state=environment.state(), mutations=environment.mutations(), trajectory=environment.trajectory)
        self.assertTrue(all(row["passed"] for row in rows), rows)


if __name__ == "__main__":
    unittest.main()
