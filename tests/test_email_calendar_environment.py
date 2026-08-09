from __future__ import annotations

import copy
import unittest

from parm_bench.workflows import get_environment
from parm_bench.workflows.verify import evaluate_assertions


FIXTURE = {
    "account": "alex@example.test",
    "messages": [
        {
            "id": "m-1",
            "thread_id": "t-1",
            "from": "pat@example.test",
            "to": ["alex@example.test"],
            "subject": "Launch review",
            "body": "Can we confirm the launch review?",
            "folder": "inbox",
            "sent_at": "2026-08-10T09:00:00Z",
        }
    ],
    "events": [
        {
            "id": "e-1",
            "title": "Launch review",
            "start": "2026-08-11T10:00:00Z",
            "end": "2026-08-11T10:30:00Z",
            "attendees": ["alex@example.test", "pat@example.test"],
            "description": "Review launch readiness.",
            "status": "confirmed",
            "comments": [],
        }
    ],
}


class EmailCalendarFixtureEnvironmentTest(unittest.TestCase):
    def _environment(self):
        return get_environment("email_calendar_fixture", copy.deepcopy(FIXTURE))

    def test_schema_exposes_only_the_email_and_calendar_tool_surface(self) -> None:
        environment = self._environment()
        specs = {tool.name: tool for tool in environment.tools()}
        self.assertEqual(
            set(specs),
            {
                "list_messages",
                "get_message_thread",
                "list_events",
                "get_event",
                "create_draft_reply",
                "send_reply",
                "assign_follow_up",
                "create_event",
                "update_event",
                "decline_event",
                "add_event_comment",
            },
        )
        self.assertFalse(specs["list_messages"].mutating)
        self.assertTrue(specs["send_reply"].mutating)
        self.assertEqual(specs["create_draft_reply"].parameters["required"], ["thread_id", "body"])
        self.assertEqual(specs["update_event"].parameters["required"], ["event_id"])

    def test_read_tools_render_details_without_mutating(self) -> None:
        environment = self._environment()
        self.assertIn("Launch review", environment.invoke("list_messages", {"folder": "inbox"}).text)
        self.assertIn("Can we confirm", environment.invoke("get_message_thread", {"thread_id": "t-1"}).text)
        self.assertIn("Launch review", environment.invoke("list_events", {}).text)
        self.assertIn("Review launch readiness", environment.invoke("get_event", {"event_id": "e-1"}).text)
        self.assertEqual(environment.mutations(), ())
        self.assertEqual(len(environment.trajectory), 4)

    def test_deep_reset_restores_fixture_and_does_not_alias_input(self) -> None:
        fixture = copy.deepcopy(FIXTURE)
        environment = get_environment("email_calendar_fixture", fixture)
        environment.invoke("send_reply", {"thread_id": "t-1", "body": "Confirmed."})
        environment.invoke("update_event", {"event_id": "e-1", "title": "Updated review"})
        fixture["messages"][0]["body"] = "externally modified"
        environment.reset()
        state = environment.state()
        self.assertEqual(len(state["messages"]), 1)
        self.assertEqual(state["messages"]["m-1"]["body"], "Can we confirm the launch review?")
        self.assertEqual(state["events"]["e-1"]["title"], "Launch review")
        self.assertEqual(environment.mutations(), ())
        self.assertEqual(environment.trajectory, ())

    def test_message_mutations_preserve_audit_state(self) -> None:
        environment = self._environment()
        self.assertTrue(environment.invoke("create_draft_reply", {"thread_id": "t-1", "body": "Draft response"}).ok)
        self.assertTrue(environment.invoke("send_reply", {"thread_id": "t-1", "body": "Confirmed."}).ok)
        self.assertTrue(environment.invoke("assign_follow_up", {"thread_id": "t-1", "assignee": "maya", "due_date": "2026-08-12"}).ok)
        state = environment.state()
        self.assertEqual([message["folder"] for message in state["messages"].values()], ["inbox", "drafts", "sent"])
        self.assertEqual(state["follow_ups"]["t-1"]["assignee"], "maya")
        self.assertEqual([row["kind"] for row in environment.mutations()], ["create_draft_reply", "send_reply", "assign_follow_up"])

    def test_event_mutations_capture_lifecycle_and_reject_unknown_event_without_mutation(self) -> None:
        environment = self._environment()
        created = environment.invoke(
            "create_event",
            {"title": "Decision", "start": "2026-08-12T12:00:00Z", "end": "2026-08-12T12:30:00Z", "attendees": ["pat@example.test"]},
        )
        self.assertTrue(created.ok)
        event_id = environment.mutations()[0]["event_id"]
        self.assertTrue(environment.invoke("update_event", {"event_id": event_id, "description": "Final decision."}).ok)
        self.assertTrue(environment.invoke("add_event_comment", {"event_id": event_id, "body": "Bring metrics."}).ok)
        self.assertTrue(environment.invoke("decline_event", {"event_id": event_id, "comment": "Scheduling conflict"}).ok)
        self.assertEqual(environment.state()["events"][event_id]["status"], "declined")
        before = environment.mutations()
        self.assertFalse(environment.invoke("update_event", {"event_id": "missing", "title": "Nope"}).ok)
        self.assertEqual(environment.mutations(), before)


class EmailCalendarAssertionTest(unittest.TestCase):
    def _rows_after(self, calls):
        environment = get_environment("email_calendar_fixture", copy.deepcopy(FIXTURE))
        for name, arguments in calls:
            self.assertTrue(environment.invoke(name, arguments).ok)
        assertions = [
            {"id": "draft", "role": "workflow", "kind": "draft_reply_exists", "thread_id": "t-1", "body_keywords": ["draft"]},
            {"id": "sent", "role": "decisive", "kind": "sent_reply_exists", "thread_id": "t-1", "body_keywords": ["confirmed"]},
            {"id": "followup", "role": "workflow", "kind": "follow_up_assigned", "thread_id": "t-1", "assignee": "maya", "due_date": "2026-08-12"},
            {"id": "event", "role": "decisive", "kind": "event_matches", "event_id": "e-1", "title_keywords": ["rescheduled"], "status": "declined", "comment_keywords": ["conflict"]},
        ]
        return {row["id"]: row for row in evaluate_assertions(assertions, state=environment.state(), mutations=environment.mutations(), trajectory=environment.trajectory)}

    def test_reply_and_follow_up_assertions_score_deterministically(self) -> None:
        rows = self._rows_after([
            ("create_draft_reply", {"thread_id": "t-1", "body": "Draft for Maya"}),
            ("send_reply", {"thread_id": "t-1", "body": "Confirmed with Pat"}),
            ("assign_follow_up", {"thread_id": "t-1", "assignee": "maya", "due_date": "2026-08-12"}),
        ])
        self.assertTrue(rows["draft"]["passed"])
        self.assertTrue(rows["sent"]["passed"])
        self.assertTrue(rows["followup"]["passed"])
        self.assertFalse(rows["event"]["passed"])

    def test_event_assertion_scores_update_decline_and_comment(self) -> None:
        rows = self._rows_after([
            ("update_event", {"event_id": "e-1", "title": "Rescheduled launch review"}),
            ("add_event_comment", {"event_id": "e-1", "body": "Conflict with customer call"}),
            ("decline_event", {"event_id": "e-1"}),
        ])
        self.assertFalse(rows["draft"]["passed"])
        self.assertFalse(rows["sent"]["passed"])
        self.assertFalse(rows["followup"]["passed"])
        self.assertTrue(rows["event"]["passed"])


if __name__ == "__main__":
    unittest.main()
