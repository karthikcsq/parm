from __future__ import annotations

import copy
from typing import Any

from .environment import (
    ToolInvocationError,
    ToolResult,
    ToolSpec,
    TrajectoryStep,
    _TrajectoryRecorder,
    register_environment,
)


ADAPTER_NAME = "email_calendar_fixture"
ADAPTER_VERSION = "email_calendar_fixture_v1"
_STRING = {"type": "string"}
_STRING_ARRAY = {"type": "array", "items": _STRING}


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("list_messages", "List messages, optionally filtered by mailbox folder.", _object({"folder": {"type": "string", "enum": ["inbox", "drafts", "sent", "all"]}}, [])),
    ToolSpec("get_message_thread", "Read every message in one email thread.", _object({"thread_id": _STRING}, ["thread_id"])),
    ToolSpec("list_events", "List calendar events.", _object({}, [])),
    ToolSpec("get_event", "Read one calendar event including attendees and comments.", _object({"event_id": _STRING}, ["event_id"])),
    ToolSpec("create_draft_reply", "Create a draft reply in an existing thread.", _object({"thread_id": _STRING, "body": _STRING}, ["thread_id", "body"]), mutating=True),
    ToolSpec("send_reply", "Send a reply in an existing thread.", _object({"thread_id": _STRING, "body": _STRING}, ["thread_id", "body"]), mutating=True),
    ToolSpec("assign_follow_up", "Assign a follow-up for an email thread.", _object({"thread_id": _STRING, "owner": _STRING, "due_date": _STRING}, ["thread_id", "owner"]), mutating=True),
    ToolSpec("create_event", "Create a calendar event.", _object({"title": _STRING, "start": _STRING, "end": _STRING, "attendees": _STRING_ARRAY, "description": _STRING}, ["title", "start", "end", "attendees"]), mutating=True),
    ToolSpec("update_event", "Update selected fields on an existing calendar event.", _object({"event_id": _STRING, "title": _STRING, "start": _STRING, "end": _STRING, "attendees": _STRING_ARRAY, "description": _STRING}, ["event_id"]), mutating=True),
    ToolSpec("decline_event", "Decline an existing calendar event, optionally with a comment.", _object({"event_id": _STRING, "comment": _STRING}, ["event_id"]), mutating=True),
    ToolSpec("add_event_comment", "Add a comment to an existing calendar event.", _object({"event_id": _STRING, "body": _STRING}, ["event_id", "body"]), mutating=True),
)
_TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


class EmailCalendarFixtureEnvironment:
    """Deterministic in-process email and calendar fixture environment.

    Public fixture schema v1 is nested: ``actor``, ``mailbox.threads``, and
    ``calendar.events``.  Messages use ``sender``/``recipients`` and events
    may use attendee objects with an ``email``. The action API names follow-up
    responsibility ``owner``; verifier kinds are documented in
    ``docs/email-calendar-fixture-contract.md``.

    The seed fixture is copied before normalization and ``reset`` rebuilds all
    mutable state from that private snapshot. This permits independent workflow
    variants to run in the same process without leaking replies or events.
    """

    adapter_name = ADAPTER_NAME
    adapter_version = ADAPTER_VERSION

    def __init__(self, fixture: dict[str, Any]) -> None:
        self._seed = copy.deepcopy(fixture)
        self.reset()

    def reset(self) -> None:
        fixture = copy.deepcopy(self._seed)
        actor = fixture.get("actor")
        if not isinstance(actor, dict):
            raise ValueError("email calendar fixture needs an actor object")
        account = actor.get("email")
        if not isinstance(account, str) or not account.strip():
            raise ValueError("email calendar fixture needs a non-empty account")
        self.account = account
        self._actor = str(actor.get("name") or actor.get("id") or account)
        self._messages = {
            str(message["id"]): _normalize_message(message)
            for message in _fixture_messages(fixture)
        }
        mailbox = fixture.get("mailbox")
        calendar = fixture.get("calendar")
        if not isinstance(mailbox, dict) or not isinstance(mailbox.get("threads"), list):
            raise ValueError("email calendar fixture needs mailbox.threads")
        if not isinstance(calendar, dict) or not isinstance(calendar.get("events"), list):
            raise ValueError("email calendar fixture needs calendar.events")
        events = calendar["events"]
        self._events = {
            str(event["id"]): _normalize_event(event)
            for event in events
        }
        self._follow_ups = {
            str(thread_id): copy.deepcopy(detail)
            for thread_id, detail in fixture.get("follow_ups", {}).items()
        }
        self._next_message_id = _next_id(self._messages, "m")
        self._next_event_id = _next_id(self._events, "e")
        self._recorder = _TrajectoryRecorder()

    def tools(self) -> tuple[ToolSpec, ...]:
        return TOOLS

    @property
    def trajectory(self) -> tuple[TrajectoryStep, ...]:
        return tuple(self._recorder.steps)

    def mutations(self) -> tuple[dict[str, Any], ...]:
        return self._recorder.mutations

    def state(self) -> dict[str, Any]:
        return {
            "account": self.account,
            "messages": copy.deepcopy(self._messages),
            "events": copy.deepcopy(self._events),
            "follow_ups": copy.deepcopy(self._follow_ups),
        }

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        spec = _TOOLS_BY_NAME.get(name)
        step = self._recorder.next_step_index
        if spec is None:
            result = ToolResult(False, f"error: unknown tool {name!r}")
            self._recorder.record(name, arguments, result, mutating=False)
            return result
        try:
            _validate_schema(spec, arguments)
            result = getattr(self, f"_tool_{name}")(step, dict(arguments))
        except ToolInvocationError as exc:
            result = ToolResult(False, f"error: {exc}")
        self._recorder.record(name, arguments, result, mutating=spec.mutating)
        return result

    def _tool_list_messages(self, step: int, args: dict[str, Any]) -> ToolResult:
        folder = str(args.get("folder", "all"))
        rows = [m for m in self._messages.values() if folder == "all" or m["folder"] == folder]
        if not rows:
            return ToolResult(True, "No messages match that folder.")
        lines = [f"{len(rows)} message(s):"]
        for message in rows:
            lines.append(f"{message['id']} [{message['folder']}] {message['subject']} ({message['thread_id']})")
        return ToolResult(True, "\n".join(lines))

    def _tool_get_message_thread(self, step: int, args: dict[str, Any]) -> ToolResult:
        thread_id = _required_str(args, "thread_id")
        rows = [message for message in self._messages.values() if message["thread_id"] == thread_id]
        if not rows:
            raise ToolInvocationError(f"no message thread {thread_id!r}")
        lines = [f"Thread {thread_id}:"]
        for message in rows:
            lines.extend((f"{message['id']} [{message['folder']}] {message['from']} -> {', '.join(message['to'])}", f"Subject: {message['subject']}", message["body"]))
        return ToolResult(True, "\n".join(lines))

    def _tool_list_events(self, step: int, args: dict[str, Any]) -> ToolResult:
        if not self._events:
            return ToolResult(True, "No events.")
        return ToolResult(True, "\n".join([f"{len(self._events)} event(s):"] + [f"{event['id']} [{event['status']}] {event['title']} {event['start']} - {event['end']}" for event in self._events.values()]))

    def _tool_get_event(self, step: int, args: dict[str, Any]) -> ToolResult:
        event = self._event(_required_str(args, "event_id"))
        lines = [f"Event {event['id']}: {event['title']}", f"Status: {event['status']}", f"When: {event['start']} - {event['end']}", f"Attendees: {', '.join(event['attendees']) or 'none'}", "Description:", event["description"]]
        if event["comments"]:
            lines.append("Comments:")
            lines.extend(f"- {comment['author']}: {comment['body']}" for comment in event["comments"])
        return ToolResult(True, "\n".join(lines))

    def _tool_create_draft_reply(self, step: int, args: dict[str, Any]) -> ToolResult:
        message = self._reply(args, "drafts")
        self._recorder.record_mutation(step, "create_draft_reply", {"message_id": message["id"], "thread_id": message["thread_id"]})
        return ToolResult(True, f"Created draft reply {message['id']} in thread {message['thread_id']}.")

    def _tool_send_reply(self, step: int, args: dict[str, Any]) -> ToolResult:
        message = self._reply(args, "sent")
        self._recorder.record_mutation(step, "send_reply", {"message_id": message["id"], "thread_id": message["thread_id"]})
        return ToolResult(True, f"Sent reply {message['id']} in thread {message['thread_id']}.")

    def _tool_assign_follow_up(self, step: int, args: dict[str, Any]) -> ToolResult:
        thread_id = _required_thread(self._messages, args)
        detail = {"owner": _required_str(args, "owner")}
        if "due_date" in args:
            detail["due_date"] = _required_str(args, "due_date")
        self._follow_ups[thread_id] = detail
        self._recorder.record_mutation(step, "assign_follow_up", {"thread_id": thread_id, **detail})
        return ToolResult(True, f"Assigned follow-up for {thread_id} to {detail['owner']}.")

    def _tool_create_event(self, step: int, args: dict[str, Any]) -> ToolResult:
        event_id = f"e-{self._next_event_id}"
        self._next_event_id += 1
        event = _normalize_event({"id": event_id, "title": _required_str(args, "title"), "start": _required_str(args, "start"), "end": _required_str(args, "end"), "attendees": _string_list(args, "attendees"), "description": str(args.get("description", "")), "status": "confirmed"})
        self._events[event_id] = event
        self._recorder.record_mutation(step, "create_event", {"event_id": event_id})
        return ToolResult(True, f"Created event {event_id}.")

    def _tool_update_event(self, step: int, args: dict[str, Any]) -> ToolResult:
        event = self._event(_required_str(args, "event_id"))
        updated = [key for key in ("title", "start", "end", "description") if key in args]
        for key in updated:
            event[key] = _required_str(args, key)
        if "attendees" in args:
            event["attendees"] = _string_list(args, "attendees")
            updated.append("attendees")
        if not updated:
            raise ToolInvocationError("update_event needs at least one field to update")
        self._recorder.record_mutation(step, "update_event", {"event_id": event["id"], "fields": updated})
        return ToolResult(True, f"Updated event {event['id']}.")

    def _tool_decline_event(self, step: int, args: dict[str, Any]) -> ToolResult:
        event = self._event(_required_str(args, "event_id"))
        event["status"] = "declined"
        if "comment" in args:
            event["comments"].append({"author": self._actor, "body": _required_str(args, "comment")})
        self._recorder.record_mutation(step, "decline_event", {"event_id": event["id"]})
        return ToolResult(True, f"Declined event {event['id']}.")

    def _tool_add_event_comment(self, step: int, args: dict[str, Any]) -> ToolResult:
        event = self._event(_required_str(args, "event_id"))
        event["comments"].append({"author": self._actor, "body": _required_str(args, "body")})
        self._recorder.record_mutation(step, "add_event_comment", {"event_id": event["id"]})
        return ToolResult(True, f"Added a comment to event {event['id']}.")

    def _reply(self, args: dict[str, Any], folder: str) -> dict[str, Any]:
        thread_id = _required_thread(self._messages, args)
        parent = next(message for message in self._messages.values() if message["thread_id"] == thread_id)
        message_id = f"m-{self._next_message_id}"
        self._next_message_id += 1
        message = _normalize_message({"id": message_id, "thread_id": thread_id, "from": self.account, "to": [parent["from"]], "subject": parent["subject"], "body": _required_str(args, "body"), "folder": folder, "sent_at": ""})
        self._messages[message_id] = message
        return message

    def _event(self, event_id: str) -> dict[str, Any]:
        event = self._events.get(event_id)
        if event is None:
            raise ToolInvocationError(f"no event {event_id!r}")
        return event


def _validate_schema(spec: ToolSpec, args: dict[str, Any]) -> None:
    if not isinstance(args, dict):
        raise ToolInvocationError("arguments must be an object")
    parameters = spec.parameters
    allowed = set(parameters["properties"])
    extra = sorted(set(args) - allowed)
    if extra:
        raise ToolInvocationError(f"unexpected argument(s): {', '.join(extra)}")
    missing = [key for key in parameters["required"] if key not in args]
    if missing:
        raise ToolInvocationError(f"missing required argument(s): {', '.join(missing)}")
    for key, value in args.items():
        schema = parameters["properties"][key]
        if schema["type"] == "string" and (not isinstance(value, str) or not value.strip()):
            raise ToolInvocationError(f"{key} must be a non-empty string")
        if schema["type"] == "array" and (not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value)):
            raise ToolInvocationError(f"{key} must be a list of non-empty strings")
        if "enum" in schema and value not in schema["enum"]:
            raise ToolInvocationError(f"{key} must be one of: {', '.join(schema['enum'])}")


def _fixture_messages(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the public nested mailbox fixture schema into runtime rows."""
    rows: list[dict[str, Any]] = []
    for thread in fixture["mailbox"]["threads"]:
        folder = "inbox" if "inbox" in thread.get("labels", []) else "all"
        for message in thread.get("messages", []):
            rows.append(
                {
                    "id": message["id"],
                    "thread_id": thread["id"],
                    "from": message.get("sender", thread["sender"]),
                    "to": message.get("recipients", thread["recipients"]),
                    "subject": thread.get("subject", ""),
                    "body": message.get("body", ""),
                    "folder": folder,
                    "sent_at": thread.get("date", ""),
                }
            )
    return rows


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(message["id"]), "thread_id": str(message["thread_id"]), "from": str(message["from"]), "to": [str(value) for value in message.get("to", [])], "subject": str(message.get("subject", "")), "body": str(message.get("body", "")), "folder": str(message.get("folder", "inbox")), "sent_at": str(message.get("sent_at", ""))}


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    attendees = [
        str(value.get("email", "")) if isinstance(value, dict) else str(value)
        for value in event.get("attendees", [])
    ]
    return {"id": str(event["id"]), "title": str(event["title"]), "start": str(event["start"]), "end": str(event["end"]), "attendees": attendees, "description": str(event.get("description", "")), "status": str(event.get("status", "confirmed")), "comments": [{"author": str(row.get("author", "unknown")), "body": str(row["body"])} for row in event.get("comments", [])]}


def _next_id(rows: dict[str, Any], prefix: str) -> int:
    values = [int(key[len(prefix) + 1:]) for key in rows if key.startswith(f"{prefix}-") and key[len(prefix) + 1:].isdigit()]
    return max(values, default=0) + 1


def _required_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolInvocationError(f"{key} must be a non-empty string")
    return value


def _string_list(args: dict[str, Any], key: str) -> list[str]:
    value = args.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ToolInvocationError(f"{key} must be a list of non-empty strings")
    return list(value)


def _required_thread(messages: dict[str, dict[str, Any]], args: dict[str, Any]) -> str:
    thread_id = _required_str(args, "thread_id")
    if not any(message["thread_id"] == thread_id for message in messages.values()):
        raise ToolInvocationError(f"no message thread {thread_id!r}")
    return thread_id


register_environment(ADAPTER_NAME, EmailCalendarFixtureEnvironment)
