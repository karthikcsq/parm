# Email + Calendar fixture contract (v1)

`data/workflows_email_calendar_v1` is a deterministic local workflow dataset. Its public contract is deliberately small: tracked fixtures use one nested schema, agents see one stable action API, and every case assertion comes from the verifier vocabulary below. There are no flat-fixture or legacy assertion aliases.

## Fixture schema

Each JSON fixture has these top-level fields:

```json
{
  "schema_version": 1,
  "actor": {"id": "...", "name": "...", "email": "..."},
  "mailbox": {"threads": []},
  "calendar": {"events": []},
  "contacts": [],
  "attachments": []
}
```

A mailbox thread has `id`, `sender`, `recipients`, `subject`, `labels`, and `messages`; messages use `id`, `sender`, `recipients`, and `body`. Calendar events have `id`, `title`, `start`, `end`, `attendees`, and `description`; attendees can be objects containing `email`.

The adapter normalizes this nested source format to private runtime state. Its `state()` representation is not a public fixture format.

## Actions

The read actions are `list_messages`, `get_message_thread`, `list_events`, and `get_event`. Mutations are `create_draft_reply`, `send_reply`, `assign_follow_up`, `create_event`, `update_event`, `decline_event`, and `add_event_comment`.

`assign_follow_up` requires `{ "thread_id", "owner" }`; `due_date` is optional. Public action and assertion fields use **`owner`**, never `assignee`.

## Assertions

The Email + Calendar-specific decisive kinds are:

- `message_sent`: `thread_id`, optional `body_keywords`
- `follow_up_assigned`: `thread_id`, optional `owner` and `due_date`
- `event_response`: `event_id`, `expected` status
- `event_attendees`: `event_id`, optional `includes` and `excludes`

Generic workflow kinds (for example `no_mutation`, with `kinds` and optional `where`) remain available. Dataset validation verifies structural triplets; the end-to-end scenario test additionally builds every fixture, reaches each declared cue, invokes its declared decisive action, and requires every assertion to pass.

## Determinism and fairness

Run `python3 scripts/build_workflows_email_calendar_v1_cases.py` to regenerate cases and fixtures. It writes sorted JSON with LF newlines and is byte-idempotent. Positive and cue-ablated fixture pairs differ only at their declared late cue path; their decisive assertion signatures must differ, including the accessibility triplet.
