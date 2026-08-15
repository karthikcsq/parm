# Email + Calendar fixture contract (v1 and natural v2)

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

The read actions are `list_messages`, `get_message_thread`, `list_events`, and `get_event`. Mutations are `create_draft_reply`, `send_reply`, `assign_follow_up`, `resolve_thread`, `set_thread_hold`, `create_event`, `update_event`, `mark_event_pending_exception`, `decline_event`, and `add_event_comment`.

`assign_follow_up` requires `{ "thread_id", "owner" }`; `due_date` is optional. Public action and assertion fields use **`owner`**, never `assignee`.

Natural-pilot extensions are additive and use resolved fixture identities:

- `list_routing_targets` lists normalized `contacts` entries with `id`, `name`, `email`, and `kind` (`contact` or `queue`).
- `route_follow_up` requires `{ "thread_id", "target_id" }`; `target_id` must come from that list, and the final follow-up state records the resolved target identity rather than an arbitrary label.
- `reschedule_event` requires `{ "event_id", "start", "end" }` and accepts optional `approval_reference`. It changes only the event time and leaves its response/status intact.
- `resolve_thread` requires `{ "thread_id" }` and records ordinary resolution in private runtime state.
- `set_thread_hold` requires `{ "thread_id", "hold_kind", "target_id" }`. `hold_kind` is `review` or `approval`; `target_id` must resolve through `list_routing_targets`. It records a target-bound pending hold, visible through `get_message_thread`.
- `mark_event_pending_exception` requires `{ "event_id", "exception_type" }`; the current documented exception type is `change_window`. It records a pending exception state, visible through `get_event`.

These actions do not alter the v1 fixture schema or replace any v1 action.

## Assertions

The Email + Calendar-specific decisive kinds are:

- `message_sent`: `thread_id`, optional `body_keywords`
- `follow_up_assigned`: `thread_id`, optional `owner` and `due_date`
- `event_response`: `event_id`, `expected` status
- `event_attendees`: `event_id`, optional `includes` and `excludes`
- `event_rescheduled`: `event_id`, optional final `start`/`end`, and optional `approval_reference`; requires a matching `reschedule_event` mutation.
- `thread_message_content`: `thread_id`, `folder` (`drafts` or `sent`), optional `includes` and `excludes`; one message must contain every included phrase and none of the excluded phrases.
- `thread_hold_state`: `thread_id`, `hold_kind`, `target_id`; requires that exact target-bound pending hold.
- `thread_not_on_hold`: the same fields; requires that exact pending hold not be present, without requiring any particular ordinary action.
- `event_pending_exception`: `event_id`, `exception_type`; requires that exact target event hold.
- `event_not_pending_exception`: the same fields; requires that exact event hold not be present.

For example, a legal handoff can require a draft with `includes: ["legal", "review"]` and `excludes: ["we confirm"]`, making the handoff content deterministic without forcing an external send.

Generic workflow kinds (for example `no_mutation`, with `kinds` and optional `where`) remain available. Dataset validation verifies structural triplets; the end-to-end scenario test additionally builds every fixture, reaches each declared cue, invokes its declared decisive action, and requires every assertion to pass.

## Natural v2 environmental-state contracts

`data/workflows_email_calendar_v2` grades final target-bound environment state,
not generated prose, drafts, sent replies, keyword matching, or an exact option
string. The natural target locators remain work-item IDs such as
`mail-customer-request-march-4` and `event-delivery-session-dec-23`; adapter
behavior itself has no scenario or customer identifiers.

- **Atlas positive/oracle:** the target thread is pending a `review` hold at
  the resolved Legal target. **Control:** that target is not in that Legal hold;
  it may be resolved normally or take another ordinary path.
- **Trellis positive/oracle:** the target event is pending the `change_window`
  exception. **Control:** that target event is not pending that exception.
- **Orion positive/oracle:** the target reimbursement thread is pending an
  `approval` hold at the resolved Incident Manager target. **Control:** that
  target is not in that approval hold.

The v2 score is deterministic from final environment state. A model may use the
documented state-setting operations, while read tools expose the resulting
state. Control scoring deliberately does not require an arbitrary canned
normal action.

## Determinism and fairness

Run `python3 scripts/build_workflows_email_calendar_v1_cases.py` to regenerate cases and fixtures. It writes sorted JSON with LF newlines and is byte-idempotent. Positive and cue-ablated fixture pairs differ only at their declared late cue path; their decisive assertion signatures must differ, including the accessibility triplet.

Run `python3 scripts/build_workflows_email_calendar_v2_cases.py` for v2.  Its
positive/control fixture pairs retain that exact one-late-fact symmetry; only
the oracle/memory-included goal receives the verbatim source text.
