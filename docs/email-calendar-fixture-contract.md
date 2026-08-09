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

The read actions are `list_messages`, `get_message_thread`, `list_events`, and `get_event`. Mutations are `create_draft_reply`, `send_reply`, `assign_follow_up`, `create_event`, `update_event`, `decline_event`, and `add_event_comment`.

`assign_follow_up` requires `{ "thread_id", "owner" }`; `due_date` is optional. Public action and assertion fields use **`owner`**, never `assignee`.

Natural-pilot extensions are additive and use resolved fixture identities:

- `list_routing_targets` lists normalized `contacts` entries with `id`, `name`, `email`, and `kind` (`contact` or `queue`).
- `route_follow_up` requires `{ "thread_id", "target_id" }`; `target_id` must come from that list, and the final follow-up state records the resolved target identity rather than an arbitrary label.
- `reschedule_event` requires `{ "event_id", "start", "end" }` and accepts optional `approval_reference`. It changes only the event time and leaves its response/status intact.

These actions do not alter the v1 fixture schema or replace any v1 action.

## Assertions

The Email + Calendar-specific decisive kinds are:

- `message_sent`: `thread_id`, optional `body_keywords`
- `follow_up_assigned`: `thread_id`, optional `owner` and `due_date`
- `event_response`: `event_id`, `expected` status
- `event_attendees`: `event_id`, optional `includes` and `excludes`
- `event_rescheduled`: `event_id`, optional final `start`/`end`, and optional `approval_reference`; requires a matching `reschedule_event` mutation.
- `thread_message_content`: `thread_id`, `folder` (`drafts` or `sent`), optional `includes` and `excludes`; one message must contain every included phrase and none of the excluded phrases.

For example, a legal handoff can require a draft with `includes: ["legal", "review"]` and `excludes: ["we confirm"]`, making the handoff content deterministic without forcing an external send.

Generic workflow kinds (for example `no_mutation`, with `kinds` and optional `where`) remain available. Dataset validation verifies structural triplets; the end-to-end scenario test additionally builds every fixture, reaches each declared cue, invokes its declared decisive action, and requires every assertion to pass.

## Natural v2 semantic constraint contracts

`data/workflows_email_calendar_v2` uses `constraint_compliance` for its
positive and oracle cases.  It deliberately accepts a bounded family of
source-conforming outcomes instead of one canned API expression.  The contract
has `allowed_outcomes` (any one may satisfy the scenario) and
`forbidden_outcomes` (any one vetoes success).  Each nested outcome uses a
target-specific thread or event locator from the fixture; these locators are
natural work items such as `mail-customer-request-march-4` and
`event-delivery-session-dec-23`, not policy labels.

The current v2 scenarios are deliberately strict about the source wording:

- **Atlas:** a draft/sent non-confirming Legal-pending response, or resolved
  routing to Legal, is allowed.  A sent external “we confirm” on the Atlas
  request is forbidden because the source says Legal review is required first.
- **Trellis:** the December 23 delivery session must be rescheduled wholly
  outside the December 20–January 5 production window, or carry a
  target-specific approved-production-exception note.  Rescheduling a context
  event does not satisfy it.
- **Orion:** a target-specific incident-manager-pending draft/reply or resolved
  route to the Incident Manager is allowed.  A routine “being processed” reply
  without that approval path is forbidden.

The supporting assertion kinds are `follow_up_routed`,
`event_rescheduled_outside_window`, and `event_comment_content`.  They inspect
final fixture state and mutation evidence, rather than requiring one exact
tool-call body.

## Determinism and fairness

Run `python3 scripts/build_workflows_email_calendar_v1_cases.py` to regenerate cases and fixtures. It writes sorted JSON with LF newlines and is byte-idempotent. Positive and cue-ablated fixture pairs differ only at their declared late cue path; their decisive assertion signatures must differ, including the accessibility triplet.

Run `python3 scripts/build_workflows_email_calendar_v2_cases.py` for v2.  Its
positive/control fixture pairs retain that exact one-late-fact symmetry; only
the oracle/memory-included goal receives the verbatim source text.
