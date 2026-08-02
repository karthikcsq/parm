# Event naming conventions - 2026-03-19

Written up after the third review where Dan and I argued about this in comments
instead of pointing at a rule.

Shape is `<subject>_<verb_past_tense>`, lowercase, underscores:

- `session_started`, `session_compact_triggered`, `tool_invoked`
- not `startSession`, not `Session.Start`, not `compact-triggered`

Properties are flat. No nested objects, no arrays. If you want to send a list,
send a count and a separate event per item, or do not send it.

Never put free text in a property. Property values are enums, counts, booleans,
or durations in milliseconds. The moment someone puts a file path or a prompt
in a property we have a review problem that is much more expensive than the
metric was worth.

Names are permanent in practice. Renaming an event means every saved query and
every dashboard that used it silently returns nothing, and nobody notices for a
month. Get the name right the first time or leave the event out.

This note is about what an event is called. It says nothing about when one may
ship.
