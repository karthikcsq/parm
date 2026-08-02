# Decision log - 2026-03-05 - no consent banner in the CLI

Present: me, Priya, Sofia.

Rina asked for a first-run consent prompt so we could count more. We decided
against it.

Reasoning:

- A CLI that asks a question before it does anything is a CLI people stop
  using. Sofia's read is that a blocking prompt on install costs us more than
  the extra telemetry is worth.
- Priya's position is that consent theatre is worse than no consent: a banner
  people dismiss without reading does not give us permission for anything we
  would not otherwise be comfortable collecting.

So the rule stays: collect only what we would be comfortable defending without
having asked, document it in the docs site, and make opt-out a single flag.

Revisit if we ever want to collect something that fails that test. We have not
wanted to yet.
