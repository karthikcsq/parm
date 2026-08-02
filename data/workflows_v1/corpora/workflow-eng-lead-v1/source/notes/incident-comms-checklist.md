# Incident comms - who says what - 2026-03-07

Separate from fixing. During a SEV1 the person fixing does not write the
updates; that is how you get a 40 minute silence.

Roles:

- **Incident lead.** Coordinates, decides severity, decides when it is over.
  Usually the on-call.
- **Comms.** Writes the status page and the internal updates on a fixed cadence
  regardless of whether there is news. "No change, next update in 30 minutes"
  is a valid update and prevents six people asking.
- **Fixer.** Fixes. Talks only to the lead.

For anything customer-facing, Aditi writes the customer-facing wording, not
engineering. We are bad at it: we either understate to the point of being
misleading or include detail that raises questions nobody asked.

Status page goes up for anything SEV1 or SEV2 lasting more than 15 minutes,
even if we expect to fix it in 20. Putting it up late looks worse than putting
it up and resolving quickly.
