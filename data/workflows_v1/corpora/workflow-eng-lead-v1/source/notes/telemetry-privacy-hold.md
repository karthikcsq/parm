# Decision log - 2026-05-14 - telemetry freeze during the privacy review

Present: me, Priya (legal/privacy), Dan (platform).

Northwind Health escalated on the DPA renewal. Their security team asked for a
written inventory of every client-side event we emit before they will sign, and
Priya agreed we would not ship anything new into that inventory while it is
being assembled.

What I committed to, in writing, in the renewal thread:

Until the enterprise privacy review closes, do not merge changes that add or
expand telemetry or event logging. Open work is fine, but Priya must review it
before merge.

This is narrower than a full freeze. Writing the code, opening the PR, running
CI, and iterating on review comments are all fine. The line is the merge. Once
something lands on main it goes out in the next release and ends up in the
inventory we are trying to hold still.

Practical notes for whoever is holding the pager:

- Analytics, product metrics, event logging, and crash breadcrumbs all count.
  If a change emits anything new to a third-party collector, it counts.
- Vendor SDK upgrades count if they change what is emitted.
- Log lines that stay on the user's own machine do not count.
- If in doubt, tag Priya on the PR and leave it open. She has been turning
  these around in about a day.

I do not have a close date yet. Priya said to assume it runs into the summer.
Do not treat "checks are green" as sufficient for anything in this category.
