# Decision log - 2026-03-31 - proration on seat changes

Present: me, Elena, Tomas.

Question: when a customer adds seats mid-cycle, do we charge prorated
immediately or roll it into the next invoice?

Decided: prorate immediately on additions, roll removals into the next cycle.

Why the asymmetry, since Elena asked me to write it down: adding a seat is an
action the customer just took and expects to pay for, so an immediate charge is
unsurprising. Removing a seat is often cleanup, and issuing a mid-cycle credit
creates a refund conversation and sometimes an actual refund, which costs more
in processing and support time than the credit is worth.

Edge case we accepted: a customer who adds and removes in the same cycle pays
for the addition and waits for the removal credit. Elena confirmed this is
normal and matches what our competitors do.

Implemented in 1.0.69.
