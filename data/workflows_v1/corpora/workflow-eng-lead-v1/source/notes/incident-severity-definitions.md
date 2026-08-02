# Incident severity - 2026-03-06

Written after the auth outage, where we spent ten minutes arguing about
severity while it was ongoing.

- **SEV1.** Users cannot work. Total outage, data loss, or a security breach.
  Page immediately, incident channel, someone owns comms separately from the
  person fixing it.
- **SEV2.** A major feature is broken for many users, or any customer with a
  contractual SLA is materially affected. Page during business hours, next
  morning otherwise.
- **SEV3.** Degraded, workaround exists. No page. Normal issue flow.

Declare high and downgrade. Downgrading is free; realising an hour in that it
was worse than you thought costs the hour.

The person who declares is whoever notices. You do not need permission and you
will not be second-guessed for declaring a SEV1 that turned out to be a SEV3.
We have never had that problem; we have had the opposite twice.

The March auth outage was a SEV1 and we called it a SEV2 for the first 15
minutes.
