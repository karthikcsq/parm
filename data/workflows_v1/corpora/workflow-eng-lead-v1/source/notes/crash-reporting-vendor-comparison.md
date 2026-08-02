# Crash reporting: staying on Sentry - 2026-03-26

Spent an afternoon on whether to move off Sentry. Conclusion: no.

What I compared:

- **Sentry**, current. Self-host option exists. Scrubbing config is ours. Costs
  more than it should at our volume.
- **Bugsnag**. Cheaper. Their scrubbing is allowlist-based which is arguably
  safer, but migrating the deny list we already trust is a week of work for a
  marginal gain.
- **Roll our own.** Tomas offered. It would be one endpoint and a symbol store.
  I said no: the moment we own it, we own retention, deletion requests, and
  someone getting paged when the ingest queue backs up.

The real cost of switching is not the integration, it is that every processor
change means another conversation with every customer who has asked about our
subprocessors. That is the part people forget when they price these.

Staying put. Revisit if Sentry's pricing changes again.
