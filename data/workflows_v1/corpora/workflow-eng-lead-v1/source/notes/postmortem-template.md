# Postmortem template and rules - 2026-03-08

Sections, in order: Impact, Cause, Detection, Fix, Actions.

**Impact** is in user terms and includes duration and scope. "41 minutes, ~2,100
users" not "the auth service was down".

**Detection** is its own section on purpose. How long until we knew, and how we
found out. If a user told us before monitoring did, that goes here in those
words. Detection failures are usually more actionable than the cause.

**Actions** have an owner and a date, or they are not actions. An action
without an owner is a wish.

Rules:

- Blameless means we do not name who made the change. It does not mean we
  avoid saying what the change was.
- Write it within three days. A postmortem written a fortnight later is
  fiction assembled from a chat log.
- Every action gets filed as an issue before the document is considered done.

We do postmortems for SEV1 and SEV2. SEV3 only if someone asks.
