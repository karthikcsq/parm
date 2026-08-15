# Postmortem - 2026-03-02 - auth outage, 41 minutes

**Impact.** All authenticated requests failed for 41 minutes. Roughly 2,100
users affected.

**Cause.** A certificate on the internal auth service expired. It was issued
manually 13 months earlier by someone who has since left, and it was not in the
renewal automation because the automation reads from a list that this
certificate was never added to.

**Detection.** A user in the community channel, nine minutes before our
monitoring noticed. We alert on error rate over a five minute window with a
three-window trigger, so the floor on detection is fifteen minutes.

**Fix.** Reissued, deployed. Tomas had it in hand in 20 minutes from being
paged.

**Actions.**

- Every certificate into the automation, and an alert on any cert expiring in
  under 30 days. Done 2026-03-04.
- Reduce the error rate alert to two windows. Done.
- Audit for other manually-provisioned infrastructure created by people who
  have left. Tomas, found two more.

No customer contract implications; Aditi confirmed nobody was in a window that
mattered.
