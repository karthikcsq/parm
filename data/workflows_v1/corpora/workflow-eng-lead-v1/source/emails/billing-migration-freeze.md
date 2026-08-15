Subject: billing DB migration - change freeze 2026-04-13 to 2026-04-20
From: me
To: eng
Date: 2026-04-09

We are moving the billing tables to the new cluster the week of the 13th.

For that week only, nothing that touches `services/billing` merges. Not fixes,
not tests, not documentation in that directory. The migration replays writes
and a schema change landing mid-replay is how people lose a weekend.

Everything else in the repository is unaffected. Normal review, normal merges,
normal release cadence. If your change does not touch `services/billing`, this
email is not about you.

Tomas is running the migration and has the rollback. If you think you have an
exception, ask him, not me.

Lifting Monday the 20th. I will send a note when it is actually done rather
than when it is scheduled to be done.
