# Triage rotation - how it runs - 2026-03-17

Whoever holds the pager also holds triage. One pass per day, timeboxed to 30
minutes.

For each new issue:

1. Is it reproducible from what is written? If not, ask once, apply
   `needs-repro`, move on.
2. Is it a duplicate? Close and link. Do not merge the discussion.
3. Does it match the `critical` bar? Data loss, crash on a common path, or a
   regression we shipped. If yes, label `critical` and say so in the daily
   note.
4. Otherwise leave it unlabelled. Unlabelled means triaged and ordinary.

The 30 minute box is the important part. Triage expands to fill whatever time
it is given, and a thorough triage pass that happens twice a week is worse than
a shallow one that happens daily.

Do not fix things during triage. The temptation is enormous on anything that
looks like a ten minute fix, and it is how a 30 minute pass becomes an
afternoon.
