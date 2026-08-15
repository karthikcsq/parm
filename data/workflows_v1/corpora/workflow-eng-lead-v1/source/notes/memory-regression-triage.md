# Triage notes - long-session memory regressions

Two separate reports keep getting filed as one bug and they are not the same
thing.

The heap exhaustion is real and reproducible: a session that runs past roughly
four hours with a large workspace open climbs until V8 gives up. Dan bisected
it to the transcript buffer never being trimmed - we keep every tool result in
memory so the transcript view can scroll back to the start.

The auto-compact complaint is different. Compaction does fire, but the trigger
reads a token estimate that is computed before the tool results are appended,
so on a big repo the estimate is stale by the time it matters and compaction
runs a few turns too late to help.

My preference for the fix: cap the transcript buffer with a hard byte limit and
stream older entries to disk rather than trying to make the estimator smarter.
The estimator will always be behind. A buffer limit is a bound we can state.

Whoever picks this up should write the reasoning down somewhere in docs/ rather
than only in the PR description. We have lost this analysis twice already.
