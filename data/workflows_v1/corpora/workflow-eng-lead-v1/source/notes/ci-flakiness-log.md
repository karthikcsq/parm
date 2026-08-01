# CI flakiness log

Running tally of tests that have failed without a code change, so we stop
re-running blind.

- `session/compaction.spec.ts` - three failures in April, all on the Windows
  runner, all timing out at exactly 30s. Suspect the temp directory cleanup.
- `indexer/symlinks.spec.ts` - one failure, traced to a stale fixture cache.
  Fixed.
- `telemetry/emit.spec.ts` - two failures, both after Dan's SDK bump. He is
  aware. The test asserts on a payload shape the SDK changed.
- `cli/completion.spec.ts` - noisy on macOS only, roughly one run in fifteen.

Nobody should merge on a re-run alone. If a test fails twice on the same
branch, it goes in this file before the branch merges.
