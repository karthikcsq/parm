# What the regression suite does not cover - 2026-04-14

Audited after a regression shipped in 1.0.69 that a test should have caught.

Genuinely uncovered:

- Long-running sessions. Every test session is under a hundred turns. The
  memory behaviour we care about only shows up after several hundred, and we
  have no test that runs that long because it would take twenty minutes.
- Concurrent tool calls. Almost all tests issue one call at a time. The token
  refresh race was invisible to the suite for exactly this reason.
- Anything involving the real filesystem being slow. All our IO is against a
  temp dir on a fast disk.

Covered but weakly:

- Config precedence. Tested, but only the two-layer case, not the five-layer
  precedence chain that actually exists.

The long-session gap is the one that matters right now given #46 and #49. A
soak test that runs nightly rather than per-commit would close it without
costing eleven minutes per change. Filed as #41.
