# Testing pyramid targets - SUPERSEDED 2026-04-14

Replaced by the testing expectations note. Kept because the coverage numbers
here still appear in old pull request templates.

What this said:

> 70% unit, 20% integration, 10% end-to-end. Overall line coverage target 85%.

Why it went: the 85% line coverage target produced tests written to touch lines
rather than to catch failures. The clearest example is `src/session/`, which sat
at 91% coverage and did not have a single test that ran a session long enough to
exercise the transcript growth in #46.

The current expectation is stated in behaviours rather than ratios: every bug
fix gets a test that fails before the fix, and every module has at least one
test that exercises it the way a user would.

Coverage is still measured. It is no longer a target, and no pull request is
blocked on it.
