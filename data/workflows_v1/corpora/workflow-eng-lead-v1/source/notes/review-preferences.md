# How I want reviews handled

Small PRs, one concern each. If a PR has a refactor and a behavior change in
it, I will ask for it to be split before I read either half.

I do not need to be a reviewer on everything. Route by area:

- Platform and build: Dan.
- Anything touching the session loop: Mei.
- Anything with a legal, privacy, or contractual angle: Priya.
- Docs-only changes: any one of us, and do not block on it.

Approve-with-comments is fine for nits. Use a blocking review only when the
change would be wrong to ship, not when you would have written it differently.

If a PR sits for more than two days, ping in the channel rather than adding
another reviewer. Adding reviewers makes everyone assume someone else has it.
