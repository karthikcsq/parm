# Postmortem - 2026-04-22 - v1.0.68 shipped a broken indexer

What happened: v1.0.68 went out Tuesday morning with an indexer change that
crashed on repositories containing symlink loops. Roughly nine percent of
sessions failed to start for four hours until we pulled the release.

Why it got through: the change had a green CI run, one approval, and no manual
smoke test. The symlink case is not in our fixtures.

What I changed afterwards:

- The smoke job now runs against the tag, not the branch, because the branch
  job had been passing against a merge base that no longer existed.
- We added a symlink-loop fixture.

What I did not change: I did not add more required reviewers. The problem was
not that too few people looked at it. Green checks are not evidence that a
change is safe to ship, and I would rather we say that out loud than add
process nobody reads.
