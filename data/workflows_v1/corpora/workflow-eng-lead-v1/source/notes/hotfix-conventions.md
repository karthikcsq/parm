# Hotfix conventions

A hotfix is a change we would not normally ship mid-week, shipped mid-week
because leaving it alone is worse.

Shape I expect:

- One branch off main named `hotfix/<topic>-v<target-version>`.
- The smallest change that removes the user-visible failure. Not the change we
  would make with a month.
- A written explanation in `docs/` of what was wrong, not only in the pull
  request body. Pull request bodies are hard to find a year later.
- A tracking issue that names every issue number the hotfix addresses, so the
  people who filed them find out it is being worked on.
- The tracking issue stays open until the fix is released, then gets a closing
  comment with the release tag.

If a hotfix turns out to need a design discussion, it was not a hotfix. Stop,
open the issue, and go back to the normal path.
