# Issue triage rules

Labels I actually use:

- `critical` - users lose work or cannot start a session. Goes on the next
  hotfix regardless of what else is planned.
- `bug` - wrong behavior, no data loss.
- `enhancement` - everything else.

Only I add `critical`. It has been diluted twice by well-meaning people and
each time it took a month to clean up.

When two issues describe the same underlying defect, keep both open and
cross-reference them rather than closing one as a duplicate. The reporters
described different symptoms and both symptoms need to be verified fixed.

A tracking issue for a multi-issue fix should list the issue numbers in its
body, not only in a comment. Search finds the body.
