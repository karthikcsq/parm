# Who resolves a merge conflict - 2026-03-24

Trivial, but it was causing friction so it is written down.

The author of the later change rebases and resolves. Not the reviewer, not
whoever merged first, not me.

Corollary: if your branch has been open long enough to conflict with something
that landed, that is information. Two conflicts in a week usually means the
change is too big or has been open too long, and the fix is to split it rather
than to get better at rebasing.

Exception: if resolving requires understanding the other change well enough to
be dangerous, ask the other author to pair on it for ten minutes. That is
cheaper than a bad resolution that compiles.

Never resolve a conflict by taking your whole side. I have seen a fix silently
reverted this way twice, and both times the tests passed because the fix had no
test.
