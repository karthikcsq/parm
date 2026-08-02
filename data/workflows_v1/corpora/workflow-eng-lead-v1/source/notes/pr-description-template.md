# What a pull request description needs - 2026-02-11

Four things. Anything else is optional.

1. **What changes for a user.** One sentence, in the language a user would use.
   If nothing changes for a user, say that explicitly.
2. **Why now.** Link the issue, or explain the trigger if there is no issue.
3. **What you did not do.** The scope you deliberately left out. This is the
   one people skip and the one reviewers need most, because a reviewer's first
   instinct is to ask about the thing you already decided against.
4. **How you know it works.** Not "tests pass". Which test, or what you ran by
   hand.

Not required: a summary of the diff. The diff is right there. A description
that walks through the changed files line by line is a reviewer telling you
they did not want to read it either.

Reviewers: it is reasonable to ask for a description before reviewing. It is
not reasonable to approve without one and then ask afterwards.
