# Hotfix or wait for the next release - 2026-04-06

The question comes up every time and we keep answering it from scratch.

Hotfix if any of these is true:

- data loss or corruption, at any frequency
- a crash on a common path with no workaround
- a security issue, regardless of severity
- a regression we shipped in the last release that a user can hit accidentally

Wait for the next weekly release otherwise. Including for things that feel
urgent. A degraded-but-working behaviour is not a hotfix even when it is
embarrassing.

The cost people underweight: a hotfix skips the soak the weekly release gets,
so it carries more risk per line than the same change would a few days later.
Twice we have hotfixed a fix for a hotfix.

#46 and #49 are degradation, not crashes, and both have workarounds
(restarting the session). By this rule they are not hotfix-urgent. They are on
the hotfix path anyway because they have been open long enough that "next
release" has stopped meaning anything.
