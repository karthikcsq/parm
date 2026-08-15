# Feature flags: the cleanup rule - 2026-02-17

We have 34 flags. Eleven of them have been at 100% for over six months. That is
not flag-driven development, that is dead branches with extra steps.

Rule: every flag gets an owner and an expiry date when it is created. At
expiry, either the flag is removed and the behaviour becomes permanent, or the
feature is removed. Extending is allowed once, with a reason.

Practically:

- Flag names go in the flag registry with owner and expiry. Not optional.
- A monthly job opens an issue for every expired flag, assigned to the owner.
- Removing a flag means removing the losing branch, not just the check.

The eleven long-lived ones: Dan is taking six, I have five. Both done by end of
March.

The cost of leaving them is not the check, it is that every one doubles the
states a reader has to hold in their head, and after a while nobody knows which
combination is actually running in production.
