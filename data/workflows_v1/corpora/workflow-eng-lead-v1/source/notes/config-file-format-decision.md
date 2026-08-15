# Decision log - 2026-02-06 - config stays JSON with comments

Present: me, Dan, Yuki.

Long argument about TOML versus YAML versus what we have.

Decided: keep JSONC. Not because it is the best format but because every
config file our users already have is JSONC, and a migration would mean
supporting both formats forever while gaining nothing a user can feel.

Positions on the record:

- Dan wanted TOML, correctly noting our schema is flat and TOML is nicer for
  flat.
- Yuki wanted YAML because the docs examples would be shorter.
- I care most that there is exactly one format, and the cheapest way to have
  one format is to keep the one that exists.

If we ever do change, the trigger should be a feature that JSONC genuinely
cannot express, not aesthetics. We have not hit one in two years.
