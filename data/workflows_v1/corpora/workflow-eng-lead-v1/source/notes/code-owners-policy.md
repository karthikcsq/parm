# Code owners - 2026-04-11

`CODEOWNERS` is set up. What it means and does not mean.

Current ownership:

- `src/session/**` - Mei
- `src/auth/**`, `services/billing/**` - Dan
- `src/search/**` - me
- `docs/**` - Yuki
- infrastructure - Tomas

An owner's review is required to merge in their area. That is the whole
mechanism.

What it does not mean: that the owner must do the work, that they are the only
person who may change the code, or that they are accountable for every bug in
it. Ownership here is about who must have seen a change, not who is to blame
for it.

Deliberately not covered: `src/cli/**` and the top-level scripts. Everyone
touches those and requiring an owner would just add a hop.

Reviewing your own area's change still needs a second pair of eyes. Owners
cannot self-approve.
