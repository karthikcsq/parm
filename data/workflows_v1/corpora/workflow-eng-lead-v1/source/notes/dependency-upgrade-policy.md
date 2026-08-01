# Dependency upgrades

Patch and minor upgrades go in on the weekly bot pull request, batched, one
merge. I do not read them individually and I do not expect anyone else to.

Major upgrades get their own pull request with a paragraph on what changed.

Two categories come out of the batch and get looked at by a person:

- Anything in the build or release path. A compromised build dependency is the
  whole product.
- Anything that changes what leaves the user's machine. A client SDK bump can
  silently start sending a new field.

The bot is configured to exclude both categories from the batch. If you see one
in there, the configuration has drifted and I want to know.
