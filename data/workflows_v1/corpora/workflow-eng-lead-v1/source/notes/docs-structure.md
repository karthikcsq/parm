# Where documentation goes

`docs/` holds explanations that outlive a pull request. Anything that answers
"why is it like this" belongs there.

Naming: uppercase with underscores for standalone analyses
(`MEMORY_OPTIMIZATION.md`, `INDEXING.md`), lowercase with hyphens for reference
material (`cli-reference.md`).

The CLI reference is generated in part and hand-written in part. Do not
regenerate the whole file; the hand-written sections have no source.

README stays short. It is the first thing a new user reads and every paragraph
we add to it makes the quickstart harder to find.

CHANGELOG entries are written by whoever ships the release, not by whoever
wrote the change. One voice reads better.
