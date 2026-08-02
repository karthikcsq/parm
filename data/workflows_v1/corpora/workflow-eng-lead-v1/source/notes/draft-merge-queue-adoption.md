# Draft - adopting a merge queue - NOT ADOPTED

Draft from 2026-04-05. We did not do this. Keeping the analysis because the
question comes back every few months.

The proposal was to put every change through a merge queue that rebases and
runs the full suite before landing, so `main` is never broken.

Why I did not take it:

- `main` breaks about once a month, and it is usually noticed and fixed inside
  twenty minutes. The queue would cost every change an extra full suite run,
  which is eleven minutes.
- At five people, conflicts between changes in flight are rare enough that the
  queue would mostly be serialising things that did not need serialising.

What would change my mind: team size past about ten, or a full suite fast
enough that the added latency is not felt. Neither is true yet.

See also the merge queue evaluation note, which reached the same conclusion
independently and with actual numbers.
