# Context window budget

Where the tokens go in a typical long session, measured on the monorepo
checkout:

- System prompt and tool definitions: about 6k, fixed.
- Workspace summary: 3k to 9k depending on repository size.
- Transcript: everything else, and it is almost all tool results.

Compaction targets the transcript because it is the only part that grows. The
current trigger fires at 85 percent of the window.

The bug in #49 is not the threshold. It is that the estimate feeding the
threshold is computed at the top of the turn, before the turn's tool results
are appended, so a turn that returns a 40k-token file overshoots by a whole
turn. Compute the estimate after appending and #49 mostly goes away.

I would still cap the transcript. The estimator being correct does not stop
the array from growing without bound.
