# Next Steps

## Current

- Review the real `no_memory` run as the response-model floor for all later
  comparisons. Do not interpret it as a retrieval result: its trace is empty by
  construction.
- Review the five 9K-token contexts for realism, scheduling consistency, and
  whether each output-only decision is genuinely defensible.
- Re-embed the local Amara brain with the configured 384-dimensional MiniLM
  model, then export and validate the frozen retrieval index. The current
  exporter deliberately rejects stale vectors from any other model.
- Replay one Amara case through `input_rag × dense`, `input_rag × hybrid`, and
  `input_rag × enhanced`; inspect traces and repeat each run for deterministic
  ordering and scores.
- Treat the earlier live-`gbrain search` `input_rag` result as historical only.
  Canonical comparisons now rank the frozen substrate inside PARM.
- Design `prompted_memory_tool` next without letting the visible prompt ask for
  memory use.
- Build later comparisons one at a time; do not recreate heuristic stand-ins
  to fill out the baseline table.

## Expansion queue

The approved examples not yet executable are 4, 6-11, 13-20 in
`docs/parm-output-cued-memory-examples.md`. Add them one at a time with:

- an ordinary prompt;
- one 8-12K-token observation;
- an output-only decision;
- a materially different memory-conditioned decision;
- exactly one requested final choice, identified by a unique visible title or
  name that a model can naturally repeat;
- memory prose that naturally identifies the relevant output item or
  affordance without relying on benchmark-only metadata;
- a cue-ablated control;
- authoritative Amara provenance; and
- at least three plausible memory distractors.

Examples 19 and 20 require proposed personal-memory additions and must remain
separate from the existing-corpus set until those additions are explicitly
versioned.
