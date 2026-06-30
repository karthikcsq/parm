# Next Steps

## Current

- Design and implement the first baseline end to end. Register it only after its
  agent behavior, retrieval policy, model configuration, and trace contract
  have been explicitly reviewed.
- Review the five 9K-token contexts for realism, scheduling consistency, and
  whether each output-only decision is genuinely defensible.
- Add a model response adapter that preserves the model's natural-language
  answer. The scorer should continue recognizing the visible option title or
  name directly; optional source traces remain diagnostic only.
- Build later comparisons one at a time; do not recreate heuristic stand-ins to
  fill out the baseline table.

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
