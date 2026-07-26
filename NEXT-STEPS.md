# Next Steps

## Current

- Review the real `no_memory` run as the response-model floor for all later
  comparisons. Do not interpret it as a retrieval result: its trace is empty by
  construction.
- Review the five 9K-token contexts for realism, scheduling consistency, and
  whether each output-only decision is genuinely defensible.
- Treat the tracked schema-v2 Amara index as the canonical substrate. It holds
  600 chunk vectors and 2,498 deterministic sentence vectors at 512
  dimensions using `text-embedding-3-small`.
- Keep the retrieval-only PARM acceptance bar at five of five gold positives
  admitted and five of five cue-ablated controls empty. Recheck that table
  whenever the seed extractor, sentence segmenter, graph policy, or thresholds
  change.
- Use the tracked `prompted_memory_tool × enhanced` run as the naive-agent
  comparison. It never called memory on any positive or cue-ablated pilot case,
  so the PARM condition must demonstrate output-cue retrieval without relying
  on the response model to elect tool use.
- Treat the tracked `parm` run as the completed in-sample response-model gate:
  five of five positive decisions, five of five cue-ablated controls, and five
  of five memory-included ceilings are correct. Retrieval correctness and
  downstream decision correctness remain separate measurements.
- Build a held-out expansion set before changing the claim from pilot
  convergence to general retrieval quality. Keep thresholds and judgment
  instructions frozen while scoring that set.

## Expansion evaluation

Examples 4, 6-11, and 13-18 are now executable as the frozen `expansion`
split. Evaluate them without changing the pilot-tuned retrieval thresholds or
judgment instructions before inspecting per-case failures. Each includes:

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

Examples 19 and 20 still require proposed personal-memory additions and must
remain separate from the existing-corpus set until those additions are
explicitly versioned.
