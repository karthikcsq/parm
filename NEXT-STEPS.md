# Next Steps

## Preserve the current evidence

- Keep the `*-v2-gpt-5-mini` artifacts as the immutable first expansion pass.
- Treat all revised-fixture and post-analysis runs as development results.
- Keep the no-memory fixture-fairness run as the floor for the repaired 54-case
  dataset.
- Version retrieval conditions, judgment prompts, response caches, and result
  namespaces together.

## Improve PARM without hiding failure modes

- Replace or tighten the semantic anchored-singleton path. The current
  expansion shows that a generic review note can select the same secondary
  listing in both a positive and its cue-ablated control.
- Develop a contrastive dense region-to-memory channel for semantic cues that
  BM25 cannot express, especially relationship, hiring, and proactive-priority
  cases.
- Revisit control construction when another surviving listing still has a
  legitimate relationship to the same memory, as in the human-factors case.
- Calibrate any new threshold on development cases and score it once on a new
  held-out batch.
- Keep retrieval admission metrics separate from downstream choice metrics.

## Add external validity

- Build a small paired end-to-end suite from natural agent or tool traces.
- Start with five human-reviewed case studies if a statistically meaningful
  suite is not yet available.
- Measure task success, causal memory lift, false intervention, faithfulness,
  privacy restraint, and calibration.
- Keep deterministic PARMBench as the mechanism and regression gate.
- Add an LLM judge only after measuring agreement with human reviewers and
  freezing the judge protocol.

See [Real-World Evaluation Strategy](docs/real-world-evaluation.md) for the
recommended design and [How to Construct a PARMBench Scenario](docs/benchmark-construction.md)
for the current controlled-case procedure.
