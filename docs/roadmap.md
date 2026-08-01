# PARM Roadmap

This page contains unfinished work only. Completed experiment history belongs
in the result reports and decision log.

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

- Add corpus-scoped adapters and index isolation for multi-person datasets.
- Convert 30 PersonaMem-v2 development records before freezing the generation
  and acceptance procedure.
- Publish a sealed, persona-disjoint set of 500 human-audited PersonaMem-v2
  triplets.
- Build 50 to 100 paired end-to-end cases from LongMemEval-V2 web and
  enterprise trajectories.
- Measure task success, causal memory lift, false intervention, faithfulness,
  privacy restraint, and calibration.
- Keep deterministic PARMBench as the mechanism and regression gate.
- Add an LLM judge only after measuring agreement with human reviewers and
  freezing the judge protocol.

## Grow PARMBench Workflows

The pilot is one scenario in one environment. What it needs next, in order:

- Probe whether the semantic-judge rubric admits on confirmed
  non-applicability. Under an earlier wording of the pilot goal the judge
  admitted the telemetry hold on the documentation-only control, reasoning that
  a passage saying "no runtime code paths change" *satisfies* the freeze
  condition when it establishes the opposite. It stopped reproducing once the
  goal was reworded, so the rubric is not currently failing this way and the
  observation is unconfirmed rather than a known defect. It is cheap to test
  directly and worth knowing. Rubric `parm_pair_admission_v3` is frozen into
  published PersonaMem results, so any change needs a new version.
- Decide whether per-observation admission needs a trajectory-level budget.
  The judge admits at most one memory per observation, which reads as restraint
  on a single-turn case and accumulates over twenty-odd observations in a
  trajectory.
- Grow the corpus past the point where dumping all of it is a viable strategy.
  At 24 records a broad output-RAG policy admits essentially the whole history
  and still reaches the right decision, so the decision metric cannot separate
  it from selective retrieval on this scenario; only admission precision can.
  The corpus needs to be large enough that indiscriminate retrieval does not
  fit in context, which is the condition under which precision starts to buy
  decisions rather than just tidiness.
- Add the restaurant-expense, legal-review, and running-shoe conversions from
  the same MCPMark sources, each with its own environment adapter.
- Run more than one trajectory per case. This is the first thing to fix, not a
  refinement. The one pilot condition sampled twice flipped its decisive
  control result between samples, so a single trajectory cannot separate a
  policy difference from agent variance and every current workflow number is
  one trajectory.
- Tighten the workflow-role assertions or accept them as a competence signal
  rather than a target. They currently reward following the user's own written
  hotfix conventions, which memory-equipped systems can read and others cannot.
- Add a second model family. One model cannot show that the effect is not a
  quirk of how gpt-5-mini reads a pull request body.
- Decide whether an `mcpmark_live` adapter is worth building, and what it would
  buy over the local fixture beyond fidelity to a real API.

See [Real-World Evaluation Strategy](real-world-evaluation.md) for the
recommended design and [How to Construct a PARMBench Scenario](benchmark-construction.md)
for the current controlled-case procedure. The
[larger dataset survey](dataset-candidates.md) explains the source selection,
conversion rules, and retrieval-index changes.
