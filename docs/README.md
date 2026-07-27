# PARM Documentation

Start with the document that matches the question you are trying to answer.

## Understand the system

- [Architecture](architecture.md): the end-to-end data flow, retrieval
  conditions, PARM admission channels, caches, and scoring boundary.
- [Research proposal](../parm-proposal.md): the original motivation, claim, and
  research hypotheses.
- [Decision log](../DECISIONS.md): dated decisions that changed the benchmark
  or retrieval design.

## Build and run the benchmark

- [Run PARMBench](running-parmbench.md): install, validate, run, score, replay,
  and inspect a case.
- [Construct benchmark cases](benchmark-construction.md): the procedure for
  turning an approved memory/cue example into a positive, cue-ablated control,
  and memory-included ceiling.
- [Evaluation contract](benchmark-evaluation.md): deterministic metrics,
  correctness, and the failure taxonomy.
- [Real-world evaluation strategy](real-world-evaluation.md): what the current
  benchmark proves, where it stops, and how to add a calibrated end-to-end
  judge layer.
- [Example catalog](parm-output-cued-memory-examples.md): the 20 approved
  output-cued memory scenarios and their source relationships.
- [Rebuild the Amara substrate](gbrain-amara-local-setup.md): prepare and freeze
  the neutral GBrain-derived retrieval index.

## Read results

- [Expanded benchmark first pass](results/benchmark-expansion-first-pass.md):
  the immutable first score on the 13-scenario expansion, before tuning on it.
- [Benchmark artifacts](../data/benchmark-results/README.md): names and
  interpretation of tracked prediction, configuration, and metric files.

## Historical design records

These files explain how the implemented design was reached. They are useful
for archaeology, but they are not the current reference.

- [Convergence retrieval V1 design](history/parm-convergence-retrieval-v1.md)
- [Retrieval-mode axis implementation plan](history/retrieval-mode-axis-plan.md)

The current source of truth is [Architecture](architecture.md), the CLI, and
the tests.
