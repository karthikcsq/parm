# Benchmark Results

This directory contains tracked replay artifacts for canonical PARM benchmark
runs. Retrieval-backed runs should point at the tracked frozen index in
`data/retrieval-indexes/amara-life-v1`; enhanced runs should use the tracked
frozen expansion cache in `data/expansion-caches/amara-life-v1`.

## No-memory floor

The first real baseline run is published as:

- `no-memory-gpt-5-mini.jsonl`: ten prediction rows from five positive/control
  pairs;
- `no-memory-gpt-5-mini.metrics.json`: deterministic benchmark scores.

The run used requested model `gpt-5-mini`, resolved by OpenAI as
`gpt-5-mini-2025-08-07`, on 2026-06-30. It produced an identifiable answer for
all ten cases, changed 0/5 positive decisions toward the memory-conditioned
choice, and caused 0/5 false interventions on cue-ablated controls. All memory
admission fields are empty by construction.

This establishes the response-model floor, not a retrieval result.

## Input-RAG

The prompt-only retrieval baseline is published as:

- `input-rag-gpt-5-mini.jsonl`: ten prediction rows;
- `input-rag-gpt-5-mini.config.json`: run configuration; and
- `input-rag-gpt-5-mini.metrics.json`: deterministic benchmark scores.

The run used requested model `gpt-5-mini`, resolved as
`gpt-5-mini-2025-08-07`, GBrain `amara-life-v1`, and retrieval limit 5 on
2026-07-01. It retrieved zero memories for all ten original prompts. Therefore
it changed 0/5 positive decisions toward the memory-conditioned choice, caused
0/5 false interventions on controls, and had zero memory admission recall,
poison admission, and privacy overexposure. Nine responses retained an exactly
identifiable choice; one AI-news response omitted the `Lead Story —` prefix.

All five positive/control pairs produced identical empty retrieval traces. A
representative prompt also returned no results under GBrain's broader
`tokenmax` search mode, so the empty retrieval is not merely the active
conservative mode truncating a candidate list. This is the expected limitation
of input-only RAG for a benchmark whose decisive cue appears only in the later
observation. It remains a degenerate context-injection comparison because no
retrieved memory reached the response model.

This directory should receive results only after a baseline has:

1. been implemented as an explicit `Baseline` registered in
   `src/parm_bench/baselines.py`;
2. been reviewed as a faithful implementation of the intended comparison;
3. run against the declared response model and any required memory backend; and
4. recorded its configuration alongside its predictions.

The runner now writes this configuration automatically as a sibling
`<output-stem>.config.json`. Historical runs created before that behavior may
record equivalent configuration in this README.

## Retrieval-mode comparison

The dense, hybrid, and enhanced input-RAG artifacts use the same frozen
`amara-life-v1` retrieval index and top-k 5. The enhanced frozen replay uses
the tracked expansion cache populated from the same five distinct benchmark
prompts. The recorded scores show zero gold-memory admission in all three
modes for the current input-only RAG condition, which is the expected failure
mode when the decisive entity appears only in the later observation.
