# Benchmark Results

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

This directory should receive results only after a baseline has:

1. been implemented as an explicit `Baseline` registered in
   `src/parm_bench/baselines.py`;
2. been reviewed as a faithful implementation of the intended comparison;
3. run against the declared response model and any required memory backend; and
4. recorded its configuration alongside its predictions.
