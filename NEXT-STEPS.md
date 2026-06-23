# PARM Benchmark Next Steps

## 1. Qualitatively Review The Dataset

Review 10 to 15 cases by hand before adding smarter baselines.

For each case, check:

- Does it have a real hidden-join moment?
- Is the trigger entity absent from the user goal and present in generated output or a tool response?
- Is the expected suggestion useful for the stated goal?
- Are distractors tempting but genuinely wrong?
- Does the case feel like a PARM case rather than a template exercise?

Use:

```powershell
$env:PYTHONPATH='src'
python -m parm_bench.cli inspect data/benchmark_v1 --case parm-v1-004
```

## 2. Add The First Non-Oracle PARM Baseline

Current `parm_oracle_monitor` proves the dataset is recoverable, but it uses
gold path access. The next baseline should use simple real mechanics:

- Extract trigger entities from output/tool text by label match.
- Traverse the graph up to 3 hops.
- Ignore `valid: false` edges.
- Score candidates with a simple goal/actionability heuristic.

Expected behavior:

- `input_rag` should still miss most cases.
- `prompted_memory_tool` should still mostly miss.
- Simple PARM should recover many cases, but not all.
- `parm_oracle_monitor` remains the ceiling.

## 3. Compare And Diagnose

Run the Core 4 baselines plus the new simple PARM baseline over the 50-case set.

Inspect:

- Cases simple PARM recovers.
- Cases only oracle PARM recovers.
- Cases where simple PARM emits useless suggestions.
- Whether failures come from entity extraction, graph traversal, or scoring.

## 4. Expand Baselines After The Dataset Feels Right

Only after the case set feels credible, add:

- FLARE-style uncertainty-triggered retrieval.
- A proactive-agent heuristic baseline.
- Relevance-only output-conditioned retrieval.
- Goal-grounded graph scoring without oracle path access.

## 5. Add Latency And Context Metrics Later

V1 does not call models or stream tokens, so latency claims are out of scope for
now. Add latency and context-overhead fields once model-backed baselines exist.
