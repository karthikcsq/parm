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

Suggested review set:

| Case | Why Review It |
| --- | --- |
| `parm-v1-003` | Tool-response warning, 3-hop join, stale/invalid distractor. |
| `parm-v1-004` | Tool-response travel warning, 1-hop join, goal-irrelevant distractor. |
| `parm-v1-006` | Tool-response enrichment case, 3-hop join, graph-proximity distractor. |
| `parm-v1-009` | Tool-response finance warning, 3-hop join, semantic distractor. |
| `parm-v1-011` | Generated-output general-travel warning, 2-hop join, stale/invalid distractor. |
| `parm-v1-013` | Generated-output fitness warning, 1-hop join, semantic distractor. |
| `parm-v1-016` | Tool-response intro case, 1-hop join, goal-irrelevant distractor. |
| `parm-v1-018` | Tool-response opportunity-risk warning, 3-hop join, graph-proximity distractor. |
| `parm-v1-021` | Tool-response learning/research enrichment, 3-hop join, semantic distractor. |
| `parm-v1-024` | Tool-response finance warning, 3-hop join, goal-irrelevant distractor. |
| `parm-v1-028` | Generated-output fitness warning, 1-hop join, goal-irrelevant distractor. |
| `parm-v1-033` | Tool-response opportunity-risk warning, 3-hop join, semantic distractor. |
| `parm-v1-039` | Tool-response finance warning, 3-hop join, stale/invalid distractor. |
| `parm-v1-043` | Generated-output fitness warning, 1-hop join, stale/invalid distractor. |
| `parm-v1-048` | Tool-response opportunity-risk warning, 3-hop join, goal-irrelevant distractor. |

## 2. Add The First Non-Oracle PARM Baseline

Before implementing real PARM behavior, make the Core 3 floor baselines
explicit and easy to compare:

- `no_memory`: emits no memory interventions.
- `input_rag`: retrieves only from the initial user goal.
- `prompted_memory_tool`: calls memory only when the initial prompt obviously
  asks for memory, contacts, relationships, or introductions.

These are intentionally weak. Their job is to prove the benchmark is not
solvable by ordinary input-conditioned retrieval or by a generic "search memory
when relevant" prompt.

Expected behavior:

- `no_memory` abstains on all cases.
- `input_rag` misses because V1 keeps the trigger entity out of the user goal.
- `prompted_memory_tool` mostly abstains because relevance appears only after
  assistant output or tool context.

Run and refresh the Core 3 metrics before adding real PARM:

```powershell
$env:PYTHONPATH='src'
foreach ($b in @('no_memory','input_rag','prompted_memory_tool')) {
  python -m parm_bench.cli run data/benchmark_v1 --baseline $b --out ".Codex/benchmark-results/$b.jsonl"
  python -m parm_bench.cli score ".Codex/benchmark-results/$b.jsonl" --gold data/benchmark_v1 --out ".Codex/benchmark-results/$b.metrics.json"
}
```

## 3. Add The First Real PARM Baseline

Current `parm_oracle_monitor` proves the dataset is recoverable, but it uses
gold path access. The first real PARM baseline should use simple deterministic
mechanics:

- Extract trigger entities from output/tool text by label match.
- Traverse the graph up to 3 hops.
- Ignore `valid: false` edges.
- Score candidates with a simple goal/actionability heuristic.
- Emit natural suggestion text plus structured trace fields for diagnostics.

Expected behavior:

- Core 3 floor baselines should still miss or abstain.
- Simple PARM should recover many cases, but not all.
- `parm_oracle_monitor` remains the ceiling.

## 4. Compare And Diagnose

Run Core 3, simple PARM, and `parm_oracle_monitor` over the 50-case set.

Inspect:

- Cases simple PARM recovers.
- Cases only oracle PARM recovers.
- Cases where simple PARM emits useless suggestions.
- Whether failures come from cue detection, memory fact recovery, graph
  traversal, ranking/selection, actionability, decision effect, or scoring.

## 5. Expand Baselines After The Dataset Feels Right

Only after the case set feels credible, add:

- Relevance-only output-conditioned retrieval.
- Goal-grounded graph scoring without oracle path access.
- FLARE-style uncertainty-triggered retrieval.
- A proactive-agent heuristic baseline.

## 6. Add Latency And Context Metrics Later

V1 does not call models or stream tokens, so latency claims are out of scope for
now. Add latency and context-overhead fields once model-backed baselines exist.
