# PARM Benchmark V1 Plan

## Summary

Build a Python CLI benchmark harness for PARM's first deliverable: a controlled synthetic cross-source-join benchmark. V1 should prove whether output/tool-response-conditioned entity watching can surface useful personal-memory joins that input-triggered memory systems miss.

## Benchmark Shape

- Create a 50-case synthetic seeded dataset.
- Each case includes a user goal, simulated assistant or tool output containing the trigger entity, a personal memory graph, distractors, an expected surfaced suggestion, and a gold rationale path.
- Headline success metric: surfaced join quality, measured by correct cross-source suggestions and low useless-intervention rate.

## Harness And Interfaces

- Use a Python CLI with JSON/JSONL artifacts.
- Commands:
  - `parm-bench validate <dataset_dir>`
  - `parm-bench run <dataset_dir> --baseline <name> --out results.jsonl`
  - `parm-bench score results.jsonl --gold <dataset_dir>`
- Required baselines:
  - `no_memory`
  - `input_rag`
  - `prompted_memory_tool`
  - `parm_oracle_monitor`

## Dataset Taxonomy

- Include warm-intro cases and two additional goal families: customer-discovery prioritization and opportunity/risk surfacing.
- Label trigger source, join depth, distractor type, and actionability for every case.
- Include semantic, graph-proximity, stale/invalid-edge, and goal-irrelevant distractors.

## Scoring

- Primary metrics: correct surfaced suggestion rate, useless intervention rate, precision, and recall.
- Secondary metrics: path correctness, trigger detection, memory-fact detection, and optional context overhead.
- A suggestion is correct only if it identifies the trigger-side entity, the memory-side fact, the useful action, and the gold graph path.

## Test Plan

- Unit-test validation failures for missing fields, duplicate case IDs, invalid graph references, and broken gold paths.
- Unit-test scorer behavior with hand-written predictions.
- Smoke-test each required baseline on a 3-case fixture.
- Run all Core 4 baselines on the full 50-case dataset.

## Assumptions And Defaults

- V1 uses synthetic seeded cases, not real user memory.
- V1 size target is 50 cases.
- Python is the implementation stack.
- Core 4 baselines are required; FLARE/proactive baselines are deferred extension work.
- The PARM baseline uses oracle linking/scoring so v1 validates the benchmark phenomenon before noisy implementation details.
