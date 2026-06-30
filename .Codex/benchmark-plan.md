# PARM Benchmark V1 Plan

## Summary

Build a Python CLI benchmark harness for PARM's first deliverable: a controlled synthetic output/tool-conditioned memory benchmark. V1 should prove whether a system can identify memory-relevant cues inside large noisy agent outputs or tool responses, retrieve useful personal memories, and avoid spurious correlations that input-triggered memory systems and naive output-RAG miss.

## Benchmark Shape

- Create a 50-case synthetic seeded dataset.
- Each case includes a user goal, simulated assistant or tool output containing
  many possible cue candidates, a simple personal memory store/graph,
  distractors, and an expected surfaced suggestion.
- Headline success metric: correct memory intervention with low
  useless-intervention rate under noisy output/tool context.

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

- Include multiple everyday and work goal families where output/tool context can
  introduce cues the agent would not search from the initial prompt alone.
- Label trigger source, distractor type, actionability, and whether the case
  includes optional graph-path diagnostics.
- Include semantic, graph-proximity, stale/invalid-edge, and goal-irrelevant distractors.

## Scoring

- Primary metrics: correct surfaced suggestion rate, useless intervention rate, precision, and recall.
- Secondary metrics: cue detection, memory-fact detection, affected-item
  detection, optional path correctness for trace-emitting systems, and optional
  context overhead.
- A suggestion is correct if it identifies the trigger-side cue, the memory-side
  fact, and the useful action. Gold paths are diagnostic, not required for
  natural responses.

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
