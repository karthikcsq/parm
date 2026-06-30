# PARM Benchmark V1 Guide

This guide explains what the benchmark components do, how data flows through
the harness, and how to run or extend it.

## What This Benchmark Measures

PARM V1 isolates a specific memory-retrieval failure mode:

1. The user asks for a concrete task with enough context for a generic answer.
2. If the task requires local, current, private, or otherwise non-obvious
   option discovery, a tool/local-context response gathers that context by
   comparing many plausible options. Generated-output-only cue cases are
   appropriate only when the cue is something a capable base model could
   plausibly supply from general knowledge.
3. The generated/tool context contains many possible entities, sentences, and
   options, most of which would produce spurious or low-value memory matches.
4. One or a few visible cues connect to latent personal-memory facts that
   change, qualify, or improve the recommendation.
5. Input-conditioned memory retrieval misses the connection because the cue was
   absent from the original user prompt, and naive output-RAG is too noisy.

The benchmark asks whether a system can proactively select the right
output/tool-side cues, retrieve useful memory, and avoid flooding the user with
useless interventions. Multi-hop graph paths are optional diagnostics; they are
not the core V1 challenge.

For the active/passive recall framing behind this benchmark, see
`.Codex/active-passive-recall.md`.

## Repository Components

### `pyproject.toml`

Defines the Python package metadata and CLI entrypoint:

```text
parm-bench = "parm_bench.cli:main"
```

The package has no external runtime dependencies. For local development, set
`PYTHONPATH=src` and run the module directly.

### `src/parm_bench/cli.py`

The command-line interface. It wires together dataset loading, validation,
baseline execution, and scoring.

Available commands:

```powershell
python -m parm_bench.cli generate <dataset_dir> --count 50
python -m parm_bench.cli validate <dataset_dir>
python -m parm_bench.cli inspect <dataset_dir> --case parm-v1-004
python -m parm_bench.cli run <dataset_dir> --baseline <name> --out results.jsonl
python -m parm_bench.cli score results.jsonl --gold <dataset_dir>
```

The CLI intentionally uses JSONL for predictions so every case result is easy
to inspect, diff, stream, or post-process.

### `src/parm_bench/synthetic.py`

Generates the controlled 50-case synthetic dataset.

Each generated case varies across:

- `goal_family`: introductions, customer discovery, opportunity/risk, travel
  planning, health admin, learning/research, event planning, hiring, personal
  finance, and home operations.
- `trigger_source`: where the cue first becomes visible. Tool-response cases
  are used when the option set is local, current, private, or synthetic.
  Generated-output cases are used when the cue is plausibly available from
  general model knowledge.
- `join_depth`: optional 1-hop, 2-hop, or 3-hop graph diagnostics. This is not
  the headline difficulty.
- `distractor_type`: semantic, graph proximity, stale/invalid edge, or
  goal-irrelevant.
- `actionability`: intro, prioritization, warning, follow-up question, or
  enrichment.

The generator creates synthetic selected items, bridge/cue entities, people,
personal-memory facts, distractors, and gold graph paths across multiple task
domains. The user prompt is intentionally answerable without memory. The key
design choice is that the trigger cue is introduced later by assistant output or
tool content inside a noisy context, while the memory side supplies
user-specific edge cases, warnings, introductions, or enrichments. The graph
should remain a simple memory-store scaffold, not a perfect reasoning
substrate.

### `data/benchmark_v1/cases.jsonl`

The canonical generated V1 dataset.

Each line is one benchmark case. Important fields:

- `case_id`: stable case identifier.
- `user_goal`: the original user request.
- `assistant_output`: simulated generated content.
- `tool_response`: simulated external tool result.
- `context_requirement`: whether the case requires tool/local-context discovery
  or can rely on general model knowledge.
- `graph.nodes`: synthetic personal-memory graph nodes.
- `graph.edges`: graph relations, including `valid: false` for stale/invalid
  distractors.
- `trigger_entity`: the output/tool-side cue PARM should notice.
- `expected.suggestion`: the gold surfaced suggestion.
- `expected.gold_path`: optional diagnostic graph path from trigger entity to
  memory fact for oracle/debug traces.
- `expected.decision_effect`: the item-level decision impact, such as avoiding
  or reconsidering an otherwise attractive item.
- `expected.memory_fact_node_id`: the latent memory fact that makes the
  suggestion useful.
- `expected.action_keywords`: terms the scorer uses to confirm actionability.

### `src/parm_bench/dataset.py`

Handles dataset I/O and validation.

Validation checks:

- Required fields are present.
- Case IDs are unique.
- Taxonomy labels are valid.
- Graph edges reference known nodes.
- The trigger entity and memory fact exist in the graph.
- When present, the gold path starts at the trigger entity.
- When present, the gold path ends at the expected memory fact.
- When present, every step in the gold path uses a valid graph edge.

If validation fails, the CLI prints case-specific errors and exits nonzero.

### `src/parm_bench/baselines.py`

Implements the Core 4 V1 baselines.

`no_memory`

Returns no suggestions. This establishes the floor: without memory access, the
latent memory intervention should not be recoverable.

`input_rag`

Only retrieves from the initial user prompt. Because V1 cases keep the trigger
entity out of the prompt, this baseline should miss the join. If a future case
does include the trigger in the prompt, this baseline can recover it via the
oracle suggestion helper.

`prompted_memory_tool`

Models the objection, "Just tell the model to search memory when relevant."
The current V1 implementation is deliberately conservative: it only calls into
input retrieval when the prompt itself contains obvious memory-search cues.
This captures the structural problem PARM targets: the relevance often becomes
visible only after the lookup.

Future baselines should include naive output-RAG and salience-filtered
output-RAG. Those are the most important comparisons for the updated V1 target:
they test whether cue selection and spurious-correlation control add value over
searching every output entity or sentence.

`parm_oracle_monitor`

Watches output/tool entities and uses gold linking/scoring to emit the expected
suggestion. This is not a production PARM implementation. Its job is to prove
the dataset contains recoverable signal before later work adds noisy sentence
segmentation, entity extraction, hybrid memory search, reranking, FLARE-style
triggers, or proactive-agent baselines.

### `src/parm_bench/scoring.py`

Scores predictions against the gold dataset.

A suggestion is correct only when it satisfies all of these:

- Identifies the trigger entity.
- Identifies the memory-side fact.
- Contains an action keyword tied to the user goal.

Primary metrics:

- `correct_surfaced_suggestion_rate`: fraction of cases with at least one
  correct suggestion.
- `precision`: correct suggestions divided by all suggestions.
- `recall`: same as case-level recovery for this one-gold-suggestion V1 setup.
- `useless_intervention_rate`: non-correct suggestions divided by all
  suggestions.

Secondary metrics:

- `path_correct_rate`
- `trigger_found_rate`
- `memory_fact_found_rate`
- `total_suggestions`

The scorer also emits per-case rows so failures can be inspected without
rerunning the benchmark.

### `tests/`

The test suite covers:

- Generated dataset validity.
- Missing required fields.
- Duplicate case IDs.
- Invalid graph references.
- Broken gold paths.
- Scorer behavior for correct, empty, and wrong-path predictions.
- CLI smoke tests for all Core 4 baselines on a 3-case fixture.

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover
```

### `.Codex/benchmark-plan.md`

The implementation plan that this V1 harness follows.

### `.Codex/fairness-checklist.md`

Review checklist for adding or auditing cases. Use it to make sure baseline
failures are caused by the intended benchmark mechanics, not accidental tuning
against one implementation.

### `.Codex/benchmark-results/`

Contains generated Core 4 baseline predictions and metrics for the canonical
50-case dataset.

The current sanity-check result:

| Baseline | Expected V1 Behavior |
| --- | --- |
| `no_memory` | Misses all interventions. |
| `input_rag` | Misses interventions because triggers are absent from prompts. |
| `prompted_memory_tool` | Misses interventions because relevance is not visible before lookup. |
| `parm_oracle_monitor` | Recovers all interventions using gold output-conditioned monitoring. |

## End-To-End Data Flow

1. `generate` creates synthetic benchmark cases.
2. `validate` checks schema, taxonomy, graph integrity, and gold paths.
3. `inspect` pretty-prints individual cases for qualitative review.
4. `run` loads cases and applies one baseline to each case.
5. The baseline writes one JSONL prediction row per case.
6. `score` compares predictions against `expected`.
7. Metrics are printed or written to JSON.

In short:

```text
cases.jsonl -> validate -> baseline predictions.jsonl -> score -> metrics.json
```

## How To Run The Full V1 Benchmark

From the repo root:

```powershell
$env:PYTHONPATH='src'
python -m parm_bench.cli validate data/benchmark_v1
python -m parm_bench.cli inspect data/benchmark_v1 --case parm-v1-004

foreach ($b in @('no_memory','input_rag','prompted_memory_tool','parm_oracle_monitor')) {
  python -m parm_bench.cli run data/benchmark_v1 --baseline $b --out ".Codex/benchmark-results/$b.jsonl"
  python -m parm_bench.cli score ".Codex/benchmark-results/$b.jsonl" --gold data/benchmark_v1 --out ".Codex/benchmark-results/$b.metrics.json"
}

python -m unittest discover
```

## Prediction JSONL Format

Each baseline writes rows shaped like:

```json
{
  "case_id": "parm-v1-001",
  "baseline": "parm_oracle_monitor",
  "suggestions": [
    {
      "text": "Acme 100 connects to Professor Iyer through High Alpha; ask Professor Iyer for an intro before cold outreach.",
      "trigger_entity_id": "n_trigger",
      "memory_fact_node_id": "n_memory",
      "path": ["n_trigger", "n_memory"]
    }
  ],
  "notes": "Oracle monitor watches output/tool entities and uses gold graph linking."
}
```

Baselines may emit zero suggestions. Future baselines can emit multiple
suggestions; the scorer will count non-correct suggestions as useless
interventions.

## How To Add A New Baseline

1. Add a prediction function to `src/parm_bench/baselines.py`.
2. Register it in `available_baselines()`.
3. Return the standard prediction shape with `case_id`, `suggestions`, and
   `notes`.
4. Add or update smoke tests if the baseline is required for V1.
5. Run the full benchmark and store results under `.Codex/benchmark-results/`.

Future likely baselines:

- Naive output-RAG over every extracted output/tool entity or sentence.
- Salience-filtered output-RAG over selected cue candidates.
- FLARE-style uncertainty-triggered generation-time retrieval.
- Proactive-agent heuristic over the same graph store.
- Relevance-only output-conditioned retrieval.

## How To Add Or Regenerate Cases

Regenerate a synthetic dataset:

```powershell
$env:PYTHONPATH='src'
python -m parm_bench.cli generate data/benchmark_v1 --count 50
python -m parm_bench.cli validate data/benchmark_v1
```

When editing cases manually, rerun validation and use
`.Codex/fairness-checklist.md`.

For V1, keep the canonical target at 50 cases unless the benchmark plan is
updated. The balance does not need to be perfectly even, but each taxonomy axis
should remain represented.

## Design Limitations

- The PARM baseline is oracle-assisted; it is a recoverability check, not a real
  cue selector, memory searcher, or reranker.
- The synthetic data is controlled and intentionally repetitive. That is useful
  for causal testing, but later versions need more realistic curated cases.
- `prompted_memory_tool` is a heuristic stand-in, not a full model-agent
  implementation.
- There is no latency measurement yet because V1 does not call models or stream
  tokens.
- The current scorer expects one gold suggestion per case.

## Recommended Next Steps

- Add naive output-RAG and salience-filtered output-RAG baselines.
- Replace oracle monitor pieces incrementally: sentence/item segmentation,
  entity extraction, hybrid memory search, cue/actionability reranking, then
  optional graph diagnostics.
- Add the FLARE-style and proactive-agent baselines.
- Add model-call adapters while preserving the same JSONL prediction contract.
- Add latency and context-overhead fields once streaming/model baselines exist.
- Expand the dataset with realistic curated cases after the synthetic harness is
  stable.
