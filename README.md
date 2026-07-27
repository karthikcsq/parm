# PARM

PARM evaluates output cue-triggered personal memory. An ordinary prompt is
followed by a large tool result or agent output. A small incidental cue inside
that output makes a stored memory newly useful, and the memory should improve
the final decision without changing the cue-ablated control.

PARMBench currently contains 18 scenarios and 54 cases:

- 18 positive cases with a decision-relevant cue;
- 18 cue-ablated controls; and
- 18 memory-included ceilings.

The comparison conditions include no memory, input RAG, naive output RAG,
all-entity output RAG, a prompted memory-tool agent, and PARM convergence
retrieval. Every condition uses the same frozen `amara-life-v1` memory
substrate.

## Quick start

Use the Anaconda interpreter in this checkout:

```powershell
$python = 'C:\Users\karth\anaconda3\python.exe'
& $python -m pip install -e .
& $python -m spacy download en_core_web_sm

parm-bench validate data\benchmark_v1
parm-bench inspect data\benchmark_v1 `
  --case parm-amara-conference-agenda-positive

$env:PYTHONPATH = 'src'
& $python -m unittest discover -s tests
```

Run and score PARM:

```powershell
parm-bench run data\benchmark_v1 `
  --baseline parm `
  --retrieval-index data\retrieval-indexes\amara-life-v1 `
  --retrieval-limit 5 `
  --response-cache data\response-caches\amara-life-v4\parm `
  --response-policy populate `
  --model gpt-5-mini `
  --out data\benchmark-results\parm-v4-gpt-5-mini.jsonl

parm-bench score `
  data\benchmark-results\parm-v4-gpt-5-mini.jsonl `
  --gold data\benchmark_v1 `
  --out data\benchmark-results\parm-v4-gpt-5-mini.metrics.json
```

See [How to Run PARMBench](docs/running-parmbench.md) for replay caches,
comparison commands, the browser workbench, and troubleshooting.

## How the system fits together

GBrain prepares a neutral memory substrate: pages, chunks, embeddings, and
links. Canonical runs use its tracked frozen export and never call GBrain
search.

Each baseline controls when retrieval happens, what becomes a query, and which
memories are admitted. PARM splits the later observation into listing regions,
combines store-backed entity links, task-conditioned semantic evidence, and
contrastive durable-note matching, then may admit one memory, several, or
none. The response model receives the triggering listing beside focused memory
evidence.

Read [Architecture](docs/architecture.md) for the complete waterfall and
module map.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [How to Run PARMBench](docs/running-parmbench.md)
- [How to Construct a Scenario](docs/benchmark-construction.md)
- [Evaluation Contract](docs/benchmark-evaluation.md)
- [Real-World Evaluation Strategy](docs/real-world-evaluation.md)
- [Output-Cued Memory Examples](docs/parm-output-cued-memory-examples.md)
- [Expanded Benchmark First Pass](docs/results/benchmark-expansion-first-pass.md)
- [Benchmark Result Artifacts](data/benchmark-results/README.md)
- [GBrain and Amara Setup](docs/gbrain-amara-local-setup.md)
- [Research Proposal](parm-proposal.md)
- [Decision Log](DECISIONS.md)

## Repository map

```text
src/parm_bench/               benchmark package and CLI
tests/                        unit and CLI smoke tests
scripts/                      dataset and retrieval-artifact builders
data/benchmark_v1/            54 executable cases and large contexts
data/retrieval-indexes/       frozen neutral memory substrate
data/expansion-caches/        frozen enhanced-mode query expansions
data/response-caches/         replayable model calls
data/benchmark-results/       predictions, configs, and metrics
docs/                         current guides, results, and history
```

The root `.env` is ignored. Start from `.env.example` and set
`OPENAI_API_KEY` for live model-backed runs.
