# How to Run PARMBench

This guide validates the dataset, runs one retrieval condition, scores it, and
replays the exact response calls.

## Prerequisites

The supported interpreter in this checkout is:

```text
C:\Users\karth\anaconda3\python.exe
```

It has the required OpenAI, NumPy, and tiktoken dependencies. The bare
`python` on `PATH` may resolve to an incompatible MSYS2 build.

Install the package and spaCy model once:

```powershell
$python = 'C:\Users\karth\anaconda3\python.exe'
& $python -m pip install -e .
& $python -m spacy download en_core_web_sm
```

Copy `.env.example` to the ignored root `.env` and set `OPENAI_API_KEY` for
live model-backed runs.

## Validate and inspect the dataset

```powershell
parm-bench validate data\benchmark_v1
parm-bench inspect data\benchmark_v1 `
  --case parm-amara-conference-agenda-positive
```

Validation should report 54 cases.

## Run PARM

Replay the published V5 response requests without making new response-model
calls:

```powershell
$replay = Join-Path $env:TEMP 'parm-v5-replay.jsonl'
parm-bench run data\benchmark_v1 `
  --baseline parm `
  --retrieval-index data\retrieval-indexes\amara-life-v1 `
  --retrieval-limit 5 `
  --response-cache data\response-caches\amara-life-v5\parm `
  --response-policy frozen `
  --model gpt-5-mini `
  --out $replay
```

The command writes predictions plus a sibling config sidecar in the temporary
directory. Compare it with the tracked
`data\benchmark-results\parm-v5-gpt-5-mini.jsonl`. A live development run uses
`--response-policy populate` and a new cache namespace; do not add new requests
to the published V5 cache.

## Score the run

```powershell
$replay = Join-Path $env:TEMP 'parm-v5-replay.jsonl'
$metrics = Join-Path $env:TEMP 'parm-v5-replay.metrics.json'
parm-bench score `
  $replay `
  --gold data\benchmark_v1 `
  --out $metrics
```

Read primary accuracy as positive decisions plus cue-ablated controls. Read
retrieval precision and recall separately. A system that changes every
positive and every control has zero net primary advantage.

## Replay responses from cache

The V5 command above is already a frozen replay. For a new experiment, populate
a new cache first, then rerun the identical request with only the policy
changed from `populate` to `frozen`.

```powershell
$replay = Join-Path $env:TEMP 'parm-v5-second-replay.jsonl'
parm-bench run data\benchmark_v1 `
  --baseline parm `
  --retrieval-index data\retrieval-indexes\amara-life-v1 `
  --retrieval-limit 5 `
  --response-cache data\response-caches\amara-life-v5\parm `
  --response-policy frozen `
  --model gpt-5-mini `
  --out $replay
```

A frozen cache miss is an error. That is intentional: official replay must not
silently make a new nondeterministic response-model call. PARM still embeds
runtime cue queries, so the replay requires `OPENAI_API_KEY` unless query
embeddings are separately cached.

## Run comparison conditions

Enhanced input RAG:

```powershell
parm-bench run data\benchmark_v1 `
  --baseline input_rag `
  --retrieval-mode enhanced `
  --retrieval-index data\retrieval-indexes\amara-life-v1 `
  --retrieval-limit 5 `
  --expansion-cache data\expansion-caches\amara-life-v2-input-rag-enhanced `
  --expansion-policy frozen `
  --response-cache data\response-caches\amara-life-v3\input-rag-enhanced `
  --response-policy frozen `
  --model gpt-5-mini `
  --out data\benchmark-results\input-rag-enhanced-replay.jsonl
```

Naive output RAG:

```powershell
parm-bench run data\benchmark_v1 `
  --baseline naive_output_rag `
  --output-rag-flow tool_then_model_output `
  --retrieval-mode hybrid `
  --retrieval-index data\retrieval-indexes\amara-life-v1 `
  --retrieval-limit 5 `
  --response-cache data\response-caches\amara-life-v3\naive-output-tool-then-model-hybrid `
  --response-policy frozen `
  --model gpt-5-mini `
  --out data\benchmark-results\naive-output-rag-replay.jsonl
```

Naive memory-tool agent:

```powershell
parm-bench run data\benchmark_v1 `
  --baseline prompted_memory_tool `
  --retrieval-mode hybrid `
  --retrieval-index data\retrieval-indexes\amara-life-v1 `
  --retrieval-limit 5 `
  --response-cache data\response-caches\amara-life-v3\prompted-memory-tool-hybrid `
  --response-policy frozen `
  --model gpt-5-mini `
  --out data\benchmark-results\prompted-memory-tool-replay.jsonl
```

## Inspect retrieval in the browser

```powershell
parm-bench serve-workbench `
  --retrieval-index data\retrieval-indexes\amara-life-v1 `
  --expansion-cache data\expansion-caches\amara-life-v2-input-rag-enhanced
```

The workbench shows the final choice, expected decision, admitted sources,
selected chunks, ranking diagnostics, and complete trace JSON.

## Verification

Run all tests from the repository root:

```powershell
$env:PYTHONPATH = 'src'
& 'C:\Users\karth\anaconda3\python.exe' -m unittest discover -s tests
parm-bench validate data\benchmark_v1
```

## Troubleshooting

### Import errors for OpenAI, NumPy, or tiktoken

The wrong Python executable is active. Use the Anaconda path shown above or
activate that environment before running commands.

### Frozen expansion-cache miss

The query or expansion prompt differs from the cached experiment. Use the
declared cache for an exact replay. Use `--expansion-policy populate` only when
intentionally building a new versioned cache.

### Frozen response-cache miss

The model request changed. Instructions, observation text, memory context, and
model name are all part of the key. Populate a new cache namespace and retain
the old namespace with the old result artifacts.

### Retrieval index validation error

Do not edit files inside a frozen index by hand. Rebuild it through the
documented exporter and review the manifest and hash changes together.
