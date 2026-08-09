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

## Controlling cost during development

Two levers keep live spend down while iterating; neither changes what a run
produces.

Set `OPENAI_SERVICE_TIER=flex` in `.env` to send every eligible gpt-5-family
call at OpenAI's discounted flex tier (about half price, higher latency,
occasional queuing). The tier is deliberately excluded from every cache key,
so entries populated under flex replay identically without it. Embedding
calls are unaffected because the API does not tier them.

Pass `--limit N` to `parm-bench run` to restrict a development iteration to
the first N base scenarios. The cut is scenario-level, so triplets stay whole
and paired metrics remain meaningful on the subset. Full, reportable runs
must omit `--limit`.

Replays are always free: any run against a populated response, expansion, or
admission cache with a `frozen` policy makes no live calls at all.

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

The semantic-pair path adds a frozen admission cache. It is the workflow
default, so the replay that exercises it is a workflow run:

```powershell
& $python -m parm_bench.cli workflow run data\workflows_v1 `
  --policy parm `
  --retrieval-index data\retrieval-indexes\workflow-eng-lead-v1-100 `
  --parm-admission-cache data\workflow-caches\parm-admission-tier-100 `
  --parm-admission-policy frozen `
  --trajectory-cache data\workflow-caches\trajectories-gpt5mini `
  --trajectory-policy frozen `
  --model gpt-5-mini `
  --out data\benchmark-results\workflows-v1-replay.jsonl
```

Use `--parm-admission-policy populate` only to construct a new versioned judge
cache. The final choice cache is separate because it contains the downstream
response after the admitted region-memory pair is injected.

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

## Run PARMBench Workflows

The workflow suite has its own subcommands and its own dataset. It shares the
interpreter, the `.env`, and the retrieval index format.

```powershell
parm-bench workflow validate data\workflows_v1
parm-bench workflow inspect data\workflows_v1 `
  --case parm-workflow-github-telemetry-hotfix-positive
```

Run one condition. Every case rebuilds its environment from the tracked
fixture, so `--workers` is safe:

```powershell
parm-bench workflow run data\workflows_v1 `
  --policy parm `
  --retrieval-index data\retrieval-indexes\workflow-eng-lead-v1 `
  --parm-admission-cache data\workflow-caches\parm-admission-v1 `
  --trajectory-cache data\workflow-caches\trajectories-gpt5mini `
  --model gpt-5-mini `
  --workers 3 `
  --out data\benchmark-results\workflows-v1\parm.jsonl

parm-bench workflow score data\benchmark-results\workflows-v1\parm.jsonl `
  --gold data\workflows_v1 `
  --out data\benchmark-results\workflows-v1\parm.metrics.json
```

`--trajectory-cache` freezes each agent turn keyed by the whole conversation so
far, which is what makes a multi-step comparison replayable. PARM admission
entries are keyed by the complete judge input (prompt, observation, candidate
pairs, model, and rubric), so a change to candidate generation intentionally
requires a new admission cache. Add `--trajectory-policy frozen` to forbid new
live calls; a miss is then an error.

The whole ladder, one condition after another:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_workflows_v1_matrix.ps1
& 'C:\Users\karth\anaconda3\python.exe' scripts\summarize_workflows_v1_results.py
```

PARM defaults to `--parm-retriever semantic-judge` here. The workflow corpora
are ordinary personal histories with no link graph and no review/reflection
filename convention, which is the shape the semantic path exists for. Pass
`--parm-retriever convergence` to run the deterministic waterfall instead.

Rebuild the dataset and its index after editing the corpus:

```powershell
$env:PYTHONPATH = 'src'
& 'C:\Users\karth\anaconda3\python.exe' scripts\build_workflows_v1_cases.py
Remove-Item -Recurse -Force data\retrieval-indexes\workflow-eng-lead-v1
& 'C:\Users\karth\anaconda3\python.exe' scripts\build_workflows_v1_index.py
```

Then check that no goal can reach its own gold memory:

```powershell
& 'C:\Users\karth\anaconda3\python.exe' scripts\evaluate_workflows_v1_fairness.py
```

For another workflow environment, pass both the dataset and its frozen index;
`--index` takes precedence over the legacy tier mapping. This check embeds the
goals, so it needs `OPENAI_API_KEY`; the structural workflow tests do not.

```powershell
& 'C:\Users\karth\anaconda3\python.exe' scripts\evaluate_workflows_v1_fairness.py `
  --dataset data\workflows_email_calendar_v1 `
  --index data\retrieval-indexes\ops-lead-email-calendar-v1
```

Keep first-pass outputs and caches in environment-specific namespaces. Start a
new namespace whenever the dataset, index, model, or prompt changes; do not
reuse a cache from GitHub workflows for Email + Calendar. The v1 runner keeps
its historic defaults, but accepts explicit values for a new environment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_workflows_v1_matrix.ps1 `
  -Dataset data\workflows_email_calendar_v1 `
  -Index data\retrieval-indexes\ops-lead-email-calendar-v1 `
  -Results data\benchmark-results\workflows-email-calendar-v1-first-pass `
  -Caches data\workflow-caches `
  -CacheNamespace email-calendar-v1-first-pass
& 'C:\Users\karth\anaconda3\python.exe' scripts\summarize_workflows_v1_results.py `
  --results data\benchmark-results\workflows-email-calendar-v1-first-pass `
  --dataset data/workflows_email_calendar_v1
```

For a genuinely fresh run, use a new, empty `-CacheNamespace` and result root
rather than deleting a replay cache that may be needed to reproduce an older
report.

### Natural Email + Calendar v2 rates

The v2 natural pilot measures target-bound environmental state, not reply prose
or one exact normal action. After scoring independently sampled workflow runs,
render its per-triplet and aggregate positive, cue-ablated control, and
memory-included oracle rates with:

```powershell
$env:PYTHONPATH = 'src'
python scripts\summarize_email_calendar_v2_results.py `
  data\benchmark-results\workflows-email-calendar-v2-diagnostic
```

The report includes `cue_triggered_lift` (positive minus control) and, when
both conditions are present, `parm_minus_no_memory_positive` plus the
PARM-minus-no-memory cue-lift difference.  These are rates over samples; do
not impose a fixed `3/3` gate.  Any claim about a retrieval effect requires
fresh independent samples and an explicit threshold preregistered before the
comparison.

Run this after any edit to a goal or a corpus, before spending a run. If
prompt-only retrieval finds a gold source from the goal alone, the scenario is
no longer testing late-cued retrieval and input RAG will win for the wrong
reason. Two things trip it: a corpus small enough that top-five covers much of
it, and a goal written in the memory's own vocabulary.

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
