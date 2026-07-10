# PARM

PARM evaluates output cue-triggered personal memory: an ordinary prompt is
followed by one large agent output or tool result, an incidental cue makes a
stored memory newly relevant, and that memory should materially improve the
final decision.

## Benchmark V1 pilot

The executable pilot contains five approved Amara Life cases and five
cue-ablated controls:

1. conference agenda -> unresolved NovaMind diligence;
2. AI news digest -> CoreWeave dependency;
3. podcast feed -> personal burnout pattern;
4. vendor report -> NovaTech counterparty risk; and
5. lunch search -> interrupt the desk-lunch pattern.

Each observation is 8,000-12,000 `cl100k_base` tokens. Prompts do not ask for
memory, cues appear only in the later observation, and success requires a
different decision rather than a relevant-sounding aside.

Every case asks for exactly one final choice in ordinary language. The answer
key is the option's visible title, company, or restaurant name, not an opaque
benchmark ID. The same identifying entity or pattern is readable in the Amara
memory prose, so a model can connect memory to the noisy output without access
to hidden metadata.

The compact raw Amara fixture is tracked under `data/amara-life-v1/`. GBrain
prepares the neutral memory substrate; PARM owns ranking, cue selection,
retrieval timing, admission, and decision evaluation.

## Run

Install the repository once in editable mode:

```powershell
python -m pip install -e .
python -m spacy download en_core_web_sm
```

Then use the installed command without setting `PYTHONPATH`:

```powershell
parm-bench validate data/benchmark_v1
parm-bench inspect data/benchmark_v1 --case parm-amara-conference-agenda-positive
python -m unittest discover -s tests
```

### Inspect one prompt in the browser

Start the local retrieval workbench with a validated frozen index:

```powershell
parm-bench serve-workbench `
  --retrieval-index data\retrieval-indexes\amara-life-v1 `
  --expansion-cache data\expansion-caches\amara-life-v1
```

The browser opens automatically. Choose one of the benchmark cases or enter a
custom prompt, compare `no_memory`, `input_rag`, `naive_output_rag`, and
`all_entity_output_rag`. Mode-matched conditions expose `dense`, `hybrid`, or
`enhanced` ranking; all-entity output RAG instead extracts entities from the
case observation and runs fixed exact-match retrieval per entity. A selected case runs
with its complete observation. The result leads with the generated response,
then reports decision pass/fail against the condition-appropriate expected
choice separately from gold-memory retrieval status. Ordered memories,
selected chunks, score diagnostics, and complete run JSON remain available
below. Enhanced mode is enabled when the server is started with
`--expansion-cache PATH`; use the tracked `data\expansion-caches\amara-life-v1`
cache for replay, or add `--expansion-policy populate` only while rebuilding
that cache.

## Baseline status

Four baselines are implemented:

- `no_memory` sends only the ordinary prompt and resolved observation to the
  response model. It has no retriever and emits an empty retrieval trace.
- `input_rag` sends only the original sanitized prompt to the shared PARM
  retriever, admits every top-k page, and appends each selected chunk in a
  separate retrieved-memory section. It does not retrieve from the later
  observation.
- `naive_output_rag` retrieves from output text without cue selection. Use
  `--output-rag-flow` to choose where output-triggered retrieval runs:
  `tool_output_only`, `model_output_only`, or `tool_then_model_output`.
- `all_entity_output_rag` extracts every visible output entity, retrieves with
  exact-match page lookup per extracted entity over the frozen index, merges the flat union, and
  admits every deduped hit. It does not use `--retrieval-mode`.

Retrieval condition and retrieval mode are separate experiment axes. The
mode-matched conditions (`input_rag` and `naive_output_rag`) explicitly choose
`dense`, `hybrid`, or `enhanced` and use the same frozen index for comparisons.
For `naive_output_rag`, `--output-rag-flow` is the separate axis for where
output-triggered retrieval happens; `--retrieval-mode` still controls how
memories are ranked. `all_entity_output_rag` is a fixed exact-match entity
baseline, so it requires `--retrieval-index` but rejects `--retrieval-mode`.

Run the five positive/control pairs and score them:

```powershell
parm-bench run data/benchmark_v1 `
  --baseline no_memory `
  --model gpt-5-mini `
  --out data/benchmark-results/no-memory-gpt-5-mini.jsonl
parm-bench score `
  data/benchmark-results/no-memory-gpt-5-mini.jsonl `
  --gold data/benchmark_v1 `
  --out data/benchmark-results/no-memory-gpt-5-mini.metrics.json
```

Run input-RAG over the same cases:

```powershell
parm-bench run data/benchmark_v1 `
  --baseline input_rag `
  --retrieval-mode dense `
  --retrieval-index data\retrieval-indexes\amara-life-v1 `
  --retrieval-limit 5 `
  --model gpt-5-mini `
  --out data/benchmark-results/input-rag-gpt-5-mini.jsonl
```

Run naive output-RAG with retrieval on the observed tool/output text:

```powershell
parm-bench run data/benchmark_v1 `
  --baseline naive_output_rag `
  --output-rag-flow tool_output_only `
  --retrieval-mode dense `
  --retrieval-index data\retrieval-indexes\amara-life-v1 `
  --retrieval-limit 5 `
  --model gpt-5-mini `
  --out data/benchmark-results/naive-output-rag-tool-output-gpt-5-mini.jsonl
```

Run all-entity output-RAG over the observed tool/output text:

```powershell
parm-bench run data/benchmark_v1 `
  --baseline all_entity_output_rag `
  --retrieval-index data\retrieval-indexes\amara-life-v1 `
  --retrieval-limit 5 `
  --model gpt-5-mini `
  --out data/benchmark-results/all-entity-output-rag-gpt-5-mini.jsonl
```

The CLI automatically loads the ignored repo-root `.env` without overriding
variables already set in the process. Start from `.env.example`; set
`OPENAI_API_KEY`, and override `GBRAIN_HOME`, `PARM_GBRAIN_CWD`, or
`PARM_OPENAI_MODEL` only when needed. GBrain is contacted only by the explicit
index-export command, never during a canonical benchmark run.

Every run writes the common prediction JSONL plus a sibling `.config.json`
recording its baseline, output-RAG flow where applicable, retrieval mode or
fixed retrieval condition detail, ranking constants where applicable,
dependency versions, index-manifest hash, and expansion-cache hash where
applicable.
Input-RAG traces retain complete ranking diagnostics and perturbation labels,
but labels and IDs are not shown to the response model.

Canonical benchmark replay does not require a live GBrain checkout or PGLite
database. It reads the tracked frozen index under
`data\retrieval-indexes\amara-life-v1`, the tracked expansion cache under
`data\expansion-caches\amara-life-v1`, and writes or compares tracked result
artifacts under `data/benchmark-results`.

Use GBrain only when rebuilding the frozen retrieval artifact from source:

```powershell
parm-bench prepare-amara
parm-bench export-retrieval-index `
  --out data\retrieval-indexes\amara-life-v1 `
  --chunker-version gbrain-0.42.53.0-default
```

The exporter rejects a brain whose stored vectors are not 512-dimensional
`openai:text-embedding-3-small` embeddings. Query embeddings use the same
model and dimensions through the OpenAI API. Enhanced runs additionally
require `--expansion-cache`; official runs use `--expansion-policy frozen`.

Later baselines will be developed and reviewed one at a time.

See `docs/benchmark-evaluation.md` for the scoring contract and
`docs/parm-output-cued-memory-examples.md` for the 20-example expansion set.
