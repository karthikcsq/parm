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
key is the option's visible title, company, or restaurant name—not an opaque
benchmark ID. The same identifying entity or pattern is readable in the Amara
memory prose, so a model can connect memory to the noisy output without access
to hidden metadata.

The compact raw Amara fixture is tracked under `data/amara-life-v1/`. GBrain
prepares the neutral memory substrate; PARM owns ranking, cue selection,
retrieval timing, admission, and decision evaluation.

## Run

```powershell
$env:PYTHONPATH='src'
python -m parm_bench.cli validate data/benchmark_v1
python -m parm_bench.cli inspect data/benchmark_v1 --case parm-amara-conference-agenda-positive
python -m unittest discover -s tests
```

## Baseline status

Two baselines are implemented:

- `no_memory` sends only the ordinary prompt and resolved observation to the
  response model. It has no retriever and emits an empty retrieval trace.
- `input_rag` sends only the original sanitized prompt to the shared PARM
  retriever, admits every top-k page, and appends each selected chunk in a
  separate retrieved-memory section. It does not retrieve from the later
  observation.

Retrieval condition and retrieval mode are separate experiment axes. Every
memory-using condition must explicitly choose `dense`, `hybrid`, or `enhanced`
and use the same frozen index for mode-matched comparisons.

Run the five positive/control pairs and score them:

```powershell
$env:PYTHONPATH='src'
python -m parm_bench.cli run data/benchmark_v1 `
  --baseline no_memory `
  --model gpt-5-mini `
  --out .Codex/benchmark-results/no-memory-gpt-5-mini.jsonl
python -m parm_bench.cli score `
  .Codex/benchmark-results/no-memory-gpt-5-mini.jsonl `
  --gold data/benchmark_v1 `
  --out .Codex/benchmark-results/no-memory-gpt-5-mini.metrics.json
```

Run input-RAG over the same cases:

```powershell
python -m parm_bench.cli run data/benchmark_v1 `
  --baseline input_rag `
  --retrieval-mode dense `
  --retrieval-index .gbrain-local\indexes\amara-life-v1 `
  --retrieval-limit 5 `
  --model gpt-5-mini `
  --out .Codex/benchmark-results/input-rag-gpt-5-mini.jsonl
```

The CLI automatically loads the ignored repo-root `.env` without overriding
variables already set in the process. Start from `.env.example`; set
`OPENAI_API_KEY`, and override `GBRAIN_HOME`, `PARM_GBRAIN_CWD`, or
`PARM_OPENAI_MODEL` only when needed. GBrain is contacted only by the explicit
index-export command, never during a canonical benchmark run.

Every run writes the common prediction JSONL plus a sibling `.config.json`
recording its baseline, retrieval mode, fixed ranking constants, dependency
versions, index-manifest hash, and expansion-cache hash where applicable.
Input-RAG traces retain complete ranking diagnostics and perturbation labels,
but labels and IDs are not shown to the response model.

Prepare the repo-local GBrain corpus before running a memory baseline:

```powershell
python -m parm_bench.cli prepare-amara
python -m parm_bench.cli export-retrieval-index `
  --out .gbrain-local\indexes\amara-life-v1 `
  --chunker-version gbrain-0.42.53.0-default
```

The exporter rejects a brain whose stored vectors are not 384-dimensional
MiniLM embeddings. Enhanced runs additionally require `--expansion-cache`;
official runs use `--expansion-policy frozen`.

Later baselines will be developed and reviewed one at a time.

See `docs/benchmark-evaluation.md` for the scoring contract and
`docs/parm-output-cued-memory-examples.md` for the 20-example expansion set.
