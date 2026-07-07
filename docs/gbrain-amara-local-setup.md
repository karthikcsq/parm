# GBrain + Amara Life Local Setup

Recorded against GBrain `0.42.53.0` and the `amara-life-v1` fixture.

## Roles

```text
Amara Life = fictional personal-history corpus
GBrain     = import, chunking, embedding, and graph substrate
PARM       = ranking, output-cue policy, and decision evaluation
```

The tracked source fixture contains 424 fictional artifacts across email,
Slack, calendar, meetings, notes, and reference documents. It is compact and
safe to version. Canonical frozen PARM artifacts are tracked under `data/`:
the retrieval index in `data/retrieval-indexes/amara-life-v1`, the enhanced
query expansion cache in `data/expansion-caches/amara-life-v1`, and benchmark
outputs in `data/benchmark-results`. GBrain runtime code, PGLite state,
dependencies, model caches, and rebuild scratch state remain under ignored
`.gbrain-local/`.

## Prepare the corpus

From the repo root:

```powershell
python -m pip install -e .
parm-bench prepare-amara
```

This converts the mixed source formats into Markdown, canonicalizes links,
creates deterministic entity endpoints, and writes the result to
`.gbrain-local/corpus/amara-life-v1`.

The tracked converter preserves:

- source item IDs and timestamps;
- source-content hashes;
- `poison`, `contradiction`, and `stale-fact` perturbation labels;
- poison fixture IDs where present; and
- meeting attendee versus body-mention semantics.

That provenance is essential. Earlier local normalization stripped poison
labels, making unsafe records indistinguishable from ordinary memory. Gold
benchmark sources now fail validation if labeled `poison`.

Expected prepared count:

```text
424 source artifacts + 176 generated entity pages = 600 pages
```

## Initialize GBrain

Keep `GBRAIN_HOME`, database state, model caches, and the embedding endpoint
inside `.gbrain-local`. The canonical configuration uses:

- PGLite;
- `openai:text-embedding-3-small`;
- 512-dimensional embeddings;
- hosted reranking disabled;
- self-upgrade disabled; and
- MCP skill publication disabled.

Switch an existing PGLite brain to the canonical model. GBrain preserves the
old database as `.gbrain-local\db.bak`; `--no-sync` keeps reinitialization
separate from the explicit corpus import:

```powershell
$repo = (Resolve-Path '.').Path
$env:OPENAI_API_KEY = python -c "from dotenv import dotenv_values; print(dotenv_values('.env')['OPENAI_API_KEY'])"
$env:GBRAIN_HOME = "$repo\.gbrain-local\home"
$env:GBRAIN_LOCAL_FS_WALK = 'true'

Set-Location .gbrain-local\runtime\gbrain
bun run src\cli.ts reinit-pglite `
  --embedding-model openai:text-embedding-3-small `
  --embedding-dimensions 512 `
  --no-sync `
  --yes
bun run src\cli.ts import "$repo\.gbrain-local\corpus\amara-life-v1" --no-embed --workers 4 --json
bun run src\cli.ts embed --all --json
bun run src\cli.ts extract links --source db --include-frontmatter --json
bun run src\cli.ts extract timeline --from-meetings --source db --json
Set-Location $repo
```

`OPENAI_API_KEY` must be present in the shell that runs GBrain; unlike
`parm-bench`, GBrain does not load the repo-root `.env`. GBrain `0.42.53.0`
accepts 512 as an OpenAI embedding width but rejects 384 during provider
preflight. The expected shape is 600 pages and 600 chunks; graph and timeline
counts should be verified after rebuilding because they can vary with the
pinned GBrain version. The OpenAI migration recorded here produced 600 embedded
chunks, 407 links, and 55 timeline entries.

## Freeze the retrieval artifact

Canonical runs never invoke `gbrain search` and do not require a live GBrain
checkout. They load the tracked frozen index directly. Use this export step
only when intentionally refreshing that tracked artifact from the neutral
database state:

```powershell
parm-bench export-retrieval-index `
  --out data\retrieval-indexes\amara-life-v1 `
  --chunker-version gbrain-0.42.53.0-default
```

The exporter reads `pages`, `content_chunks`, and `links`, verifies that every
stored vector uses the configured 512-dimensional OpenAI model, attaches
benchmark perturbation labels, and writes hashed pages, chunks, embeddings,
links, and manifest files. The PARM loader revalidates hashes, IDs, references,
counts, model identity, and dimensions before ranking.

GBrain may leave a legacy value in `content_chunks.model` after `embed --all`.
The exporter therefore validates the newer per-page `embedding_signature`
first and uses the chunk model only as a fallback for older databases.

The local database may contain vectors created by an earlier model. The
exporter fails loudly in that state; reinitialize and rebuild the PGLite brain
before freezing the artifact.

## What this setup proves

It proves that the complete Amara corpus can be rebuilt into a local,
provenance-aware memory substrate. It does not itself prove cue-triggered
recall. PARM must still detect a cue that was absent from the prompt, retrieve
after the cue appears, and improve the final decision without triggering on the
cue-ablated control.
