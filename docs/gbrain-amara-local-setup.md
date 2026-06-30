# GBrain + Amara Life Local Setup

Recorded against GBrain `0.42.53.0` and the `amara-life-v1` fixture.

## Roles

```text
Amara Life = fictional personal-history corpus
GBrain     = storage and candidate-retrieval substrate
PARM       = output-cue policy and decision evaluation
```

The tracked source fixture contains 424 fictional artifacts across email,
Slack, calendar, meetings, notes, and reference documents. It is compact and
safe to version. GBrain runtime code, PGLite state, embeddings, dependencies,
models, and caches remain under ignored `.gbrain-local/`.

## Prepare the corpus

From the repo root:

```powershell
$env:PYTHONPATH='src'
python -m parm_bench.cli prepare-amara
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
inside `.gbrain-local`. The current verified configuration uses:

- PGLite;
- `sentence-transformers/all-MiniLM-L6-v2`;
- 384-dimensional embeddings;
- hosted reranking disabled;
- self-upgrade disabled; and
- MCP skill publication disabled.

Import and extract:

```powershell
$repo = (Resolve-Path '.').Path
$env:GBRAIN_HOME = "$repo\.gbrain-local\home"
$env:GBRAIN_LOCAL_FS_WALK = 'true'

Set-Location .gbrain-local\runtime\gbrain
bun run src\cli.ts import "$repo\.gbrain-local\corpus\amara-life-v1" --no-embed --workers 4 --json
bun run src\cli.ts embed --stale --json
bun run src\cli.ts extract links --source db --include-frontmatter --json
bun run src\cli.ts extract timeline --from-meetings --source db --json
Set-Location $repo
```

The previously verified local state contained 600 pages, 600 chunks, 407 graph
edges, and 55 timeline entries. Those counts describe the pinned setup, not a
guarantee for later GBrain versions.

## GBrain adapter

The GBrain adapter parses the CLI's ranked text output and chunks observations
that exceed Windows command-line limits before merging results.

No baseline currently consumes the adapter. A future implementation must report
its corpus version, embedding model, retrieval configuration, response model,
and whether perturbation filtering occurred.

## What this setup proves

It proves that the complete Amara corpus can be rebuilt into a local,
provenance-aware memory substrate. It does not itself prove cue-triggered
recall. PARM must still detect a cue that was absent from the prompt, retrieve
after the cue appears, and improve the final decision without triggering on the
cue-ablated control.
