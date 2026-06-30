# GBrain + Amara Life: local personal-memory retrieval setup

Recorded on June 26, 2026.

This document explains the personal-memory substrate created for PARM: what
GBrain is, what the Amara Life corpus is, why they were paired, how the source
data was converted into a form GBrain could ingest, the exact setup sequence,
and what the resulting system establishes for cue-triggered memory research.

## Short version

We wanted an existing, realistic personal-memory collection rather than a
small synthetic example invented specifically for PARM. Amara Life supplies
that collection. It represents the working life of a fictional person across
email, Slack, calendar, meeting transcripts, notes, and reference documents.

We also wanted to study retrieval without first building an entire memory
capture and consolidation system. GBrain supplies that substrate. It imports
pages into Postgres-compatible storage, chunks and embeds their contents, and
supports hybrid retrieval over the resulting memory collection.

The completed local brain contains the original artifacts plus deterministic
entity pages needed to materialize their references as graph edges:

| Item | Result |
|---|---|
| GBrain version | `0.42.53.0` |
| Corpus | BrainBench `amara-life-v1` |
| Source artifacts | 424 |
| Generated entity pages | 176 |
| Pages imported | 600 |
| Chunks embedded | 600 |
| Graph edges | 407 |
| Timeline entries | 55 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector dimensions | 384 |
| Database | PGLite |
| Local footprint | approximately 632 MB |

Everything—source corpus, normalized pages, GBrain runtime, configuration,
database, dependencies, model weights, and caches—is under the repository's
ignored `.gbrain-local/` directory.

## What GBrain is

[GBrain](https://github.com/garrytan/gbrain) is a Postgres-native personal
knowledge system designed to act as an agent's persistent “brain.” Its core
data unit is a page, usually represented as Markdown. It stores those pages in
PGLite or Postgres, creates searchable content chunks, and can layer several
retrieval signals:

- vector similarity over embeddings;
- keyword or full-text matching;
- reciprocal-rank fusion of retrieval channels;
- graph relationships derived from page links;
- source and recency signals; and
- optional reranking and query expansion.

GBrain exposes two relevant retrieval surfaces:

- `gbrain search` returns ranked source pages. This is the raw retrieval layer
  used in this setup.
- `gbrain think` retrieves evidence and then asks a language model to synthesize
  a cited answer with gap analysis.

The distinction matters for PARM. We are initially interested in which
memories become available and when, not in outsourcing the final response to
GBrain's synthesis model. `gbrain search` therefore gives us the cleaner
retrieval substrate.

GBrain also has capture, enrichment, schema, timeline, graph, MCP, and
background “dream cycle” functionality. None of those broader agent features
were required for this first setup. We intentionally used the smallest useful
slice:

```text
Markdown pages -> PGLite -> chunks -> embeddings -> ranked retrieval
```

## What Amara Life is

Amara Life is the `amara-life-v1` fixture from the
[GBrain evaluation project](https://github.com/garrytan/gbrain-evals). It is a
fictional, multi-channel personal history built for BrainBench.

The central person is Amara Okafor, a venture investor at Halfway Capital. The
corpus records her work across several kinds of artifacts rather than reducing
her “memory” to a list of isolated facts. It contains:

| Source | Format supplied upstream | Items |
|---|---|---:|
| Slack messages | JSONL | 300 |
| Emails | JSONL | 50 |
| Notes | Markdown | 40 |
| Calendar events | ICS | 20 |
| Meeting transcripts | Markdown | 8 |
| Reference documents | Markdown | 6 |
| **Total** |  | **424** |

The corpus manifest identifies it as schema version 1, seed 42, MIT licensed,
and generated with `amara-life-gen` using Claude Opus 4.5 for prose expansion.
Its artifacts contain recurring people, companies, deals, commitments, and
preferences across channels. For example, a meeting may establish an
introduction Amara promised to make, while a later Slack message refers to one
of the same people.

This cross-channel structure makes it much closer to a personal-memory problem
than a conventional question-answering dataset. Relevant information can be:

- stated directly in one artifact;
- repeated or paraphrased elsewhere;
- connected through a person or company;
- time-sensitive;
- contradicted by another artifact; or
- useful only when a later cue makes it salient.

Because Amara and every artifact are fictional, the corpus can be inspected,
transformed, and benchmarked without exposing a real person's private data.

## Why we chose GBrain and Amara Life together

The pair separates two problems that are easy to blur:

1. **Memory substrate:** how personal information is represented, stored,
   embedded, linked, and searched.
2. **Recall policy:** what event causes retrieval, which memories become
   candidates, and whether they influence a model response.

GBrain is the substrate in this experiment. Amara Life supplies a realistic
memory collection. PARM can then focus on the recall policy.

### Why not invent a tiny memory graph?

A hand-authored toy graph would make the implementation easy but the result
hard to trust. Its vocabulary, graph shape, and retrieval cues would all be
chosen with the intended test in mind. That invites accidental benchmark
telegraphing and makes retrieval look cleaner than it is in ordinary life.

Amara Life already has:

- hundreds of artifacts;
- several source types;
- repeated entities;
- realistic prose;
- temporal ordering;
- both short messages and long-form notes; and
- an upstream evaluation context.

This lets us start with an existing personal-memory environment and author
PARM probes against it instead of authoring both the world and the probes.

### Why not use Amara Life without GBrain?

The raw files are useful as a corpus, but they do not provide a working memory
system. We would still need to decide how to parse each format, chunk content,
produce embeddings, store vectors, rank results, and eventually expose graph
or timeline relationships.

GBrain already implements those mechanics. Reusing it avoids spending the
first phase of PARM on ingestion infrastructure that is not the research
question.

### Why not copy all of GBrain's agent behavior?

The goal is not to adopt GBrain's capture loop, proactive agent, or response
synthesis wholesale. Those features would mix memory formation, retrieval,
and answer generation together.

For now, we copied the useful lower layer:

```text
Amara artifacts
      |
      v
normalized memory pages
      |
      v
GBrain chunks + embeddings + PGLite
      |
      v
ranked candidate memories
      |
      v
future PARM cue-trigger policy
```

This keeps us free to develop a new memory representation or cue mechanism
later without first rebuilding storage and semantic candidate generation.

## What this setup proves—and what it does not

The successful search smoke test proves that:

- the complete Amara corpus can be represented as GBrain pages;
- every imported page has a local embedding;
- semantic retrieval can recover a relevant personal-memory artifact; and
- the entire retrieval substrate can remain scoped to this repository.

It does **not** yet prove cue-triggered recall.

The smoke test explicitly called:

```text
gbrain search "Who did Amara offer to introduce for the Capacitor Labs Series B?"
```

That is query-conditioned retrieval: the user query itself requests the
memory. PARM's target is different. A normal interaction or model output
should contain a cue, a recall policy should detect it, and relevant memories
should become available without a prompt saying “search memory.”

The next research layer therefore sits above GBrain:

```text
ordinary prompt or generated text
            |
            v
       cue detector
            |
            v
 retrieval query construction
            |
            v
       GBrain search
            |
            v
 memory admission into model context
            |
            v
 observable change in the response
```

The current brain is a retrieval baseline and experimental substrate for that
layer, not the finished cue-triggered system.

## Conversion pipeline

### The ingestion mismatch

GBrain's directory import command consumes Markdown files. Amara Life mixes
Markdown with two JSONL streams and one ICS calendar.

Importing only the existing Markdown would have silently discarded 370 of 424
artifacts:

- 300 Slack messages;
- 50 emails; and
- 20 calendar events.

To preserve the full personal history, a local adapter converted each
non-Markdown item into one Markdown page.

### Conversion order

The data moved through these stages:

```text
upstream Amara Life fixture
  |-- inbox/emails.jsonl
  |-- slack/messages.jsonl
  |-- calendar.ics
  |-- meetings/*.md
  |-- notes/*.md
  `-- doc/*.md
            |
            v
normalize_amara.py
            |
            v
424 Markdown pages
            |
            v
prepare_amara_graph.py
            |
            v
424 artifacts + 176 entity pages
            |
            v
gbrain import --no-embed
            |
            v
600 PGLite pages + 600 unembedded chunks
            |
            v
gbrain embed --stale
            |
            v
600 chunks with 384-dimensional vectors
            |
            v
extract links + timeline
            |
            v
407 graph edges + 55 timeline entries
```

Separating import from embedding made both stages independently observable.
We could verify corpus completeness before involving the model and retry
failed embedding requests without re-importing or rewriting pages.

### Email conversion

Each JSONL email became `emails/<id>.md`.

The frontmatter records:

- message ID;
- page type `email`;
- timestamp; and
- source `amara-life-v1`.

The Markdown body contains:

- subject as the page heading;
- sender;
- recipients;
- date;
- thread ID; and
- original `body_text`.

The body text was not summarized or paraphrased.

### Slack conversion

Each JSONL Slack item became `slack/<id>.md`.

The frontmatter records:

- message ID;
- page type `slack`;
- timestamp; and
- source `amara-life-v1`.

The body contains:

- channel;
- author name and handle;
- date; and
- original message text.

Again, the message itself was preserved rather than rewritten.

### Calendar conversion

Each `VEVENT` in `calendar.ics` became `calendar/<event-id>.md`.

The frontmatter records:

- event ID;
- page type `calendar-event`;
- start time; and
- source `amara-life-v1`.

The body contains:

- summary;
- start and end times;
- attendees; and
- location when one exists.

### Existing Markdown

Meeting transcripts, notes, and reference documents were initially copied
without paraphrasing their bodies. Graph preparation later normalized link
destinations and added meeting type/title metadata. Visible labels, prose,
decisions, and action items remained intact.

### Entity and graph preparation

The 424 artifacts referenced people, companies, deals, documents, and concepts
that were not themselves included as pages. GBrain deliberately skips edges
whose target page does not exist, so the first extraction dry run produced
zero links.

`prepare_amara_graph.py` made those references materializable:

- normalized inconsistent prefixes such as `user/` to `people/`,
  `company/` to `companies/`, and `docs/` to `source/`;
- created 176 minimal entity pages from the canonical target slugs and visible
  link labels;
- typed the eight meeting pages explicitly as `meeting`;
- canonicalized meeting attendee slugs in frontmatter; and
- copied each meeting's H1 into its title metadata.

These entity pages are deterministic reference endpoints, not additional
biographical claims. They retain aliases observed in the corpus and state only
that the entity was referenced by Amara Life.

The copied GBrain runtime was also narrowed so meeting-body links produce
`mentions`; only the explicit frontmatter attendee list produces `attended`.
Without that correction, GBrain classified discussed companies and proposed
introductions as meeting attendees.

### Chunking result

The completed fixture produced one GBrain chunk per page:

```text
600 pages -> 600 chunks
```

That is an observed result of this corpus and GBrain version, not a general
promise that GBrain always creates one chunk per page. Longer or differently
structured pages can produce multiple chunks.

## Local embedding design

No hosted embedding API key was available, and the setup had to stay inside
the project. The embedding step therefore used the Python, PyTorch, and
Transformers packages already installed on the machine.

The local server:

- loaded `sentence-transformers/all-MiniLM-L6-v2`;
- cached the downloaded model under `.gbrain-local/models/`;
- tokenized input with a 512-token maximum;
- mean-pooled the model's token representations using the attention mask;
- L2-normalized the resulting vector;
- returned 384-dimensional embeddings through an OpenAI-compatible
  `/v1/embeddings` endpoint;
- listened only on `127.0.0.1:18080`; and
- required a randomly generated bearer token.

GBrain's `llama-server` provider was pointed at this endpoint. The temporary
server was stopped after embedding and search verification. Amara content was
not sent to OpenAI, Anthropic, or another hosted embedding provider.

## Commands used

The following is the PowerShell-oriented execution sequence used for this
repository. The downloaded source snapshots were copied into
`.gbrain-local/`; neither GBrain nor its configuration was installed globally.

### 1. Create the local directory structure

```powershell
$local = Join-Path (Get-Location) ".gbrain-local"
$dirs = @(
  $local,
  "$local\runtime",
  "$local\source",
  "$local\corpus",
  "$local\tools",
  "$local\home",
  "$local\db",
  "$local\models",
  "$local\cache"
)

foreach ($dir in $dirs) {
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
```

The repository `.gitignore` was updated with:

```gitignore
.gbrain-local/
```

### 2. Copy the downloaded source snapshots locally

The current GBrain and GBrain eval repositories were downloaded through the
browser, expanded in the Windows temporary directory, and copied into the
project:

```powershell
Copy-Item `
  -LiteralPath "$env:LOCALAPPDATA\Temp\gbrain-master-current-unpacked\gbrain-master" `
  -Destination ".gbrain-local\runtime\gbrain" `
  -Recurse -Force

Copy-Item `
  -LiteralPath "$env:LOCALAPPDATA\Temp\gbrain-evals-main-unpacked\gbrain-evals-main\eval\data\amara-life-v1" `
  -Destination ".gbrain-local\source\amara-life-v1" `
  -Recurse -Force
```

### 3. Normalize Amara Life into Markdown

The adapter was created at:

```text
.gbrain-local/tools/normalize_amara.py
```

It was run with:

```powershell
python .gbrain-local\tools\normalize_amara.py
```

Output:

```text
normalized 424 Amara Life items into
C:\Users\karth\CodingFiles\PARM\parm\.gbrain-local\corpus\amara-life-v1
```

The first normalization pass revealed that the upstream reference-document
folder is named `doc`, singular. The adapter was corrected from `docs` to
`doc`, then rerun. The final count was:

```text
calendar    20
doc          6
emails      50
meetings     8
notes       40
slack      300
```

### 4. Install the local GBrain dependencies

The Bun cache was redirected into the project:

```powershell
$env:BUN_INSTALL_CACHE_DIR = (
  Resolve-Path ".gbrain-local\cache\bun"
).Path
```

The initial frozen install reached GBrain's postinstall hook:

```powershell
bun install --frozen-lockfile
```

That hook uses Unix shell syntax and failed under Windows while checking for a
global `gbrain` command. The check was unnecessary here because a global
installation was explicitly out of scope. Dependencies were completed with
lifecycle scripts disabled:

```powershell
bun install --frozen-lockfile --ignore-scripts
bun run src\cli.ts --version
```

Version output:

```text
gbrain 0.42.53.0
```

### 5. Start the temporary local embedding endpoint

The server was created at:

```text
.gbrain-local/tools/local_embedding_server.py
```

Before starting it, all relevant caches and runtime state were pointed inside
the repository:

```powershell
$root = (Resolve-Path ".gbrain-local").Path
$token = [guid]::NewGuid().ToString("N")

$env:HF_HOME = Join-Path $root "models\huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:XDG_CACHE_HOME = Join-Path $root "cache"
$env:EMBED_AUTH_TOKEN = $token
$env:EMBED_PORT = "18080"

$process = Start-Process `
  -FilePath "python" `
  -ArgumentList @(
    (Join-Path $root "tools\local_embedding_server.py")
  ) `
  -WindowStyle Hidden `
  -PassThru `
  -RedirectStandardOutput (
    Join-Path $root "run\embedding.stdout.log"
  ) `
  -RedirectStandardError (
    Join-Path $root "run\embedding.stderr.log"
  )
```

Readiness was checked through the authenticated local health endpoint:

```powershell
$headers = @{ Authorization = "Bearer $token" }
Invoke-RestMethod `
  -Uri "http://127.0.0.1:18080/health" `
  -Headers $headers
```

The endpoint returned:

```json
{
  "status": "ok",
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### 6. Initialize the repo-local PGLite brain

For every GBrain command, `GBRAIN_HOME` was set to an absolute directory
inside this repository:

```powershell
$repo = (Resolve-Path ".").Path
$env:GBRAIN_HOME = "$repo\.gbrain-local\home"
$env:LLAMA_SERVER_BASE_URL = "http://127.0.0.1:18080/v1"
$env:LLAMA_SERVER_API_KEY = $token
```

From `.gbrain-local\runtime\gbrain`, initialization used:

```powershell
bun run src\cli.ts init `
  --pglite `
  --path "$repo\.gbrain-local\db" `
  --embedding-model `
    "llama-server:sentence-transformers/all-MiniLM-L6-v2" `
  --embedding-dimensions 384 `
  --non-interactive
```

Relevant output:

```text
Embedding: llama-server:sentence-transformers/all-MiniLM-L6-v2 (384d)
Brain ready at C:\Users\karth\CodingFiles\PARM\parm\.gbrain-local\db
0 pages. Engine: PGLite (local Postgres).
```

The effective local configuration was then tightened:

```powershell
bun run src\cli.ts config set mcp.publish_skills false
bun run src\cli.ts config set self_upgrade.mode off
bun run src\cli.ts config set search.reranker.enabled false
```

`gbrain config get` confirms the effective values are `false`, `off`, and
`false`, respectively. These effective settings can live in GBrain's database
configuration plane even when the bootstrap JSON still shows its original
file-plane defaults.

### 7. Import all 424 pages without embeddings

Import was deliberately separated from embedding:

```powershell
$env:GBRAIN_LOCAL_FS_WALK = "true"

bun run src\cli.ts import `
  "$repo\.gbrain-local\corpus\amara-life-v1" `
  --no-embed `
  --workers 4 `
  --json
```

Output:

```text
Found 424 markdown files
Using 4 parallel workers
{"status":"success","duration_s":13.1,"imported":424,
"skipped":0,"errors":0,"chunks":424,"total_files":424}
```

### 8. Embed stale chunks

The embedding backfill command was:

```powershell
bun run src\cli.ts embed --stale --json
```

The first pass embedded 260 chunks. GBrain issued enough parallel localhost
requests to saturate Python's default server queue, so failed pages remained
stale rather than being partially written.

The server was changed to:

- use a request queue of 512; and
- serialize PyTorch inference with a lock.

Running the same stale command again embedded 163 chunks:

```text
Embedded 163 chunks across 164 pages
```

One connection closed during that pass. A final retry embedded the remaining
chunk:

```text
Embedded 1 chunks across 1 pages
```

The zero-stale verification was:

```powershell
bun run src\cli.ts embed --stale --json
```

Result:

```text
Embedded 0 chunks (0 stale found)
```

### 9. Run a retrieval smoke test

The semantic retrieval check was:

```powershell
bun run src\cli.ts search `
  "Who did Amara offer to introduce for the Capacitor Labs Series B?" `
  --json
```

The first result was:

```text
[0.8308] meetings/mtg-0000
```

That meeting contains the relevant promise to introduce Sarah Chen and Marcus
Reid, showing that the local vector index could recover the right artifact.

### 10. Stop the temporary embedding server

After verification, the Python process was stopped:

```powershell
$proc = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
if (
  $proc -and
  $proc.Path -eq "C:\Users\karth\anaconda3\python.exe"
) {
  Stop-Process -Id $proc.Id -Force
}
```

The final check found no listener on port 18080.

### 11. Materialize graph edges and entity timelines

The first extraction preview correctly returned zero because the referenced
entity pages did not exist:

```powershell
bun run src\cli.ts extract links --source db --dry-run --json
bun run src\cli.ts extract timeline --source db --dry-run --json
```

After creating the entity layer:

```powershell
python .gbrain-local\tools\prepare_amara_graph.py
```

the expanded 600-page corpus was imported:

```powershell
$env:GBRAIN_LOCAL_FS_WALK = "true"

bun run src\cli.ts import `
  "$repo\.gbrain-local\corpus\amara-life-v1" `
  --no-embed `
  --workers 4 `
  --json
```

Links were previewed and then committed with frontmatter enabled so explicit
meeting attendees retained stronger provenance than ordinary body mentions:

```powershell
bun run src\cli.ts extract links `
  --source db `
  --include-frontmatter `
  --dry-run `
  --json

bun run src\cli.ts extract links `
  --source db `
  --include-frontmatter `
  --json
```

The final graph contains:

```text
382  mentions       markdown
 16  attended       frontmatter
  5  invested_in    markdown
  4  founded        markdown
407  total
```

Entity timelines were derived from the eight meeting pages:

```powershell
bun run src\cli.ts extract timeline `
  --from-meetings `
  --source db `
  --dry-run `
  --json

bun run src\cli.ts extract timeline `
  --from-meetings `
  --source db `
  --json
```

This created 55 dated timeline entries. Repeating both committed extraction
commands created zero additional rows, confirming idempotency.

The changed and generated pages were embedded with the same local model:

```powershell
bun run src\cli.ts embed --stale --json
```

Final verification:

```text
pages=600
chunks=600
embedded=600
links=407
timeline_entries=55
stale_embeddings=0
```

## Repo-local GBrain compatibility changes

The current GBrain version needed three small changes for this deliberately
ignored, user-supplied local model setup. All three changes exist only in the
copied runtime under `.gbrain-local/runtime/gbrain/`.

### Accept the model's real dimensions

The generic `llama-server` recipe declared `default_dims: 0` because GBrain
cannot know which model a user will serve. Initialization simultaneously
required `--embedding-dimensions` and rejected 384 as different from that zero
default.

The copied recipe was pinned to this setup:

```typescript
// Repo-local runtime is pinned to all-MiniLM-L6-v2.
default_dims: 384,
```

### Recognize the configured user-provided model

The embedding preflight treated an empty built-in model catalog as proof that
no model had been configured. That is not valid for a user-provided provider:
the full model ID was present in `embedding_model`.

The copied runtime was changed so the preflight rejects only an actually empty
parsed model ID:

```typescript
if (
  Array.isArray(tp.models) &&
  tp.models.length === 0 &&
  (recipe.id === "litellm" || isUserProvided) &&
  !parsed.modelId
) {
  // report user_provided_model_unset
}
```

### Import an intentionally Git-ignored corpus

GBrain's importer uses `git ls-files --exclude-standard` inside a Git
worktree. That normally prevents dependencies and generated artifacts from
being indexed. Here it also correctly hid `.gbrain-local/`, leaving zero
discoverable Markdown files.

The copied runtime received an explicit filesystem-walk escape hatch:

```typescript
if (process.env.GBRAIN_LOCAL_FS_WALK === "true") return null;
```

This allowed GBrain to read the local corpus without removing it from
`.gitignore`.

### Preserve meeting-edge semantics

GBrain's generic meeting rule classified every body link as `attended`.
Amara meetings also link to discussed companies and people who were merely
mentioned or proposed for introductions.

The copied runtime was changed so:

```text
meeting body link       -> mentions
frontmatter attendee    -> attended
```

The attendee frontmatter mapping was made outgoing from the meeting to the
person, matching the direction consumed by GBrain's
`extract timeline --from-meetings` implementation.

These are compatibility patches to the disposable local runtime, not changes
to PARM's source code or to any global GBrain installation.

## Local layout

```text
.gbrain-local/
|-- cache/       Bun and process caches
|-- corpus/      Normalized Markdown corpus
|-- db/          PGLite database
|-- home/        GBrain configuration and state
|-- models/      Hugging Face model cache
|-- run/         Temporary embedding-server state and logs
|-- runtime/     Local GBrain source snapshot and dependencies
|-- source/      Original Amara Life fixture
`-- tools/       Corpus normalizer and local embedding server
```

The directory is intentionally untracked. The tracked document you are
reading records how that local state was produced.

## Isolation verification

The completed setup was checked against the project-local requirement:

- `GBRAIN_HOME` pointed to `.gbrain-local/home`;
- PGLite data lives in `.gbrain-local/db`;
- the model cache lives in `.gbrain-local/models`;
- runtime dependencies live in `.gbrain-local/runtime`;
- generated corpus pages live in `.gbrain-local/corpus`;
- MCP skill publication is disabled in effective local config;
- GBrain self-upgrade is disabled in effective local config;
- hosted reranking is disabled in effective local config;
- no `~/.gbrain/config.json` was created;
- no global `gbrain` CLI was installed;
- no MCP client was registered;
- no hosted embedding API received Amara content; and
- the temporary localhost embedding process was stopped.

The only tracked setup-level change outside this document is the
`.gbrain-local/` entry in `.gitignore`.

## Recommended next step for PARM

Keep this brain fixed as a reproducible memory substrate and build the first
cue-triggered retrieval adapter above `gbrain search`.

A minimal experiment would:

1. present an ordinary, non-leading interaction;
2. monitor the model's generated text for a defined cue;
3. convert that cue into a GBrain search query;
4. admit the returned memory only after the cue fires;
5. continue generation or run a second response turn; and
6. compare the response with a no-retrieval control.

That experiment would preserve the conceptual boundary:

```text
GBrain = memory storage and candidate retrieval
PARM   = cue detection, retrieval timing, and causal evaluation
```
