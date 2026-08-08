# PARM Architecture

PARM tests a specific failure mode in personal-memory systems: a prompt does
not contain enough information to retrieve the right memory, but a later tool
result or agent output does. The system must notice the new cue, retrieve only
the memory that now matters, and use it to make a better decision.

## One case in plain language

1. The user asks for one conference session.
2. A 9,000-token agenda arrives. One row mentions NovaMind's Texas grid pilot.
3. A durable note says Amara needs that pilot evidence before recommending an
   investment.
4. PARM links the agenda row to the note and hands both to the response model.
5. The model selects the NovaMind session.
6. In the cue-ablated twin, the row names an unrelated company. PARM should
   admit nothing and preserve the agenda's ordinary best choice.

The positive and control together test both halves of the claim. Retrieval
must help when the cue exists and stay quiet when it does not.

## End-to-end waterfall

```mermaid
flowchart TB
    A["Tracked Amara source corpus"] --> B["Frozen retrieval index<br/>pages, chunks, sentences, links, embeddings"]
    C["Benchmark case<br/>prompt plus large observation"] --> D["Dataset loader and validator"]
    D --> E["Baseline selects when and why retrieval runs"]
    B --> F["Retrieval resource"]
    E --> F
    F --> G["Retrieved candidates"]
    G --> H["Condition-specific admission policy"]
    H --> I["Short memory handoff<br/>triggering output plus evidence"]
    C --> J["Response-model request"]
    I --> J
    J --> K["One visible label or name"]
    K --> L["Deterministic scorer"]
    D --> L
    L --> M["Prediction JSONL"]
    L --> N["Metrics JSON"]
    E --> O["Run configuration sidecar"]
    H --> M
```

The frozen index is neutral. It does not decide when retrieval should run or
which pages should reach the response model. Each baseline supplies that
policy so the benchmark can compare mechanisms over the same memory substrate.

## The benchmark layer

`data/benchmark_v1/cases.jsonl` contains 54 cases: 18 scenarios, each with
three variants.

`data/workflows_v1/cases.jsonl` is the executable workflow suite: three
scenarios over one engineering-lead corpus, scored on the environment an agent
leaves behind rather than on a single label. It uses the same triplet contract.

| Variant | Observation | Expected behavior |
| --- | --- | --- |
| Positive | Contains the decisive incidental cue | Retrieve and select the memory-conditioned choice |
| Cue-ablated | Replaces only the decisive cue | Preserve the output-only choice |
| Memory-included | Reuses the positive observation and puts the memory in the prompt | Select the memory-conditioned choice without depending on retrieval |

`src/parm_bench/dataset.py` resolves context files and replacements, validates
the schema, checks source hashes, and rejects unsafe gold-source definitions.
`src/parm_bench/scoring.py` then scores the visible final choice and the
retrieval trace independently.

The three variants separate different failures:

- Positive failure can mean retrieval missed, judgment failed, or the fixture
  lacks enough separation.
- Control failure means memory changed a decision without the intended cue.
- Ceiling failure means the response model or fixture cannot use explicit
  memory reliably, so retrieval should not receive all the blame.

## Retrieval conditions and modes

The baseline is the retrieval condition. It controls when retrieval happens
and what text becomes the query.

| Condition | Retrieval trigger | Admission behavior |
| --- | --- | --- |
| `no_memory` | Never | No memory |
| `input_rag` | Original prompt only | Admit fixed top-k |
| `prompted_memory_tool` | Model elects to call a tool after seeing the observation | Admit fixed top-k when called |
| `naive_output_rag` | Whole tool output, model output, or both | Admit fixed top-k |
| `all_entity_output_rag` | Every exact entity found in the observation | Admit the deduplicated union |
| `parm` | Cue regions inside the observation | Selectable convergence or semantic-pair admission |

`input_rag`, `prompted_memory_tool`, and `naive_output_rag` use
`IndexRetriever`. Their independent retrieval-mode axis controls how one query
is ranked:

- `dense`: OpenAI embedding cosine over frozen chunks.
- `hybrid`: body BM25 and dense rankings fused with reciprocal rank fusion.
- `enhanced`: hybrid ranking plus three frozen query expansions, title BM25,
  and a small graph rerank.

`all_entity_output_rag` uses `EntityExactRetriever`. `parm` defaults to
`PARMConvergenceRetriever` and can select the development
`PARMSemanticJudgeRetriever` with `--parm-retriever semantic-judge`. These are
PARM-specific mechanisms, so the CLI rejects `--retrieval-mode` for them.

## Deterministic PARM retrieval waterfall

The default retriever treats the large observation as visible regions. Existing
catalog fixtures preserve one-listing-per-region parsing; general prose and
mixed markdown use paragraph-like semantic blocks. PARM creates several cheap
views of those regions, then admits only durable pages supported by a
channel-specific rule.

```mermaid
flowchart TB
    A["Large observation"] --> B["Split into listings or semantic blocks"]
    B --> C["Task anchors from the prompt"]
    B --> D["Store-backed entity surfaces"]
    B --> E["Rare concepts per listing"]
    B --> F["Full-region lexical text"]

    D --> G["Entity graph candidates"]
    G --> H["Region cosine plus rare lexical tie-break"]
    H --> I["Best graph-linked durable page per region"]

    C --> J["Anchor plus concept queries"]
    E --> J
    J --> K["Frozen sentence cosine"]
    K --> L["Review/reflection convergence rules"]

    F --> M["BM25 against durable notes"]
    M --> N["Best page must beat runner-up page"]
    N --> O["Best region must beat runner-up region"]

    I --> P["Priority merge and page deduplication"]
    L --> P
    O --> P
    P --> Q["At most top-k admitted memories<br/>zero is allowed"]
    Q --> R["Evidence projection"]
    R --> S["Triggering region plus focused memory evidence"]
```

### Entity graph channel

The entity extractor uses an Aho-Corasick gazetteer built from the frozen
store. If a listing contains known entities, PARM follows inbound links to
durable pages. Candidates must clear a region cosine floor. When several pages
share the same entities, rare exact terms in the listing provide a lexical
tie-break before cosine.

This channel handles relations such as a named founder, company, or
counterparty connected to an existing meeting or email.

### Task-conditioned semantic channel

spaCy supplies task anchors and rare noun concepts without an LLM in the
retrieval loop. Short anchor-concept queries are embedded and compared with the
frozen sentence matrix. The selector currently restricts this path to
review/reflection pages and accepts multi-concept convergence or a high
anchored singleton.

This channel handles patterns such as burnout, recovery, and recurring
behavior when no entity link is sufficient.

### Contrastive durable-note channel

Each listing region is scored with BM25 against unperturbed durable notes. A
note is admitted only when two contrasts agree:

1. the best note for that region beats the second-best note; and
2. that region's winning score beats the best score from every other region.

Both ratios default to `1.25`. This opens coverage to notes whose filenames are
not reviews while still requiring the observation to contain one unusually
specific memory-shaped region.

### Merge and evidence handoff

Admissions are deduplicated by page. Channel priority is:

1. direct-note contrast;
2. entity graph;
3. semantic multi-concept;
4. semantic anchored singleton.

The response model receives the exact triggering region beside a focused
memory excerpt. It never receives benchmark gold IDs, perturbation labels, or
the retrieval scores.

## Semantic-pair admission development path

Ordinary personal histories do not have the link graph or review/reflection
filename conventions that make the deterministic waterfall effective on Amara.
The workflow corpus is one such history, which is why the semantic path is the
workflow default. It keeps corpus isolation and the frozen raw index but
changes candidate generation and admission. Its canonical replay waterfall is:

```mermaid
flowchart TB
    A["1. Corpus scope"] --> B["2. Observation regions"]
    B --> C["3. Distinctive-region filter"]
    C --> D["4. Query embeddings"]
    D --> E["5. Five candidate views"]
    E --> F["6. Top-seven union"]
    F --> G["7. Contrastive admission judge"]
    G --> H["8. Frozen selection validation"]
    H --> I["9. Zero-or-one retrieval hit"]
    I --> J["10. Evidence projection"]
    J --> K["11. Final visible choice"]
```

The retriever first restricts the index to the requested persona corpus.
Existing catalogs keep their listing rows, while mixed Markdown and prose
become complete semantic blocks. Blocks without distinctive concepts are
dropped. The remaining block descriptions are embedded and compared with raw
history through sentence-max, sentence-top-three, chunk-max, blended dense, and
BM25 views. The union of the best seven pairs from each view is a high-recall
prefilter, not an admission threshold.

The judge sees the complete observation so it can distinguish real candidates
from archival noise and identify the ordinary evidence winner. It must find a
fact in the user's own words, an explicit affordance in the visible block, and
a reason that fact would favor a lower-ranked candidate. It does not receive
case gold IDs, expected choices, source-support grades, or hidden persona
annotations.

This LLM use is narrow. Dense and lexical candidate generation, corpus
filtering, the final choice scorer, and artifact validation stay deterministic.
Canonical runs require `--parm-admission-cache` and normally use
`--parm-admission-policy frozen`. Populate runs are for explicit experiment
construction.

Admission cache keys intentionally do not serialize the live ranked candidate
list. They bind the prompt and complete observation to the judge model, rubric,
retrieval-index hash, candidate depth, and candidate-policy version. Each
cached admission stores the selected corpus page, region ID, and exact region
text. Replay accepts a shifted live ranking only when that page is still in the
scoped corpus and the exact region still exists in the observation. This
prevents harmless embedding-order jitter from breaking replay without allowing
a cached decision to cross a corpus or observation boundary.

## Frozen artifacts and replay

Canonical runs depend on tracked, hashed artifacts:

| Artifact | Purpose |
| --- | --- |
| `data/retrieval-indexes/amara-life-v1` | Neutral pages, chunks, sentences, links, and embeddings |
| `data/retrieval-indexes/workflow-eng-lead-v1[-100]` | Schema-v3 workflow corpus indexes, one per declared scale tier |
| `data/expansion-caches/...` | Frozen enhanced-mode query alternatives |
| `data/workflow-caches/parm-admission-*` | Frozen semantic-pair admission decisions, namespaced per retrieval index |
| `data/workflow-caches/trajectories-*` | Replayable agent turns keyed by the full request |
| `data/response-caches/...` | Reusable response-model calls keyed by the full request |
| `data/benchmark-results/*.jsonl` | One prediction and trace per case |
| `data/benchmark-results/*.config.json` | Exact condition, model, constants, and artifact hashes |
| `data/benchmark-results/*.metrics.json` | Deterministic aggregate, split, and per-case scores |

The CLI loads GBrain only while intentionally rebuilding the index. A canonical
benchmark run reads the frozen artifact directly.

## PARMBench Workflows

The deterministic suite compresses a trajectory into one observation and one
label. That makes it exact and cheap, and it is also its ceiling: it cannot
show whether late-cued memory changes what an agent *does* when it has tools
and several turns to use them.

PARMBench Workflows is the executable second suite. The agent gets an ordinary
goal and a seeded tool environment, works through a real trajectory, and is
scored on the environment it leaves behind. The triplet contract is unchanged:
every scenario has a positive, a cue-ablated twin, and a memory-included
ceiling.

```mermaid
flowchart TB
    A["Ordinary user goal"] --> B["Agent turn"]
    B --> C{"Tool call or<br/>final summary?"}
    C -->|tool call| D["Environment adapter<br/>seeded from a tracked fixture"]
    D --> E["Tool observation<br/>step index recorded"]
    E --> F["Memory policy<br/>offered the observation"]
    F -->|admits nothing| B
    F -->|admits memory| G["Memory handoff<br/>triggering region plus evidence"]
    G --> B
    C -->|final summary| H["Final environment state"]
    D --> H
    H --> I["Assertion engine<br/>decisive, workflow, restraint"]
    E --> J["Timing record<br/>cue, admission, decisive action"]
    F --> J
    I --> K["Prediction JSONL"]
    J --> K
    K --> L["Deterministic workflow scorer"]
```

The memory policy is a sidecar, not a step in the agent's plan. It is offered
every observation as it becomes visible and its admissions reach the agent
before the next turn. That models parallel retrieval optimistically and it
models it *identically* for every output-triggered condition, so a difference
between them is a difference in admission, not in plumbing.

### Environment adapters

An adapter owns the tool surface, the state, and the mutation log. The pilot
adapter is `github_fixture`: issues, pull requests, branches, files, comments,
and reviewers, driven by fifteen GitHub-shaped tools.

Every case rebuilds its environment from a tracked JSON fixture, so a positive
and its cue-ablated twin never share mutable state and cases can run
concurrently. Nothing reaches a live account.

The pilot fixture derives its repository shape and its verification style from
an MCPMark task under Apache-2.0, pinned by revision in every case. MCPMark's
own GitHub service duplicates a seed repository into a real private org over
the REST API, which needs credentials, mutates account-visible state, and
cannot give paired variants independent resets inside one parallel run. What
the local adapter gives up in exchange is real pagination, rate limits, and API
error taxonomies. A `mcpmark_live` adapter can register under the same protocol
when running against a real org is worth that cost.

### What the workflow scorer reads

Three independent records, so a regression is attributable:

| Record | Question |
| --- | --- |
| Final environment state | Did the agent do the right thing? |
| Retrieval trace | Was the right memory admitted, and only it? |
| Step log | Did the memory arrive after the cue and before the action it governs? |

Assertions carry a role. `decisive` assertions are the outcome memory is
supposed to change and they differ between the positive and its control, which
validation enforces. `workflow` assertions measure ordinary task competence and
are shared. `restraint` assertions catch collateral damage and false
intervention. Scoring them separately keeps a system from looking good on the
decision because it completed more boilerplate.

There is no LLM judge anywhere in this scorer.

## Module map

| Module | Responsibility |
| --- | --- |
| `dataset.py` | Load, resolve, and validate benchmark cases |
| `baselines.py` | Define retrieval timing, query source, admission handoff, and baseline registry |
| `retrieval.py` | Load the frozen index and implement all ranking/retrieval mechanisms |
| `semantic_parm.py` | Generate block-memory candidate pairs and replay versioned admission judgments |
| `models.py` | OpenAI response calls, memory-tool decisions, and response caching |
| `scoring.py` | Deterministic choice, admission, poison, privacy, and split metrics |
| `cli.py` | Validate experiment axes, run cases, write config sidecars, and score results |
| `workbench.py` and `web/` | Local browser inspection of cases, traces, and responses |
| `retrieval_export.py` | Freeze neutral GBrain data into the validated index format |
| `workflows/case.py` | Load and validate workflow cases against the triplet contract |
| `workflows/environment.py` | Adapter protocol, tool specs, and the shared step log |
| `workflows/github_env.py` | The isolated GitHub-shaped environment |
| `workflows/agent.py` | Tool-calling trajectory turns, memory handoff, and replay cache |
| `workflows/policies.py` | When a system may search memory and what it may see |
| `workflows/verify.py` | Declarative final-state assertion kinds |
| `workflows/runner.py` | Build the retrieval resource, run a case, verify the result |
| `workflows/scoring.py` | Decision, admission, timing, and restraint metrics |

## Design trade-offs

PARM optimizes precision under noise, so it accepts lower coverage rather than
injecting many weak memories. This is why zero admission is a valid result.
The baseline matrix measures the cost of the opposite choice: broad input or
output RAG can retrieve more gold sources while also admitting hundreds of
spurious pages and changing controls.

The current ranker is deliberately inspectable. BM25, cosine, graph links,
fixed thresholds, and exact traces make a failure attributable. A learned
ranker could cover more cue shapes, but it would require a larger train and
held-out evaluation split to avoid turning the benchmark into its training
set.

The deterministic final-choice scorer is narrow by design. It gives exact,
repeatable primary metrics, but it does not measure whether a long natural
response was helpful, calibrated, or appropriately private. Those broader
questions belong in an additional realistic end-to-end evaluation layer, not
in an unversioned replacement for the core triplets.

## Related documents

- [Research claim and scope](research-scope.md)
- [Run PARMBench](running-parmbench.md)
- [Construct benchmark cases](benchmark-construction.md)
- [Evaluation contract](benchmark-evaluation.md)
- [Rebuild the memory index](rebuilding-memory-index.md)
- [Larger dataset candidates](dataset-candidates.md)
- [Expanded benchmark first pass](results/benchmark-expansion-first-pass.md)
