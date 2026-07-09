# PARM Convergence Retrieval Design

## Status

Design, pre-validation. This describes the PARM retrieval condition (proposal
baseline 8, the cue selector), not an implementation plan. It runs under the
existing frozen-index and retrieval-mode-axis contracts in
`docs/retrieval-mode-axis-plan.md`, and it deprecates that document's universal
fixed-top-k selection rule (see "Relationship to the frozen-retrieval contract").
No claim here is validated on the benchmark yet; the first experiment that would
justify building the ranker is in the last section.

## Goal

Retrieve decision-relevant personal memory from a large, noisy observation
without an LLM in the retrieval loop, and win on precision under noise rather than
on recall. The observation contains many entities, sentences, and patterns; most
are irrelevant. PARM should admit the small number of memories that would change
the decision and admit nothing on a cue-ablated observation.

## Where this fits

Retrieval condition and retrieval mode are separate axes. The condition decides
what to query with and when; the mode (`dense`, `hybrid`, `enhanced`) decides how
a single query ranks memories. PARM is a condition. It does not reuse the
single-query, fixed-top-k shape of the mode-matched baselines. It generates many
seeds from the observation and applies its own ranking and selection over the
same frozen substrate.

The two component baselines PARM is measured against:

- all-entity output RAG (baseline 5): search every extracted entity, merge hits.
- key-sentence output RAG (baseline 6): search selected salient sentences.

Both are candidate generators. PARM's contribution is what happens after
generation.

## Constraints (hard)

- The memory store is far too large to hand a model wholesale. "Give the model
  the whole store" is not a design or a baseline.
- Retrieval, ranking, and selection must not use an LLM. They run as scoring
  functions over the frozen index (lexical, dense, graph, rank fusion).
- An LLM may judge only the small admitted set, downstream, for the decision.
- Provenance, staleness, and poison handling belong to the substrate and existing
  retrieval policy. PARM does not reimplement them.
- More candidate hits is not a failure. Breadth is expected at the candidate
  stage; precision lives in ranking and selection.

## What PARM does not own

- Relevance or decision judgment on admitted memories. The downstream response
  model does that over the short admitted list.
- A smarter store. GBrain prepares pages, chunks, embeddings, and links; the
  frozen exporter freezes them.

## Pipeline

Two non-LLM stages over the frozen artifact, then downstream judgment.

### Stage 1: multi-seed candidate generation (wide)

Replace the single "original query" of the mode-matched baselines with a set of
seeds drawn from the observation:

- entity seeds: discrete lexical anchors, one per extracted entity, retrieved by
  body BM25.
- key-sentence seeds: dense anchors, one per selected sentence, retrieved by
  OpenAI-embedding cosine against the frozen chunk matrix.
- graph expansion: from each seed's hit pages, expand one hop over `links.jsonl`
  to pull in linked pages as additional candidates.

The union of all seed hits plus their graph expansion is the candidate pool.
This stage maximizes recall; a spurious single-seed hit is acceptable here
because it is demoted, not admitted, in Stage 2.

### Stage 2: convergence ranking and threshold selection (the core)

Rank candidates by convergence — how many independent seeds point at a page and
how strongly — rather than by any single best match:

1. For each candidate page, collect its per-seed contributions across all seed
   lists (fixed `K=60` reciprocal rank fusion, consistent with the existing
   hybrid and enhanced definitions).
2. Aggregate into a convergence score that rewards agreement across distinct
   seeds and across graph-linked seeds, not a single strong hit. This
   generalizes enhanced mode's inbound-link adjacency boost from one query to
   many seeds: a page that several seeds' hits link to scores higher.
3. Select every page whose convergence score exceeds threshold `T`.

`T` is a single tunable hyperparameter. It can admit several memories, one, or
zero. The downstream model sees only the admitted set.

Convergence is a soft ranking weight, not a hard intersection gate. A memory that
matches one seed strongly and two weakly stays a candidate and floats up on its
aggregate score; a hard AND would drop it, and that shape is exactly how a
distributed pattern cue looks.

## Seed extraction (cheap, non-LLM by default)

PARM does not need open-domain NER or abstract key-sentence summarization. It
needs the parts of the observation that touch the memory store. That is a
store-anchored lookup, not an ML task, and it drives extraction cost close to
zero.

Entities, cheapest first:

- Gazetteer match. Extract every entity surface form from the memory graph once,
  offline, into an Aho-Corasick automaton. Matching an observation is one linear
  pass — sub-millisecond to low-millisecond over a 10-12K token observation, with
  power draw you would struggle to measure. This finds the entities that exist in
  memory, which are the ones worth searching on.
- Noun-phrase chunking (POS-based, e.g. spaCy small). Catches entities not yet in
  memory so PARM is not blind to new names. CPU, tens of milliseconds.
- Tiny fine-tuned NER, then an SLM, are higher rungs used only if the two above
  miss too much.

Key sentences, cheapest first:

- Segment into sentences and drop boilerplate with a classical salience prefilter
  (TF-IDF or TextRank). No model.
- Embed the survivors with the same MiniLM encoder already used for retrieval and
  use them as dense seeds. The embedder is the only model in the extraction path
  and is already in the stack; embedding roughly 50-150 sentences is the heaviest
  step, and it is small and NPU-friendly.

Design principle: high recall, not smart. Because precision lives in convergence
ranking downstream, the extractor's job is to enumerate many candidate seeds
cheaply — including weak ones — not to pre-select good ones. A precise extractor
is counterproductive: convergence exploits cues that individually match memory
weakly and only agree jointly (the burnout pattern), so a smart extractor that
drops weak single matches removes exactly the signal the ranker needs. Tune for
recall, keep the noun-phrase catch-all, accept the noise, and let convergence pay
it back.

SLM demoted to optional booster. Extraction and key-sentence ID do not need a
language model. The only step that plausibly did — composition, deciding which
disconnected entities belong together — is answered by the load-bearing graph:
entities that co-activate the same memory region are combinable by structure,
which is the convergence computation itself. An SLM cue proposer stays available
as an optional upstream booster for cross-entity cues the graph cannot link on its
own. It runs only in the proposer role (it proposes queries; the store decides
what converges; a hallucinated seed fails to converge and is not admitted) and
under the existing freeze/cache contract (`CachingLanguageModel`, keyed on the
observation). It is not on the V1 critical path.

## Why convergence beats the component baselines

The two ways to combine seeds have opposite precision behavior:

- disjunctive (OR): merge every seed's hits. This is baseline 5. More seeds means
  more spurious matches reaching the decision.
- convergent: rank by cross-seed agreement. Single-seed noise ranks low and falls
  below `T`; the memory that several independent signals agree on rises.

Neither component computes cross-seed agreement, and neither forms a query for a
cue with no single salient entity or sentence. The burnout case is the worked
example: "desk lunches," "missed workouts," and "weekend work" are disconnected
in the observation but converge on one memory region through the graph. A
single-entity search never forms the query; a disjunctive union drowns it;
convergence surfaces it.

## Behavior on cue-ablated twins

Precision on the controls falls out of the same mechanism, through ranking rather
than gating. Cue ablation removes one converging signal from the observation. The
true memory loses a seed, or a graph path, so its convergence score drops below
`T`, and PARM admits nothing. No separate suppression rule and no LLM are
required for the twin to come back empty.

## Relationship to the frozen-retrieval contract

PARM runs under `docs/retrieval-mode-axis-plan.md` with these points of contact:

- Same frozen artifact: `pages.jsonl`, `chunks.jsonl`, `embeddings.npy`,
  `links.jsonl`. No canonical run invokes `gbrain search`.
- Deterministic and fully traced: record the seed set, each seed's candidate
  list and per-seed ranks, the convergence components per admitted page, `T`, and
  the admitted set, so a run reconstructs from the index hash.
- The graph is load-bearing here, a conscious departure from that document's
  "graph traversal is optional" hedge. Relational and pattern cues need traversal
  to converge disconnected surface entities onto shared memory regions.

Deprecation: that document's shared-behavior rules "use a final top-k of 5,"
"admit all retrieved memories," and "do not use dynamic score thresholds" were
written as universal. They are no longer universal. Selection policy is
condition-dependent. The mode-matched baselines may keep fixed top-k for
comparability; PARM selects by threshold `T` and may admit zero. Fixed top-k
cannot express "admit nothing," which is the whole point of the ablated control,
so a universal top-k rule is incompatible with the PARM condition.

## Open questions

- Convergence aggregation: the exact function over seed count and seed strength,
  and how to normalize across heterogeneous seed types (lexical entity vs dense
  sentence) so neither dominates by construction.
- Seed independence: correlated seeds drawn from the same sentence should not
  count as multiple votes. Convergence needs seeds from dispersed regions to be
  meaningful.
- Seed extraction boundary (resolved): extraction is non-LLM by default —
  gazetteer entity match, noun-phrase chunking, classical salience prefilter, and
  the existing MiniLM embedder. No SLM on the V1 critical path; it stays an
  optional upstream booster in the proposer role only. Open sub-question: whether
  the graph alone proposes enough cross-entity composition, or the booster earns
  its power cost on the pattern cases.
- `T` calibration: whether one threshold holds across cases, or whether the
  distribution-adaptive fallback (a dominant convergence peak fires, a flat
  distribution admits nothing) is needed. Fixed `T` is the V1 choice; adaptive is
  deferred.
- Heuristic vs learned convergence weights.

## First experiment

Before building the ranker, test the one assumption everything rests on: that
convergence beats flat retrieval on the controls. Under threshold selection over
the frozen index, compare:

- flat: all-entity plus key-sentence union, ranked by best single seed.
- convergent: the same candidate pool, ranked by cross-seed convergence.

Score both on positives (does the gold memory get admitted) and on cue-ablated
twins (does the admitted set stay empty, and what is the false-intervention rate).
It is cheap, needs no new machinery beyond the seed union and the two ranking
functions, and it decides whether PARM's precision claim is real before any
selector is built. If flat already clears the twins, the contribution is thin and
worth knowing early.
