# How to Construct a PARMBench Scenario

This page defines what a valid PARMBench scenario is and how to build one from
a raw persona history. It is the canonical construction contract for the next
benchmark generation. The frozen Amara `benchmark_v1` procedure is preserved at
the end as a historical record of how the existing 54 cases were produced.

## What PARMBench supplies

A scenario gives a system under test exactly five things:

1. a raw, time-ordered persona history;
2. an ordinary task prompt;
3. a later visible observation or tool output;
4. a deterministic expected final output; and
5. paired causal controls.

Nothing else is model-visible.

## Retrieval-agnostic boundary

PARMBench prescribes no memory representation. It does not ship a required
graph, ontology, entity schema, embedding index, or gold retrieval path to
systems under test.

Systems may build any substrate they want from the same raw history: lexical or
sparse search, dense semantic search, knowledge graphs, agentic memory tools,
rerankers, or hybrids. Two systems are comparable because they read the same
history under the same budget, not because they share an index format.

Evaluator-only evidence annotations may identify source spans and minimal
witness chains. They exist so a run can be scored for grounding. They are never
retrieval inputs and never reach a system under test. Any graph, embedding
index, or cache is a system artifact, not benchmark truth.

## Acceptance criteria

A scenario is valid only when all twelve hold.

| # | Criterion | Requirement |
| --- | --- | --- |
| 1 | Raw-source support | The decisive personal fact is explicitly stated by the user or is a conservative, independently auditable inference. Assistant-authored suggestions, topical questions, and fabricated habitual claims are invalid. |
| 2 | Persona isolation | Retrieval can access only that persona's raw history. |
| 3 | Ordinary prompt opacity | The initial prompt alone does not reveal a useful memory query. |
| 4 | Late visible cue | A later observation contains a concrete affordance that makes the memory actionable. |
| 5 | Decision change | Without memory, ordinary visible evidence supports output A. With the grounded memory and the cue, output B is best. |
| 6 | Cue-ablated control | Removing only the decisive visible affordance makes memory retrieval unnecessary and restores output A. |
| 7 | Memory-included ceiling | Exposing the supported personal fact directly produces output B. |
| 8 | Deterministic scoring | The expected final output is a visible natural-language label that can be scored exactly. An LLM judge may be a secondary analysis tool, never the primary correctness oracle. |
| 9 | No construction signature | Do not repeat a fixed score pattern, option order, phrase such as "narrower choice," cue position, envelope layout, or answer role. |
| 10 | No judge leakage | Retrieval or admission prompts must not encode the benchmark's answer construction, ordinary winner, target rank, or expected decision change. |
| 11 | Faithful evidence handoff | A retrieved hit must expose the raw source text or a strictly attributable span. A synthesized preference must never silently replace weaker raw evidence. |
| 12 | Sealed evaluation | Freeze construction rules before creating held-out personas. Do not tune retrieval against the final test cases. |

Criteria 5, 6, and 7 are established by running `no_memory` on all three
variants before the scenario is used to judge any retrieval mechanism. The
desired pattern is output-only choice on the positive, output-only choice on
the control, and memory-conditioned choice on the ceiling. Repair or reject a
case that misses this pattern; do not change the underlying source fact to make
it pass.

## Evidence-support rules

Criterion 1 fails more often than any other, so gate it automatically before
human review.

- The decisive personal fact must be user-stated or a conservative auditable
  inference from user-authored text.
- A user asking about a topic is not a durable preference. A question about why
  royal courts used ceremonial codes does not support "the user enjoys
  historical dramas."
- One instance never justifies "usually," "often," "regularly," or
  "exclusively." Frequency words need repeated user-authored evidence.
- Assistant suggestions, rewrites, and recommendations are not user facts
  unless the user explicitly adopts them.
- Dataset-supplied preference fields do not override the raw conversation. Raw
  user-authored evidence always wins.

Every accepted scenario records the exact source span that supports the fact.
An expected final answer is not a success when the retrieved evidence does not
support that fact.

## Supply gates

Source support is one of two separate supply judgments, and the v1
relevance audit (`data/parmbench-v1-supply/relevance-audit-v1.md`) showed
it is the weakest gate on its own: a claim can be perfectly supported and
still be an anecdote no ordinary task turns on. Before construction, every
candidate fact passes two versioned, cached gates that record their
rejection reasons in provenance:

1. **Source support** (`evidence_gate`, rubric
   `personamem_source_support_v2`): did the user say it? A row whose grade
   carries any deterministic rejection, such as `topical_question_only`, is
   also a hard construction reject.
2. **Memory quality** (`memory_quality`, rubric
   `parmbench_memory_quality_v1`): is the fact durable or currently
   operative — a preference, constraint, exclusion, active commitment,
   stable relationship, owned item, accessibility need, or concrete
   schedule? Editing requests, questions, and in-session states are
   rejected deterministically before any model call. The gate also labels
   sensitive facts so `memory.sensitive_terms` is populated at the source.

The failure taxonomy shared by the gates lives in
`parm_bench.relevance_taxonomy`. Fairness (steps 6-7 below) proves a model
follows the intended A/A/B pattern; the supply gates are what make the
pattern worth following.

## Direct scenario generation

Construction is one generation call per gated fact, implemented in
`scripts/build_parmbench_simple_v1.py` under the versioned prompt family
`parmbench_construction_simple_v*`. The call receives the plain claim, the
raw user-authored span, and the triplet requirements, and performs the
reasoning a person would:

1. Name one plausible request in which the fact could change which option
   the person should pick. Plausible means a normal thing somebody might
   ask an assistant, even if it only comes up occasionally.
2. Write an ordinary option set for that request.
3. Give the winner a default advantage unrelated to the fact: popularity, a
   recommendation, price, convenience, condition, or availability.
4. Give the target one concrete property matching the fact, delivered as a
   late cue sentence.
5. Without the fact, the winner is the sensible answer; with the fact and
   the cue, the target is; with the cue replaced by a neutral detail, the
   winner is again, even for a reader who knows the fact.

The model may decline a fact it cannot turn into a plausible decision after
a serious attempt, and a declined row is dropped, not repaired. It is never
asked to classify the fact, select from a task ontology, or prove an
entailment chain first. Task family, relation type, and capability labels
may be derived afterward for analysis; they are not construction inputs,
and disagreement about them cannot reject an otherwise valid scenario.

Generation responses are cached by the sha256 of
`{prompt_version, model, input}` in
`data/construction-caches/parmbench-simple-v1`, so a build replays offline.
Per-row request hints and candidate variation numbers are part of the input
and are recorded in provenance. A seeded, model-free assembly then wraps
the core in an observation envelope, rotates the filler vocabulary per
scenario, and reassembles offenders until the batch-level
construction-signature checks pass.

Accepted scenarios face a short human checklist rather than an LLM validity
judge. A case is good when all answers are yes:

1. Did the user actually state the memory in their own words?
2. Is the request something a real person might ask?
3. Does the winner have a clear default advantage unrelated to the memory?
4. Does the target's late cue plainly connect to the memory?
5. Would a person without the memory still choose the winner?
6. Would a person with the memory reasonably switch to the target?
7. Does cue ablation restore the winner without redirecting the memory to
   another option?
8. Is the prompt ordinary and silent about memory retrieval?
9. Does the scenario avoid invented possessions, relationships, medical
   needs, or plans?

Deterministic assertions cover what they can (label uniqueness, cue and
ablation symmetry, leak checks, prompt contracts); the semantic questions
are answered by manual review, recorded per scenario in the pilot's
adjudication files. The first calibration round and its measured failure
classes are in `data/benchmark_parmbench_pilot_simple_v1/README.md`, and
each class became a counter-instruction in the next prompt version.

### Superseded: selection-predicate stage

Earlier construction inserted a mapper between the fact and the builder: a
selection predicate, a bounded task-family registry, relation-type routing,
an ordinary mechanism, and a capability label, followed by an LLM
decision-validity judge. Over 48 mapped supply rows this chain retained one
scenario, which manual adjudication then rejected; the causes are recorded
in `data/benchmark_parmbench_pilot_v5/README.md` and
`data/parmbench-v1-supply/pilot-v6-adjudication.jsonl`. The stage survives
for frozen replay only: `parm_bench.selection_predicate`,
`scripts/build_parmbench_v1_benchmark.py` versions v4 through v6, their
caches, and the anchor file
`data/parmbench-v1-supply/selection_predicate_anchors.json` (whose verified
raw spans the current path still reuses). None of its fields are inputs to
the direct generation path.

## Capability coverage

Keep the benchmark broad enough to compare retrieval methods without making
any structure mandatory:

- direct lexical personal facts;
- paraphrased semantic personal facts;
- schedules and prior commitments;
- relationships and named entities;
- negative preferences and exclusions;
- updated, contradicted, forgotten, or stale facts;
- one-hop relational cases encoded only in raw history; and
- abstention cases where related memory exists but should not affect the
  decision.

Each category needs positive and cue-ablated cases. Relational cases may later
reveal graph advantages, but a system must derive any graph from raw history
itself.

## Variation axes

Vary every scenario batch along these axes and measure the distribution
automatically. Manual inspection misses repeated templates and rank
signatures.

| Axis | What to vary |
| --- | --- |
| Task domain | Scheduling, purchasing, reading, travel, hiring, health logistics, and others |
| Source age | Recent, mid-history, and old user statements |
| Cue position | Early, middle, and late in the observation |
| Option ordering | Position of the memory-conditioned choice among visible options |
| Observation format | Catalogs, prose reports, tables, transcripts, mixed markdown |
| Visible evidence strength | Narrow and wide margins for the ordinary winner |
| Persona-history size | Multiple history-size buckets, not one corpus size |
| Lexical overlap | High and low surface overlap between cue and source span |
| Distractors | Number and plausibility of competing memories |

The ordinary evidence must not always work the same way. Do not build every
case as a lower-rated personalized option defeating a higher-rated generic
option. Rotate the mechanism that makes the ordinary winner defensible:
schedule fit, feasibility, stated requirements, recency, quality, cost, and
explicit constraints. A repeated 9.8-versus-8.8 rating pattern is a
construction signature and fails criterion 9.

Evaluate retrieval across history-size buckets of roughly 25k, 100k, 500k, and
1M or more tokens per persona. If the source dataset cannot support a bucket,
document the limitation rather than synthesizing filler.

## Scale and splits

Build in two stages.

Stage 1, construction calibration. Create and independently audit at least 100
supported base scenarios. These are development data. Their purpose is to
calibrate the construction rules, the evidence gate, and the fairness run.

Stage 2, credible release. Freeze the construction procedure, then use it to
create at least 500 independent base scenarios across at least 100 personas,
with persona-disjoint development, validation, and sealed test splits. A
1,000-scenario release is preferable if source quality permits.

Triplet variants are controls, not independent samples. All three variants
share one base scenario, so statistical analysis operates at the scenario
level. Always report the count of independent base scenarios alongside any case
count.

If cost or source quality prevents reaching the release target, finish a
rigorously validated 100-scenario development set plus a replayable frozen
builder for the remaining held-out construction. A 30-scenario tuned slice is
not publication-ready.

## Construction procedure

### Step 1: Select a persona and a supported fact

Read the raw history. Find a fact the user stated themselves, record the exact
span, and write a short faithful summary entailed by that span. Run the
evidence gate from the rules above before doing any further work on the case.

### Step 2: Write the ordinary task

The prompt states an ordinary task with an exact one-choice answer contract. It
must not name the person, project, entity, or preference that would make the
memory retrievable from the prompt alone.

### Step 3: Build the observation and the decisive cue

The observation is a realistic tool result or agent output. It contains a
defensible output-only winner supported by ordinary visible evidence, plus one
region carrying the affordance that makes the personal fact actionable. Plant
near-miss decoys so the target is not selectable by salience, unusual phrasing,
or length.

### Step 4: Generate the triplet

Each scenario produces three cases:

- positive: original prompt, original observation, memory-conditioned gold;
- cue-ablated: original prompt, one exact observation replacement, output-only
  gold; and
- memory-included: the positive observation plus an explicit memory preamble,
  memory-conditioned gold.

Positive and memory-included observations are byte-identical. The control
differs from the positive by exactly one declared replacement of similar length
and register.

### Step 5: Validate structure

Automated validators must check unique stable case IDs, source hashes against
the tracked corpus, identical positive and ceiling observations, exactly one
control replacement, cue and replacement symmetry, visibility of every expected
choice in its resolved observation, absence of the decisive entity or
affordance from the control, persona isolation, triplet completeness, prompt
opacity, source support, and construction-pattern repetition across the batch.

### Step 6: Establish fixture fairness

Run `no_memory` on all three variants and confirm the criteria 5 to 7 pattern.
Rerun every condition after any fixture change.

Deterministic construction and scoring do not establish semantic fairness. If
repairs accumulate, templates become topic-specific, or reviewers cannot
resolve failures from the declared choices alone, stop tuning the template. Add
a versioned judge rubric for construction review only, calibrate it against
sampled human review, and keep the deterministic checks as the provenance and
symmetry gates. The judge does not become the correctness oracle.

### Step 7: Review qualitatively

The structural validator cannot decide whether a choice is realistic. Human
review answers:

- Would a reasonable person accept the output-only choice without memory?
- Does the memory materially change the decision rather than add trivia?
- Is the target buried naturally in the observation?
- Are the decoys close enough to prevent keyword shortcuts?
- Does the control remove only the decisive relationship?
- Is the memory safe and necessary to use?
- Would a wrong but plausible answer reveal a retrieval failure, a judgment
  failure, or an ambiguous fixture?

Record any non-obvious repair in [the decision log](history/decisions.md).

### Step 8: Freeze before inspecting failures

1. Commit the cases, observations, builders, and validation tests.
2. Run the full baseline ladder against unchanged retrieval and judgment code.
3. Retain predictions, config sidecars, metrics, and response caches under a
   new namespace.
4. Publish the aggregate and split results.

After results have been inspected, that batch is development data. Preserve the
first-pass artifacts as the honest generalization measurement and give every
tuned result a new version. Build sealed test personas only from the frozen
procedure, and never tune retrieval against them.

## Frozen Amara benchmark_v1 procedure

The 18 scenario triplets in `data/benchmark_v1` were built before this contract
existed. The procedure is recorded here for reproducibility. Do not extend the
Amara set with it; new construction follows the contract above.

Each scenario was declared once in `SPECS` in `scripts/build_pilot_cases.py`
with these fields:

| Field | Meaning |
| --- | --- |
| `slug` | Stable readable scenario identifier |
| `prompt` | Ordinary task with an exact one-choice answer contract |
| `kind` | `tool_result` or `assistant_output` |
| `cue_type` | Human-readable cue taxonomy |
| `cue` | Complete target listing text |
| `replacement` | Symmetric cue-ablated listing text |
| `query` | Diagnostic description of the intended relationship |
| `sources` | Source ID, source path, and perturbation labels |
| `memory_text` | Short faithful memory summary used by the ceiling |
| `output_choice` | Best choice without the personal memory |
| `memory_choice` | Best choice after using the memory |
| `control_choice` | Optional control answer when replacement changes the label |
| `sensitive_terms` | Private phrases the answer need not expose |
| `example_number` | Approved catalog number |

The matching large-output fixture was declared in
`scripts/build_pilot_contexts.py`. Generated contexts reach 8,000 to 12,000
`cl100k_base` tokens, use unique natural-language labels, place the cue at
listing 147, carry a credible output-only lead near the top, and decorrelate
company, speaker, topic, detail, format, and template strides.

```powershell
$env:PYTHONPATH = 'src'
& 'C:\Users\karth\anaconda3\python.exe' scripts\build_pilot_contexts.py
& 'C:\Users\karth\anaconda3\python.exe' scripts\build_pilot_cases.py
parm-bench validate data\benchmark_v1
```

Fixed cue position, a single observation format, and a single source corpus all
fail criterion 9 of the current contract. Examples 19 and 20 from the catalog
stayed outside the executable set because their proposed health memories are
not present in `amara-life-v1`, which is the same rule criterion 1 now
generalizes: do not invent history to make a case executable.

## Verification checklist

```powershell
$env:PYTHONPATH = 'src'
& 'C:\Users\karth\anaconda3\python.exe' -m unittest discover -s tests
parm-bench validate data\benchmark_v1
git diff --check
```

A scenario is ready only when the generated files, builder specification,
recorded source span, evidence gate, no-memory fairness run, and first-pass
namespace agree.
