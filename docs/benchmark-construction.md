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

## Relevance gates

Source support is one of three separate judgments, and the v1 relevance
audit (`data/parmbench-v1-supply/relevance-audit-v1.md`) showed it is the
weakest gate on its own: a claim can be perfectly supported and still be an
anecdote no ordinary task turns on, or a well-built scenario can wrap a real
fact in an invented permission. Construction therefore runs three versioned
gates, each cached and each recording its rejection reasons in provenance:

1. **Source support** (`evidence_gate`, rubric
   `personamem_source_support_v2`): did the user say it?
2. **Memory quality** (`memory_quality`, rubric
   `parmbench_memory_quality_v1`, pre-construction): is the fact durable or
   currently operative — a preference, constraint, exclusion, active
   commitment, stable relationship, owned item, accessibility need, or
   concrete schedule? Editing requests, questions, and in-session states are
   rejected deterministically before any model call. The gate also labels
   sensitive facts so `memory.sensitive_terms` is populated at the source.
3. **Decision validity** (`decision_validity`, rubric
   `parmbench_decision_validity_v1`, post-construction): does this exact
   task, cue, and choice change follow from the fact without invented
   assumptions? The auditor sees the fixture roles — it audits dataset
   quality and never touches benchmark answer scoring. The construction
   model must also return an evaluator-only causal chain (why A wins, why
   the cue is neutral without memory, why memory plus cue prefers B, what
   assumptions are required, why the control removes the advantage), and any
   material assumption rejects the core before the gate is called.
   Deterministic backstops reject lexical residue of the claim in the
   target's name or ablated body and relational capability labels whose
   answer is named after the memory's own words.

The failure taxonomy shared by the gates lives in
`parm_bench.relevance_taxonomy`. Fairness (steps 6-7 below) proves a model
follows the intended A/A/B pattern; the relevance gates are what make the
pattern worth following.

## Selection-predicate stage

The v3 pilot showed that gates alone cannot rescue a bad construction
premise: asking one model call to invent the task, the affordance, and the
causal relationship around a fact makes valid scenarios needlessly rare.
Construction therefore starts from the decision rule the fact directly
implies, and only then generates a familiar task around it:

```text
raw user history -> supported durable fact -> concrete selection predicate
-> compatible task family -> ordinary winner A + compatible target B
-> cue-ablated control
```

Before any scenario is constructed, a versioned mapper produces
evaluator-only fields. The names below are the schema contract between the
mapper and the builder:

| field | meaning |
| --- | --- |
| `selection_predicate` | a concise condition that follows directly from the raw fact |
| `task_family` | a bounded family in which that condition naturally affects a choice |
| `target_affordance` | the visible, checkable property of B that satisfies the predicate |
| `ordinary_mechanism` | the visible reason A wins for a person without this memory |
| `control_affordance` | a neutral replacement that removes the predicate match without creating a new personalized match |
| `relation_type` | direct_constraint, compatibility, active_project_relevance, schedule_fit, accessibility_need, stable_preference, or relationship_obligation |
| `material_assumptions` | must be empty; a non-empty list rejects the mapping |

The predicate must express an action-relevant implication, never a topic
restatement. "The user owns an NES" maps to "prefer vintage items usable
with an NES", not to "the user likes retro games". A supported durable fact
with no direct selection implication is a correct abstention, not a mapping
failure.

Task families are routed by compatibility, from a small auditable registry:
food preferences or exclusions to menu, dessert, grocery, or catering
selection; active research or work topics to books, talks, courses,
archives, or reading groups; owned devices or items to compatible games,
accessories, parts, repair, or vintage finds; hobbies to events, supplies,
clubs, or workshops; concrete schedules to appointment, event, delivery, or
travel slots; accessibility needs to rooms, routes, seating, transport, or
delivery; stable relationships or obligations to gifts, visits,
communications, or travel, only when the relationship itself is directly
supported. Global domain balancing is subordinate to compatibility: variety
is applied to envelopes, names, ordering, and surface form after the causal
structure is valid, and the construction-signature checks still run on the
finished set.

Three user-approved anchor scenarios define the quality bar and are frozen
with their verified raw spans in
`data/parmbench-v1-supply/selection_predicate_anchors.json`: a dessert menu
against a stated fresh-fruit habit, a reading-group book against an active
theology-and-pluralism writing project, and a vintage-store find against an
owned NES. A generator version that cannot produce these three examples is
not fit to generate anything else.

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
