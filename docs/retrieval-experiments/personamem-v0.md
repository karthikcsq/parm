# PersonaMem V0 retrieval experiments

**Status note:** this log records development diagnostics over the legacy
`benchmark_personamem_v0` and `benchmark_personamem_mixed_v0` slices, not
benchmark results. Both slices are development-only: 13 of 30 labeled
personal-memory claims are unsupported by raw source, and the mixed slice
carries a repeated 9.8/8.8 construction signature plus admission-judge
leakage of construction details. None of the results below support a
superiority claim for any retrieval method. See
`docs/benchmark-construction.md` for the canonical construction contract.

This log records each retrieval design tried against the frozen 30-scenario
PersonaMem development benchmark. The benchmark cases, persona corpora, gold
source IDs, and frozen Amara artifacts do not change during these experiments.

## Evaluation discipline

- Use the 30 positive and 30 cue-ablated cases for retrieval diagnostics.
- Keep the 30 memory-included cases as the response-model ceiling.
- Keep V0 frozen as the clean-listing diagnostic slice. Do not accept a
  retrieval design based only on V0 gains.
- Build a separate mixed-format development slice with prose, noisy markdown,
  pasted threads, uneven tables, and working notes. Candidate names may remain
  visible for deterministic decision scoring, but retrieval cannot depend on a
  `Listing ...` prefix, one-item-per-line structure, or a fixed target position.
- Report gold admission recall on positives, admission precision, and false
  admissions on controls before spending response-model calls.
- Prefer rules that operate on region and memory evidence rather than scenario
  names, listing positions, benchmark choices, or gold annotations.
- Run the response model only after an offline retrieval design improves the
  admission frontier.
- Record failed attempts. A rejected design is still useful evidence.

The first-pass end-to-end PARM result is the fixed comparison point:

| Metric | First pass |
| --- | ---: |
| Positive decision accuracy | 5/30 |
| Cue-ablated control accuracy | 28/30 |
| Gold admission recall | 4/30 |
| Admission precision | 50.0% |
| Spurious admission rate | 50.0% |

All six positive admissions came from the direct-note BM25 contrast channel.
Four were gold. The channel found the gold page as its best candidate in only
six positive cases, so threshold tuning alone cannot solve the recall problem.
The prompt-anchor semantic channel admitted no PersonaMem pages because it was
designed around Amara review/reflection page conventions that the new corpus
does not use. PersonaMem has no graph links, so its entity-graph channel is
also inactive.

The first pass also exposes a benchmark-shape problem: all PersonaMem
observations are orderly recommendation lists. Optimizing only on these cases
would reward catalog parsing rather than cue-triggered reading over general
agent context. The next benchmark version therefore changes the observation
envelope while preserving the same causal triplet:

```text
ordinary prompt
-> messy later context mentions an affordance in prose
-> prior memory becomes useful
-> one visible action or name becomes better
```

The V0 lists remain useful because their exact symmetry makes retrieval
failures easy to diagnose. They are no longer sufficient evidence for a
general PARM improvement.

### Source-support audit

The upstream `preference` label and `related_conversation_snippet` are not
always equivalent evidence. One current example labels a preference for going
to the gym three times per week, while its related snippet states only that the
user has a standing early-morning commitment on Monday, Wednesday, and Friday.
The system persona block explains the missing relation, but the settled corpus
contract correctly excludes that hidden profile from the retrieval index.

Exact gold-source recall is therefore not achievable on every selected row
from raw history alone. A `gpt-5-mini` audit using the versioned
`personamem_source_support_v1` rubric graded 8 sources explicit, 9 inferable,
and 13 unsupported. The audit reads only the memory claim and indexed
conversation. It does not receive the hidden persona profile, benchmark
choice, or observation.

This semantic boundary is an appropriate use of an LLM judge. A keyword rubric
would reproduce the same source-label error in a less visible form. The audit
is still not a replacement for human review: one concurrent first pass moved
the borderline morning-run source between inferable and unsupported. Its
current unsupported grade is defensible because the source says "motion" on
quiet streets but never says running or fitness. Report retrieval on the 17
supported or inferable rows separately, sample the judge rationales manually,
and replace unsupported rows in the next benchmark revision.

## Experiment 1: mutual region-memory alignment

Status: candidate generator retained; admission rule rejected.

Hypothesis: the useful cue is a distinctive output region whose complete
meaning aligns with one part of the user's history. Embed the complete
distinctive region instead of many generic prompt-anchor and noun pairs.
Compare it with sentence and page embeddings inside the requested persona
corpus. A candidate is stronger when:

1. it is the best memory page for that region;
2. that region is also the best output region for the memory page;
3. it beats the next memory page for the same region; and
4. it beats the same memory page's score on other distinctive regions.

This is a bipartite matching view of retrieval. It preserves zero-admission
behavior and does not require the old GBrain graph export. Sentence-level
memory units retain the useful part of GBrain-style fine-grained indexing
without treating a whole four-turn page as one vector. The production version
must align general semantic blocks rather than listing rows.

The first probe compares sentence-max, sentence-top-three, chunk, and blended
dense scores. It writes feature rows before any thresholds are selected so the
same cached embeddings can be replayed.

### Result

Complete-region dense retrieval is a much better candidate generator than the
current PARM channels, but it is not a safe admission rule by itself:

| Candidate method | Gold page and cue in top 1 | Top 3 | Top 10 |
| --- | ---: | ---: | ---: |
| Blended sentence/page dense | 8/30 | 19/30 | 23/30 |
| BM25 page text | 6/30 | 11/30 | 13/30 |
| Union of dense and BM25 probes, supported subset | - | - | 16/17 |

The union result counts the gold cue-region and gold page appearing together
in the top-ten outputs from any of the five retained candidate methods. Dense
alignment recovers useful evidence that the first pass never considered.
Adding more BM25 weight usually makes the ranking worse.

The same format-agnostic probe remains viable on the mixed slice. Its five-way
candidate union contains the exact cue-region and source for 15/17 supported
positives. The blended dense method alone has 13/17 in its top ten. This is
slightly below the clean slice, but it shows that semantic blocks retain most
candidate recall without a `Listing` parser.

A scalar threshold cannot cleanly turn this into admission. The top blended
score for supported positives ranges from 0.376 to 0.543; controls range from
0.339 to 0.500. Their medians are 0.436 and 0.407. Threshold selection inside
that overlap would be benchmark-specific fiddling and would discard many real
positives while retaining hard controls.

## Experiment 2: semantic admission judge

Status: V1 rejected; V2 retained as a development candidate.

Hypothesis: use the dense and lexical methods only as a high-recall prefilter,
then use a small, versioned semantic judge as a cross-encoder. The judge sees
candidate pairs consisting of one visible region and one raw history source.
It must find all three of:

1. a durable fact supported by the user's own words;
2. a specific connection from that fact to the visible passage; and
3. enough decision relevance that recalling it could change the choice.

It returns one admitted pair or no retrieval. Assistant suggestions, generic
topic overlap, and invented relations are explicit rejection conditions. This
tests whether an LLM is useful at the semantic boundary where score thresholds
overlap, without letting it replace deterministic choice scoring or browse the
gold annotations.

### V1 result: reject non-contrastive relevance

The first rubric recovered 15 of the 17 supported positives and missed two,
but it also admitted 25 of the 43 cases that should be empty under strict
gold-source scoring. It regularly found a real personal fact that shared a
topic with a control passage, even when the fact did not create the intended
choice reversal. Relevance alone is too permissive.

The errors also exposed a second fixture problem. Several V0 controls replace a
positive affordance with its opposite. A memory for plain jars remains relevant
when the twin describes ornate jars, but it reinforces the already-leading
choice instead of changing it. Other controls collide with unrelated, genuine
facts elsewhere in the persona history. Exact non-gold admission metrics count
both as equally spurious even though they are semantically different.

### V2 result: contrastive admission

V2 receives the complete observation and must establish that a source-backed
fact gives a reason to choose a lower-ranked viable option over the ordinary
evidence winner. A title or topic match is insufficient, and memory that only
reinforces the existing winner must be rejected.

On V0, V2 admitted all 17 supported positives and no supported positive was
missed. It admitted 25/30 positives overall and 12/30 controls. Eighteen
admissions selected the exact intended cue region and gold source. The
remaining positives include cases where the judge found different user-history
evidence that genuinely supports the benchmark target, which is useful for
decision accuracy but is scored as spurious by exact source ID. The 12 control
admissions remain too many for retrieval-only acceptance.

The downstream decision run confirms a real gain with a precision cost:

| Metric | First-pass PARM | V2 semantic admission |
| --- | ---: | ---: |
| Positive decision accuracy | 5/30 | 25/30 |
| Cue-ablated control accuracy | 28/30 | 19/30 |
| All retrieval-case decisions | 33/60 | 44/60 |
| Gold source recall | 4/30 | 18/30 |
| Exact-source admission precision | 50.0% | 48.6% |

The response model followed all 25 positive admissions to the intended choice.
Eleven of the 12 control admissions changed the control decision incorrectly.
V2 therefore improves total decision accuracy by 11 cases and positive
coverage by 20 cases, but it does not yet meet PARM's precision objective.
All 17 positives whose intended source was explicit or inferable were correct.
The other eight correct positives used different history evidence in scenarios
whose labeled source was unsupported.

The next evaluation uses the repaired mixed slice, whose neutral candidate
names and genuinely neutral cue ablations remove the opposite-affordance
artifact. If control precision remains poor there, the judge itself needs
another design change; if it improves, V0's control construction is the main
confound and should stay as a documented legacy diagnostic.

### Mixed-context result

The mixed slice changes the result materially. With semantic blocks, the same
five candidate views, candidate depth seven, and the frozen V2 admission
cache, the judge admits 22/30 positives and 4/30 cue-ablated controls. Fifteen
admissions use the exact intended source and cue region. The downstream run
gets 21/30 positive decisions and 27/30 controls correct, or 48/60 retrieval
cases overall:

| Metric | No memory | Deterministic PARM | Semantic PARM |
| --- | ---: | ---: | ---: |
| Positive decision accuracy | 0/30 | 3/30 | 21/30 |
| Cue-ablated control accuracy | 30/30 | 29/30 | 27/30 |
| All retrieval-case decisions | 30/60 | 32/60 | 48/60 |
| Cue-ablated false intervention rate | 0/30 | 1/30 | 3/30 |
| Exact gold source recall | 0/30 | 2/30 | 15/30 |
| Exact-source admission precision | - | 40.0% | 57.7% |

The original listing-only parser would produce no regions on these contexts.
Combined with the validated no-memory outputs, that corresponds to the 30/60
starting point above. Replacing the parser with semantic blocks lets the
existing deterministic waterfall gain two decisions. Rebuilding candidate
generation and adding contrastive admission gains another 16. The final path
therefore improves 18 decisions without depending on a `Listing` prefix or
orderly rows.

The strict source-supported subset remains the clearest diagnostic. The
candidate union and judge select the exact intended pair for 15/17 explicit or
inferable sources. One miss is a candidate-generation failure: the raw history
quotes “The night has ears,” while the cue calls the target an African-proverb
anthology, so neither lexical nor embedding retrieval brings the labeled page
into the candidate set. The other is an admission failure: a source asks why
screen-free days improve Monday clarity, but V2 declines to turn that question
into a durable preference. Fourteen of those 17 produce the intended downstream
choice; one correct spice-jar retrieval is followed by a response-model error
that copies an archival checklist label instead of the eligible target.

The three control interventions are also informative. V2 overinterprets weak
phrases such as “general audience,” “narrower,” and “specialized option” as
decision-changing affordances for unrelated real memories. Those are rubric
errors, not score-threshold errors. Editing another scalar cutoff or adding
case-specific forbidden terms would now be benchmark fitting, so this
iteration stops there. A V3 should be compared on a fresh held-out construction
split and explicitly require a concrete action, property, schedule, subject,
or relationship rather than a generic evaluative adjective.

The admission cache is keyed by the prompt, complete observation, model,
rubric, index hash, and candidate-policy namespace. The cached result stores
the selected page and exact region text. On replay, the page must still belong
to the requested corpus and the region text must still exist in the
observation. This keeps the expensive semantic judgment reproducible without
making harmless floating-point changes in live embedding ranks invalidate the
cache.
