# PARMBench V5 Development Result

## Outcome

On the repaired 54-case benchmark, PARM scores 30/36 primary decisions.
The strongest comparison conditions score 18/36.

PARM beats:

- enhanced input RAG by 15 decisions;
- naive hybrid output RAG by 12 decisions;
- all-entity output RAG by 12 decisions; and
- the prompted hybrid memory-tool agent by 13 decisions.

The result uses `parm_convergence_v2`, `parm_judgment_v3`,
`cue_region_memory_v1`, the frozen `amara-life-v1` retrieval index, top-k 5,
and requested response model `gpt-5-mini`.

## Aggregate matrix

| Condition | Primary decisions | Positive lift | Controls correct | Ceiling | Admission precision | Admission recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PARM V5 | **30/36** | **14/18** | 16/18 | 17/18 | **15/21 (71.43%)** | 15/20 (75.00%) |
| No memory | 18/36 | 0/18 | **18/18** | 18/18 | n/a | n/a |
| All-entity output RAG | 18/36 | 7/18 | 11/18 | **18/18** | 16/1,377 (1.16%) | **16/20 (80.00%)** |
| Naive output RAG, hybrid tool-then-model | 18/36 | 7/18 | 11/18 | 17/18 | 9/335 (2.69%) | 9/20 (45.00%) |
| Prompted memory tool, hybrid | 17/36 | 0/18 | 17/18 | **18/18** | 0/0 | 0/20 |
| Enhanced input RAG | 15/36 | 6/18 | 9/18 | 17/18 | 2/180 (1.11%) | 2/20 (10.00%) |

Primary decisions are positives plus cue-ablated controls. Positive lift alone
is not enough: a condition that injects memory everywhere can change many
positives while also changing the controls.

PARM admits 15 gold and 6 spurious sources across primary cases. All-entity
output RAG retrieves one more gold source, but admits 1,361 spurious sources to
do it. This is the central precision result.

Poison admission, privacy overexposure, and abstention are zero for PARM.

## Pilot and expansion splits

| Condition | Pilot decisions | Pilot positives | Pilot controls | Expansion decisions | Expansion positives | Expansion controls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PARM V5 | **10/10** | **5/5** | 5/5 | **20/26** | **9/13** | 11/13 |
| No memory | 5/10 | 0/5 | 5/5 | 13/26 | 0/13 | **13/13** |
| All-entity output RAG | 5/10 | 0/5 | 5/5 | 13/26 | 7/13 | 6/13 |
| Naive output RAG | 6/10 | 2/5 | 4/5 | 12/26 | 5/13 | 7/13 |
| Prompted memory tool | 5/10 | 0/5 | 5/5 | 12/26 | 0/13 | 12/13 |
| Enhanced input RAG | 4/10 | 1/5 | 3/5 | 11/26 | 5/13 | 6/13 |

PARM's expansion source admission precision is 62.5% and recall is 71.43%.
Pilot precision remains 100%.

## Improvement over the frozen first pass

The immutable first expansion pass remains the honest generalization record.
After analyzing it, repairing three fixture-fairness problems, and treating the
expansion as development data, PARM changed as follows:

| Metric | Frozen first pass | V5 development result |
| --- | ---: | ---: |
| Primary decisions | 26/36 | **30/36** |
| Positive lift | 10/18 | **14/18** |
| Controls correct | 16/18 | 16/18 |
| Admission precision | 60.00% | **71.43%** |
| Admission recall | 45.00% | **75.00%** |
| Expansion decisions | 17/26 | **20/26** |
| Expansion positive lift | 6/13 | **9/13** |

The implemented changes are:

- exact lexical tie-breaking among same-entity graph candidates;
- a contrastive BM25 channel over durable notes;
- precomputed direct-note BM25 documents;
- response judgment anchored to the retrieved decision locus;
- repaired no-memory fairness for phone, weekend, and human-factors fixtures;
  and
- bounded batching and retry for transient empty OpenAI response bodies.

## Remaining failures

Four positives still miss:

- `webinar-catalog`;
- `phone-feature-digest`;
- `weekend-events`; and
- `essay-digest`.

The first, second, and fourth receive no admitted memory. They need a
contrastive semantic region-to-memory channel rather than more lexical or
entity tuning.

`weekend-events` admits the gold weekly-review note against the wrong listing
region, `Event E-085`, which is present in both the positive and control. The
same region causes the weekend control failure.

The second control failure is `human-factors-event`. Its cue-ablated observation
still contains a confidence-display listing semantically close to the
orange-mode note. PARM admits that note against the remaining listing and the
model changes its answer.

The only ceiling miss is `vendor-report-memory-included`. It does not affect
the primary comparison, but it shows that retrieval and explicit prompt memory
can still interact nondeterministically in the response layer.

These failures argue against further tuning on this expansion. The next
retrieval mechanism should be calibrated on development cases and measured
once on a new held-out batch.

## Interpretation

This result supports the original mechanism claim. Prompt-only retrieval does
not see the late cue. Whole-output and all-entity retrieval recover some gold
memory but flood the response with unrelated pages and change controls. The
prompted agent does not search memory on any positive case. PARM uses the same
frozen store and response model but produces a substantially better
positive/control trade-off.

It is not yet a broad real-world product claim. The cases are controlled,
fictional, and choice-based, and V5 is tuned on the expansion. See
[Real-World Evaluation Strategy](../real-world-evaluation.md) for the proposed
paired end-to-end layer.

## Artifacts

PARM:

- `data/benchmark-results/parm-v5-gpt-5-mini.jsonl`
- `data/benchmark-results/parm-v5-gpt-5-mini.config.json`
- `data/benchmark-results/parm-v5-gpt-5-mini.metrics.json`
- `data/response-caches/amara-life-v5/parm`

Comparison artifacts use the revised-fixture V3 namespaces:

- `input-rag-enhanced-v3-gpt-5-mini.*`
- `naive-output-tool-then-model-hybrid-v3-gpt-5-mini.*`
- `all-entity-output-rag-v3-gpt-5-mini.*`
- `prompted-memory-tool-hybrid-v3-gpt-5-mini.*`
- `no-memory-v3-fixture-fairness-gpt-5-mini.*`
