# Parallel Asynchronous Retrieval of Memory

## Motivating examples

Amara asks an agent to choose one Thursday-afternoon conference session. The
agenda is thousands of tokens long and contains hundreds of plausible names and
topics. Buried in it is a session by Chen Wei of NovaMind on Texas grid
reliability pilots. Nothing in Amara's prompt suggests searching personal
memory. Only after the agenda arrives does a stored diligence note become
relevant: she wants NovaMind's Texas pilot evidence before presenting the deal
to her partnership. A useful memory system should answer with Chen's session
label instead of the agenda-only favorite.

The same mechanism can be personal and quiet. A podcast feed mentions skipped
workouts, desk lunches, and repeated weekend work as burnout signals. Separate
notes show that pattern has recently appeared in Amara's life. The useful action
is to change the episode choice, not to recite her diary back to her.

These are not query-conditioned recall problems. The original prompts are
ordinary and independently answerable. The reason to retrieve appears only
inside a later agent output or tool result.

## 1. Thesis

PARM monitors generated output and tool responses for newly visible cues,
selects the few cues worth searching, retrieves relevant personal memory, and
admits it asynchronously when it should materially change the decision.

The central research question is precision under noise. Large outputs contain
many entities, relations, claims, and behavioral patterns. Searching all of
them produces spurious matches and unsafe admissions. PARM should outperform
both input-only retrieval and naive whole-output RAG by identifying the small
number of cues that change what the agent should do.

GBrain is the initial memory substrate: storage, provenance, timestamps,
embeddings, keyword search, and graph metadata. PARM is the recall policy:
cue detection, retrieval timing, memory admission, and causal decision
evaluation.

## 2. Scope boundary

PARM requires a visible cue introduced after the prompt:

```text
ordinary prompt
-> one large output or tool result
-> incidental cue
-> memory retrieval
-> changed final decision
```

This differs from MemGuide-style intent-conditioned retrieval, which identifies
an unfilled task slot from the current dialogue and retrieves memories useful
for completing it. That behavior is complementary, but it does not depend on a
later cue and is outside the core PARM score.

PARM also does not claim that output-conditioned retrieval, proactive
assistance, personal memory, or graph retrieval are individually new. The
proposed contribution is the benchmark and system problem at their
intersection: output/tool-cued personal-memory retrieval under large noisy
contexts, evaluated by beneficial decision change and false intervention.

## 3. Hypotheses

- **H1, coverage:** Output-cued retrieval recovers decision-relevant memories
  missed by prompt-only retrieval.
- **H2, precision:** Cue selection beats whole-output and all-entity retrieval
  on memory-admission precision and spurious-intervention rate.
- **H3, causal usefulness:** Admitting the right memory improves the selected
  action, while removing the cue eliminates the intervention.
- **H4, latency:** Asynchronous monitoring can improve decisions without adding
  the full latency of a blocking retrieve-then-generate loop.
- **H5, privacy:** A response policy can use private memory without exposing
  unnecessary source details.

## 4. System

```text
generated/tool context
-> sentence and list-item segmentation
-> entity, relation, event, and pattern extraction
-> cue-worthiness ranking
-> GBrain hybrid search
-> provenance, temporal, poison, and actionability filtering
-> asynchronous memory admission
-> final decision
```

Named entities are only one cue type. Useful triggers may be relational
(`CoreWeave -> pricing changed`), event-shaped (`Chen Wei -> grid session`), or
semantic patterns spanning several phrases (`desk lunches + missed workouts +
weekend work`).

Graph traversal is optional. It may improve provenance, temporal reasoning, and
diagnostics, but V1 should live or die on cue selection and decision change,
not on elegant multi-hop paths.

## 5. Benchmark

The first executable pilot uses the fictional Amara Life corpus and five
approved scenarios:

1. conference agenda and unresolved NovaMind diligence;
2. AI news digest and NovaMind's CoreWeave pricing dependency;
3. podcast feed and a personal burnout pattern;
4. warehouse vendor report and NovaTech counterparty risk; and
5. lunch search and an unsustainable desk-lunch pattern.

Each case has:

- an ordinary prompt;
- one committed 8-12K-token observation;
- exactly one requested final choice;
- a unique option title or name that is visible and naturally outputtable;
- an output-only decision;
- a materially different memory-conditioned decision;
- authoritative Amara sources and perturbation metadata;
- at least 25 visible cue candidates and three memory distractors; and
- a cue-ablated twin.

The remaining approved examples form the expansion queue. Health constraints
that do not yet exist in Amara remain proposed additions rather than being
silently inserted into the corpus.

## 6. Baselines

Planned comparisons should share the same memory store. None is considered
implemented merely because its name appears here:

1. no memory;
2. prompt-only RAG;
3. an agent with a memory tool available but no explicit user request to use it;
4. whole-output RAG;
5. retrieval over every extracted entity;
6. selected key-sentence retrieval;
7. gold-cue retrieval as a recoverability ceiling; and
8. the learned or heuristic PARM cue selector.

Later comparisons can add FLARE-style uncertainty triggers and proactive-agent
policies. They should not replace the simpler baselines that isolate the actual
V1 failure mode.

## 7. Evaluation

Primary metrics are beneficial decision-change rate, cue-ablated false
intervention rate, memory-admission precision/recall, spurious admission,
poison admission, abstention, and privacy overexposure.

Primary scoring is deterministic. It recognizes the gold title or name directly
in the model's natural-language response; models do not emit benchmark-only
IDs. A correct response may use memory silently and need not expose the
retrieval path. Optional traces record detected cues and retrieved/admitted
source IDs for diagnosis. LLM judging is reserved for scorer-disagreement
audits.

Hand-authored prediction rows validate benchmark mechanics without pretending
to be retrieval baselines. Publication claims require deliberately implemented
policies over the prepared GBrain corpus and the same response model.

## 8. Risks

- Cue-worthiness may be as hard as retrieval itself.
- Whole-output retrieval may look competitive on recall while being unusably
  noisy.
- Corpus perturbations can poison naive retrieval if provenance metadata is
  discarded.
- Counterfactual controls may be too easy if cue removal changes the document's
  style or length.
- Private-memory explanations may become invasive even when the decision is
  correct.
- Asynchronous injection may arrive too late or invalidate generation caches.
- The strongest simple baseline may erase the need for graph-aware scoring.

## 9. Milestones

1. Validate the five-case pilot and counterfactual controls.
2. Run all reference retrieval policies over the real GBrain index.
3. Add a model response adapter and human-review the paired decisions.
4. Implement cue selection and measure it against whole-output/entity search.
5. Expand the approved examples one at a time.
6. Measure latency and injection timing after retrieval quality is established.

## Selected references

References below were spot-checked against their official arXiv records on
June 30, 2026.

- Park et al. (2023). *Generative Agents: Interactive Simulacra of Human
  Behavior.* arXiv:2304.03442.
- Jiang et al. (2023). *Active Retrieval Augmented Generation.* arXiv:2305.06983.
- Gutiérrez et al. (2024). *HippoRAG: Neurobiologically Inspired Long-Term
  Memory for Large Language Models.* arXiv:2405.14831.
- Chhikara et al. (2025). *Mem0: Building Production-Ready AI Agents with
  Scalable Long-Term Memory.* arXiv:2504.19413.
- Du et al. (2025). *MemGuide: Intent-Driven Memory Selection for Goal-Oriented
  Multi-Session LLM Agents.* arXiv:2505.20231.
- *SYNAPSE: Empowering LLM Agents with Episodic-Semantic Memory via Spreading
  Activation.* arXiv:2601.02744.
- *RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for
  Knowledge Graph Question Answering.* arXiv:2601.19225.
- *HyperGraphPro: Progress-Aware Reinforcement Learning for Structure-Guided
  Hypergraph RAG.* arXiv:2601.17755.
- *ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large
  Language Models.* arXiv:2604.08064.
- *PASK: Toward Intent-Aware Proactive Agents with Long-Term Memory.*
  arXiv:2604.08000.
- *ProActor: Timing-Aware Reinforcement Learning for Proactive Task Scheduling
  Agents.* arXiv:2605.24900.
- *Ask Now, Use Later: Benchmarking the Proactivity Gap in Long-Lived LLM
  Agents.* arXiv:2605.28108.
