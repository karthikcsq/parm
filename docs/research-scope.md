# PARM Research Claim and Scope

## Problem

Many personal-memory systems retrieve from the user's prompt. That works when
the prompt contains the person, project, or preference needed to form a useful
query. It misses a different class of cases:

```text
ordinary prompt
-> large tool result or agent output
-> newly visible cue
-> relevant personal memory
-> better final decision
```

For example, a user asks an agent to choose one conference session. The prompt
contains no company name. A 9,000-token agenda later includes a NovaMind
session, and a stored diligence note says that NovaMind's Texas pilot evidence
is needed before the next investment discussion. The memory becomes useful
only after the agenda arrives.

Large outputs also contain many irrelevant entities and patterns. Searching
the whole output can recover the right memory while admitting hundreds of
wrong ones. PARM therefore studies precision under output noise, including the
ability to admit no memory when the decisive cue is absent.

## Thesis

PARM v1 is graphless. Its contribution is parallel, output-triggered
personal-memory retrieval:

1. detect cue-sized regions as output becomes visible;
2. launch memory searches for those regions in parallel;
3. ground candidate memories in raw user-authored evidence;
4. admit memory only when the visible region makes it useful to the current
   task; and
5. abstain otherwise.

Parallelism is an implementation property. The research claim is the
output-triggered timing plus the selective intervention policy. The final
response receives the triggering region beside a focused memory excerpt.

PARM is not defined by any retrieval substrate. Lexical search, dense search,
graphs, agentic memory tools, rerankers, and hybrids are competing ways to
generate candidates from the same raw history, and PARMBench treats them as
interchangeable submissions. Graph retrieval may later become another
candidate-generation channel behind the same admission interface; it is not
part of v1 and the benchmark never requires it.

The proposed contribution is the benchmark and retrieval problem at the
intersection of:

- a cue introduced after the prompt;
- personal long-term memory;
- large noisy tool or agent outputs;
- selective memory admission grounded in raw user-authored evidence; and
- paired causal evaluation through a cue-ablated control.

Output-conditioned retrieval, proactive assistance, graph retrieval, and
personal memory each exist independently. PARM does not claim ownership of
those broad categories.

## Scope boundaries

PARM requires a visible cue introduced after the original prompt. It does not
cover every useful form of personal-memory retrieval.

MemGuide-style intent-conditioned retrieval identifies a missing task slot
from the current dialogue and fetches a memory that can fill it. That behavior
is complementary. It does not require a later tool or agent output, so it is
outside the core PARMBench score.

PARMBench supplies raw, time-ordered persona history and nothing else that is
model-visible. Building a substrate from that history is the system's job, not
the benchmark's. In the current implementation GBrain prepares PARM's own
substrate of imported pages, chunks, embeddings, links, provenance, and
timestamps; a competing submission may prepare something entirely different
from the same history. PARM owns retrieval timing, cue-region selection,
candidate ranking, admission, focused evidence handoff, and causal evaluation.
Canonical benchmark runs load the tracked export directly and do not call
GBrain search.

The current benchmark evaluates one visible final choice. It does not yet
establish natural drafting quality, multi-step planning, asynchronous latency,
or broad product usefulness.

## Hypotheses and current evidence

| Hypothesis | Current evidence | Status |
| --- | --- | --- |
| H1: output-cued retrieval recovers memories missed by prompt-only retrieval | PARM V5 scores 14/18 positives; enhanced input RAG scores 6/18 | Supported on controlled development cases |
| H2: selective admission beats whole-output and all-entity retrieval on precision | PARM admission precision is 71.43%; broad RAG conditions range from 1.11% to 2.69% | Supported on controlled development cases |
| H3: the right memory improves decisions while cue removal suppresses intervention | PARM scores 30/36 positive/control decisions and 16/18 controls | Supported, with two remaining false interventions |
| H4: asynchronous monitoring improves decisions without blocking retrieval latency | The current runner evaluates retrieval and response quality, not live injection timing | Unevaluated |
| H5: private memory can guide a decision without unnecessary disclosure | Deterministic sensitive-term checks report no privacy exposure in the published run | Partially supported by a narrow check |

The [frozen expansion first pass](results/benchmark-expansion-first-pass.md)
measures generalization before tuning on the 13 added scenarios. The
[V5 result](results/benchmark-v5.md) measures the improved mechanism after
those failures informed fixture and retrieval changes.

## What the benchmark establishes

The present evidence supports a narrow claim: late output cues create a
retrieval opportunity that input RAG cannot see, broad output retrieval handles
with poor precision, and a prompted agent may fail to act on. A selective PARM
policy produces a better positive/control trade-off over the same memory store
and response model.

The evidence does not support a claim that PARM already works across real
personal-agent tasks, users, corpora, and model families. The Amara corpus is
fictional, the outputs are generated catalogs, and V5 uses the expansion as
development data. The next external-validity step is described in
[Real-World Evaluation](real-world-evaluation.md).

## Evaluation principles

Primary scoring stays deterministic:

- the positive must select the declared memory-conditioned choice;
- the cue-ablated control must preserve the output-only choice;
- the memory-included ceiling tests whether the response model can use the
  fact when retrieval is removed from the path;
- source admission is scored separately from the final decision; and
- run sidecars pin retrieval, prompt, model, and cache versions.

An LLM judge can extend this evaluation to natural responses after agreement
with human reviewers is measured. It should not replace the paired,
attributable mechanism test.

## Selected references

References are included as research context, not as claims that the cited
systems implement PARM's benchmark contract.

- Park et al. (2023), *Generative Agents: Interactive Simulacra of Human
  Behavior*, arXiv:2304.03442.
- Jiang et al. (2023), *Active Retrieval Augmented Generation*,
  arXiv:2305.06983.
- Gutiérrez et al. (2024), *HippoRAG: Neurobiologically Inspired Long-Term
  Memory for Large Language Models*, arXiv:2405.14831.
- Chhikara et al. (2025), *Mem0: Building Production-Ready AI Agents with
  Scalable Long-Term Memory*, arXiv:2504.13413.
- Du et al. (2025), *MemGuide: Intent-Driven Memory Selection for Goal-Oriented
  Multi-Session LLM Agents*, arXiv:2505.20231.
- *ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large
  Language Models*, arXiv:2604.08064.
- *PASK: Toward Intent-Aware Proactive Agents with Long-Term Memory*,
  arXiv:2604.08000.
- *Ask Now, Use Later: Benchmarking the Proactivity Gap in Long-Lived LLM
  Agents*, arXiv:2605.28108.
