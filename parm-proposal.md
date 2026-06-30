# Parallel Asynchronous Retrieval of Memory

*A research proposal for ambient, goal-aware memory retrieval.*
*Draft v0.4 - for discussion*

---

## Motivating example

A founder doing customer-discovery research asks an LLM to surface candidate startups by web search. One returned profile notes that a company was spun out of High Alpha, a venture studio. The founder keeps a personal knowledge graph that records, among other things, the years their entrepreneurship professor spent at High Alpha. He would very likely have a warm connection into the company.

The useful action, asking the professor for an intro, requires joining two facts from two sources: an entity that arrived in a tool response and got restated in the model's output, and an attribute sitting in a personal knowledge graph. The founder only found it by reading the company profile by hand. The model had both halves in front of it. It had emitted "High Alpha," and the professor's record was already in the personal knowledge graph. Nothing connected them, because memory lookups key off what the user asks, and the user asked about customer discovery, not about the professor.

PARM targets that gap. It does output- and tool-response-conditioned recall: after generated or tool context introduces new cues, it decides which of those cues deserve memory lookup, retrieves relevant personal memory, and surfaces a suggestion the user never thought to request.

## 1. Summary

Most LLM memory is explicit and input-triggered. A retrieval step fires on the user's query, looks up a store, and prepends the result before generation begins. PARM makes two narrow bets and runs the loop asynchronously alongside generation.

First, condition retrieval on the model's own output stream and on the contents of tool responses, not only the user's query, so an entity, sentence, option, or tool-result fact the model merely mentions can trigger a lookup even when it was never the focus of the request. Second, add a precision layer that selects which output/tool cues are worth searching and suppresses spurious memory matches, instead of naively searching every extracted entity.

The second bet exists for precision. Large agent outputs contain many plausible names, products, places, people, and constraints. Searching all of them against personal memory creates a flood of weak matches. PARM's central problem is cue selection under noisy output/tool context: surface the few memories that should change the answer, and suppress the rest.

Most of the ingredients here are not new. Neither output-conditioned retrieval nor proactive surfacing is. Generation-time retrieval is established (FLARE, Self-RAG), and proactive memory-grounded assistants are an active 2025 to 2026 subfield (ProActor, PASK). Goal- and intent-conditioned retrieval also exists (MemGuide, PKG, RPO-RAG), and graph memory systems offer useful stores. PARM's contribution is narrower: a benchmark and system loop for output/tool-conditioned cue selection, asynchronous memory injection, and spurious-correlation control. The first deliverable isolates this case, which current memory benchmarks do not.

## 2. Background and the gap

The relevant prior work falls into six rough strands.

- **Scoring foundations.** Generative Agents (Park et al., 2023) set the standard retrieval score combining recency, relevance, and importance, with top-ranked memories inserted into context.
- **Graph-structured memory.** HippoRAG (Gutiérrez et al., NeurIPS 2024; HippoRAG 2, 2025) pairs a knowledge graph with Personalized PageRank for associative, multi-hop recall. Production systems (Mem0, ECAI 2025; Zep/Graphiti; Cognee; LangMem) combine embeddings, entity graphs, and key-value stores with automatic write pipelines.
- **Memory as a subsystem.** MemGPT/Letta and memory-OS work (MemOS) treat memory as a managed, tiered resource outside model weights.
- **Goal- and intent-conditioned retrieval (2025 to 2026).** A fast-growing line argues semantic similarity is the wrong objective and retrieval should condition on task intent. MemGuide (AAAI-26) does intent-aligned retrieval plus a filter that reranks memory units by marginal slot-completion gain, reporting large task-success gains over similarity retrieval. PKG pre-stores meta-paths for goal-relevant traversal. RPO-RAG (WWW'26) learns relation-aware path preferences and argues against shortest-path heuristics. SYNAPSE (2026) revives spreading activation for associative recall, and ProGraph-R1 trains graph retrieval with a progress-toward-goal reward. This line is the closest prior art to PARM's salience mechanism, and PARM borrows from it directly. Every system in it is input- or intent-conditioned: it infers the goal from the current query and fires on the user's turn.
- **Generation-time retrieval.** FLARE (EMNLP 2023) already conditions retrieval on the model's generation, using the predicted upcoming sentence as the query and triggering when the model emits low-confidence tokens. Self-RAG learns an on-demand variant. Output-conditioned retrieval is established. Both are uncertainty-triggered and corpus-grounding for factuality. They retrieve when the model is unsure, to support what it is about to assert. PARM runs close to the inverse. It triggers on confident entity mentions (the model is not unsure about "High Alpha," which is the point), retrieves from personal memory, and surfaces an unsolicited suggestion rather than rewriting a sentence.
- **Proactive and anticipatory agents (2025 to 2026).** A subfield now studies assistants that surface help before being asked. ProActor (ACL 2026) builds agents that monitor conversations, identify emerging opportunities, and intervene when appropriate. PASK targets intent-aware proactive agents with long-term memory. Related work names a proactivity gap in long-lived agents (LLaMAPIE, GUM). PARM is one instance of this paradigm and does not invent it. What this work mostly leaves open is how to evaluate output/tool-conditioned memory triggers inside noisy agent contexts.

Seen together, the three leave a gap. None of them directly evaluates the combination of large noisy output/tool context, cue selection, personal-memory retrieval, asynchronous injection, and resistance to spurious matches. Two practical limitations also recur in the memory line PARM builds its store from. Retrieval is usually input- or query-conditioned (ImplicitMemBench, 2026, reports explicit-retrieval mechanisms miss implicit memory), and it is serial and blocking (the retrieve-then-generate bottleneck noted in MemOS), which discourages frequent fine-grained retrieval.

The natural objection is that none of this is needed, that you could give the model a memory-search tool and tell it to retrieve when a memory seems relevant. That fails on the cases that motivate PARM. A self-triggered search fires only on relevance the model already suspects, but the useful cue often appears later as a throwaway detail in a long output or tool result. The model emits "High Alpha" as one descriptor among many; to choose to search on it, it would have to already suspect the connection it has not yet retrieved. Two further problems compound this. Self-directed tool use is unreliable, with tool selection easily biased (Faghih et al., 2025). And a prompt fires on the model's reasoning, not its emitted output, so many tool-result or generated cues never reach the trigger. Prompting reaches input-cued, model-salient retrieval, which is roughly what shipping product memory already does. Rather than threatening the work, this objection defines a primary baseline, alongside naive output-RAG over every extracted cue.

## 3. Core idea and hypotheses

The shift is from "retrieve once, on the input, before generating" to "retrieve continuously, on the unfolding output and on tool responses, during generation," a setting FLARE and the proactive-agent line already occupy. The specific elements PARM adds are cue selection in large noisy output/tool contexts, personal-memory retrieval after those cues become visible, and asynchronous injection with enough precision to be useful.

- **H1 (coverage).** Output/tool-conditioned retrieval surfaces relevant memories that input-conditioned retrieval misses, improving accuracy on tasks where the cue is absent from the prompt.
- **H2 (latency).** An asynchronous design adds negligible wall-clock latency versus a blocking retrieve-then-generate baseline at equal retrieval frequency.
- **H3 (cue selection, the headline).** A PARM monitor over generated/tool context beats both input-only memory and naive output-RAG that searches every extracted entity or sentence. The gain should come from finding the few cues that matter while avoiding spurious correlations.
- **H4 (precision under high-frequency watching).** Salience filtering and reranking beat relevance-only output-conditioned retrieval on surfaced-suggestion precision and useless-intervention rate.

## 4. How the system works, end to end

The pipeline has six steps. Steps 2 to 5 carry the contribution; the memory store can be a simple page/fact store with extracted entities, keyword indexes, embeddings, timestamps, and optional light graph metadata.

1. **Watch the output and tool responses.** As the model generates, read the tokens as they emit, and read entities arriving in tool results. This is the output-conditioning. PARM attends to what the model says and what its tools return, not only what the user asked.
2. **Segment and extract candidate cues.** Split output/tool text into sentences, list items, options, and claims. NER and phrase extraction pull named items and distinctive constraints from the stream ("High Alpha," "Acme," "UPS Store first at 9 AM").
3. **Select search-worthy cues.** Rank cue candidates by how likely they are to affect the user's current decision, not merely by whether they are named entities. This is where PARM avoids querying every item in a long tool result.
4. **Search memory with a simple hybrid stack.** For selected cues, use vector similarity, keyword/BM25-style search, and entity overlap over a realistic personal-memory store. A graph can help diagnostics or reranking, but multi-hop traversal is not required for V1.
5. **Rerank and surface asynchronously.** Suppress spurious matches and raise only actionable suggestions ("spun out of High Alpha; your professor worked there, want a warm intro?") while the model keeps writing.
6. **Write back.** New facts can be added to the memory store for next time.

## 5. The salience mechanism (the precision layer)

For V1, salience is a practical retrieval-filtering problem, not a commitment to a learned graph reasoner. The default implementation can be deliberately simple: cue segmentation, NER, hybrid memory search, and reranking by actionability. Graph structure remains useful for provenance, stale-edge handling, diagnosis, and later ablations, but the benchmark should not require beautiful multi-hop paths.

A minimal PARM monitor can run this loop:

```text
output/tool text
-> sentence and item segmentation
-> NER and distinctive phrase extraction
-> cue salience filter
-> hybrid memory search: vector + keyword + entity overlap
-> actionability and spurious-match rerank
-> asynchronous memory injection
```

Graph-aware scoring is an optional later rung. It can help explain why a cue connected to a memory, identify stale edges, and diagnose failure modes. But the V1 paper should live or die on cue selection in noisy output/tool context, not on whether a learned graph model beats a random walk.

## 6. Research questions

- RQ1: How much accuracy on cue-in-output tasks comes from output-conditioning specifically, holding store and scorer fixed?
- RQ2: Does cue selection plus reranking beat naive output-RAG that searches every extracted entity or sentence?
- RQ3: Do graph-aware salience methods beat a simple Mem0-style hybrid retrieval pipeline, and on which cases?
- RQ4: What injection mechanism keeps added latency negligible without polluting context?
- RQ5: How sensitive is everything to cue-extraction and reranking quality?

## 7. Evaluation plan

**The benchmark (first deliverable).** A noisy-output cue-selection set modeled on the High Alpha case: the initial user prompt does not make the agent search for the relevant memory, but generated output or a tool response later introduces many possible cues. One or a few of those cues should trigger useful personal-memory retrieval; many others are spurious correlations. The set is built first, both because it is the instrument every later experiment needs and because it answers the prior question of whether the phenomenon is frequent enough to be worth a method. Its value does not depend on PARM working: it isolates a retrieval timing and cue-selection case nothing currently measures. Distractor design and an independent pass over the items to confirm the baselines fail for principled reasons are part of the construction, not an afterthought. Alongside it, use LongMemEval and LoCoMo for long-session conversational memory and ImplicitMemBench for implicit behavioral adaptation.

**Baselines.** (i) no memory; (ii) input-conditioned RAG; (iii) prompted memory tool, a model instructed to retrieve when relevant; (iv) naive output-RAG that searches every extracted output/tool entity or sentence; (v) salience-filtered output-RAG with simple heuristic cue selection; (vi) a Mem0-style hybrid search baseline using vector, keyword, and entity overlap; (vii) a FLARE-style generation-time retriever, output-conditioned but uncertainty-triggered; (viii) a proactive-agent baseline (ProActor or PASK style) over the same store; (ix) PARM with synchronous blocking retrieval, to isolate async; (x) full PARM with asynchronous cue selection and reranking. Baselines (iii) to (vi) are the headline comparisons for H3.

**Headline experiment.** PARM versus input-only memory, the prompt baseline, naive output-RAG, and simple hybrid output retrieval on the noisy-output cue-selection set. The theory predicts input-only and prompted retrieval miss cues that appear only after output/tool context is produced. Naive output-RAG retrieves too many spurious memories. Simple hybrid output retrieval improves coverage but still needs cue/actionability filtering to avoid noise. Beating these on correct surfaced suggestions and useless-intervention rate is the result; failing to means there is no paper.

**Metrics.** Task accuracy; added wall-clock latency and tokens per second (H2); surfaced-suggestion precision and recall, plus a recommendation-fatigue rate, the fraction of injected items judged useless (H3, H4); retrieval precision and recall; context-overhead tokens.

Published benchmark numbers in this area are largely vendor-reported and not independently replicated. Re-run all baselines in-house rather than cite leaderboard figures.

## 8. Contributions (claimed)

1. **A noisy-output cue-selection benchmark.** An evaluation set isolating the case where relevant memory cues appear only in output or tool-response content, surrounded by many tempting but spurious cue candidates. Nothing currently measures this directly. It stands whether or not the method below wins.
2. **Output- and tool-response-conditioned, asynchronous retrieval.** A trigger on generated/tool context, with surfacing that runs alongside generation without blocking, positioned inside active retrieval (FLARE/Self-RAG) and proactive agents (ProActor/PASK) rather than claimed as a new paradigm, with a structural argument for why a "retrieve when relevant" prompt cannot reach this class.
3. **Cue salience and spurious-correlation control for precision.** A filtering and reranking layer that keeps high-frequency output watching from flooding the user. The default version can use segmentation, NER, hybrid search, and actionability reranking; graph-aware scoring remains an optional extension rather than the V1 dependency.
4. A prototype and injection mechanism that integrates surfaced memory mid-generation without blocking.

## 9. Risks and limitations

- **Cue selection is the crux.** Long tool outputs contain many entities and sentences that are plausible memory queries. A weak cue selector either misses the relevant cue or floods the user with spurious matches.
- **Training signal for usefulness is sparse.** "This surfaced memory was useful" is expensive to label, which puts learned rerankers in weak-supervision and outcome-reward territory, with cold-start.
- **Graph-aware scoring may not beat simple hybrid retrieval.** If a tuned vector/keyword/entity pipeline is enough, the contribution narrows to output-conditioning, asynchronous injection, cue-selection evaluation, and the benchmark. That would still be a useful result.
- **Latency may not hide cleanly.** If injection forces KV-cache invalidation or re-decoding, the async claim weakens, so this is worth measuring early.
- **Linking and validity errors compound.** NER and linking errors on the output stream, plus stale memory facts, can feed garbage into an otherwise-correct monitor. Temporal validity and provenance matter.
- **Context pollution.** Frequent injection can hurt, since LLMs are sensitive to proactive interference.
- **Privacy.** Ambient, non-prompted memory takes away the user's explicit recall trigger. That is where the usefulness comes from, and it is also what makes the system hard to audit, so a consent and inspection surface belongs in the design.
- **Scope versus in-weights memory.** A competing school (Titans, memory-layer architectures) puts memory inside the model. PARM is deliberately external and non-parametric, and I should say so up front.

## 10. Rough milestones

1. *Weeks 1 to 4* - Build the noisy-output cue-selection set and the independent fairness pass. Stand up a simple memory store with vector, keyword, entity, and light graph metadata. Reproduce the input-conditioned, prompted-memory, and naive output-RAG baselines. By the end of this block, know whether the phenomenon is frequent enough to justify the rest.
2. *Weeks 5 to 8* - Output and tool-response monitor, sentence/item segmentation, NER, hybrid memory search, and cue/actionability reranking. Validate H1 and the H3 headline against the prompt and naive output-RAG baselines before optimizing anything.
3. *Weeks 9 to 12* - Async injection; measure H2; resolve the race-condition policy; add context-overhead and fatigue metrics.
4. *Weeks 13 to 15* - Optional graph-aware salience ablations, FLARE/proactive comparisons, and write-up.
## Selected references

*(arXiv IDs verified against live listings during drafting where shown. Re-confirm the recent 2026 IDs at submission.)*

- Park et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* arXiv:2304.03442.
- Gutiérrez et al. (2024). *HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs.* NeurIPS 2024.
- Gutiérrez et al. (2025). *From RAG to Memory: Non-Parametric Continual Learning for LLMs* (HippoRAG 2). arXiv:2502.14802.
- Chhikara et al. (2025). *Mem0.* ECAI 2025; arXiv:2504.19413.
- Rasmussen et al. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory.* arXiv:2501.13956.
- Packer et al. (2023). *MemGPT: Towards LLMs as Operating Systems.*
- *MemOS: A Memory OS for AI Systems* (2025). arXiv:2507.03724.
- *ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in LLMs* (2026). arXiv:2604.08064.
- Du et al. (2026). *MemGuide: Intent-Driven Memory Selection for Goal-Oriented Multi-Session LLM Agents.* AAAI-26; arXiv:2505.20231.
- *Pseudo-Knowledge Graph: Meta-Path Guided Retrieval and In-Graph Text for RAG-Equipped LLM* (2025). arXiv:2503.00309.
- Um et al. (2026). *RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for KGQA.* WWW'26; arXiv:2601.19225.
- *SYNAPSE: Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation* (2026). arXiv:2601.02744.
- *ProGraph-R1: Progress-aware Reinforcement Learning for Graph Retrieval-Augmented Generation* (2026). arXiv:2601.17755.
- Faghih et al. (2025). *Tool Preferences in Agentic LLMs are Unreliable.* EMNLP 2025; arXiv:2505.18135.
- Jiang et al. (2023). *Active Retrieval Augmented Generation* (FLARE). EMNLP 2023; arXiv:2305.06983.
- Asai et al. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024.
- *ProActor: Timing-Aware Reinforcement Learning for Proactive Task Scheduling Agents* (2026). ACL 2026; arXiv:2605.24900.
- *PASK: Toward Intent-Aware Proactive Agents with Long-Term Memory* (2026). arXiv:2604.08000.
- *Ask Now, Use Later: Benchmarking the Proactivity Gap in Long-Lived LLM Agents* (2026). arXiv:2605.28108.
