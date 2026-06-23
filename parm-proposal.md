# Parallel Asynchronous Retrieval of Memory

*A research proposal for ambient, goal-aware memory retrieval.*
*Draft v0.3 · for discussion*

---

## Motivating example

A founder doing customer-discovery research asks an LLM to surface candidate startups by web search. One returned profile notes that a company was spun out of High Alpha, a venture studio. The founder keeps a personal knowledge graph that records, among other things, the years their entrepreneurship professor spent at High Alpha. He would very likely have a warm connection into the company.

The useful action, asking the professor for an intro, requires joining two facts from two sources: an entity that arrived in a tool response and got restated in the model's output, and an attribute sitting in a personal knowledge graph. The founder only found it by reading the company profile by hand. The model had both halves in front of it. It had emitted "High Alpha," and the professor's record was already in the personal knowledge graph. Nothing connected them, because memory lookups key off what the user asks, and the user asked about customer discovery, not about the professor.

PARM targets that gap. It does output- and tool-response-conditioned recall, joining a just-seen entity to a latent personal fact and surfacing a suggestion the user never thought to request.

## 1. Summary

Most LLM memory is explicit and input-triggered. A retrieval step fires on the user's query, looks up a store, and prepends the result before generation begins. PARM makes two narrow bets and runs the loop asynchronously alongside generation.

First, condition retrieval on the model's own output stream and on the contents of tool responses, not the user's query, so an entity the model merely mentions can trigger a lookup even when it was never the focus of the request. Second, score what to surface by how much it advances the user's goal, computed over a knowledge graph, instead of by raw semantic similarity.

The second bet exists for precision. Watching every entity the model emits against a personal graph throws off mostly weak matches, and the goal-grounded scoring is there to keep the few that surface from being noise.

Most of the ingredients here are not new. Neither output-conditioned retrieval nor proactive surfacing is. Generation-time retrieval is established (FLARE, Self-RAG), and proactive memory-grounded assistants are an active 2025 to 2026 subfield (ProActor, PASK; see §2). Goal- and intent-conditioned retrieval also exists (MemGuide, PKG, RPO-RAG), and PARM reuses that machinery rather than inventing it. The contribution is narrower, a combination of both: triggering on confident entity mentions in the output or a tool response, then scoring the resulting cross-source join against personal memory by goal-advancement over a graph, run asynchronously and at high enough precision to be usable. The first deliverable is a benchmark that isolates this case, which does not currently exist.

## 2. Background and the gap

The relevant prior work falls into six rough strands.

- **Scoring foundations.** Generative Agents (Park et al., 2023) set the standard retrieval score combining recency, relevance, and importance, with top-ranked memories inserted into context.
- **Graph-structured memory.** HippoRAG (Gutiérrez et al., NeurIPS 2024; HippoRAG 2, 2025) pairs a knowledge graph with Personalized PageRank for associative, multi-hop recall. Production systems (Mem0, ECAI 2025; Zep/Graphiti; Cognee; LangMem) combine embeddings, entity graphs, and key-value stores with automatic write pipelines.
- **Memory as a subsystem.** MemGPT/Letta and memory-OS work (MemOS) treat memory as a managed, tiered resource outside model weights.
- **Goal- and intent-conditioned retrieval (2025 to 2026).** A fast-growing line argues semantic similarity is the wrong objective and retrieval should condition on task intent. MemGuide (AAAI-26) does intent-aligned retrieval plus a filter that reranks memory units by marginal slot-completion gain, reporting large task-success gains over similarity retrieval. PKG pre-stores meta-paths for goal-relevant traversal. RPO-RAG (WWW'26) learns relation-aware path preferences and argues against shortest-path heuristics. SYNAPSE (2026) revives spreading activation for associative recall, and ProGraph-R1 trains graph retrieval with a progress-toward-goal reward. This line is the closest prior art to PARM's salience mechanism, and PARM borrows from it directly. Every system in it is input- or intent-conditioned: it infers the goal from the current query and fires on the user's turn.
- **Generation-time retrieval.** FLARE (EMNLP 2023) already conditions retrieval on the model's generation, using the predicted upcoming sentence as the query and triggering when the model emits low-confidence tokens. Self-RAG learns an on-demand variant. Output-conditioned retrieval is established. Both are uncertainty-triggered and corpus-grounding for factuality. They retrieve when the model is unsure, to support what it is about to assert. PARM runs close to the inverse. It triggers on confident entity mentions (the model is not unsure about "High Alpha," which is the point), retrieves from personal memory, and surfaces an unsolicited suggestion rather than rewriting a sentence.
- **Proactive and anticipatory agents (2025 to 2026).** A subfield now studies assistants that surface help before being asked. ProActor (ACL 2026) builds agents that monitor conversations, identify emerging opportunities, and intervene when appropriate. PASK targets intent-aware proactive agents with long-term memory. Related work names a proactivity gap in long-lived agents (LLaMAPIE, GUM). PARM is one instance of this paradigm and does not invent it. What this work mostly leaves open is the mechanism for cross-source-join salience and a benchmark that isolates it.

Seen together, the three leave a gap. None of them assembles the full combination of triggering on confident entity mentions in the output or a tool response (not model uncertainty, not the query), retrieving a cross-source join against personal memory (not corpus facts for grounding), and scoring by goal-advancement over a graph. Two practical limitations also recur in the memory line PARM builds its store from. Retrieval is usually input- or query-conditioned (ImplicitMemBench, 2026, reports explicit-retrieval mechanisms miss implicit memory), and it is serial and blocking (the retrieve-then-generate bottleneck noted in MemOS), which discourages frequent fine-grained retrieval.

The natural objection is that none of this is needed, that you could give the model a memory-search tool and tell it to retrieve when a memory seems relevant. That fails on the cases that motivate PARM. A self-triggered search fires only on relevance the model already suspects, but a cross-source join's relevance becomes visible only after the lookup. The trigger lives in the memory rather than in the model's current reasoning. The model emits "High Alpha" as a throwaway descriptor; to choose to search on it, it would have to already suspect the connection it has not yet retrieved. Two further problems compound this. Self-directed tool use is unreliable, with tool selection easily biased (Faghih et al., 2025). And a prompt fires on the model's reasoning, not its emitted output, so throwaway entities never reach the trigger. Prompting reaches input-cued, model-salient retrieval, which is roughly what shipping product memory already does, and structurally cannot reach the output-cued cross-source case. Rather than threatening the work, this objection just defines its primary baseline (§7).

## 3. Core idea and hypotheses

The shift is from "retrieve once, on the input, before generating" to "retrieve continuously, on the unfolding output and on tool responses, during generation," a setting FLARE and the proactive-agent line already occupy. The specific elements PARM adds are that the trigger is a confident entity mention, the target is a cross-source join against personal memory, and salience is goal-advancement over a graph.

- **H1 (coverage).** Entity-triggered, output-conditioned retrieval surfaces relevant memories that input-conditioned retrieval misses, improving accuracy on tasks where the cue is absent from the prompt.
- **H2 (latency).** An asynchronous design adds negligible wall-clock latency versus a blocking retrieve-then-generate baseline at equal retrieval frequency.
- **H3 (cross-source surfacing, the headline).** Goal-grounded graph matching surfaces useful cross-source joins (the High Alpha pattern) that none of the obvious baselines produce: a "retrieve when relevant" prompt, a FLARE-style uncertainty-triggered retriever, or a proactive-agent baseline. This is the experiment the paper lives or dies on.
- **H4 (precision under high-frequency watching).** Scoring candidates by how much they advance the unfilled part of the goal, computed over the graph, beats relevance-only retrieval and a MemGuide-style unit-level filter on surfaced-suggestion precision, and suppresses the N×M spurious-match flood at the scoring stage rather than after retrieval.

## 4. How the system works, end to end

The pipeline has six steps. Steps 3 and 4 carry the contribution; the rest can be borrowed from existing graph-memory systems.

1. **Watch the output and tool responses.** As the model generates, read the tokens as they emit, and read entities arriving in tool results. This is the output-conditioning. PARM attends to what the model says and what its tools return, not only what the user asked.
2. **Extract and link distinctive entities.** NER pulls named items from the stream ("High Alpha," "Acme"). Entity linking maps each surface form to a graph node by nearest-neighbor in embedding space.
3. **Ground the active goal onto the graph (§5.1).** Decompose the user's goal into sub-goals and link each to real existing nodes (contacts, target companies). These grounded anchors are where salience is measured from.
4. **Score salience, off the critical path (§5.2 to §5.3).** For a linked entity, compute how much it advances the goal, using the graph structure between it and the grounded anchors. Done in a side worker so it never blocks generation.
5. **Surface asynchronously (§5.4).** If salient, raise a side-channel suggestion ("spun out of High Alpha; your professor worked there, want a warm intro?") while the model keeps writing.
6. **Write back.** New facts (the company, that it is a High Alpha spinout) are added to the graph for next time.

## 5. The salience mechanism (the precision layer)

### 5.1 Grounding the goal into the graph

This step is borrowed. MemGuide infers a goal and reranks by slot-completion, PKG pre-stores goal-relevant meta-paths, and RPO-RAG learns relation preferences. PARM uses the same idea. Decompose the goal into sub-goals, then link each sub-goal to real nodes by embedding similarity, reusing the entity-linking from the output text but pointed at goal fragments. "Find a warm intro to a target company" decomposes into roughly two anchors: people I know (grounds to existing contact nodes) and the company I'm targeting (grounds to a company node). The walk is seeded from those real anchors. The goal is a small set of seed nodes plus a preference for which edge types count.

The part that is not borrowed is that the concrete target end of the goal often arrives from the model's own output. "Find intros" has no specific company attached until the model surfaces Acme mid-answer; that is the moment Acme becomes a live anchor. PARM seeds from {your contacts} and {Acme}, and the bridge between them (Professor to High Alpha) lights up. The output stream supplies half the seeds. Existing goal-conditioned systems seed entirely from the query.

Grounding an open-ended goal into anchors is fragile, and the fragility is inherited from the prior work above. It is a known difficulty in intent-conditioned retrieval rather than a new risk PARM introduces, which is some reassurance, since MemGuide and PKG ship with it and still report gains. PARM treats it as a first-class risk (§9) and tests sensitivity to it (RQ5).

### 5.2 Scoring salience: the random-walk-to-GNN ladder

Salience is one scoring function. Given the grounded goal and a node the model mentioned, it asks how much that node helps the goal. It can be instantiated at three levels of "how learned," all doing the same job. Think of it as one idea with a dial rather than three competing mechanisms.

**Rung 1, a goal-biased random walk (cheap, no training).** Run a random walk with restart (Personalized PageRank, as in HippoRAG) seeded at the grounded anchors. Where the walk spends its time marks nodes close to the goal. What makes it goal-aware is that it does not cross all edges equally. Crossing an edge depends on whether that edge's vector fits the goal, so for an intro goal the walk flows across `worked_at`, `knows`, and `spun_out_of` edges and stalls on irrelevant ones. Run once, and every node carries a goal-relevance score read off at match time.

**Rung 2, template meta-paths (goal-aware, still no training).** If you can name which edge types matter for a goal, encode them as a pre-stored meta-path template per goal type, the PKG approach. This gives goal-aware walks with no learned component. You can sit on this rung for a long time.

**Rung 3, a goal-conditioned GNN (learned).** Same job, learned weights instead of fixed ones. Ordinary message passing, each node gathering from its neighbors over a few hops, except every message is conditioned on the goal vector, so the network learns which relations matter for which goals. A small readout head turns each node's representation into a salience score. The random walk is the untrained special case.

**Why Rung 1 is the default.** It is cheap, needs no labels, and already works. The GNN earns its place only when you cannot hand-name the goal-relevant edge types and want to learn them, or when you want the score to reflect outcome-driven usefulness (this kind of connection tends to pay off; that one looks close but dead-ends), which a parameterless walk cannot learn. Whether Rung 3 beats a well-tuned Rung 1 or 2 is an open empirical question. The proposal treats the GNN as a hypothesis to test against the walk, not as the design, and the contribution does not depend on the GNN winning. If it loses, that result is reported plainly and the contribution stands on grounding, output-conditioning, and the benchmark.

### 5.3 From relevance to usefulness (the unfilled-goal layer)

A node can be on-topic and still not move you forward. So seed the walk from the still-unfilled parts of the goal (you have the company, not the way in), and score a node by how much it covers a sub-goal that is still open, with diminishing returns so a filled sub-goal stops attracting score. The diminishing-returns property is what stops the system re-suggesting the same thing. This coverage rule can be a plain heuristic, or a learned value where a head predicts how much progress surfacing a candidate makes, rewarded by task outcomes, in the ProGraph-R1 direction.

This also makes salience live. As sub-goals fill, the goal representation changes and scores re-rank on their own, so relevance tracks where the user is in the task rather than being computed once and frozen.

### 5.4 Running it ambiently (compute, injection, races)

You do not fire a full GNN every token. Precompute the cheap walk-field once per goal, refresh it when a sub-goal fills via incremental updates, then at match time run a light message-passing pass only on the small neighborhood around the matched node.

Two design questions to resolve early. First, the injection mechanism: prefix update, interleaved side-note, or constrained re-decode of the current span, and its interaction with KV-cache reuse. Second, race conditions: what to do when scoring returns after the model has emitted past the relevant point (re-decode, defer to the next sentence, or accept lag). These determine whether the negligible-latency claim (H2) holds.

## 6. Research questions

- RQ1: How much accuracy on cue-in-output tasks comes from output-conditioning specifically, holding store and scorer fixed?
- RQ2: Does goal-grounded salience (§5.2 to §5.3) beat the "just prompt it" baseline on cross-source joins, and by how much?
- RQ3: Does the learned GNN (Rung 3) beat a tuned random walk or meta-path field (Rungs 1 and 2), and on which cases?
- RQ4: What injection mechanism keeps added latency negligible without polluting context?
- RQ5: How sensitive is everything to goal-decomposition quality (§5.1)?

## 7. Evaluation plan

**The benchmark (first deliverable).** A cross-source-join set modeled on the High Alpha case: items where an entity appears only in retrieved or generated content and the payoff requires connecting it, possibly multi-hop, to a stored personal-memory attribute, plus distractor entities that look matchable but lead nowhere. The set is built first, both because it is the instrument every later experiment needs and because it answers the prior question of whether the phenomenon is frequent enough to be worth a method. Its value does not depend on PARM working: it isolates a retrieval case nothing currently measures. Distractor design and an independent pass over the items to confirm the baselines fail for principled reasons (rather than because the set was tuned against them) are part of the construction, not an afterthought. Alongside it, use LongMemEval and LoCoMo for long-session conversational memory and ImplicitMemBench for implicit behavioral adaptation.

**Baselines.** (i) no memory; (ii) input-conditioned RAG; (iii) input-conditioned graph memory (Mem0 or Zep style); (iv) the prompt baseline, a model with a memory-search tool instructed to retrieve when relevant; (v) a FLARE-style generation-time retriever, output-conditioned but uncertainty-triggered and similarity-scored, to show the difference is the trigger and the scoring rather than merely retrieving during generation; (vi) a proactive-agent baseline (ProActor or PASK style) over the same store; (vii) PARM with synchronous blocking retrieval, to isolate async; (viii) PARM with relevance-only scoring, to isolate goal-grounding; (ix) a MemGuide-style unit-level slot-completion filter, to isolate edge-level versus unit-level; (x) full PARM. Baselines (iv) to (vi) are the headline comparisons for H3.

**Headline experiment.** PARM versus the prompt baseline, the FLARE-style retriever, and the proactive-agent baseline on the cross-source-join set. The theory predicts all three fall short there for distinct reasons. The prompt cannot trigger on relevance visible only after the lookup. The FLARE-style retriever fires on uncertainty rather than confident entities and scores by similarity rather than goal-advancement. The proactive baseline lacks the graph mechanism to find the multi-hop join. Each should look competitive on ordinary input-cued retrieval. Beating all three on the join set is the result; failing to means there is no paper.

**Metrics.** Task accuracy; added wall-clock latency and tokens per second (H2); surfaced-suggestion precision and recall, plus a recommendation-fatigue rate, the fraction of injected items judged useless (H3, H4); retrieval precision and recall; context-overhead tokens.

Published benchmark numbers in this area are largely vendor-reported and not independently replicated. Re-run all baselines in-house rather than cite leaderboard figures.

## 8. Contributions (claimed)

1. **A cross-source-join benchmark.** An evaluation set isolating the case where a relevant entity appears only in output or tool-response content and the payoff requires joining it to a latent personal-memory attribute, with distractors and an independent fairness pass. Nothing currently measures this. It stands whether or not the method below wins.
2. **Output- and tool-response-conditioned, asynchronous retrieval.** A trigger on confident entity mentions in the output and in tool responses, with surfacing that runs alongside generation without blocking, positioned inside active retrieval (FLARE/Self-RAG) and proactive agents (ProActor/PASK) rather than claimed as a new paradigm, with a structural argument for why a "retrieve when relevant" prompt cannot reach this class.
3. **Goal-grounded graph salience for precision.** A scoring function that keeps high-frequency output watching from flooding the user, spanning a cheap goal-biased random walk, a no-training meta-path field, and a learned goal-conditioned GNN, with an unfilled-goal layer that makes relevance track task progress. The grounding machinery is reused from MemGuide, PKG, and RPO-RAG; the new element is seeding part of the goal from output entities. The GNN is tested against the cheap walk rather than assumed to win.
4. A prototype and injection mechanism that integrates surfaced memory mid-generation without blocking.

## 9. Risks and limitations

- **Goal grounding is the crux (§5.1).** Decomposing an open-ended goal into seed anchors is fragile, and a bad decomposition degrades every downstream score, with no obvious supervised target. The fragility is inherited from the intent-conditioned line, which ships with it and still reports gains, so it is a managed risk rather than a novel one, and RQ5 tests for it.
- **Training signal for usefulness is sparse.** "This surfaced join was useful" is expensive to label, which puts the learned scorer in weak-supervision and outcome-reward territory, with cold-start.
- **The learned rung may not beat the cheap one.** If the GNN does not beat a tuned random walk or meta-path field, the contribution narrows to grounding, output-conditioning, and the benchmark, which I would report plainly rather than bury.
- **Latency may not hide cleanly.** If injection forces KV-cache invalidation or re-decoding, the async claim weakens, so this is worth measuring early.
- **Linking and edge-validity compound.** NER and linking errors on the output stream, plus temporally stale edges (the professor's `worked_at` may be expired), feed garbage into an otherwise-correct walk. Temporal validity on join edges matters.
- **Context pollution.** Frequent injection can hurt, since LLMs are sensitive to proactive interference.
- **Privacy.** Ambient, non-prompted memory takes away the user's explicit recall trigger. That is where the usefulness comes from, and it is also what makes the system hard to audit, so a consent and inspection surface belongs in the design.
- **Scope versus in-weights memory.** A competing school (Titans, memory-layer architectures) puts memory inside the model. PARM is deliberately external and non-parametric, and I should say so up front.

## 10. Rough milestones

1. *Weeks 1 to 4* — Build the cross-source-join set and the independent fairness pass. Stand up a graph and vector store on an existing backend, and reproduce the input-conditioned and prompt baselines. By the end of this block, know whether the phenomenon is frequent enough to justify the rest.
2. *Weeks 5 to 8* — Output and tool-response monitor, entity linking, goal grounding (§5.1), and Rung-1 random-walk salience. Validate H1 and the H3 headline against the prompt baseline before optimizing anything.
3. *Weeks 9 to 12* — Async injection; measure H2; resolve the race-condition policy; add the unfilled-goal layer (§5.3).
4. *Weeks 13 to 15* — Rung-3 GNN and the Rung-1-versus-Rung-3 ablation (RQ3); goal-decomposition sensitivity (RQ5); write-up.

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
