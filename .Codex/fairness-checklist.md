# PARM Benchmark Fairness Checklist

Use this checklist when adding or reviewing V1 benchmark cases.

- The trigger entity is absent from the initial user goal and present in either generated output or a tool response.
- The initial user goal contains enough task context for a useful generic response without memory.
- If the task requires local, current, private, or otherwise non-obvious option discovery, the fixture includes a tool/local-context response.
- Generated-output-only cues are reserved for cases where a capable base model could plausibly supply the cue from general knowledge.
- Generated output or tool response is large enough to contain many plausible
  cue candidates, not a placeholder or single clue sentence.
- The attractive item is plausible from prompt plus tool/output context before memory is considered.
- The memory-augmented suggestion changes, qualifies, or improves the final recommendation for that item.
- The initial user goal does not ask for memory search, personal context, latent context, or hidden-memory surfacing.
- The initial user goal is an ordinary task request; it should not directly ask to flag the same warning, constraint, follow-up, or enrichment that the gold suggestion provides.
- The gold suggestion requires selecting the right visible cue and retrieving a
  latent memory-side fact.
- Naive output-RAG over every entity or sentence would retrieve tempting but
  wrong memories, not just the gold one.
- The expected suggestion includes both the connection and the action implied by the user's goal.
- Distractors are plausible enough to test precision, but they do not accidentally satisfy the goal.
- Invalid or stale distractor edges are marked with `valid: false`.
- Gold paths, when present, use only valid graph edges and are diagnostic rather
  than the main benchmark challenge.
- Cases are balanced across goal family, trigger source, distractor type, and
  actionability. Multi-hop depth is optional and should not drive the case.
- Baseline failures are explained by benchmark mechanics, not by quirks of one implementation.
- No case relies on private or real personal memory; V1 cases are synthetic.
