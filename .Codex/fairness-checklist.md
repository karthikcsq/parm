# PARM Case Fairness Checklist

- The prompt is independently answerable and does not mention memory, GBrain,
  personal context, or the eventual intervention.
- The decisive cue is absent from the prompt and appears only in the one later
  output or tool result.
- The observation is 8,000-12,000 tokens under the pinned tokenizer.
- The prompt requests exactly one final choice.
- Every option has a unique, natural label such as a session title, story
  label, episode title, company name, or restaurant name.
- The relevant identity or behavioral pattern appears in readable memory
  prose, not only in source metadata or a benchmark-only ID.
- The output-only choice is credible before memory is admitted.
- The memory-conditioned choice is materially different and better.
- A cue-ablated twin keeps the noise and task intact but removes the reason to
  intervene.
- At least 25 visible cue candidates and three plausible memory distractors are
  present.
- Gold sources exist, hashes match, provenance is recorded, and no poison
  record is authoritative.
- Stale or contradictory sources are labeled rather than silently flattened.
- Naive whole-output retrieval can produce tempting false matches.
- A correct response may use memory quietly; it need not repeat private facts.
- Mentioning the memory without changing the decision fails.
- Functional prompt leakage is rejected even when the word "memory" is absent.
- A normal model can answer by repeating the requested title or name; success
  does not depend on emitting a synthetic ID or hidden JSON field.
