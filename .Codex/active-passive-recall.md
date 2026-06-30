# Deliberate And Cue-Triggered Recall

Deliberate recall begins with a known information need. A query, plan, or agent
explicitly searches memory because it already knows what is missing.

Cue-triggered recall begins when new information arrives. An entity, relation,
event, or behavioral pattern inside an agent output or tool result makes a
stored memory relevant even though the original task gave no reason to search
for it.

PARM tests the second mode:

```text
ordinary prompt
-> one large output or tool result
-> incidental visible cue
-> memory retrieval and admission
-> materially changed final decision
```

The executable cases ask for one ordinary-language choice. Titles and names
serve as the identifiers a model already sees and can repeat; source IDs remain
provenance only.

This is distinct from MemGuide-style intent-conditioned retrieval. Inferring a
missing task slot from the prompt without a later visible cue is useful, but it
is outside the core PARM score.
