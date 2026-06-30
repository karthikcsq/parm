# PARM Benchmark V1 Plan

## Objective

Measure whether an agent can notice a small number of memory-worthy cues inside
large noisy outputs, retrieve the right personal memory, and improve a decision
without flooding the response with spurious or private context.

## Pilot

- Five Amara-backed concepts, each with positive and cue-ablated variants.
- One ordinary prompt and one 8-12K-token output/tool observation per instance.
- Tracked source provenance and perturbation labels.
- Exactly one final choice, keyed by the title or name visible to the model.
- Explicit output-only and memory-conditioned natural-language choices.

## Required comparisons

Develop no-memory, input-RAG, prompted-tool, whole-output RAG, all-entity RAG,
key-sentence RAG, and oracle-cue recovery one at a time over the same memory
substrate. Do not publish a comparison until its implementation is reviewed.

Primary outcomes are beneficial decision change, control false interventions,
memory-admission precision/recall, spurious admissions, poison admissions, and
privacy overexposure. Graph paths are optional diagnostics, not a correctness
requirement.

## Exit criteria

The pilot is ready for model experiments when all ten instances validate and
human review confirms the prompts do not functionally telegraph the memory.
After real baselines exist, require oracle recovery, control restraint, and
measurable whole-output noise before drawing comparisons.
