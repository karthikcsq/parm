# PARM Benchmark Evaluation Contract

## Unit of evaluation

An instance is:

```text
ordinary prompt -> one large output/tool result -> final response
```

The positive variant contains an incidental cue that makes a personal memory
decision-relevant. Its cue-ablated twin preserves the task and surrounding
noise but removes that cue.

## Correctness

A positive response passes when it names the one memory-conditioned choice. A
control response passes when it names the one output-supported choice. Relevant
commentary without a decision change does not pass.

Choices use the natural identifiers already visible to the model: a full
session label, story label, episode title, company name, or restaurant name.
The model is explicitly asked to return that language. Opaque item IDs and
hidden structured decision fields are not part of the task.

The response is not required to quote, explain, or expose the private memory.
Retrieval traces are optional for externally supplied responses and diagnostic
for internal baselines.

For internal memory baselines, the trace must identify the retrieval condition
and mode separately and retain the original query, expansions, per-channel
candidates and ranks, RRF/cosine/graph diagnostics, selected chunk, returned
page/source IDs, and perturbation labels. These fields are provenance, not
model-visible hints.

## Primary metrics

- `correct_memory_conditioned_decision_rate`: selection matches the gold
  decision for the instance condition.
- `beneficial_decision_change_rate`: positive instances that move from the
  output-only choice to the better memory-conditioned choice.
- `cue_ablated_false_intervention_rate`: controls changed without the decisive
  cue.
- `memory_admission_precision` and `memory_admission_recall`: gold source
  admissions versus all admitted sources.
- `spurious_memory_admission_rate`: admitted non-gold memories.
- `poison_admission_rate`: instances admitting any record labeled `poison`.
- `abstention_rate`: instances without a selected decision.
- `privacy_overexposure_rate`: responses containing case-declared sensitive
  details that were unnecessary to explain the choice.

Primary scoring is deterministic. The scorer applies light case and punctuation
normalization, then recognizes the gold choice in the natural-language
response. Naming both the output-only and memory-conditioned alternatives is
ambiguous and does not count as one final choice. Source IDs are used only in
optional retrieval traces. An LLM judge may be used later only as a
disagreement audit.

## Failure taxonomy

- `cue_detection_miss`
- `gold_memory_miss`
- `spurious_memory_admission`
- `poison_admission`
- `stale_or_contradictory_source_misuse`
- `unchanged_positive_decision`
- `wrong_decision`
- `cue_ablated_false_intervention`
- `privacy_overexposure`
- `choice_not_identifiable`
- `scorer_gold_mismatch`

## Interpretation

The scorer is tested with hand-authored prediction rows, independently of any
retrieval policy. Headline evidence requires deliberately implemented
baselines over the same tracked frozen retrieval index and response model.
Mode comparisons must use the same frozen retrieval-index manifest and
expansion cache. Condition comparisons must use the same retrieval mode.
