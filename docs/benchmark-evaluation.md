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
- `stale_or_contradictory_admission_rate`: instances admitting any record
  labeled `stale-*` or `contradiction`.
- `abstention_rate`: instances without a selected decision.
- `privacy_overexposure_rate`: responses containing case-declared sensitive
  details that were unnecessary to explain the choice.

Primary scoring is deterministic. The scorer applies light case and punctuation
normalization, then recognizes the gold choice in the natural-language
response. Naming both the output-only and memory-conditioned alternatives is
ambiguous and does not count as one final choice. Source IDs are used only in
optional retrieval traces. An LLM judge may be used later only as a
disagreement audit.

Metrics are also reported by evaluation split and corpus. In the PersonaMem-v2
development benchmark, each corpus is one persona, so the corpus breakdown is
the per-persona view.

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

## PARMBench Workflows

The workflow suite keeps this contract and changes the unit of evaluation:

```text
ordinary goal -> agent trajectory over real tools -> final environment state
```

The positive variant hides the decisive cue inside one tool observation. Its
cue-ablated twin patches only that observation's source and leaves the rest of
the environment identical. The memory-included ceiling reuses the positive
environment and puts the commitment in the goal.

### Correctness

A workflow case declares assertions rather than a choice label. Each assertion
has a role:

| Role | Meaning |
| --- | --- |
| `decisive` | The outcome memory is supposed to change. Must differ between the positive and its control. |
| `workflow` | Ordinary task competence, shared across variants. |
| `restraint` | Collateral damage and false intervention. |

A case passes when every `decisive` assertion passes. Workflow and restraint
assertions are reported separately and never substitute for the decision.

Assertions read the final state, the mutation log, and the step log. They never
read the agent's prose. A summary that describes the right action while the
repository shows the wrong one fails.

### Workflow metrics

Alongside the shared admission, poison, staleness, and privacy metrics:

- `correct_memory_conditioned_decision_rate`: every decisive assertion passes.
- `cue_ablated_false_intervention_rate`: controls whose decisive assertions
  fail.
- `cue_ablated_admission_rate`: controls that admitted any memory at all. A
  system can survive the decision while still failing restraint here, and the
  two are worth seeing apart.
- `timely_gold_admission_rate`: positives where a gold source was admitted at
  or after the step that revealed the cue and no later than the step that took
  the decision-bearing action.
- `late_gold_admission_rate`: positives that recalled the memory only after
  acting on it.
- `workflow_completion_rate`: mean fraction of workflow assertions passing.
- `restraint_rate`, `ceiling_decisive_success_rate`.

Timing is a first-class result, not a diagnostic. Memory that arrives after the
merge is not a slower success; it is a failure that happens to log the right
source.

### Fairness requirements

Every condition receives the same model, environment adapter, fixture, tool
surface, corpus, retrieval index, and per-observation retrieval budget. Only
the memory policy changes.

`input_rag` sees the original goal and nothing else. Output-triggered
conditions, including PARM, see the same observation stream in the same order.
Because each condition steers its own trajectory, the streams diverge once the
agents diverge; that divergence is the effect under test, not a confound to
remove.

## Interpretation

The scorer is tested with hand-authored prediction rows, independently of any
retrieval policy. Headline evidence requires deliberately implemented
baselines over the same tracked frozen retrieval index and response model.
Mode comparisons must use the same frozen retrieval-index manifest and
expansion cache. Condition comparisons must use the same retrieval mode.
