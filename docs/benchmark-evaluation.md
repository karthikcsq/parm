# PARM Benchmark Evaluation

This document is the repo-visible contract for PARM V1 scoring. It captures the
ABC-Bench guidance we are adopting, the metrics we report, the scorer behavior,
and the failure taxonomy used to diagnose benchmark runs.

## ABC-Bench Guidance For PARM

ABC-Bench evaluates backend agents through the full lifecycle: repository
exploration, implementation, environment setup, deployment, and external
end-to-end tests. The useful lesson for PARM is not the backend domain itself;
it is the evaluation posture.

For PARM, the full behavior is:

1. A user asks a normal task.
2. A large generated output or tool response introduces many entities,
   sentences, options, and distractors.
3. The system identifies the few output/tool-side cues that warrant memory
   lookup.
4. The system retrieves the relevant personal-memory fact without overreacting
   to spurious correlations.
5. The system surfaces an actionable intervention asynchronously, without
   excessive noise.

The oracle is a quality gate, not a production baseline. If
`parm_oracle_monitor` emits the gold cue and memory fact, it should score as the
recoverability ceiling. If the oracle fails, the scorer or gold contract needs
review before comparing non-oracle baselines.

## Metrics

Primary metrics:

- `correct_surfaced_suggestion_rate`: fraction of cases where at least one
  suggestion surfaces the right memory intervention.
- `precision`: correct suggestions divided by all suggestions.
- `recall`: fraction of gold cases recovered. In V1, this matches case-level
  recovery because each case has one gold intervention.
- `useless_intervention_rate`: non-correct suggestions divided by all
  suggestions.
- `abstention_rate`: fraction of cases where the system produces no scorable
  intervention.

Stage metrics:

- `format_valid_rate`: prediction rows that are structurally usable.
- `trigger_found_rate`: response mentions or identifies the output/tool-side
  cue.
- `memory_fact_found_rate`: response surfaces the hidden memory-side fact.
- `affected_item_found_rate`: response applies the memory to the correct item.
- `action_found_rate`: response contains expected action keywords.
- `decision_effect_found_rate`: response expresses the expected intervention
  type.
- `path_correct_rate`: exact gold path rate for structured oracle/debug traces.
  Natural responses without traces are not penalized for missing paths. This is
  diagnostic, not the headline V1 metric.

## Scoring Policy

Primary V1 scoring is deterministic. We do not require model responses to use
JSON or a verbal template, because that would hint that memory retrieval is
expected. Natural model outputs can use `response_text`; structured oracle or
debug baselines can use `suggestions`.

The scorer uses normalized text matching plus hidden gold fields:

- Structured fields (`trigger_entity_id`, `memory_fact_node_id`, `path`) are
  scored exactly when present.
- Natural text is scored by normalized labels and action terms.
- `path` is required only when a structured trace is emitted. It helps diagnose
  retrieval mechanics, but V1 does not require a production system to perform
  multi-hop graph reasoning.
- Decision effects use small controlled vocabularies, not fuzzy semantic
  grading.

V1 does not use an LLM judge, BERT model, embeddings, or fuzzy matching for
primary metrics. Those can be added later as audit signals for scorer
disagreement, not as the source of truth.

## Failure Taxonomy

- `format_failure`: prediction row is malformed or unusable.
- `abstained`: no memory intervention is surfaced.
- `cue_detection_miss`: the later output/tool cue is not noticed.
- `memory_fact_miss`: the cue is noticed, but the relevant memory is not
  surfaced.
- `wrong_memory_fact`: a memory is surfaced, but it is not the gold memory.
- `wrong_affected_item`: the memory is applied to the wrong option or item.
- `invalid_or_stale_edge_used`: the system relies on an invalid graph relation.
- `graph_traversal_miss`: a trace-emitting system fails to recover the gold
  path.
- `ranking_or_selection_error`: multiple candidates are found, but a distractor
  is selected over the gold intervention.
- `actionability_miss`: the memory is mentioned without useful advice.
- `decision_effect_mismatch`: the advice has the wrong effect, such as
  enriching when it should warn or reconsider.
- `overtrigger_noise`: the system surfaces irrelevant or excessive memory
  interventions.
- `scorer_gold_mismatch`: the gold/oracle output appears semantically right, but
  deterministic scoring rejects it.
