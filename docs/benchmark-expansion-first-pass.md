# PARMBench Expansion: Frozen First Pass

## Purpose

This report records the first score on the 13-scenario expansion before any
retrieval thresholds, source-class rules, judgment instructions, or new
fixtures were changed in response to expansion results.

The complete benchmark contains 18 scenario triplets:

- 5 pilot scenarios used while developing `parm_convergence_v1`;
- 13 expansion scenarios reproduced from approved examples 4, 6-11, 13-18;
- one positive, one cue-ablated control, and one memory-included ceiling per
  scenario.

All runs use the same tracked `amara-life-v1` retrieval index, requested model
`gpt-5-mini`, and deterministic scorer. Retrieval-backed conditions use top-k
5. This makes the first pass a useful generalization measurement even though
the expansion cases become development cases after the analysis below.

## Aggregate results

| Condition | Primary decisions | Positive lift | Controls correct | Ceiling | Admission precision | Admission recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PARM convergence v1 | 26/36 | 10/18 | 16/18 | 18/18 | 60.00% | 45.00% |
| Prompted memory tool, hybrid | 19/36 | 2/18 | 17/18 | 18/18 | 0.00% | 0.00% |
| No memory | 18/36 | 2/18 | 16/18 | 18/18 | n/a | n/a |
| All-entity output RAG | 17/36 | 7/18 | 10/18 | 18/18 | 1.16% | 80.00% |
| Naive output RAG, hybrid tool-then-model | 16/36 | 7/18 | 9/18 | 17/18 | 2.70% | 45.00% |
| Enhanced input RAG | 14/36 | 6/18 | 8/18 | 17/18 | 1.11% | 10.00% |

Primary decisions include positives and cue-ablated controls. A system can
raise positive lift by injecting many memories, but it loses primary accuracy
when the same behavior changes controls. PARM has the best aggregate decision
score and the strongest precision by a wide margin.

## Split results

| Condition | Pilot decisions | Expansion decisions | Expansion positive lift | Expansion controls correct |
| --- | ---: | ---: | ---: | ---: |
| PARM convergence v1 | 9/10 | 17/26 | 6/13 | 11/13 |
| Prompted memory tool, hybrid | 5/10 | 14/26 | 2/13 | 12/13 |
| No memory | 5/10 | 13/26 | 2/13 | 11/13 |
| All-entity output RAG | 5/10 | 12/26 | 7/13 | 5/13 |
| Naive output RAG, hybrid tool-then-model | 6/10 | 10/26 | 5/13 | 5/13 |
| Enhanced input RAG | 4/10 | 10/26 | 5/13 | 5/13 |

PARM admitted a gold source on 4/13 expansion positives and 5/5 pilot positives.
Its expansion source-ID recall was 4/14. The main generalization failure is
retrieval coverage, followed by two false admissions.

## What the first pass exposed

The expansion was generated before scoring, but three cases have weak
output-only separation:

- `phone-feature-digest`: no-memory selected the positive target without
  memory, so the case cannot measure beneficial decision change reliably;
- `human-factors-event`: no-memory selected the positive target and preferred a
  strong alternative over the declared lead in the control;
- `weekend-events`: no-memory preferred the cue-ablated replacement over the
  declared lead.

These are fixture-fairness findings, not retrieval wins or losses. Any revised
fixtures must retain the approved positive memory/cue relationship, keep the
large-output structure symmetric, and be rerun across every condition.

The frozen PARM trace also exposed two mechanism-specific limits:

- the semantic selector only admitted review/reflection filenames, excluding
  authoritative notes such as `threshold-terms`, `next-quarter-plan`, and
  `orange-mode`;
- the entity graph chose among same-entity candidate pages using region cosine
  alone, which selected the wrong Marcus Reid meeting for `startup-expo`.

Changes prompted by these observations must receive a new retrieval-condition
version and a new result namespace. The files ending in
`-v2-gpt-5-mini` remain the frozen first-pass record.

## Baseline notes

The enhanced prompted-memory-tool condition failed three populate attempts with
`expansion alternatives must be non-empty, unique, and distinct`. The report
uses its completed hybrid fallback. The hybrid agent made no tool calls on any
positive or control case, so its primary admission precision and recall are
zero.

The all-entity run recovered 80% of gold source IDs while admitting 1,360
spurious sources. Naive output RAG recovered 45% while changing half of the
controls. These conditions confirm that recall alone does not solve the
benchmark.

One all-entity attempt stopped after a transient DNS failure. The retry reused
the isolated response cache and completed all 54 rows without final-run errors.

