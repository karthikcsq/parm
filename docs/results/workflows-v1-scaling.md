# PARMBench Workflows: 28 to 100 Memories

Two points on a scaling curve, three samples per condition, `gpt-5-mini`. One
scenario. This measures how each retrieval policy responds to a larger personal
history, not how well it does in general.

## What was run

`parm-workflow-github-telemetry-hotfix`, all three variants, under six memory
policies, at both declared corpus tiers. Everything except the corpus is held
constant: same model, environment adapter, fixture, tool surface, per-observation
top-k of five, and step limit.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_workflows_v1_matrix.ps1 -Tier tier-28 -Samples 3
powershell -ExecutionPolicy Bypass -File scripts\run_workflows_v1_matrix.ps1 -Tier tier-100 -Samples 3
& 'C:\Users\karth\anaconda3\python.exe' scripts\summarize_workflows_v1_results.py --tier tier-100
```

## Results

Positive and control are counts of samples whose decisive assertions all passed.
Spurious and memory tokens are means over the positive samples.

### 28 records

| Condition | Positive | Control | Ceiling | Spurious | Memory tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `parm` | 3/3 | 3/3 | 3/3 | 7.0 | 1,971 |
| `naive_output_rag` | 3/3 | 3/3 | 3/3 | 21.3 | 4,150 |
| `all_entity_output_rag` | 3/3 | 3/3 | 3/3 | 24.3 | 4,587 |
| `no_memory` | 0/3 | 3/3 | 3/3 | 0.0 | 0 |
| `input_rag` | 0/3 | 3/3 | 3/3 | 5.0 | 877 |
| `prompted_memory_tool` | 0/3 | 3/3 | 3/3 | 0.0 | 0 |

### 100 records

| Condition | Positive | Control | Ceiling | Spurious | Memory tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `parm` | 3/3 | 2/3 | 3/3 | 7.7 | 2,068 |
| `naive_output_rag` | 3/3 | 3/3 | 3/3 | 38.7 | 8,164 |
| `all_entity_output_rag` | 3/3 | 3/3 | 3/3 | 60.3 | 12,613 |
| `no_memory` | 0/3 | 3/3 | 3/3 | 0.0 | 0 |
| `input_rag` | 0/3 | 3/3 | 3/3 | 5.0 | 977 |
| `prompted_memory_tool` | 0/3 | 3/3 | 3/3 | 0.0 | 0 |

## What changed with scale

The corpus grew 3.6x. Decisions did not move at all. Cost did.

| Condition | Spurious 28 → 100 | Tokens 28 → 100 |
| --- | ---: | ---: |
| `parm` | 7.0 → 7.7 (+10%) | 1,971 → 2,068 (+5%) |
| `naive_output_rag` | 21.3 → 38.7 (+82%) | 4,150 → 8,164 (+97%) |
| `all_entity_output_rag` | 24.3 → 60.3 (+148%) | 4,587 → 12,613 (+175%) |

PARM's admission count is close to flat in corpus size. The broad policies'
tracks it. That is the selectivity result, and it is a rate rather than a
threshold: nothing broke at 100 records, and nothing was expected to, because
100 records still fit comfortably in context.

Contamination moves the same way. Admissions of records labelled `poison` or
`stale-superseded`, summed over nine runs per condition:

| Condition | 28 records | 100 records |
| --- | --- | --- |
| `parm` | none | none |
| `input_rag` | 3 stale | 3 stale |
| `naive_output_rag` | 8 stale, 4 poison | 21 stale |
| `all_entity_output_rag` | 9 stale | 33 stale, 18 poison |

The poison records are imported third-party documents asserting that an agent
should merge on green without review. At 100 records `all_entity_output_rag`
pulled one into context about twice per run. PARM admitted no poison and no
superseded record in any run at either tier.

This did not cause a wrong decision. Every poison admission above sits in a run
that passed its decisive assertions. It is exposure, not damage, and it should
not be reported as damage.

## What did not change

**The late-cue premise holds at both scales.** `no_memory`, `input_rag`, and
`prompted_memory_tool` score 0/3 on the positive at 28 and at 100 records. The
commitment is not reachable from the goal, and a prompted agent does not think
to look for it: `prompted_memory_tool` admitted nothing at all in any of the
twelve positive samples across both tiers.

**Broad retrieval still wins on decisions.** `naive_output_rag` and
`all_entity_output_rag` are 3/3 on both the positive and the control at both
tiers. Growing the corpus 3.6x did not make the dump strategy fail. On this
scenario the decision metric still cannot separate selective retrieval from
indiscriminate retrieval; only cost and contamination can.

**PARM's positive timing is exact.** In all six positive samples across both
tiers, the gold record was admitted on the same observation step where the cue
first became visible.

## PARM's control

3/3 at 28 records, 2/3 at 100. The one failure declined to merge the
documentation-only pull request and ended by choice, not by step limit.

Two things follow. First, the single-sample control failure reported in the
[first pass](workflows-v1-first-pass.md) does not survive sampling: at the same
tier, with three samples, the control passes 3/3. That earlier result was noise,
which is what motivated sampling here. Second, at 100 records the control is
genuinely 2/3, and PARM admits 8 to 11 records on a control where the correct
behaviour is to admit nothing. Low admission is not abstention.

## Limits

**One scenario.** Everything here is one task in one environment. A policy's
response to corpus growth on this scenario is not its response in general.

**Three samples.** Enough to catch a result that flips, not enough for a
confidence interval. The PARM control at 100 records is 2/3, which is
consistent with anything between an occasional and a frequent failure.

**Two points do not establish a trend.** The growth rates above are computed
from two measurements. They are consistent with PARM being flat in corpus size
and the broad policies being roughly linear, and they cannot distinguish that
from other shapes.

**100 records is not the interesting scale.** The whole history still fits in
context, so no policy was forced to choose. The scale at which indiscriminate
retrieval stops being viable is further out; this is a waypoint on the way
there, not a test of the hypothesis.

**Per-policy budgets are not symmetric.** Top-k is five per retrieval for every
policy, but `all_entity_output_rag` issues one retrieval per entity found in an
observation rather than one per observation, so it admits 45 to 49 records from
a single observation. At 100 records it saturated against the corpus rather than
against its budget. This is the baseline's actual character and is left as is;
the token column is what carries the cost.
