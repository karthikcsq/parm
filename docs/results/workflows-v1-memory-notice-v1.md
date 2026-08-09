# PARMBench Workflows: Applicable-Memory Notice v1

## Change

The workflow agent previously treated any admitted personal memory as a standing
instruction to change its plan. That made a weakly related admission likely to
turn into a false intervention. The injected-memory notice now requires a
specific constraint that applies to the task or observed object before changing
the ordinary plan.

This changes only the handoff from retrieval to the agent. It does not change
the workflow fixture, the corpus, the retrieval index, the semantic admission
ranker, or the comparison-policy implementation.

## Experiment design

- Dataset: `data/workflows_v1` (9 cases: three scenario triplets)
- Corpus/index: `workflow-eng-lead-v1-100` / tier 100
- Policy: semantic-judge `parm`
- Model: `gpt-5-mini`
- Samples: three independent trajectories per case and condition
- Both baseline and candidate used fresh, separate admission and trajectory
  caches. No frozen decision or trajectory was reused.
- Candidate artifacts: `data/benchmark-results/workflows-v1-memory-notice-v1/tier-100/`

## Fresh comparison

| Scenario | Baseline positive | New positive | Baseline control | New control | Baseline ceiling | New ceiling | Baseline timely gold | New timely gold | Baseline spurious | New spurious |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oncall-escalation` | 1/3 | **2/3** | 2/3 | 2/3 | 3/3 | 3/3 | 0/3 | **1/3** | 3.3 | 4.7 |
| `release-freeze` | 3/3 | 3/3 | 1/3 | **2/3** | 3/3 | 3/3 | 3/3 | 3/3 | 3.0 | **2.7** |
| `telemetry-hotfix` | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 5.7 | 6.3 |

Positive and control are the count of samples whose decisive assertions passed.
Gold, spurious admissions, and memory tokens are means over positive samples.

## Result

The change improves the targeted decision metrics without reducing a positive
or ceiling count:

- On-call positive success rose from **1/3 to 2/3** and timely gold admission
  from **0/3 to 1/3**.
- Release-freeze cue-ablated control success rose from **1/3 to 2/3**.
- Telemetry control success rose from **3/3 to 3/3** in the fresh paired run;
  this remains a no-regression result rather than a claimed gain.

Spurious admissions rose on on-call and telemetry, so this is not a retrieval
precision improvement. It demonstrates that making the agent verify a memory's
scope improves downstream restraint and instruction-following. The remaining
on-call miss is a candidate/admission-coverage problem and should be optimized
separately.
