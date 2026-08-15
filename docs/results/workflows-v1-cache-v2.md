# PARMBench Workflows: Cache-Input v2 Recertification

## Why this run exists

PARM's admission cache previously keyed a judge decision only on the user task,
complete observation, model, rubric, and cache namespace. It did **not** key on
the candidate pairs actually rendered to the judge. That could replay an old
admission after a candidate-ranking change, even if the judge would have been
shown a different set of source/region pairs.

The cache now fingerprints the complete judge input: prompt, observation,
candidate IDs, region IDs and text, page IDs, memory text, model, rubric, and
instructions. A frozen replay now rejects a changed candidate set rather than
silently producing a stale result.

## Run

- Dataset: `data/workflows_v1` (9 cases; three scenario triplets)
- Corpus: `workflow-eng-lead-v1-100` (100 records)
- Policy: `parm`, semantic-pair judge
- Model: `gpt-5-mini`
- Samples: 3 independent trajectories per case
- Fresh admission and trajectory caches
- Results: `data/benchmark-results/workflows-v1-cache-v2/tier-100/`

This is a PARM recertification, not a new full ladder: the comparison-policy
artifacts from the prior scenario-set run are unchanged because their execution
and cache semantics did not change.

## Results

| Scenario | Positive | Cue-ablated control | Ceiling | Timely gold | Mean spurious admissions | Mean memory tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `oncall-escalation` | 1/3 | 3/3 | 3/3 | 0/3 | 3.7 | 978 |
| `release-freeze` | 3/3 | 1/3 | 3/3 | 3/3 | 3.0 | 1,063 |
| `telemetry-hotfix` | 3/3 | 2/3 | 3/3 | 3/3 | 6.0 | 1,708 |

Positive and control count samples whose decisive assertions all passed.
Gold, spurious admissions, and memory-token values are means over positive
samples. The ceiling is reported so that model or fixture failures are not
mistaken for retrieval failures.

## Interpretation

The corrected run confirms the existing qualitative result: PARM is selective
and substantially less memory-heavy than the broad output-RAG baselines, but it
is not yet reliable enough to claim general late-cue retrieval. It still misses
the escalation commitment and false-intervenes on some freeze and telemetry
controls. Those are retrieval/admission and downstream-agent decision failures,
not cache-replay artifacts.

The important improvement is validity: future candidate-generation changes now
require a genuine recertification instead of inheriting stale admission decisions.
