# PARMBench v1, calibration batch

127 independent base scenarios, 381 cases, 109 PersonaMem personas. Every
scenario is a triplet: a positive case, a cue-ablated control, and a
memory-included ceiling. All three share one base scenario, so any statistic
belongs at the scenario level and the case count is not a sample size.

Status: calibration development data. This is stage 1 of the two-stage plan in
[the construction contract](../../docs/benchmark-construction.md). It exists to
calibrate the construction rules, the evidence gate, and the fairness run. It
is not a sealed test set, and no retrieval system should be tuned against it
and then reported on it.

Steps 6 through 8 of the contract have not run yet. The `no_memory` fairness
sweep that establishes acceptance criteria 5, 6, and 7 is the next stage, and
until it completes no scenario here is confirmed to change a real model's
decision. What the batch does establish is criteria 1 through 4 and 8 through
11, all of which are checked mechanically.

## What a scenario contains

The raw persona history in `../personamem-v2-train-v1/source` is the only
memory substrate. Nothing in this directory ships an index, a graph, or a gold
retrieval path to a system under test. The model-visible fields are the prompt,
the observation kind, the resolved observation text, and the corpus id. The
memory block, the cue, the decisions, the distractors, and the provenance are
evaluator-only.

Each observation is a generated document of 6,756 to 12,693 `cl100k_base`
tokens holding one output-only winner, one memory-conditioned target, three to
five near-miss decoys, and combinatorial filler. The cue is a single sentence
inside the target's entry. The control expresses its ablation as one declared
replacement of that sentence, so the positive and the ceiling observations are
byte-identical and the control differs by one edit.

## How the batch was built

Supply came first. `scripts/draft_parmbench_v1_claims.py` reads every candidate
source row, asks a cached `gpt-5-mini` drafter under the versioned rubric
`parmbench_claim_draft_v1` to write one conservative personal-fact claim from
the raw user-authored turns, and requires the drafter to quote back the exact
user span that entails it. The upstream PersonaMem `preference` label is passed
only as a hint of where to look; the drafter is told it is unverified, must not
copy its wording, and may decline. Each surviving span is checked verbatim
against the persona's normalized record and against the tracked history file,
then the claim goes through `parm_bench.evidence_gate.grade_claim` with its
cache under `data/evidence-gate-caches/parmbench-v1`.

Of 178 candidate rows, 133 produced a claim the gate graded `explicit`, drawn
from 114 personas. The
drafter declined on 37 rows, the gate's deterministic frequency pre-check
rejected 7 claims that used a frequency word on a single instance, and 1 span
failed the verbatim check. No claim reached the `inferable` grade, which is a
consequence of the drafting rubric: a drafter that must quote an entailing user
span tends to write claims the judge can read straight off the text.

Construction came second. `scripts/build_parmbench_v1_benchmark.py` gives every
scenario a seed derived from its base case id, assigns each axis from a
balanced pool under one fixed axis seed, and sends the claim plus the sampled
axes to a cached `gpt-5-mini` call under `parmbench_construction_v1`. That call
returns only the semantic core: the task, the option names, the bodies, the cue
sentence, the neutral replacement, the decoys, and the memory sentence. A
seeded, model-free assembly step then wraps the core in one of fourteen
observation envelopes and pads it with filler. Both caches key on the sha256 of
`{prompt_version, model, input}`, so a rebuild replays without new calls.

Every model call this batch depends on is cached in the repository: 178 claim
drafts, 133 evidence-gate gradings, 136 construction calls, and 14 control
replacement repairs. Three of the construction entries are from an earlier
three-scenario trial run and no case references them; they are kept because
deleting a cache entry is how a frozen builder stops being replayable.

## Distributions

Run `scripts/report_parmbench_v1_distribution.py` to regenerate all of this.

Capability, one per scenario:

| capability | scenarios |
| --- | --- |
| direct lexical fact | 33 |
| one-hop relational | 27 |
| paraphrased semantic fact | 24 |
| schedule or commitment | 22 |
| relationship or named entity | 12 |
| negative preference or exclusion | 9 |

Capability follows from the drafted fact and the sampled wording relationship.
A claim that states a limitation the person has, such as an injury or a food
that disagrees with them, counts as an exclusion, because the memory earns its
keep by ruling an option out rather than by promoting one.

Two categories from the contract's coverage list are missing. There is no
updated, contradicted, or stale-fact category, because the source pool's
eligibility filter drops every PersonaMem row marked `updated`, so no
superseded fact reaches the drafter at all. Abstention is not a separate
category either: an abstention case has the same answer under both conditions
and therefore cannot satisfy the positive decision-change rule the validator
enforces. Instead, 30 scenarios carry `provenance.abstention_pressure`, meaning
their distractor memories were chosen for topical closeness to the claim rather
than at random, so a system that admits memory on similarity alone has
something to trip over.

Other axes, over 127 scenarios:

- 14 envelope styles, 8 to 10 scenarios each, with openings, section markers,
  entry shapes, and filler shapes randomised inside each style;
- 16 task domains, 6 to 9 scenarios each;
- 7 ordinary-evidence mechanisms, 17 to 19 scenarios each, so the winner is
  defensible through schedule fit, feasibility, stated requirements, recency,
  quality, cost, or an explicit task constraint rather than always the same way;
- wording relationship split 64 sharing content words with the user's span and
  63 paraphrase-only;
- observation kind split 74 `tool_result` and 53 `assistant_output`;
- numeric ratings absent from 105 scenarios and present in 22;
- cue position spread across all ten deciles of the document, p10 at 0.14 and
  p90 at 0.82;
- the memory-conditioned choice appears after the output-only choice in 67
  scenarios and before it in 60;
- 3 to 5 decoys and 3 to 5 competing memories per scenario;
- 92 personas contribute one scenario, 16 contribute two, and 1 contributes
  three.

`parm_bench.construction_checks` reports no issues: no repeated numeric pair,
no six-word run shared by more than 15% of scenarios, no banned signature
phrase, enough cue-position spread, no invariant answer role, and no shared
opening line.

## Known limits

The batch runs on one history-size bucket. Every PersonaMem persona history is
roughly 32K tokens, so the contract's 100K, 500K, and 1M buckets are not
represented and cannot be without a second source.

Source age is not varied deliberately. The evidence span sits wherever the
drafter found it in a persona's history, and no scenario was built to place the
decisive fact old or recent on purpose.

`memory.sensitive_terms` is empty everywhere. 26 of the underlying source rows
come from PersonaMem's health and medical category and 17 from its therapy
category, all of them flagged non-sensitive upstream, but no per-scenario
private phrases were annotated, so the privacy-sensitive admission metric has
nothing to measure yet.

Six scenarios were dropped rather than repaired, and they are recorded with
their reasons in `dropped_scenarios.jsonl`. Two had a memory sentence that
quoted the user's raw span, one leaked the span into the observation, one
produced a cue sentence that appeared twice, and two wrote a task prompt that
shared more than two content words with the personal fact.

That last check is the builder's stand-in for acceptance criterion 3. The
validator only catches literal leakage of the memory text, the memory-conditioned
choice, or an evidence span into the prompt, which would miss a prompt that
merely happens to talk about the same subject as the fact. Measuring the shared
content words is a blunter instrument and it is worth reading as a floor rather
than a proof: 114 of the 127 prompts share no content word with their claim and
the remaining 13 share exactly one. Whether a prompt-only retriever can actually
exploit that one word is a question for the fairness and baseline stage.

## Files

- `cases.jsonl`, 381 cases, 3 per base scenario;
- `contexts/`, one observation document per base scenario;
- `construction_records.jsonl`, per scenario: the gated claim, the gate verdict,
  the evidence span, the sampled axes, the seeds, the measured metrics, and the
  hashes of both model calls;
- `dataset_manifest.json`, schema version 1, validation profile `parmbench_v1`,
  one corpus entry per persona declaring its source root and source id prefix;
- `dropped_scenarios.jsonl`, scenarios that failed a construction check.

## Rebuild

```powershell
$env:PYTHONPATH = 'src'
$python = 'C:\Users\karth\anaconda3\python.exe'

& $python scripts\draft_parmbench_v1_claims.py
& $python scripts\build_parmbench_v1_benchmark.py
& $python -m parm_bench.cli validate data\benchmark_parmbench_v1
& $python scripts\report_parmbench_v1_distribution.py
```

Both scripts replay from their caches. Add `--offline` to either one to make a
cache miss an error instead of a live call.

The source pool is PersonaMem-v2 `train_text` at revision
`b7b42b78917157afed063527a1c959e98f6109f2`, created by Bowen Jiang and
collaborators and released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
