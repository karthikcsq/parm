# PARMBench v1, calibration batch

65 independent base scenarios, 195 cases, 61 PersonaMem personas. Every
scenario is a triplet: a positive case, a cue-ablated control, and a
memory-included ceiling. All three share one base scenario, so any statistic
belongs at the scenario level and the case count is not a sample size.

Status: frozen calibration development data. This is stage 1 of the two-stage
plan in [the construction contract](../../docs/benchmark-construction.md). It
exists to calibrate the construction rules, the evidence gate, and the fairness
run. It is not a sealed test set, and no retrieval system should be tuned
against it and then reported on it.

Steps 1 through 6 of the contract have run. Every surviving scenario satisfies
the mechanical checks for criteria 1 to 4 and 8 to 11, and every one of them
passed the `no_memory` fairness sweep that establishes criteria 5, 6, and 7.
The batch is closed: nothing may be tuned against it now that the sweep has
been read.

## What a scenario contains

The raw persona history in `../personamem-v2-train-v1/source` is the only
memory substrate. Nothing in this directory ships an index, a graph, or a gold
retrieval path to a system under test. The model-visible fields are the prompt,
the observation kind, the resolved observation text, and the corpus id. The
memory block, the cue, the decisions, the distractors, and the provenance are
evaluator-only.

Each observation is a generated document of 6,762 to 12,588 `cl100k_base`
tokens holding one output-only winner, one memory-conditioned target, three to
five near-miss decoys, and combinatorial filler. The cue is a single sentence
inside the target's entry. The control expresses its ablation as one declared
replacement of that sentence, so the positive and the ceiling observations are
byte-identical and the control differs by one edit. The ceiling prompt carries
the personal fact under a `Known personal memory:` heading above the ordinary
task.

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
from 114 personas. The drafter declined on 37 rows, the gate's deterministic
frequency pre-check rejected 7 claims that used a frequency word on a single
instance, and 1 span failed the verbatim check. No claim reached the
`inferable` grade, which is a consequence of the drafting rubric: a drafter
that must quote an entailing user span tends to write claims the judge can read
straight off the text.

Construction came second. `scripts/build_parmbench_v1_benchmark.py` gives every
scenario a seed derived from its base case id, assigns each axis from a
balanced pool under one fixed axis seed, and sends the claim plus the sampled
axes to a cached `gpt-5-mini` call under `parmbench_construction_v1`. That call
returns only the semantic core: the task, the option names, the bodies, the cue
sentence, the neutral replacement, the decoys, and the memory sentence. A
seeded, model-free assembly step then wraps the core in one of fourteen
observation envelopes and pads it with filler. Both caches key on the sha256 of
`{prompt_version, model, input}`, so a rebuild replays without new calls.

Repair came third, and it is described under fixture fairness below. A repaired
scenario keeps its gated claim, its evidence span, its gold source, and its
sampled axes; only the seed and the construction request change, and the
request carries the attempt number so the rebuild cannot replay the core that
failed.

Every model call this batch depends on is cached in the repository: 178 claim
drafts, 133 evidence-gate gradings, 346 construction and repair calls, and
1,018 `no_memory` answer calls. Cache entries for scenarios that were later
repaired or dropped are kept, because deleting a cache entry is how a frozen
builder stops being replayable.

## Fixture fairness

Deterministic construction proves that a scenario is well formed. It says
nothing about whether a real model reads it the intended way. Criteria 5, 6,
and 7 are settled by running the `no_memory` baseline on all three variants and
requiring one pattern per scenario:

| variant | required answer | what it establishes |
| --- | --- | --- |
| positive | output-only choice | the cue alone does not give the memory away |
| cue-ablated | output-only choice | ordinary visible evidence carries the control |
| memory-included | memory-conditioned choice | the injected fact is actionable |

The sweep used `gpt-5-mini` through the tracked response cache at
`../response-caches/parmbench-v1-fairness`, and it is scored deterministically
by `scripts/evaluate_parmbench_v1_fairness.py` under evaluator version
`parmbench_fairness_v1`. A scenario passes only when all three variants land on
their required choice and name no other declared option.

The first sweep over 127 scenarios passed 87 positives, 85 controls, and 64
ceilings, and only 28 scenarios passed all three at once. One systematic defect
came out of that pass: the ceiling prompt concatenated the memory sentence onto
the front of the task with no label, and the answer model is instructed to work
from the supplied observation alone, so it discounted the fact even where the
cue plainly matched it. Naming the fact under its own heading, the way the
PersonaMem pilot did, was applied to every scenario and lifted the ceiling to
67. That is a construction fix rather than a per-scenario repair, and it did
not touch any claim, span, or gold source.

The rest of the failures were per-scenario, and each one was sent back to the
construction model with the variant that broke:

- a positive that already picked the target means the affordance is doing work
  on its own, so the rewrite is told to make the cue read as a plain detail;
- a control that did not pick the winner means a decoy or the target reads
  better on ordinary grounds, so the rewrite is told to state the winner's
  advantage concretely and give the target none;
- a ceiling that still picked the winner means the affordance is not decisive
  for the task, so the rewrite is told to choose an affordance that makes the
  winner unusable for a person with that fact.

Each scenario had at most two attempts. The counts across the sweeps:

| pass | scenarios | positive | cue-ablated | memory-included | all three |
| --- | --- | --- | --- | --- | --- |
| first sweep | 127 | 87 | 85 | 64 | 28 |
| ceiling heading | 127 | 87 | 85 | 67 | 33 |
| repair round 1 | 126 | 94 | 98 | 75 | 49 |
| repair round 2 | 126 | 99 | 99 | 87 | 66 |
| frozen batch | 65 | 65 | 65 | 65 | 65 |

93 scenarios were sent for a first repair and 77 of those for a second. 32 of
them ended up in the frozen batch, 15 fixed on the first attempt and 17 on the
second. 33 scenarios passed without any repair.

60 scenarios were dropped for exhausting both attempts. Their last diagnoses
were 35 unusable ceilings, 17 positives whose cue was too strong without
memory, 17 controls whose ordinary evidence was too weak, 10 positives and 10
controls that named some third entry, and 4 ceilings that named neither
declared option. The ceiling is where the batch lost most of its ground, and
the reason is visible in the supply: the evidence gate accepts a claim because
a user stated it, not because it should change a decision. A claim such as "the
user fractured their left wrist in their 20s" is auditable and useless, and no
rewrite of the options makes it decide a task.

Two further scenarios left the batch during repair. One repaired core wrote a
memory sentence that quoted the user's raw span, which the builder rejects. One
was dropped by hand after the batch shrank: a six-word run of envelope filler
that sat at 8% of 127 scenarios sat at 15.2% of the 66 survivors, over the
construction-signature ceiling, so the carrier that shared the most
high-frequency filler runs with the rest of the batch was removed and the share
fell to 13.8%.

The 100-scenario floor set for this batch was not reached. 65 survived. The
shortfall is a supply problem rather than a construction problem: all 133 gated
claims were already spent, and roughly half of them describe facts that cannot
decide an ordinary task no matter how the options are written. Reaching the
contract's stage 2 target needs a drafting rubric that asks for
decision-relevant facts, not more repair attempts on these ones.

Artifacts for every pass are preserved under
`../benchmark-results/parmbench-v1-fairness/`: the four intermediate reports
and tables, the final `fairness.json` and `fairness-table.md`, the predictions,
the deterministic scores, and the run config sidecar. The repair and drop
decisions live in `../parmbench-v1-supply/fairness_repairs.json`, which the
builder reads, so the frozen batch rebuilds from a tracked file.

## Distributions

Run `scripts/report_parmbench_v1_distribution.py` to regenerate all of this.

Capability, one per scenario:

| capability | scenarios |
| --- | --- |
| direct lexical fact | 20 |
| paraphrased semantic fact | 14 |
| one-hop relational | 12 |
| schedule or commitment | 8 |
| negative preference or exclusion | 6 |
| relationship or named entity | 5 |

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
enforces. Instead, 17 scenarios carry `provenance.abstention_pressure`, meaning
their distractor memories were chosen for topical closeness to the claim rather
than at random, so a system that admits memory on similarity alone has
something to trip over.

Other axes, over 65 scenarios:

- 14 envelope styles, 3 to 6 scenarios each, with openings, section markers,
  entry shapes, and filler shapes randomised inside each style;
- 16 task domains, 2 to 7 scenarios each;
- 7 ordinary-evidence mechanisms, 7 to 12 scenarios each, so the winner is
  defensible through schedule fit, feasibility, stated requirements, recency,
  quality, cost, or an explicit task constraint rather than always the same way;
- wording relationship split 33 sharing content words with the user's span and
  32 paraphrase-only;
- observation kind split 39 `tool_result` and 26 `assistant_output`;
- numeric ratings absent from 53 scenarios and present in 12;
- cue position spread across all ten deciles of the document, p10 at 0.19 and
  p90 at 0.81;
- the memory-conditioned choice appears after the output-only choice in 36
  scenarios and before it in 29;
- 3 to 5 decoys and 3 to 5 competing memories per scenario;
- 57 personas contribute one scenario and 4 contribute two.

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

`memory.sensitive_terms` is empty everywhere, so the privacy-sensitive
admission metric has nothing to measure yet.

68 scenarios are recorded with their reasons in `dropped_scenarios.jsonl`: 60
that could not pass the fairness sweep in two attempts, 1 removed for a filler
run that crossed the construction-signature ceiling, and 7 that failed a
construction check. Of those 7, three had a memory sentence that quoted the
user's raw span, one leaked the span into the observation, one produced a cue
sentence that appeared twice, and two wrote a task prompt that shared more than
two content words with the personal fact.

That last check is the builder's stand-in for acceptance criterion 3. The
validator only catches literal leakage of the memory text, the memory-conditioned
choice, or an evidence span into the prompt, which would miss a prompt that
merely happens to talk about the same subject as the fact. Measuring the shared
content words is a blunter instrument and it is worth reading as a floor rather
than a proof: 61 of the 65 prompts share no content word with their claim and
the remaining 4 share exactly one. Whether a prompt-only retriever can actually
exploit that one word is a question for the baseline stage.

The fairness sweep certifies the batch against one answer model at one setting.
A different model may read some of these scenarios differently, and the sweep
would have to be rerun to say so.

## Files

- `cases.jsonl`, 195 cases, 3 per base scenario;
- `contexts/`, one observation document per base scenario;
- `construction_records.jsonl`, per scenario: the gated claim, the gate verdict,
  the evidence span, the sampled axes, the seeds, the repair attempt, the
  measured metrics, and the hashes of both model calls;
- `dataset_manifest.json`, schema version 1, validation profile `parmbench_v1`,
  the `fairness_gate` block, and one corpus entry per persona declaring its
  source root and source id prefix;
- `dropped_scenarios.jsonl`, scenarios that failed a construction check or the
  fairness sweep.

## Rebuild

```powershell
$env:PYTHONPATH = 'src'
$python = 'C:\Users\karth\anaconda3\python.exe'

& $python scripts\draft_parmbench_v1_claims.py
& $python scripts\build_parmbench_v1_benchmark.py
& $python -m parm_bench.cli validate data\benchmark_parmbench_v1
& $python scripts\report_parmbench_v1_distribution.py

& $python -m parm_bench.cli run data\benchmark_parmbench_v1 `
  --baseline no_memory --model gpt-5-mini `
  --response-cache data\response-caches\parmbench-v1-fairness `
  --response-policy frozen `
  --out data\benchmark-results\parmbench-v1-fairness\no-memory.jsonl
& $python scripts\evaluate_parmbench_v1_fairness.py data\benchmark_parmbench_v1 `
  data\benchmark-results\parmbench-v1-fairness\no-memory.jsonl `
  --out data\benchmark-results\parmbench-v1-fairness\fairness.json `
  --table data\benchmark-results\parmbench-v1-fairness\fairness-table.md `
  --finalize-dataset
```

Both build scripts replay from their caches. Add `--offline` to either one to
make a cache miss an error instead of a live call. The builder rewrites
`dataset_manifest.json`, so the fairness evaluator has to run after it to put
the `fairness_gate` block back.

The source pool is PersonaMem-v2 `train_text` at revision
`b7b42b78917157afed063527a1c959e98f6109f2`, created by Bowen Jiang and
collaborators and released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
