# PARM Decision Log

This historical log records non-obvious choices and their reasoning. Newest
entries appear first. Current behavior is documented in
[Architecture](../architecture.md) and the
[Evaluation Contract](../benchmark-evaluation.md).

## 2026-07-31

### Construction generates the decision directly from the fact

**Why:** The selection-predicate chain (mapper, task-family registry,
relation-type routing, capability labels, LLM decision-validity judge)
retained one scenario out of 48 mapped supply rows, and manual adjudication
rejected that survivor too. The diagnosis was structural, not a single bug:
good scenarios were lost to metadata conflicts between stages even when a
person could write a sensible example immediately, and the ordinary
mechanism the mapper emitted kept paraphrasing the cue's own axis.

**What:** New construction is one generation call per gated fact
(`scripts/build_parmbench_simple_v1.py`, prompt family
`parmbench_construction_simple_v*`): the model reads the claim, the raw
span, and the triplet requirements, may decline, and returns the scenario
plus three plain why-sentences. Acceptance is a nine-item human checklist
with deterministic assertions, not a judge. The first calibration round
built 29 scenarios from 34 rows and manual review accepted 11, with the
measured failure classes written into the v2 prompt as
counter-instructions. The predicate stage, its caches, and the old builder
versions survive for frozen replay only.

## 2026-07-30

### Relevance is enforced by three gates, not one

**Why:** An independent audit of 29 frozen-batch scenarios against raw
sources found only 5 defensible at a paper-review bar. The failures were not
source-support failures: they were anecdotes and wording requests promoted to
memories, and real facts wrapped in invented permissions, possessions, and
hollow affordances. The fairness sweep cannot catch this because a contrived
scenario can still produce the intended A/A/B pattern.

**What:** Split relevance into source support (`evidence_gate`), memory
quality (`memory_quality`, pre-construction), and decision validity
(`decision_validity`, post-construction with an evaluator-only causal chain
from the construction model). A bounded 60-row pilot removed every upstream
failure mode from its survivors; the residual fixture-level gaps (decoy
telegraphing, prompt-instruction confounds, ablated-preference redirection)
are recorded in the audit doc as the next round's work list. The frozen
157-scenario batch is unchanged and its realism defects stay documented
rather than patched.

## 2026-07-29 (later)

### The calibration batch refroze at 157 scenarios after the v2 supplement

**Why:** The freeze at 65 was a supply failure, and the fix was upstream of
construction: `parmbench_claim_draft_v2` requires every claim to carry a
decision lever, so undecidable biographical facts are declined before they
cost a scenario. Nothing had inspected retrieval yet, so amending the drafting
rubric was construction calibration, not benchmark tuning.

**What:** 339 unspent rows from a 300-persona pool were drafted under v2; 201
passed the gate; 193 supplement scenarios were built with the round-one
machinery and `-s2` ids; 92 survived the same two-attempt fairness discipline.
The merged batch froze at 157 scenarios, 471 cases, 140 personas, with the
final frozen-replay sweep passing 157 of 157 and the 65 round-one scenarios
byte-identical throughout. The supplement ceiling first-sweep pass rate was 88
percent against 50 percent under v1, which confirms the lever requirement was
the right repair.

## 2026-07-29

### A drafted claim has to be able to decide something

**Why:** The calibration batch froze at 65 scenarios because 35 of its 60 final
drops were unusable ceilings. `parmbench_claim_draft_v1` asked for a
conservative, auditable fact and got one; it never asked whether the fact could
change a choice. "The user fractured their left wrist in their 20s" passes both
the span check and the evidence gate and then defeats every attempt to write
options around it, because no ordinary task turns on it. Repair attempts cannot
reach that: the defect is upstream in supply.

**What:** `parmbench_claim_draft_v2` keeps every soundness rule of v1 and adds a
decision-relevance test. The fact has to be a durable preference, constraint,
exclusion, requirement, schedule or commitment, or relationship that could steer
a choice among ordinary options, and the drafter has to name the
`decision_lever`, the kind of choice it could steer. Facts that are finished,
one-off, or purely opinions about abstract subjects are declined even when the
user plainly stated them.

The lever is construction metadata. It reaches the construction call, under
`parmbench_construction_v2`, so the model can pick an affordance that settles
the task rather than one that merely suits the person, and
`scripts/build_parmbench_v1_supplement.py` rejects any scenario that reuses the
lever's wording in the observation, the prompt, or the memory sentence. Nothing
model-visible may carry a construction note.

### A supplement round adds scenarios rather than rebuilding the batch

**Why:** The round-one builder assigns every axis from pools sized to its own
claim count, so adding claims to `gated_claims.jsonl` would reshuffle the axes
of all 65 frozen scenarios. Those scenarios have already been read against the
fairness sweep; regenerating them would silently retune a closed batch.

**What:** Supplement scenarios are built by a separate script that reuses the
round-one envelopes, assembly, rejection checks, distractor selection, and
caches, assigns axes under its own seed over its own claim count, and carries an
`-s2` suffix on every scenario id. Round one's case rows, construction records,
and observation files are copied through byte for byte, and repairs for the new
round are declared in their own `fairness_repairs_v2.json`.

### The ceiling prompt names the personal fact as memory

**Why:** The first `no_memory` fairness sweep of `benchmark_parmbench_v1` passed
only 64 of 127 memory-included ceilings, including scenarios whose cue repeated
the user's own wording. The builder concatenated the memory sentence onto the
front of the ordinary task with no label, and the answer model is instructed to
follow the task using only the supplied observation, so an unlabelled leading
sentence read as task text and was discounted.

**What:** `ceiling_prompt` in `scripts/build_parmbench_v1_benchmark.py` now
renders the fact under a `Known personal memory:` heading above the task, which
is the shape the PersonaMem pilot used. Applied to every scenario at once; no
claim, evidence span, or gold source changed.

### Fairness repairs are declared in a tracked file, not applied by hand

**Why:** A scenario that misses the criteria 5 to 7 pattern has to be rebuilt
from the same claim, and the rebuild has to be reproducible. Editing a case in
place would break the link between the dataset and its builder.

**What:** `data/parmbench-v1-supply/fairness_repairs.json` records, per
scenario, the repair attempt number and the variants that failed, or the reason
it was dropped. The builder reseeds a repaired scenario and adds the attempt
number plus a note about the failing variant to the construction request, so
the cache cannot replay the core that failed. Two attempts per scenario, then
the scenario is dropped with its last diagnosis.

### The parmbench_v1 calibration batch froze at 65 scenarios, below the floor

**Why:** The batch was meant to hold at least 100 scenarios. Two repair rounds
took the fairness pass count from 28 to 66 of 127, and the 60 scenarios that
exhausted both attempts were dropped, along with one carrier of an
over-represented filler run. The dominant failure was the ceiling: 35 of the 60
final drops were scenarios where a model told the personal fact still chose the
output-only winner. The evidence gate accepts a claim because a user stated it,
not because it could change a decision, so about half the supply describes
facts that cannot decide an ordinary task however the options are written.
Continuing to re-roll those scenarios would have selected on sweep noise rather
than fixing them.

**What:** Froze at 65 scenarios, 195 cases, 61 personas, and recorded the
shortfall. Stage 2 needs a drafting rubric that asks for decision-relevant
facts, not more repair attempts on the current pool.

## 2026-07-28

### PersonaMem-v2 stays the canonical raw-history source

**Why:** The mixed_v0 failures were construction failures, not source failures.
The corpus is expansive (18,549 train_text rows at the pinned revision), openly
licensed under CC BY 4.0, and its 32k-token histories carry enough
user-authored statements to support gated scenario construction. Switching
sources would discard the adapter, provenance, and audit work without fixing
the actual defect.

**What:** Expanded the local pool to 120 personas in
`data/personamem-v2-train-v1` and kept `personamem-v2-train-v0` frozen for the
legacy slices. New construction drafts each claim from raw user-authored text
and gates it with `evidence_gate`; the upstream `preference` label is only a
pointer to where to look and never becomes the claim.

## 2026-07-26

### Freeze the first expansion pass against the completed pilot

**Why:** Reusing the five pilot cases for further tuning would turn a perfect
in-sample score into a weak quality claim. Thirteen approved examples already
have authoritative Amara sources and can test the frozen mechanism on new cue
shapes. The two remaining health examples require memories that do not exist in
the corpus.

**What:** Promoted examples 4, 6-11, and 13-18 into `data/benchmark_v1` with the
same generated context, cue-ablation, memory-included, provenance, distractor,
and camouflage procedure as the pilot. Marked them as the `expansion` split,
kept examples 1, 2, 3, 5, and 12 as `pilot`, and excluded examples 19 and 20.
The first expansion score must use retrieval and judgment behavior frozen at
commit `b0e9410`.

## 2026-07-08

### PARM condition selects by convergence threshold, not fixed top-k

**Why:** The frozen-retrieval contract made fixed top-k of 5 and "no dynamic
score thresholds" universal. The PARM condition has to be able to admit zero
memories on a cue-ablated observation — that empty admission is the entire
false-intervention test. Fixed top-k structurally always admits k, so a universal
top-k rule is incompatible with the condition PARM exists to measure.

**What:** Deprecated those rules as universal in
`retrieval-mode-axis-plan.md` and made selection condition-dependent: the
mode-matched baselines keep fixed top-k for comparability; PARM selects by a
convergence-score threshold `T`. Wrote `docs/history/parm-convergence-retrieval-v1.md`
for the full condition.

### PARM's contribution is non-LLM convergence retrieval, not a smarter judge

**Why:** The store is too large to hand a model wholesale, and retrieval must stay
non-LLM. Relevance judgment (the model, downstream) and provenance/staleness (the
substrate) are not PARM's to claim. What remains, and what neither component
baseline does, is ranking candidates by cross-seed convergence over a load-bearing
graph so precision comes from ranking and selection rather than from restricting
retrieval. Breadth at the candidate stage is fine; convergence plus the threshold
carry precision, and the ablated twin comes back empty through the same
mechanism.

**What:** Recorded the two-stage design (wide multi-seed candidate generation,
then convergence ranking with threshold selection) and the graph-load-bearing
choice. First experiment before building: convergence vs flat retrieval on the
twins.

### Seed extraction stays non-LLM; SLM dropped from the critical path

**Why:** PARM needs the parts of an observation that touch the memory store, not
open-domain NER or summarization. That store-anchored framing makes extraction a
gazetteer match plus noun-phrase chunking plus classical salience plus the
embedder already in the stack — near-zero power, no new model. Because convergence
supplies precision downstream, the extractor should be high-recall and dumb, not
smart; a smart extractor would drop the weak single-match signals convergence
exists to exploit. The one step that seemed to need a small LM — composing
disconnected entities — is what the load-bearing graph already does.

**What:** Wrote the seed-extraction section in the design record with the cheap
non-LLM stack and the high-recall principle. Demoted the SLM to an optional
upstream cue-proposer booster (never in retrieval/ranking), deferred until the
data shows the graph plus cheap extractor miss the pattern cues.

## 2026-07-07

### Camouflage the memory targets in the generated fixtures

**Why:** A case was passing for the wrong reason. The memory-conditioned answer
(e.g. conference `G-147`) was the only content-rich row in a wall of templated
filler, so the model selected it by appearance, not by using memory. Cue
ablation confirmed it — swapping the target's entities but leaving its shape did
not change the selection. The uniform filler was leaking the answer.

**What:** Rewrote `scripts/build_pilot_contexts.py` to draw filler with
decorrelated per-field strides (no clones) and to plant near-miss decoy
clusters around the target so it no longer stands out. Regenerated all five
contexts.

### Only camouflage the three fixtures that actually leak

**Why:** The leak exists only where the memory target is a lone special row
(conference, ai-news, podcast). In vendor and lunch the salient row is the
*output* answer and memory overrides toward the non-salient option, so there is
no salience leak toward the memory answer.

**What:** Added decoys to the three leak-prone fixtures; gave vendor and lunch
the varied-filler upgrade only.

### Freeze answer-model outputs to make enhanced reproducible

**Why:** Enhanced retrieval caches query expansions keyed by the query text. The
`model_output_only` and `tool_then_model_output` flows derive that query from
the model's own nondeterministic output, so the expansion key changes every run
and frozen enhanced can never hit. Freezing the expansion alone cannot help when
the query feeding it is not frozen.

**What:** Added `CachingLanguageModel` (populate/frozen response cache keyed by
the full request), `--response-cache`/`--response-policy` CLI flags, and a
response-cache hash in the config sidecar. Verified byte-identical replay across
a populate-then-frozen pass.

## 2026-08-01

### Reproduce MCPMark's environments locally instead of running MCPMark

**Why:** MCPMark is the right source for realistic agent work, but its GitHub
service duplicates a seed repository into a live private org and drives it over
the REST API. That needs credentials this checkout does not have, mutates
account-visible state on every run, is validated on macOS and Linux only, and
cannot give a positive and its cue-ablated twin independent resets inside one
parallel run. Those last two properties are load-bearing for a paired
benchmark, not conveniences.

**What:** Added a `github_fixture` adapter that keeps MCPMark's artifact
shapes, tool surface, and final-state verification style while running in
process from a tracked JSON fixture. Every case records the MCPMark task,
revision, and Apache-2.0 license it derives from. What this gives up is real
pagination, rate limits, and API error taxonomies; an `mcpmark_live` adapter
can register under the same protocol if that fidelity turns out to matter.

### Score workflows from state, admission, and timing rather than a judge

**Why:** The [real-world evaluation plan](../real-world-evaluation.md) assumed
realistic open-ended tasks would need an LLM judge, and warned that a judge
would make the benchmark a general answer-quality contest. Building the suite
showed the assumption was wrong: an agent working through tools leaves a final
state that can be asserted exactly, so the judge was never needed for the
primary result.

**What:** Workflow cases declare role-tagged assertions over the final
environment state, the mutation log, and the step log. Decisive assertions must
differ between a positive and its control, which validation enforces. The judge
protocol stays documented as the plan for a secondary explanation-quality
layer.

### Treat admission timing as a result, not a diagnostic

**Why:** In a single-turn benchmark, memory either reached the model or it did
not. In a trajectory it can also arrive too late. A system that recalls the
merge prohibition one step after merging has logged the right source and caused
the wrong outcome, and a metric that counts it as a recall success is lying.

**What:** Every case declares where its cue becomes visible and which calls
dispose of the governed locus. Scoring reports `timely_gold_admission_rate` and
`late_gold_admission_rate` separately from recall.

### Check that a workflow goal cannot reach its own gold memory

**Why:** The pilot broke this twice. First the corpus was small enough that a
top-five prompt retrieval covered a quarter of it. Then a goal reworded to
force a decision on the open pull requests said "merge the ones that are safe
to merge" — the exact act the stored commitment governs — which pulled the
memory into the prompt-only results. Either way input RAG wins without the
mechanism under test existing.

**What:** `scripts/evaluate_workflows_v1_fairness.py` fails when any non-ceiling
goal retrieves a gold source in the top five under dense or hybrid. It runs
outside `workflow validate` so validation stays offline.
