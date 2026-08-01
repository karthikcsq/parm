# PARM Decision Log

This historical log records non-obvious choices and their reasoning. Newest
entries appear first. Current behavior is documented in
[Architecture](../architecture.md) and the
[Evaluation Contract](../benchmark-evaluation.md).

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
