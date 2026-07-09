# Decisions

A running log of non-obvious choices and the reasoning behind them. Newest first.

## 2026-07-08

### PARM condition selects by convergence threshold, not fixed top-k

**Why:** The frozen-retrieval contract made fixed top-k of 5 and "no dynamic
score thresholds" universal. The PARM condition has to be able to admit zero
memories on a cue-ablated observation — that empty admission is the entire
false-intervention test. Fixed top-k structurally always admits k, so a universal
top-k rule is incompatible with the condition PARM exists to measure.

**What:** Deprecated those rules as universal in
`docs/retrieval-mode-axis-plan.md` and made selection condition-dependent: the
mode-matched baselines keep fixed top-k for comparability; PARM selects by a
convergence-score threshold `T`. Wrote `docs/parm-convergence-retrieval-design.md`
for the full condition.

### PARM's contribution is non-LLM convergence retrieval, not a smarter judge

**Why:** The store is too large to hand a model wholesale, and retrieval must stay
non-LLM. Relevance judgment (the model, downstream) and provenance/staleness (the
substrate) are not PARM's to claim. What remains, and what neither component
baseline does, is ranking candidates by cross-seed convergence over a load-bearing
graph so precision comes from ranking and selection rather than from restricting
retrieval. Breadth at the candidate stage is fine; convergence plus the threshold
carry precision, and the ablated twin comes back empty through the same mechanism.

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

**What:** Wrote the seed-extraction section in the design doc with the cheap
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
