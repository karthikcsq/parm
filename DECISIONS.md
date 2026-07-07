# Decisions

A running log of non-obvious choices and the reasoning behind them. Newest first.

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
