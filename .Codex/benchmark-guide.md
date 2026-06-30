# PARM Benchmark V1 Guide

## Case flow

Each JSONL row points to a committed noisy context and records:

- `prompt`: the ordinary initial request;
- `observation`: one `assistant_output` or `tool_result`;
- `cue`: the later visible trigger and retrieval query;
- `memory`: readable Amara memory prose plus authoritative source provenance;
- `decisions`: output-only and memory-conditioned natural-language choices;
- `distractors`: cue candidates and tempting memory records; and
- `variant`: `positive` or `cue-ablated`.

Loading a case resolves its context file and applies the ablation replacement.
Validation checks token budget, prompt leakage, cue presence, one-choice
wording, unique choice labels, decision change, distractor counts, source
hashes, and poison exclusion.

## Corpus preparation

`parm-bench prepare-amara` converts the tracked mixed-format Amara source into
GBrain-ready Markdown under `.gbrain-local/corpus/amara-life-v1`. The converter
preserves source IDs, hashes, timestamps, and perturbation metadata before
materializing deterministic entity pages.

## Baseline scaffolding

No baseline is currently executable. `src/parm_bench/baselines.py` defines the
`Baseline` protocol, a registration boundary, and a non-executable inventory of
planned research comparisons. The registry is intentionally empty.

`run` accepts a name so the CLI contract is stable, but refuses any name that
has not been backed by a reviewed implementation. `--memory-backend gbrain`
will supply the pinned local GBrain CLI to a future registered baseline.

Every prediction contains:

```json
{
  "case_id": "parm-amara-conference-agenda-positive",
  "response_text": "Session G-147 — Chen Wei, NovaMind: Edge Inference for Texas Grid Reliability Pilots",
  "trace": {
    "detected_cues": [],
    "retrieved_source_ids": [],
    "admitted_source_ids": [],
    "admitted_perturbations": {}
  }
}
```

The response text is the scored answer. A model emits the same title or name a
person would use; it never needs to know a benchmark-only item ID. Responses
that name both scored alternatives are treated as ambiguous rather than as one
final choice. Responses can omit a trace, but internal retrieval baselines
should emit source provenance for diagnostics.

## Scoring

The scorer recognizes the expected visible title or name in `response_text`.
It does not require the response to state the private memory. It scores the
selected decision, positive-case improvement, control-case restraint, source
admissions, poison use, abstention, and sensitive-detail exposure.

Hand-authored predictions test the scorer independently of any baseline. No
retrieval result is published until a real implementation is registered and
run.
