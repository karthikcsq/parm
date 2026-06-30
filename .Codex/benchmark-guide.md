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

## Baseline execution

The shared runner validates and resolves every dataset row, then creates a
sanitized `BenchmarkInput` containing only:

- `case_id`;
- the ordinary prompt;
- the observation kind; and
- the resolved observation text.

Hidden cue metadata, gold decisions, memory text, distractors, and provenance
remain outside the baseline boundary. Every registered baseline receives this
same public input and returns the common prediction shape. The runner writes all
rows through a temporary file and only publishes the requested output after
every case succeeds.

`no_memory` is the first registered implementation. It invokes OpenAI Responses
once per case, does not instantiate GBrain, and emits an empty retrieval trace.
The requested model is selected by `--model`, `PARM_OPENAI_MODEL`, then
`gpt-5-mini`. Unimplemented baseline names are still refused.

Every prediction contains:

```json
{
  "case_id": "parm-amara-conference-agenda-positive",
  "baseline": "no_memory",
  "requested_model": "gpt-5-mini",
  "resolved_model": "gpt-5-mini-...",
  "provider_response_id": "resp_...",
  "response_text": "Session G-147 — Chen Wei, NovaMind: Edge Inference for Texas Grid Reliability Pilots",
  "usage": {
    "input_tokens": 10000,
    "output_tokens": 20,
    "total_tokens": 10020
  },
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
