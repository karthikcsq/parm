# Real-World Evaluation Strategy

## Recommendation

Do not replace PARMBench's deterministic scorer with an LLM judge.

Keep the current triplet benchmark as the fast, attributable mechanism test,
and add a second end-to-end suite with realistic tasks, open-ended responses,
and rubric-based judgment. Use humans as the calibration source and an LLM
judge as a scaled approximation only after its agreement and failure modes are
measured.

The two suites answer different questions:

| Suite | Question |
| --- | --- |
| Deterministic PARMBench | Did the retrieval policy notice the new cue, admit the right memory, avoid the control, and produce the declared choice? |
| Realistic end-to-end suite | Did output-triggered memory make the agent more useful, accurate, safe, and appropriately personalized in a natural task? |

## What the current benchmark proves

The current benchmark is strong evidence for the narrow original mechanism
claim:

> A memory can become decision-relevant only after a later agent or tool output,
> and selective output-cued retrieval can outperform prompt-only retrieval,
> whole-output RAG, all-entity retrieval, and waiting for an agent to elect to
> search memory.

The benchmark makes that claim testable because:

- every prompt is memory-neutral;
- the decisive cue appears only in the later observation;
- the positive and cue-ablated control differ at the intended relationship;
- the memory-included ceiling verifies that the response model can use the
  fact;
- retrieval and decision correctness are scored separately;
- every condition uses the same frozen memory substrate and response model;
- broad-recall baselines are penalized when the same memories change controls;
  and
- the frozen first expansion pass records generalization before tuning.

The benchmark therefore does more than show that PARM can retrieve a relevant
note. It demonstrates the important asymmetry behind the project: input RAG
does not see the late cue, naive output RAG admits too much, and the naive agent
often does not initiate memory retrieval at all.

## What the current benchmark does not prove

It does not yet establish broad real-world utility.

The main limits are:

- The corpus and tasks are fictional, even though the source relationships are
  coherent.
- Large outputs are generated catalogs with one-choice answer contracts.
- The target and control are carefully constructed, not naturally occurring
  agent traces.
- Exact visible-label scoring omits drafting, planning, synthesis, partial
  credit, and useful abstention.
- A short response cannot fully test privacy, explanation quality, calibrated
  uncertainty, or whether personalization feels appropriate.
- The expansion is now development data after retrieval changes were made from
  its failures.
- One response model and one memory persona cannot show robustness across
  model families, users, or domains.

For a paper or product claim phrased as "PARM works in realistic personal
agents," the current benchmark is necessary but insufficient. For the narrower
claim that output-cued selective retrieval is a real mechanism distinct from
input RAG and naive output retrieval, it is already a useful proof.

## Pros of an LLM-as-a-judge suite

An LLM judge can score tasks that exact label matching cannot:

- open-ended recommendations with several acceptable answers;
- plans that must combine memory with current constraints;
- drafts whose tone and level of disclosure matter;
- answers that use the right memory without quoting it;
- partial success when an agent identifies the right issue but chooses a
  weaker action;
- calibrated abstention when memory is ambiguous or stale; and
- natural multi-step traces where the right intervention is not one catalog
  row.

It also makes a larger, more varied suite affordable once a human-authored
rubric and calibration set exist.

## Cons of switching the primary benchmark to an LLM judge

An uncalibrated judge would weaken several things the current benchmark does
well:

- Nondeterminism makes regressions and small mechanism changes hard to trust.
- Judges may reward polished or verbose answers instead of correct memory use.
- A judge from the same model family can favor its own style and reasoning.
- Position and naming can bias pairwise comparisons.
- Giving the judge gold memory can leak the expected answer and make plausible
  but unsupported use look correct.
- Hiding the gold memory makes causal memory use hard to verify.
- The judge may conflate retrieval failure, judgment failure, and fixture
  ambiguity.
- Private details placed in the judge prompt create another exposure surface.
- Costs rise because every system response may require multiple randomized
  judge calls and human disagreement review.
- A soft score can conceal false interventions that the paired control exposes
  exactly.

The largest risk is construct drift. A benchmark intended to test selective
late-cue retrieval can quietly become a general answer-quality contest.

## Proposed two-layer evaluation

### Layer 1: deterministic mechanism benchmark

Retain the current PARMBench triplets and metrics. Add new held-out cases in
versioned batches, freeze each first pass, and keep exact source-admission and
control scoring.

This layer should remain the release gate for:

- cue-conditioned retrieval recall;
- admission precision;
- control false interventions;
- poison and stale-source admission;
- privacy term exposure; and
- reproducible comparisons among retrieval policies.

### Layer 2: realistic end-to-end benchmark

Build a smaller suite of natural agent tasks from full traces rather than
catalog templates. Each scenario should contain:

1. an ordinary user goal;
2. one or more realistic tool calls or agent outputs;
3. a late cue that may or may not make memory useful;
4. a versioned personal corpus containing the supporting and distracting
   records;
5. a PARM run and the strongest baseline run with every other component held
   constant;
6. a cue-ablated or memory-removed counterfactual; and
7. a human-authored scoring rubric.

Recommended task families:

- calendar and travel planning after a new schedule or weather constraint;
- inbox triage after a message reveals an unresolved relationship or promise;
- procurement or diligence after a report names a risky counterparty;
- research synthesis after a new paper touches an existing project decision;
- drafting a reply after a conversation surfaces a communication preference;
- daily planning after activity data intersects with a recurring wellbeing
  pattern; and
- follow-up selection after a meeting output reveals an outstanding commitment.

Use the user's real operational shape where consent permits, but redact and
version source facts. When real traces cannot be shared, replay the interaction
structure with fictionalized content and have the originating user review its
fidelity.

## Judge rubric

Score each response independently on:

| Dimension | Question |
| --- | --- |
| Task success | Did the response accomplish the user's actual goal? |
| Memory relevance | Was every used memory materially relevant to the late cue? |
| Causal lift | Did memory improve the action compared with the no-memory or baseline response? |
| Faithfulness | Is the personalized claim entailed by the stored source? |
| False intervention | Did memory alter behavior when the decisive cue was absent? |
| Privacy restraint | Did the response use the minimum private detail needed? |
| Calibration | Did the response express uncertainty or abstain when the evidence was weak? |

Do not ask for one opaque 1-10 score. Require a verdict and short evidence for
each dimension, then compute the aggregate outside the judge.

## Judge protocol

1. Blind system names and randomize A/B order.
2. Give the judge the user goal, visible trace, response, rubric, and only the
   source facts needed to assess faithfulness.
3. Run at least two judge models from different families or providers.
4. Reverse A/B order on a sample to measure position bias.
5. Have humans score an initial calibration set and every judge disagreement.
6. Report agreement by dimension, not only the final win rate.
7. Freeze judge model versions, prompts, temperatures, and raw verdicts.
8. Keep deterministic retrieval, poison, and privacy checks beside the judge
   scores.
9. Use paired bootstrap confidence intervals over scenarios.
10. Do not tune retrieval on the final held-out real-world set.

The LLM judge becomes acceptable as a scaling tool only when its agreement with
human reviewers is high enough on the dimensions used for the claim. Until
then, human judgments are the primary result.

## Minimum end-to-end proof

If a full realistic benchmark is premature, publish five deep case studies
instead. Each should show:

- the original user goal;
- the real or fidelity-reviewed tool/agent trace;
- the relevant memory and why it was not queryable from the prompt alone;
- PARM, input RAG, naive output RAG, and naive-agent retrieval traces;
- the final responses with system names hidden;
- the cue-ablated or memory-removed counterfactual;
- a human verdict on usefulness, faithfulness, and privacy; and
- the exact failure of each losing baseline.

Five well-instrumented cases do not establish population-level performance, but
they make the original PARM insight concrete. They show a real agent noticing
something new in its own output, recalling the right part of the user's history,
and taking a better action without spraying unrelated memory into context.

## Decision

The existing benchmark suffices for the mechanism proof and for iterative
retrieval engineering. It does not suffice for the broadest real-world product
or research claim.

The next investment should be a realistic paired end-to-end suite, not a
wholesale switch to LLM scoring. Preserve deterministic PARMBench as the
explainable core, use human-reviewed real-world examples to establish external
validity, and introduce an LLM judge only as a calibrated, versioned secondary
measurement.
