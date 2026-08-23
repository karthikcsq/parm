# Relevance calibration audit v1

This is a calibration audit of the frozen `benchmark_parmbench_v1` batch. It
does not modify the batch. Its job is to measure how often a scenario whose
claim passed source support still fails to be a realistic personalization
decision, to name the failure modes precisely, and to select reference
examples for the next pipeline version. The audit was performed by a Claude
model reading raw sources directly, independent of the `gpt-5-mini` model
that drafted, constructed, and fairness-tested the batch.

Rows live in `relevance-audit-v1.jsonl` (one JSON object per audited
scenario). The stratified sample is recorded in
`relevance_audit_sample.json`: 29 of 157 scenarios, containing every known
suspect case, three candidate strong cases, and at least two random samples
from each capability family.

## The relevance contract

Source support alone is not enough. A scenario is valid only when every item
holds:

1. The raw user-authored evidence supports the fact. Assistant text cannot
   establish it.
2. The fact is durable or currently operative: a preference, constraint,
   exclusion, active commitment, stable relationship, owned item,
   accessibility need, or concrete schedule. A one-off anecdote, wording
   request, hypothetical, or topic question is normally insufficient.
3. The task is an ordinary decision someone would plausibly ask an assistant
   to make.
4. Ordinary visible evidence makes choice A best without using personal
   memory.
5. A later visible cue states a concrete affordance of choice B.
6. The raw fact and that affordance make B rationally preferable for this
   person.
7. The justification requires no invented possession, permission,
   relationship, location, medical conclusion, future plan, or unstated
   preference.
8. The cue alone does not make B best for an arbitrary person.
9. Removing only the cue restores A as the correct choice.
10. The prompt does not telegraph memory retrieval or the eventual
    personalization axis.
11. The expected answer can be named exactly from the visible choices.
12. Sensitive facts are used only when necessary, are labeled, and do not
    need to be repeated in the final response.

The pipeline must enforce three separate judgments:

- source support: did the user say it? (`evidence_gate`, already enforced)
- memory quality: is it a durable or currently operative fact worth
  retaining? (new pre-construction gate)
- decision validity: does this exact task, cue, and choice change follow
  without invented assumptions? (new post-construction gate)

## Failure taxonomy

Machine-readable reasons used in the audit rows and by the new gates:

- `ephemeral_event_as_durable_memory`
- `question_as_personal_fact`
- `editing_request_as_preference`
- `topic_overlap_without_decision_relevance`
- `invented_relationship_or_permission`
- `invented_location_or_possession`
- `medical_or_sensitive_overreach`
- `cue_alone_determines_choice`
- `weak_personal_benefit_cannot_override_ordinary_evidence`
- `implausible_task_or_affordance`
- `incorrect_capability_label`
- `control_does_not_remove_personalized_advantage`

Source-support rejections keep their existing distinct reasons and are never
mixed into this list.

## Results

29 scenarios audited: 5 accepted, 8 borderline, 16 rejected. 12 of 29
capability labels are wrong. The batch's fairness gate is real but
insufficient: a scenario can follow the A/A/B pattern perfectly while its
personalization is contrived.

Failure-category counts across the 29 rows (multiple per scenario allowed):

| count | category |
| ---: | --- |
| 15 | weak_personal_benefit_cannot_override_ordinary_evidence |
| 9 | incorrect_capability_label |
| 8 | topic_overlap_without_decision_relevance |
| 6 | implausible_task_or_affordance |
| 6 | ephemeral_event_as_durable_memory |
| 4 | invented_location_or_possession |
| 4 | medical_or_sensitive_overreach |
| 3 | invented_relationship_or_permission |
| 3 | editing_request_as_preference |
| 2 | question_as_personal_fact |
| 2 | control_does_not_remove_personalized_advantage |
| 1 | cue_alone_determines_choice |

Systemic findings beyond individual rows:

- `memory.sensitive_terms` is empty on every audited case, including several
  that pivot on medical facts (childhood vaccination, childhood lung
  trouble, knee surgery, tension headaches). Item 12 currently has no
  effective coverage anywhere in the batch.
- Five of slice B's nine rejects trace to one upstream defect: the drafter
  treats the user's own editing or translation request, or a transient
  in-session state, as a personal fact. A supply-side rule that rejects
  claims whose span is the request line itself removes most of them.
- Two cases (`p417-480-s2`, `p553-44`) carry the personalized advantage in
  non-cue option text, so the cue-ablated control does not remove it. The
  decision-validity gate must check control validity against the full option
  bodies, not just the cue sentence.
- All eight known suspect cases from the handoff were confirmed. One
  handoff hunch was refuted: `p284-100` (Friday deadline) reads generic and
  dateless at source and is rejected, while `p350-39` (knee) is borderline,
  not strong, because its capability label and sensitivity handling are
  wrong.

## Approved reference examples

Five scenarios remain convincing after reading the raw source. They are the
positive references for the next pipeline version.

| scenario | evidence span (user-authored) | memory | task | A (ordinary) | B (personalized) | cue | causal justification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p19-33-s2 | "I've learned to skip the syrupy desserts... I'll choose a small bowl of fresh fruit" | Durable dietary exclusion of syrup-heavy desserts | Pick an event caterer | Cheaper caterer, explicitly no substitutions | Caterer serving plain seasonal produce | Menu note: fruit plate with no added sweeteners | The stated exclusion makes the no-substitution caterer unusable for them, paraphrase-only overlap |
| p190-172 | Season-long athlete-performance analysis owed to a partnering federation | Active analytic commitment to a federation | Choose a trip itinerary | Better-written scenic itinerary | Itinerary near the competition hall | Prearranged onsite access to the hall's event footage archive | The standing commitment makes archive access materially useful; useless to anyone else |
| p28-251 | Active drafting project on theological traditions in politically diverse societies | Current writing project on that subject | Choose a book supplier | Cheapest broad-catalogue supplier | Specialist vendor | Stocks faith-communities-and-government-institutions titles | Semantic, non-lexical mapping from project to stock; price premium justified only for this person |
| p666-244 | Performs stand-up comedy with speculative/sci-fi material | Durable creative practice and genre | Pick tonight's event | Generally best-reviewed event | Open mic night | Reserves solo slots for speculative material, genre audience | The slot is a real payoff worthless to a non-performer; clean ablation |
| p855-595-s2 | Owns an NES (first person, corroborated in a later turn) | Owned item with analog output | Choose an AV installer | Higher-rated installer | Installer retaining analog connections | Keeps legacy analog inputs wired | True one-hop inference (NES needs analog input); the cue cannot supply the ownership |

Rejected rows, the minimum invented assumption per row, and per-scenario
notes are in `relevance-audit-v1.jsonl`.

## Pilot v3 under the new gates

A bounded pilot (`data/benchmark_parmbench_pilot_v3`, see its README for the
full record) ran 60 unspent candidate rows through the implemented gates.
The comparison with the ungated pipeline:

| stage | ungated (v2) | gated (v3 pilot) |
| --- | ---: | ---: |
| rows drafted | 60 | 60 |
| passing supply | 42 | 22 |
| scenarios kept | all fairness survivors | 4 of 22 |
| manual verdicts | 5 accept / 8 borderline / 16 reject of 29 | 0 accept / 2 borderline / 2 reject of 4 |

The headline is not the acceptance rate; it is where the failures moved.
Every audited category-1/2/7 failure mode (questions as facts, ephemeral
anecdotes, editing requests, invented possessions and relationships, medical
overreach) is absent from the pilot survivors. The residual failures are
fixture-level:

1. Lexical residue of the claim in option names or non-cue bodies survives
   ablation. Now covered deterministically:
   `decision_validity.control_residual_advantage` and
   `capability_conflicts_with_lexical_target` run in the builder before any
   judge call, and retroactively flag exactly the two rejected pilot
   survivors while passing the two borderlines.
2. Decoys annotate the personalization axis ("no press access for
   reviewers"), which can leak the axis into the no-memory condition. Not
   yet checked; needs a decoy-text rule in the next construction version.
3. The prompt states the ordinary mechanism as an instruction ("based
   primarily on price and availability"), which turns memory use into
   instruction violation. Needs a prompt-shape rule.
4. The cue-ablated control is only checked as "A wins again", not as "where
   does the memory now point"; a memory-holding reader can be redirected to
   a third option. Needs an ablated-preference field in the
   decision-validity rubric.
5. Sensitive facts pass upstream (5 sensitive rows in the pilot supply) but
   none survived construction, so item 12 remains untested end to end.
6. The pilot's four survivors share one span style and one envelope band;
   the pilot is too small to exercise the axis distributions.

These six gaps are the work list for the next construction round. The gates
as implemented remove the failure modes that invalidated the frozen batch's
realism, and the pilot is deliberately too small to claim more than that.
