# PARMBench Workflows: Three Scenarios

Three scenarios, three samples per condition, 100-record corpus, `gpt-5-mini`.

The headline is that the pilot scenario's result does not generalise. Two new
scenarios built to the same contract, against the same corpus and the same
ladder, give different answers, and neither reproduces the clean separation the
telemetry scenario showed.

## Fixture validity first

The memory-included ceiling hands the commitment to the agent in the goal. A
scenario whose ceiling fails is not measuring retrieval, so the ceiling is
reported before anything else.

| Scenario | Ceiling | Step-limit deaths | Usable |
| --- | ---: | ---: | --- |
| `telemetry-hotfix` | 18/18 | 4/54 | yes |
| `oncall-escalation` | 18/18 | 0/54 | yes |
| `release-freeze` | 15/18 | 0/54 | with a caveat |

`release-freeze` sits below the 16/18 bar set before the numbers were seen.
Roughly one in six of its failures could be the fixture rather than the policy,
and PARM's own ceiling on it is 2/3, so PARM's freeze numbers specifically
carry that doubt.

Both new scenarios needed two rounds of repair to get here. What that cost is
recorded in [construction defects](#construction-defects) below, because the
defects were more instructive than the first set of numbers.

## Results

Positive and control are counts of samples whose decisive assertions all
passed. Gold and spurious are means over the positive samples.

### `telemetry-hotfix`

| Condition | Positive | Control | Gold | Spurious |
| --- | ---: | ---: | ---: | ---: |
| `parm` | 3/3 | 2/3 | 1.0 | 7.7 |
| `naive_output_rag` | 3/3 | 3/3 | 2.0 | 38.7 |
| `all_entity_output_rag` | 3/3 | 3/3 | 1.7 | 61.7 |
| `no_memory` | 0/3 | 3/3 | 0.0 | 0.0 |
| `input_rag` | 0/3 | 3/3 | 0.0 | 5.0 |
| `prompted_memory_tool` | 0/3 | 3/3 | 0.0 | 0.0 |

### `oncall-escalation`

| Condition | Positive | Control | Gold | Spurious |
| --- | ---: | ---: | ---: | ---: |
| `parm` | 1/3 | 3/3 | 0.0 | 4.3 |
| `naive_output_rag` | 0/3 | 3/3 | 1.0 | 26.0 |
| `all_entity_output_rag` | 0/3 | 3/3 | 2.0 | 56.0 |
| `no_memory` | 0/3 | 3/3 | 0.0 | 0.0 |
| `input_rag` | 1/3 | 3/3 | 0.0 | 5.0 |
| `prompted_memory_tool` | 0/3 | 3/3 | 0.0 | 0.0 |

### `release-freeze`

| Condition | Positive | Control | Gold | Spurious |
| --- | ---: | ---: | ---: | ---: |
| `parm` | 3/3 | **0/3** | 1.0 | 2.0 |
| `naive_output_rag` | 3/3 | 2/3 | 1.0 | 20.0 |
| `all_entity_output_rag` | 0/3 | 3/3 | 0.0 | 51.0 |
| `no_memory` | 0/3 | 2/3 | 0.0 | 0.0 |
| `input_rag` | 1/3 | 1/3 | 0.0 | 5.0 |
| `prompted_memory_tool` | 1/3 | 3/3 | 0.0 | 0.0 |

## What this shows

**PARM false-intervenes on every freeze control.** Three samples, three
failures, all `pr66_merged`: the agent held back a pull request that touches
the shared CSV writer and no customer's tenant. PARM admitted three records
each time and the agent declined to merge. This is the failure the control
exists to catch, and on this scenario it is not occasional. It is the result.

**PARM never reaches the escalation commitment.** Zero gold admitted across all
three positive samples, so on that scenario it behaves as `no_memory` with
extra retrieval cost. Its 3/3 control is abstention by absence rather than
restraint: there was nothing it was going to admit either way.

**Retrieving the memory is not sufficient.** On escalation,
`naive_output_rag` admitted a gold record, on time, in all three samples, and
still scored 0/3. `all_entity_output_rag` admitted two gold records and also
scored 0/3. The commitment reached the model and did not change the action.
Every result in the pilot that reads as "retrieval worked" needs this caveat:
the deciding step is the model's, and it can be handed the right memory and do
the ordinary thing anyway.

**The pilot's PARM positive is the exception, not the pattern.** 3/3 on
telemetry, 3/3 on freeze, 1/3 on escalation, against a clean control only on
telemetry. One scenario cannot support a claim about the mechanism; three
scenarios show why.

## Construction defects

Both new scenarios failed their first evaluation for reasons that were mine,
not the policies'. Recorded because the same mistakes are easy to repeat.

**An assertion that did not follow from the commitment.** The escalation
positive originally required issue #72 to stay open. Three records carry the
commitment and none of them says "do not close it"; they say a human gets told.
Agents that commented "closing here because the bug was fixed upstream, but
escalating for ownership" had done exactly what the memory asks and were scored
wrong for it. Ceiling 14/18. Scoring the thing every carrier agrees on took it
to 18/18.

**A goal and a set of assertions describing different tasks.** The freeze goal
said "land the fixes that are ready". Agents read "land" as applying changes to
a release branch themselves, wrote the files directly, and never opened pull
request #66 at all, so every assertion about pull-request state was checking
work nobody had been asked to do. Ceiling 7/18. Naming the pull requests in the
goal took it to 15/18.

**A cue that could not reach its own memory.** The escalation cue reached its
gold at rank 8 on hybrid and nowhere in the top ten on dense. No retrieval
policy could have solved it. The repair was to make gold the record whose
vocabulary matches the cue and to have the customer say "contractual" — the
word the memory answers in.

**A step budget too small for the task.** Ten of 54 freeze runs died at the
32-step limit and scored as decision failures. Scenarios now declare their own
budget. This was real, and fixing it alone changed no ceiling: truncation was a
separate defect from the two above, and fixing the visible problem first is
what made the other two take a second round to find.

## Limits

Three samples per condition. One model. One environment adapter. The three
scenarios share a repository shape and a customer, so they are not independent
draws from the space of realistic tasks. `release-freeze` has a ceiling of
15/18 and its numbers should not be quoted without it.
