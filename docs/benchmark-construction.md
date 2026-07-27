# How to Construct a PARMBench Scenario

This is the procedure used to turn approved output-cued memory examples into
the 18 executable scenario triplets in `data/benchmark_v1`.

## Acceptance contract

A scenario is eligible only when all of these statements are true:

1. The initial prompt is ordinary and does not ask for memory.
2. A later tool result or agent output introduces the decisive cue.
3. The cue maps to an authoritative memory already present in the declared
   corpus version.
4. Without memory, one visible choice is defensible.
5. With memory, a different visible choice is better.
6. The final answer can name exactly one visible label or name.
7. Removing the cue collapses the expected answer back to the output-only
   choice.
8. Giving the memory explicitly makes the memory-conditioned choice usable.

Examples 19 and 20 remain outside the executable set because their proposed
health memories are not present in `amara-life-v1`. Do not invent corpus
history to make a benchmark case executable.

## Step 1: Approve the memory-cue relationship

Start from [the example catalog](examples.md). Record:

- the ordinary task and answer format;
- the incidental output cue;
- the output-only choice;
- the memory-conditioned choice;
- the authoritative memory source IDs;
- a short model-visible memory summary;
- sensitive details that should not appear in the answer; and
- the cue type.

Read the source artifact itself. The memory summary must be entailed by it and
must identify the relevant output affordance without using benchmark-only IDs.

## Step 2: Add one declarative case specification

Add the scenario to `SPECS` in `scripts/build_pilot_cases.py`. The required
fields are:

| Field | Meaning |
| --- | --- |
| `slug` | Stable readable scenario identifier |
| `prompt` | Ordinary task with an exact one-choice answer contract |
| `kind` | `tool_result` or `assistant_output` |
| `cue_type` | Human-readable cue taxonomy |
| `cue` | Complete target listing text |
| `replacement` | Symmetric cue-ablated listing text |
| `query` | Diagnostic description of the intended relationship |
| `sources` | Source ID, source path, and perturbation labels |
| `memory_text` | Short faithful memory summary used by the ceiling |
| `output_choice` | Best choice without the personal memory |
| `memory_choice` | Best choice after using the memory |
| `control_choice` | Optional control answer when replacement changes the label |
| `sensitive_terms` | Private phrases the answer need not expose |
| `example_number` | Approved catalog number |

The builder hashes each declared source and writes provenance into every case.

## Step 3: Create the large-output fixture

Add the matching specification to `scripts/build_pilot_contexts.py`.

The generated context must:

- reach 8,000 to 12,000 `cl100k_base` tokens;
- use unique natural-language labels;
- put the cue at listing 147;
- contain a credible output-only lead near the top;
- include ordinary filler with decorrelated company, speaker, topic, detail,
  format, and template strides; and
- plant several near-miss decoys before the target.

The target must not be the only detailed or unusually phrased row. A model that
selects it by visual salience rather than memory invalidates the case.

## Step 4: Generate the triplet

Run:

```powershell
$env:PYTHONPATH = 'src'
& 'C:\Users\karth\anaconda3\python.exe' scripts\build_pilot_contexts.py
& 'C:\Users\karth\anaconda3\python.exe' scripts\build_pilot_cases.py
```

For each scenario, the case builder produces:

- `positive`: original prompt, original observation, memory-conditioned gold;
- `cue-ablated`: original prompt, one exact observation replacement,
  output-only gold; and
- `memory-included`: positive observation plus a `Known personal memory`
  preamble, memory-conditioned gold.

The positive and memory-included observations are byte-identical. The control
uses the same context file and one declared replacement.

## Step 5: Validate structural symmetry

Run:

```powershell
parm-bench validate data\benchmark_v1
```

Then verify for the new triplet:

- all three case IDs are unique and stable;
- source hashes match the tracked corpus;
- positive and ceiling observations are identical;
- control replacement occurs exactly once;
- cue and replacement preserve similar length and writing style;
- every expected choice is visible in its resolved observation;
- the control contains no leftover decisive entity or affordance;
- every declared gold source is non-poison; and
- context token counts remain inside the required range.

## Step 6: Establish fixture fairness before retrieval tuning

Run `no_memory` on all three variants before using the case to judge retrieval.
The desired pattern is:

| Variant | No-memory result |
| --- | --- |
| Positive | Output-only choice |
| Cue-ablated | Output-only choice |
| Memory-included | Memory-conditioned choice |

If the positive already selects the memory target, the case cannot measure
beneficial decision change. If the control does not select the declared
output-only choice, repair the fixture while preserving the approved
memory-cue relationship. Rerun every condition after any fixture change.

Deterministic construction and scoring do not establish semantic fairness. If
repairs begin accumulating, templates become topic-specific, or reviewers
cannot resolve failures from the declared choices alone, stop tuning the
template. Add a versioned LLM-judge rubric, calibrate it against sampled human
review, and retain the deterministic checks as provenance and symmetry gates.

## Step 7: Freeze the first pass

Before inspecting retrieval failures:

1. commit the cases, contexts, builders, and validation tests;
2. run the full baseline matrix against unchanged retrieval and judgment code;
3. retain predictions, config sidecars, metrics, and response caches under a
   new namespace; and
4. publish the aggregate and split results.

After results have been inspected, the expansion is development data. Preserve
the first-pass artifacts as the honest generalization measurement and give all
tuned results a new version.

## Step 8: Review the case qualitatively

The structural validator cannot decide whether the choice is realistic. A
human review should answer:

- Would a reasonable person accept the output-only choice without memory?
- Does the memory materially change the decision rather than add trivia?
- Is the target buried naturally in the output?
- Are the decoys semantically close enough to prevent keyword shortcuts?
- Does the control remove only the decisive relationship?
- Is the memory safe and necessary to use?
- Would a wrong but plausible answer reveal a retrieval failure, a judgment
  failure, or an ambiguous fixture?

Record any non-obvious repair in
[the decision log](history/decisions.md).

## Verification checklist

```powershell
$env:PYTHONPATH = 'src'
& 'C:\Users\karth\anaconda3\python.exe' -m unittest discover -s tests
parm-bench validate data\benchmark_v1
git diff --check
```

The scenario is ready only when the generated files, builder specification,
source provenance, no-memory fairness run, and first-pass namespace agree.
