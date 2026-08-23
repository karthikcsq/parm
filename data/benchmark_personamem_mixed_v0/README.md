# PersonaMem mixed-context development slice

**Status: legacy development experiment.** This slice is a tuned
retrieval-development fixture, not a benchmark, and it must never be cited as
canonical PARMBench evidence. A source-support audit found that 13 of the 30
labeled personal-memory claims are unsupported by the raw indexed
conversation. The generator also repeats a fixed 9.8-ordinary-versus-8.8-
personalized construction pattern, and the development admission judge tuned
against this slice used a rubric that explicitly searched for a lower-ranked
candidate to beat the ordinary winner, mirroring how the slice was
constructed. Together these mean results measured on this slice
cannot support a superiority claim for any retrieval method, including PARM.
The dataset is kept only so the recorded experiments in
`docs/retrieval-experiments/personamem-v0.md` stay reproducible. The
canonical benchmark construction contract lives in
`docs/benchmark-construction.md`; that contract, not this slice, governs what
counts as a valid PARMBench scenario.

This dataset preserves the 30 PersonaMem V0 source scenarios and triplet
decisions while replacing the clean recommendation catalog with six noisy
context envelopes:

- research notebooks;
- forwarded email threads;
- meeting transcript dumps;
- broken web clippings;
- working drafts; and
- mixed markdown, chat, table, and JSON fragments.

Candidate names remain visible so the final decision can be scored
deterministically. They do not use `Listing` prefixes, one-row-per-option
formatting, or a fixed cue position. The lower-ranked candidate uses a neutral
proper name in both twins; only the surrounding prose carries the positive
affordance. The cue-ablated twin replaces that prose with a neutral logistics
note rather than an opposite preference cue. Archival noise uses reference
labels that are not plausible final choices. The frozen
`benchmark_personamem_v0` and Amara benchmark are not modified.

Representative cases include:

- a meeting transcript whose carried-forward choice is a connected city quest,
  while a recovered line much later describes Northstar House as a completely
  screen-free weekend;
- a mixed Markdown/chat export whose general recommendation is an artisan
  pantry display, while one buried chat line says Westbridge uses plain,
  undecorated spice jars; and
- a working draft that recommends a generic cooking program, with a later
  unresolved insertion describing a Puerto Rican home-cooking format.

Each scenario has three causal variants. The positive context contains the
buried affordance but withholds memory; the cue-ablated twin replaces only that
affordance with neutral logistics; the memory-included ceiling adds the
personal fact to the prompt. This tests cue-region discovery, corpus-scoped
source retrieval, abstention on the matched control, and whether admitted
memory actually changes the final visible choice.

This slice inherits the V0 source rows. A separate semantic audit found that
13/30 labeled memory claims are not supported by the raw conversation the
retriever is allowed to index. Treat mixed V0 as a retrieval-development slice,
report the 17 supported or inferable scenarios separately, and replace the
unsupported rows before promoting a sealed benchmark.

The finalized fixture-fairness run passes all 90 cases with `gpt-5-mini`:
positive and cue-ablated cases select the ordinary evidence winner without
memory, while memory-included cases select the intended personalized option.
The semantic PARM development run scores 21/30 positive decisions and 27/30
controls. Full retrieval experiments, including failed candidates and the
source-support audit, are recorded in
`docs/retrieval-experiments/personamem-v0.md`.

Rebuild and validate:

```powershell
$env:PYTHONPATH = 'src'
& 'C:\Users\karth\anaconda3\python.exe' `
  scripts\build_personamem_mixed_v0_benchmark.py
& 'C:\Users\karth\anaconda3\python.exe' -m parm_bench.cli validate `
  data\benchmark_personamem_mixed_v0
```
