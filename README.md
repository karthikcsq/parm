# PARM

PARM evaluates output cue-triggered personal memory: an ordinary prompt is
followed by one large agent output or tool result, an incidental cue makes a
stored memory newly relevant, and that memory should materially improve the
final decision.

## Benchmark V1 pilot

The executable pilot contains five approved Amara Life cases and five
cue-ablated controls:

1. conference agenda -> unresolved NovaMind diligence;
2. AI news digest -> CoreWeave dependency;
3. podcast feed -> personal burnout pattern;
4. vendor report -> NovaTech counterparty risk; and
5. lunch search -> interrupt the desk-lunch pattern.

Each observation is 8,000-12,000 `cl100k_base` tokens. Prompts do not ask for
memory, cues appear only in the later observation, and success requires a
different decision rather than a relevant-sounding aside.

Every case asks for exactly one final choice in ordinary language. The answer
key is the option's visible title, company, or restaurant name—not an opaque
benchmark ID. The same identifying entity or pattern is readable in the Amara
memory prose, so a model can connect memory to the noisy output without access
to hidden metadata.

The compact raw Amara fixture is tracked under `data/amara-life-v1/`. GBrain is
the memory substrate; PARM is the cue-selection, retrieval-timing, admission,
and decision-evaluation layer.

## Run

```powershell
$env:PYTHONPATH='src'
python -m parm_bench.cli validate data/benchmark_v1
python -m parm_bench.cli inspect data/benchmark_v1 --case parm-amara-conference-agenda-positive
python -m unittest discover -s tests
```

## Baseline status

No baseline is currently implemented or registered. The repository contains
only the `Baseline` protocol, registry, CLI boundary, and planned comparison
inventory. `parm-bench run` refuses unimplemented names rather than executing a
placeholder.

Each baseline will be developed and reviewed as a real experiment before it can
produce results. The repo-local GBrain adapter is available as infrastructure:

```powershell
python -m parm_bench.cli prepare-amara
```

See `docs/benchmark-evaluation.md` for the scoring contract and
`docs/parm-output-cued-memory-examples.md` for the 20-example expansion set.
