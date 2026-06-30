# PARM

Parallel Asynchronous Retrieval of Memory research prototype.

## Benchmark V1

This repo currently contains the first PARM deliverable: a synthetic benchmark
for output- and tool-response-conditioned personal-memory retrieval.

The core V1 question is not whether a system can solve elegant multi-hop graph
paths. It is whether an agent can monitor large generated or tool-provided
contexts, identify the few cues that warrant memory lookup, and suppress
spurious memory matches. The memory store should stay realistic and simple:
memory pages or facts linked to extracted entities, timestamps, and light graph
metadata for diagnostics.

### Commands

Run from the repo root:

```powershell
$env:PYTHONPATH='src'
python -m parm_bench.cli validate data/benchmark_v1
python -m parm_bench.cli inspect data/benchmark_v1 --case parm-v1-004
python -m parm_bench.cli run data/benchmark_v1 --baseline parm_oracle_monitor --out .Codex/benchmark-results/parm_oracle_monitor.jsonl
python -m parm_bench.cli score .Codex/benchmark-results/parm_oracle_monitor.jsonl --gold data/benchmark_v1
python -m unittest discover -s tests
```

Available baselines:

- `no_memory`
- `input_rag`
- `prompted_memory_tool`
- `parm_oracle_monitor`

The canonical generated dataset is in `data/benchmark_v1/cases.jsonl`.
Project plans and benchmark run artifacts live under `.Codex/`.

For the public evaluation contract, metrics, scorer policy, and failure
taxonomy, see `docs/benchmark-evaluation.md`. For a full component-by-component
guide, see `.Codex/benchmark-guide.md`.
