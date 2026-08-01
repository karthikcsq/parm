# PARM

PARM evaluates personal-memory retrieval triggered by a cue that appears in a
later tool result or agent output. The original prompt gives no reason to
retrieve the memory. The later cue makes one stored fact useful, and the
retrieval policy must improve the decision without changing the cue-ablated
control.

PARMBench contains 18 scenarios and 54 cases. Each scenario has a positive
case, a cue-ablated control, and a memory-included ceiling. All retrieval
conditions use the same frozen `amara-life-v1` memory substrate.

The current repaired-fixture development result is:

| Condition | Correct positive and control decisions |
| --- | ---: |
| PARM V5 | 30/36 |
| No memory | 18/36 |
| All-entity output RAG | 18/36 |
| Naive output RAG | 18/36 |
| Prompted memory-tool agent | 17/36 |
| Enhanced input RAG | 15/36 |

PARM admitted memory at 71.43% precision and 75% recall. Read the
[V5 result](docs/results/benchmark-v5.md) before citing these numbers: V5 was
tuned after the frozen expansion first pass, so it is a development result
rather than a held-out generalization result.

## Two suites

Deterministic PARMBench answers whether the retrieval mechanism works. It is
exact, cheap, and fast, and it can only score one label from one observation.

[PARMBench Workflows](data/workflows_v1/README.md) answers whether that
mechanism changes what an agent *does*. An agent gets an ordinary goal and a
seeded tool environment, works a real trajectory, and is scored on the
environment it leaves behind: the final state, the admitted sources, and
whether the memory arrived in time to affect the action it governs. The
scoring is programmatic throughout; no LLM judge decides a result.

Both suites use the same triplet contract, so a scenario always carries its own
cue-ablated control and memory-included ceiling.

## Quick start

Use the Anaconda interpreter in this checkout:

```powershell
$python = 'C:\Users\karth\anaconda3\python.exe'
& $python -m pip install -e .
& $python -m spacy download en_core_web_sm

parm-bench validate data\benchmark_v1
parm-bench validate data\benchmark_personamem_v0
parm-bench inspect data\benchmark_v1 `
  --case parm-amara-conference-agenda-positive

$env:PYTHONPATH = 'src'
& $python -m unittest discover -s tests
```

Validation should report 54 frozen Amara cases and 90 PersonaMem-v2
development cases, and the test suite should pass. Continue with
[How to Run PARMBench](docs/running-parmbench.md) to replay the published run
or execute another retrieval condition.

## Documentation

This README is the only documentation entrypoint. Pages under `docs/` have one
defined job and link to related detail without maintaining a second index.
READMEs under `data/` describe the artifact directory in which they live; they
are manifests, not alternate project guides.

### Understand the research and system

| Page | Purpose |
| --- | --- |
| [Research claim and scope](docs/research-scope.md) | Motivation, hypotheses, boundaries, current evidence, and unresolved claims |
| [Architecture](docs/architecture.md) | End-to-end and PARM retrieval waterfalls, module ownership, and design trade-offs |
| [Interactive retrieval quiz](docs/parm-retrieval-quiz.html) | Test cue timing, query formation, selective admission, and benchmark controls in the browser |
| [Output-cued examples](docs/examples.md) | The 20 approved memory-cue relationships, including the 18 executable scenarios |

### Run and extend PARMBench

| Page | Purpose |
| --- | --- |
| [How to Run PARMBench](docs/running-parmbench.md) | Install, validate, replay, run comparisons, inspect traces, and troubleshoot |
| [Evaluation contract](docs/benchmark-evaluation.md) | Case semantics, deterministic metrics, correctness, failure taxonomy, and the workflow assertion contract |
| [Workflows pilot dataset](data/workflows_v1/README.md) | The executable scenario, its environment fixtures, and the personal corpus behind it |
| [How to Construct a Scenario](docs/benchmark-construction.md) | Generate a symmetric triplet, establish fixture fairness, and freeze a first pass |
| [Rebuild the memory index](docs/rebuilding-memory-index.md) | Prepare Amara Life with GBrain and export the neutral frozen index |

### Read the evidence and planned evaluation

| Page | Purpose |
| --- | --- |
| [V5 development result](docs/results/benchmark-v5.md) | Current repaired-fixture comparison and remaining failures |
| [Workflows first pass](docs/results/workflows-v1-first-pass.md) | The executable ladder on the pilot scenario, and what one sample can and cannot show |
| [Frozen expansion first pass](docs/results/benchmark-expansion-first-pass.md) | Untuned generalization record that motivated the V5 changes |
| [PersonaMem-v2 first pass](docs/results/personamem-v0-first-pass.md) | Untuned 30-person development comparison and per-corpus results |
| [Real-world evaluation](docs/real-world-evaluation.md) | Limits of the controlled benchmark and the human-calibrated end-to-end judge design |
| [Larger dataset candidates](docs/dataset-candidates.md) | Hugging Face survey, recommended sources, conversion procedure, and retrieval-substrate changes |
| [Roadmap](docs/roadmap.md) | Next retrieval and external-validity work, in priority order |

### Project records

| Page | Purpose |
| --- | --- |
| [Decision log](docs/history/decisions.md) | Dated decisions that changed benchmark or retrieval behavior |
| [Convergence retrieval V1](docs/history/parm-convergence-retrieval-v1.md) | Historical design record for the first PARM selector |
| [Retrieval-mode axis plan](docs/history/retrieval-mode-axis-plan.md) | Historical implementation plan for the shared retrieval substrate |

Historical records explain how the current design was reached. The current
source of truth is the architecture page, evaluation contract, CLI, and tests.

## Repository map

```text
src/parm_bench/               benchmark package and CLI
src/parm_bench/workflows/     environments, trajectory runner, and verifier
tests/                        unit and CLI smoke tests
scripts/                      dataset and retrieval-artifact builders
data/benchmark_v1/            54 executable cases and large contexts
data/benchmark_personamem_v0/ 90 development cases over 30 persona corpora
data/workflows_v1/            executable agent cases, fixtures, and corpus
data/personamem-v2-train-v0/  bounded source records and provenance
data/retrieval-indexes/       frozen neutral memory substrate
data/expansion-caches/        frozen enhanced-mode query expansions
data/response-caches/         replayable model calls
data/workflow-caches/         replayable agent turns and admission decisions
data/benchmark-results/       predictions, configs, and metrics
docs/                         canonical guides, evidence, roadmap, and history
```

The root `.env` is ignored. Start from `.env.example` and set
`OPENAI_API_KEY` for live model-backed runs.
