# PARM

**PARM is a causal benchmark suite for when an agent should use personal memory during a real workflow.**

**PARMBench Workflows is the primary evaluation track.** An agent receives an ordinary request, works through a seeded tool environment, discovers a decisive cue in a late observation, and is scored on the environment state it leaves behind. Prompt-time retrieval cannot see that cue. Naively searching the whole observation can find the right memory, but often floods the agent with unrelated personal context or changes an action when the cue is absent.

The controlled PARMBench calibration batch is the companion mechanism track: it isolates the same late-cue retrieval question in small, reproducible causal triplets. It is useful for diagnosing retrieval and admission behavior, but it is not the project’s end goal.

PARM asks a stricter question:

> Can an agent recognize when a late observation makes a specific memory actionable, retrieve it before the governed action, and stay silent when the cue is removed?

Every scenario is a causal triplet:

- **Positive:** a source-grounded late cue makes a memory relevant.
- **Cue-ablated control:** the decisive cue is removed while the rest of the task remains comparable.
- **Memory-included ceiling:** the agent receives the relevant source directly, verifying that the intended action is achievable.

## Why this is promising

PARM is built as research infrastructure rather than a one-off prompt demo:

- **157 frozen, independently constructed base scenarios** / **471 executable cases** across **140 persona-isolated histories** in the current PARMBench v1 calibration batch.
- **Raw-history, retrieval-agnostic evaluation:** PARMBench provides source material and a causal contract, not a preferred memory store or retrieval implementation.
- **Evidence and fairness gates:** scenario construction checks source support, prompt opacity, persona isolation, cue selectivity, and ceiling actionability before a case can support a claim.
- **Executable workflow evaluation:** PARMBench Workflows evaluates multi-step tool trajectories from final environment state, timely memory admission, and intervention restraint—without using an LLM judge as the scorer.
- **Replayable artifacts:** data, manifests, validation profiles, retrieval artifacts, and result sidecars are versioned so claims can be inspected and reproduced.

The benchmark is designed to make progress legible: a retrieval policy must improve the positive case *and* preserve the cue-ablated control. A system that retrieves aggressively and changes behavior everywhere does not pass.

## Current evidence

### PARMBench v1: frozen causal calibration batch

The current v1 batch contains 157 base scenarios, 471 cases, and 140 persona-isolated histories. Its final frozen replay passes **157/157** scenario contracts. The batch is intentionally retrieval-agnostic: it is a durable, auditable substrate for comparing memory systems fairly rather than an evaluation designed around PARM's own implementation.

See the [PARMBench v1 dataset record](data/benchmark_parmbench_v1/README.md) and [construction contract](docs/benchmark-construction.md).

### Amara regression suite: selective late-cue retrieval

On the repaired 54-case Amara development suite, PARM V5 achieved **30/36** positive/control decisions. The strongest comparison conditions achieved **18/36**. PARM admitted **15 gold sources with 6 spurious admissions**; all-entity output RAG admitted 16 gold sources but **1,361 spurious sources**.

| Condition | Correct positive + control decisions |
| --- | ---: |
| PARM V5 | **30/36** |
| No memory | 18/36 |
| All-entity output RAG | 18/36 |
| Naive output RAG | 18/36 |
| Prompted memory-tool agent | 17/36 |
| Enhanced input RAG | 15/36 |

This is an encouraging mechanism result, not a held-out product claim: V5 was improved using the expansion cases. The immutable first pass and the full limitations are preserved in the [V5 report](docs/results/benchmark-v5.md).

### PARMBench Workflows: memory that changes actions

The workflow suite currently validates **18 deterministic cases** across tool-using agent environments. It measures the state an agent leaves behind, the sources it admits, and whether memory arrives before the action it is meant to govern. This closes an important gap between one-shot choice evaluation and realistic agent execution.

The workflow work is deliberately candid: it records successful interventions, false interventions, ceiling failures, and retired candidates. That discipline is a feature—new workflow families are certified before their policy results are used as evidence. Start with the [workflow suite](data/workflows_v1/README.md), [scenario-set result](docs/results/workflows-v1-scenario-set.md), and [applicable-memory handoff experiment](docs/results/workflows-v1-memory-notice-v1.md).

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .

PYTHONPATH=src python -m parm_bench.cli validate data/benchmark_parmbench_v1
PYTHONPATH=src python -m parm_bench.cli workflow validate data/workflows_v1
PYTHONPATH=src python -m unittest discover -s tests
```

Live model-backed replay requires `OPENAI_API_KEY` in a local, ignored `.env`; canonical validation and deterministic scoring do not.

## Repository map

```text
src/parm_bench/                  benchmark package and CLI
src/parm_bench/workflows/        environments, runners, policies, and verifiers
tests/                           unit and CLI coverage
data/benchmark_parmbench_v1/     frozen raw-history causal calibration batch
data/workflows_v1/               executable multi-step workflow scenarios
data/retrieval-indexes/          frozen retrieval artifacts
data/benchmark-results/          replayable predictions, configs, and metrics
docs/                            claim contracts, architecture, evidence, and roadmap
```

## Documentation

| Document | What it covers |
| --- | --- |
| [Research claim and scope](docs/research-scope.md) | The late-cue retrieval problem, hypotheses, and explicit boundaries |
| [Architecture](docs/architecture.md) | Retrieval, admission, handoff, and workflow components |
| [Evaluation contract](docs/benchmark-evaluation.md) | Causal triplets, deterministic metrics, and failure taxonomy |
| [Scenario construction](docs/benchmark-construction.md) | Evidence, isolation, fairness, and ceiling gates for new cases |
| [PARMBench v1 calibration batch](data/benchmark_parmbench_v1/README.md) | 157-scenario raw-history benchmark record |
| [Workflow suite](data/workflows_v1/README.md) | Multi-step agent environments and final-state verification |
| [V5 development result](docs/results/benchmark-v5.md) | Selective-retrieval comparison, artifacts, and limitations |
| [Workflow scenario set](docs/results/workflows-v1-scenario-set.md) | Cross-scenario workflow results and construction lessons |
| [Roadmap](docs/roadmap.md) | Next evaluation and external-validity work |

## Research posture

PARM is ambitious about the problem and conservative about the evidence. It does not claim to have solved general long-term memory, broad agent reliability, or real-world personal-agent usefulness. It provides a rigorous way to test one concrete capability that current evaluations frequently blur: **whether memory should intervene only after the environment gives it a reason to.**
