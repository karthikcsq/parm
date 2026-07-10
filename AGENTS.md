# Repository Guidelines

## Project Structure & Module Organization

`src/parm_bench/` contains the Python package and CLI entrypoint. Core benchmark code lives in modules such as `dataset.py`, `baselines.py`, `retrieval.py`, `scoring.py`, and `cli.py`; the local HTML workbench is in `src/parm_bench/web/`.

`tests/` mirrors the package with focused unittest files such as `test_baselines.py`, `test_retrieval.py`, and `test_cli_smoke.py`. `docs/` holds design notes and evaluation contracts. `data/benchmark_v1/` contains benchmark cases and context fixtures, while `data/retrieval-indexes/`, `data/expansion-caches/`, and `data/benchmark-results/` hold replayable artifacts.

## Build, Test, and Development Commands

This project runs on the Anaconda interpreter at `C:\Users\karth\anaconda3\python.exe`, which has the `openai`, `numpy`, and `tiktoken` dependencies installed. The bare `python` on PATH resolves to the msys2 build, which lacks them, so invoke the Anaconda interpreter explicitly (or activate its environment) when running or testing. Tests import from `src/`, so set `PYTHONPATH=src` unless the package is installed editable.

Install locally before running commands:

```powershell
python -m pip install -e .
python -m spacy download en_core_web_sm
```

Validate and inspect the pilot benchmark:

```powershell
parm-bench validate data/benchmark_v1
parm-bench inspect data/benchmark_v1 --case parm-amara-conference-agenda-positive
```

Run the test suite:

```powershell
python -m unittest discover -s tests
```

Start the local workbench when reviewing retrieval behavior:

```powershell
parm-bench serve-workbench --retrieval-index data\retrieval-indexes\amara-life-v1 --expansion-cache data\expansion-caches\amara-life-v1
```

## Coding Style & Naming Conventions

Use Python 3.10+ and four-space indentation. Keep module names lowercase with underscores. Prefer dataclasses and typed function signatures where the surrounding code already uses them. CLI options should use kebab-case, matching existing `parm-bench` commands such as `--retrieval-index` and `--expansion-cache`.

Keep benchmark identifiers readable and stable. Expected choices should use visible labels from the prompt or observation, not hidden IDs.

Keep PARM benchmark axes separate:

- `--baseline` controls what condition is being evaluated.
- `--output-rag-flow` controls where output-triggered retrieval runs within `naive_output_rag`.
- `--retrieval-mode` controls how retrieved memories are ranked (`dense`, `hybrid`, or `enhanced`).

`all_entity_output_rag` is the exception to the retrieval-mode axis: it is a fixed exact-match entity component baseline over extracted output entities, so it requires `--retrieval-index` and rejects `--retrieval-mode`.

## Testing Guidelines

Add or update unittest coverage for behavioral changes. Name test files `test_*.py` and test methods `test_*`. For CLI changes, add smoke coverage in `tests/test_cli_smoke.py`; for retrieval or scoring changes, update the matching focused test file.

Run `python -m unittest discover -s tests` before committing. If benchmark artifacts change, also run `parm-bench validate data/benchmark_v1`.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, for example `Implement orthogonal retrieval modes` and `Track frozen retrieval benchmark artifacts`. Follow that style and keep the subject specific to the changed behavior.

Pull requests should describe the benchmark or CLI impact, list tests run, and call out any changed tracked artifacts under `data/`. Include screenshots only for workbench UI changes.

## Security & Configuration Tips

Keep secrets in the ignored root `.env`; start from `.env.example`. Set `OPENAI_API_KEY` for model-backed runs. Canonical benchmark replay should use tracked frozen indexes and caches, and should not require a live GBrain checkout unless rebuilding retrieval artifacts.
