from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "benchmark-results" / "personamem-v0-first-pass"
DATASET_MANIFEST = (
    ROOT / "data" / "benchmark_personamem_v0" / "dataset_manifest.json"
)
DATASET_CASES = ROOT / "data" / "benchmark_personamem_v0" / "cases.jsonl"
INDEX_MANIFEST = (
    ROOT
    / "data"
    / "retrieval-indexes"
    / "personamem-v2-train-v0"
    / "manifest.json"
)
JSON_OUTPUT = RESULTS / "comparison.json"
MARKDOWN_OUTPUT = ROOT / "docs" / "results" / "personamem-v0-first-pass.md"

CONDITIONS = (
    ("no-memory-gpt-5-mini", "No memory"),
    ("input-rag-enhanced-gpt-5-mini", "Enhanced input RAG"),
    (
        "naive-output-tool-then-model-hybrid-gpt-5-mini",
        "Hybrid output RAG",
    ),
    ("all-entity-output-rag-gpt-5-mini", "Exact-entity output RAG"),
    (
        "prompted-memory-tool-hybrid-gpt-5-mini",
        "Prompted memory tool",
    ),
    ("parm-gpt-5-mini", "PARM"),
)


def main() -> None:
    dataset_manifest = _read_json(DATASET_MANIFEST)
    index_manifest = _read_json(INDEX_MANIFEST)
    corpus_scenarios = {
        row["corpus_id"]: row["base_case_id"]
        for row in _read_jsonl(DATASET_CASES)
        if row["variant"] == "positive"
    }
    condition_rows: list[dict[str, Any]] = []
    corpus_rows: dict[str, dict[str, Any]] = {}
    resolved_models: set[str] = set()
    for name, label in CONDITIONS:
        metrics_path = RESULTS / f"{name}.metrics.json"
        predictions_path = RESULTS / f"{name}.jsonl"
        config_path = RESULTS / f"{name}.config.json"
        metrics = _read_json(metrics_path)
        predictions = _read_jsonl(predictions_path)
        config = _read_json(config_path)
        resolved_models.update(
            str(row["resolved_model"])
            for row in predictions
            if row.get("resolved_model")
        )
        row = {
            "condition": name,
            "label": label,
            "positive_accuracy": _variant_rate(
                metrics["rows"],
                "positive",
                "correct_memory_conditioned_decision",
            ),
            "control_accuracy": _control_accuracy(metrics["rows"]),
            "ceiling_accuracy": metrics["ceiling_accuracy"],
            "memory_admission_precision": metrics[
                "memory_admission_precision"
            ],
            "memory_admission_recall": metrics["memory_admission_recall"],
            "spurious_memory_admission_rate": metrics[
                "spurious_memory_admission_rate"
            ],
            "poison_admission_rate": metrics["poison_admission_rate"],
            "stale_or_contradictory_admission_rate": metrics[
                "stale_or_contradictory_admission_rate"
            ],
            "privacy_overexposure_rate": metrics[
                "privacy_overexposure_rate"
            ],
            "artifacts": {
                "predictions": _artifact(predictions_path),
                "config": _artifact(config_path),
                "metrics": _artifact(metrics_path),
            },
            "configuration": {
                "baseline": config["baseline"],
                "output_rag_flow": config.get("output_rag_flow"),
                "retrieval_mode": config.get("retrieval_mode"),
                "retrieval_limit": config.get("retrieval_limit"),
                "requested_model": config["requested_model"],
            },
        }
        condition_rows.append(row)
        for corpus_id, corpus_metrics in metrics["by_corpus"].items():
            corpus = corpus_rows.setdefault(
                corpus_id,
                {
                    "corpus_id": corpus_id,
                    "base_case_id": corpus_scenarios[corpus_id],
                    "conditions": {},
                },
            )
            corpus["conditions"][name] = {
                "positive_accuracy": _variant_rate(
                    corpus_metrics["rows"],
                    "positive",
                    "correct_memory_conditioned_decision",
                ),
                "control_accuracy": _control_accuracy(
                    corpus_metrics["rows"]
                ),
                "ceiling_accuracy": corpus_metrics["ceiling_accuracy"],
                "memory_admission_precision": corpus_metrics[
                    "memory_admission_precision"
                ],
                "memory_admission_recall": corpus_metrics[
                    "memory_admission_recall"
                ],
                "privacy_overexposure_rate": corpus_metrics[
                    "privacy_overexposure_rate"
                ],
            }

    comparison = {
        "experiment": "personamem-v0-first-pass",
        "dataset": {
            "path": str(DATASET_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(DATASET_MANIFEST),
            "scenarios": dataset_manifest["counts"]["scenarios"],
            "cases": dataset_manifest["counts"]["cases"],
            "corpora": dataset_manifest["counts"]["corpora"],
            "source_revision": index_manifest["dataset_revision"],
        },
        "retrieval_index": {
            "path": str(INDEX_MANIFEST.parent.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "manifest_sha256": _sha256(INDEX_MANIFEST),
            "content_hash": index_manifest["content_hash"],
            "schema_version": index_manifest["schema_version"],
            "counts": index_manifest["counts"],
        },
        "resolved_models": sorted(resolved_models),
        "conditions": condition_rows,
        "by_corpus": [
            corpus_rows[corpus_id] for corpus_id in sorted(corpus_rows)
        ],
    }
    JSON_OUTPUT.write_text(
        json.dumps(comparison, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_OUTPUT.write_text(
        _render_markdown(comparison), encoding="utf-8"
    )
    print(
        f"wrote {len(condition_rows)} conditions and "
        f"{len(corpus_rows)} corpus rows"
    )


def _variant_rate(
    rows: list[dict[str, Any]], variant: str, field: str
) -> float:
    selected = [row for row in rows if row["variant"] == variant]
    return sum(bool(row[field]) for row in selected) / max(len(selected), 1)


def _control_accuracy(rows: list[dict[str, Any]]) -> float:
    controls = [row for row in rows if row["variant"] == "cue-ablated"]
    return sum(not row["false_intervention"] for row in controls) / max(
        len(controls), 1
    )


def _artifact(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256(path),
    }


def _render_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# PersonaMem-v2 first-pass development results",
        "",
        "This is the frozen first pass over 30 persona-disjoint scenarios and "
        "90 cases. P is positive accuracy, C is cue-ablated control accuracy, "
        "and Z is the memory-included ceiling.",
        "",
        "| Condition | P | C | Z | Admission precision | Admission recall | "
        "Spurious | Poison | Stale or contradiction | Privacy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: |",
    ]
    for row in comparison["conditions"]:
        values = [
            row["positive_accuracy"],
            row["control_accuracy"],
            row["ceiling_accuracy"],
            row["memory_admission_precision"],
            row["memory_admission_recall"],
            row["spurious_memory_admission_rate"],
            row["poison_admission_rate"],
            row["stale_or_contradictory_admission_rate"],
            row["privacy_overexposure_rate"],
        ]
        lines.append(
            f"| {row['label']} | "
            + " | ".join(_percent(value) for value in values)
            + " |"
        )
    by_name = {
        row["condition"]: row for row in comparison["conditions"]
    }
    parm = by_name["parm-gpt-5-mini"]
    input_rag = by_name["input-rag-enhanced-gpt-5-mini"]
    entity = by_name["all-entity-output-rag-gpt-5-mini"]
    lines.extend(
        [
            "",
            "## First-pass readout",
            "",
            f"- PARM and enhanced input RAG tie for the highest positive "
            f"accuracy at {_percent(parm['positive_accuracy'])}. PARM keeps "
            f"{_percent(parm['control_accuracy'])} of controls stable versus "
            f"{_percent(input_rag['control_accuracy'])} for input RAG.",
            f"- PARM admits little evidence, but its "
            f"{_percent(parm['memory_admission_precision'])} precision is "
            f"substantially cleaner than every fixed top-k condition. Its "
            f"{_percent(parm['memory_admission_recall'])} recall remains too "
            f"low for strong coverage.",
            f"- Exact-entity output RAG reaches "
            f"{_percent(entity['memory_admission_recall'])} recall, but "
            f"{_percent(entity['spurious_memory_admission_rate'])} of its "
            f"admissions are non-gold and it keeps only "
            f"{_percent(entity['control_accuracy'])} of controls stable.",
            "- Every condition reaches the 100.0% explicit-memory ceiling. "
            "Together with the no-memory 0/30 positive, 30/30 control, and "
            "30/30 ceiling gate, this makes retrieval and admission the main "
            "first-pass bottleneck rather than basic task usability.",
            "",
        ]
    )
    lines.extend(
        [
            "## Interpretation boundaries",
            "",
            "- This development set uses ordinary, current, self-attributed "
            "preferences. Poison, stale-source, and privacy rates are "
            "regression checks here, not a safety stress-test result.",
            "- The 9k-token recommendation feeds are deterministic fixtures "
            "built for exact cue/control symmetry. They do not establish "
            "natural end-to-end validity.",
            "- No LLM judge was used for this first pass because all 30 "
            "triplets passed the no-memory gate without manual repair. If "
            "later batches require accumulating repairs or topic-specific "
            "templates, construction should move to a versioned judge rubric "
            "calibrated against sampled human review.",
            "- The fixtures and fairness artifacts were committed before "
            "retrieval failures were inspected. The frozen Amara benchmark "
            "was not modified.",
            "",
            "## Per-persona corpus results",
            "",
            "Each cell is P/C/Z/R, where R is gold-memory admission recall.",
            "",
            "| Corpus | Scenario | "
            + " | ".join(row["label"] for row in comparison["conditions"])
            + " |",
            "| --- | --- | "
            + " | ".join("---:" for _ in comparison["conditions"])
            + " |",
        ]
    )
    for corpus in comparison["by_corpus"]:
        cells = []
        for condition in comparison["conditions"]:
            result = corpus["conditions"][condition["condition"]]
            cells.append(
                "/".join(
                    _integer_rate(result[key])
                    for key in (
                        "positive_accuracy",
                        "control_accuracy",
                        "ceiling_accuracy",
                        "memory_admission_recall",
                    )
                )
            )
        scenario = corpus["base_case_id"].removeprefix("parm-personamem-")
        lines.append(
            f"| `{corpus['corpus_id']}` | {scenario} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "The JSON comparison beside the predictions contains the full "
            "per-corpus metric objects, exact configuration fields, and "
            "content hashes for every result artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _integer_rate(value: float) -> str:
    return str(round(value))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
