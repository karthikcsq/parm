from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import tiktoken

from personamem_v0_specs import SPECS


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "personamem-v2-train-v0"
DATASET_ROOT = ROOT / "data" / "benchmark_personamem_v0"
CONTEXT_ROOT = DATASET_ROOT / "contexts"
CASES_PATH = DATASET_ROOT / "cases.jsonl"
CONSTRUCTION_PATH = DATASET_ROOT / "construction_records.jsonl"
DATASET_MANIFEST_PATH = DATASET_ROOT / "dataset_manifest.json"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "source_manifest.json"
PILOT_SOURCES_PATH = SOURCE_ROOT / "pilot_sources.jsonl"
RECORDS_PATH = SOURCE_ROOT / "records.jsonl"
DATASET_REVISION = "b7b42b78917157afed063527a1c959e98f6109f2"
CONTEXT_BUILDER_VERSION = "personamem_context_v1"
CASE_BUILDER_VERSION = "personamem_triplet_v1"
PROMPT_VERSION = "broad_featured_selection_v1"
VARIANTS = ("positive", "cue-ablated", "memory-included")
ENCODING = tiktoken.get_encoding("cl100k_base")

ADJECTIVES = (
    "Measured",
    "Harbor",
    "Cedar",
    "Open",
    "North",
    "Practical",
    "Field",
    "Common",
    "Clear",
    "Steady",
    "Local",
    "Bright",
    "Corner",
    "Independent",
    "Civic",
    "Workshop",
    "Signal",
    "Archive",
    "Riverside",
    "Studio",
)
NOUNS = (
    "Review",
    "Exchange",
    "Guide",
    "Program",
    "Collection",
    "Session",
    "Route",
    "Series",
    "Package",
    "Circle",
    "Project",
    "Forum",
    "Catalog",
    "Briefing",
    "Lab",
    "Weekend",
    "Plan",
    "Showcase",
    "Visit",
    "Course",
)
FORMATS = (
    "a ninety-minute hosted program",
    "a self-paced half-day option",
    "a small-group reservation",
    "an editor-curated weekend slot",
    "a guided two-hour session",
    "a flexible drop-in package",
    "a limited-capacity evening event",
    "an independent afternoon route",
)
DETAILS = (
    "Reviewers praise the clear logistics, consistent pacing, and useful supporting notes.",
    "The listing includes an accessible schedule, ordinary booking terms, and a concise preparation guide.",
    "Editors describe the execution as polished, practical, and easy to evaluate against the surrounding choices.",
    "The package has transparent pricing, a standard cancellation window, and several recent participant reviews.",
    "The description gives specific timing, venue, and facilitation details without claiming a specialized personal fit.",
    "The option is well organized and credible, though its main strengths are general quality and convenience.",
    "The review highlights dependable delivery, clear instructions, and a balanced experience for a broad audience.",
    "The listing is detailed enough to compare fairly, with no hidden eligibility or membership requirement.",
)


def main() -> None:
    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    pilot_sources = {
        row["source_row_id"]: row for row in _read_jsonl(PILOT_SOURCES_PATH)
    }
    records_by_corpus: dict[str, list[dict[str, Any]]] = {}
    for record in _read_jsonl(RECORDS_PATH):
        records_by_corpus.setdefault(record["corpus_id"], []).append(record)
    CONTEXT_ROOT.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    construction_records: list[dict[str, Any]] = []
    corpus_entries: list[dict[str, str]] = []
    seen_corpora: set[str] = set()
    for scenario_index, spec in enumerate(SPECS, start=1):
        source_row_id = f"train_text:{spec['row_offset']}"
        source = pilot_sources[source_row_id]
        corpus_id = source["corpus_id"]
        context = build_context(spec)
        context_path = CONTEXT_ROOT / f"{spec['slug']}.md"
        context_path.write_text(context, encoding="utf-8")
        if corpus_id not in seen_corpora:
            seen_corpora.add(corpus_id)
            corpus_entries.append(
                {
                    "corpus_id": corpus_id,
                    "source_root": "../personamem-v2-train-v0/source",
                }
            )
        corpus_records = records_by_corpus[corpus_id]
        distractors = [
            {
                "source_id": record["source_id"],
                "text": _short_text(record["text"]),
                "perturbations": record["perturbations"],
            }
            for record in corpus_records
            if record["source_id"] != source["gold_source_id"]
        ][:3]
        scenario_cases = [
            make_case(
                spec,
                source,
                variant,
                distractors,
                scenario_index=scenario_index,
            )
            for variant in VARIANTS
        ]
        cases.extend(scenario_cases)
        control_text = context.replace(spec["cue"], spec["replacement"], 1)
        construction_records.append(
            {
                "base_case_id": f"parm-personamem-{spec['slug']}",
                "source_row_id": source_row_id,
                "persona_id": source["persona_id"],
                "corpus_id": corpus_id,
                "dataset_revision": DATASET_REVISION,
                "context_builder_version": CONTEXT_BUILDER_VERSION,
                "case_builder_version": CASE_BUILDER_VERSION,
                "prompt_version": PROMPT_VERSION,
                "generation_model": None,
                "generation_seed": scenario_index,
                "raw_model_output_path": None,
                "context_path": f"contexts/{spec['slug']}.md",
                "positive_observation_sha256": _text_sha256(context),
                "control_observation_sha256": _text_sha256(control_text),
                "source_history_sha256": source["history_sha256"],
                "gold_source_id": source["gold_source_id"],
                "gold_source_hash": source["gold_source_hash"],
                "transformation": {
                    "user_query_use": "topic_seed_only",
                    "private_memory_origin": "related_conversation_snippet",
                    "cue_created_separately": True,
                    "gold_choice_created_separately": True,
                    "control_edits": 1,
                },
                "acceptance": {
                    "status": "accepted_structurally",
                    "ordinary_prompt": True,
                    "credible_output_only_lead": True,
                    "one_decisive_affordance": True,
                    "matched_choice_detail": True,
                    "cue_control_symmetry": True,
                    "gold_source_verified": True,
                    "fairness_status": "pending",
                },
                "manual_repairs": [],
            }
        )

    _write_jsonl(CASES_PATH, cases)
    _write_jsonl(CONSTRUCTION_PATH, construction_records)
    dataset_manifest = {
        "schema_version": 1,
        "dataset_id": "benchmark_personamem_v0",
        "validation_profile": "personamem_v2_v0",
        "source_manifest": {
            "path": "../personamem-v2-train-v0/source_manifest.json",
            "sha256": _sha256(SOURCE_MANIFEST_PATH),
        },
        "source_records": {
            "path": "../personamem-v2-train-v0/records.jsonl",
            "sha256": source_manifest["artifacts"]["records"]["sha256"],
        },
        "pilot_sources": {
            "path": "../personamem-v2-train-v0/pilot_sources.jsonl",
            "sha256": source_manifest["artifacts"]["pilot_sources"]["sha256"],
        },
        "construction_records": {
            "path": "construction_records.jsonl",
            "sha256": _sha256(CONSTRUCTION_PATH),
        },
        "corpora": sorted(corpus_entries, key=lambda item: item["corpus_id"]),
        "counts": {
            "scenarios": len(SPECS),
            "cases": len(cases),
            "corpora": len(corpus_entries),
        },
        "split": {
            "source": "train_text",
            "evaluation": "development",
            "persona_disjoint": True,
        },
        "generation": {
            "context_builder_version": CONTEXT_BUILDER_VERSION,
            "case_builder_version": CASE_BUILDER_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": None,
            "method": "deterministic_template",
        },
    }
    DATASET_MANIFEST_PATH.write_text(
        json.dumps(
            dataset_manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} cases across {len(corpus_entries)} corpora")


def build_context(spec: dict[str, Any]) -> str:
    lines = [
        "# Featured Recommendation Feed",
        "",
        "Generated deterministic fixture. Every listing has a unique visible label.",
        "",
    ]
    index = 1
    while len(ENCODING.encode("\n".join(lines))) < 9_000:
        if index == 4:
            entry = spec["lead"]
        elif index == 147:
            entry = spec["cue"]
        else:
            adjective = ADJECTIVES[(index * 7) % len(ADJECTIVES)]
            noun = NOUNS[(index * 11) % len(NOUNS)]
            name = f"{adjective} {noun} {index:03d}"
            score = 7.1 + ((index * 13) % 25) / 10
            entry = (
                f"Listing {spec['code']}-{index:03d} - {name}. "
                f"Editor score {score:.1f}/10. "
                f"This option is {FORMATS[(index * 5) % len(FORMATS)]}. "
                f"{DETAILS[(index * 3) % len(DETAILS)]}"
            )
        lines.extend((entry, ""))
        index += 1
    text = "\n".join(lines).strip() + "\n"
    token_count = len(ENCODING.encode(text))
    if not 8_000 <= token_count <= 12_000:
        raise ValueError(f"{spec['slug']}: context has {token_count} tokens")
    if text.count(spec["cue"]) != 1:
        raise ValueError(f"{spec['slug']}: cue must occur exactly once")
    return text


def make_case(
    spec: dict[str, Any],
    source: dict[str, Any],
    variant: str,
    distractors: list[dict[str, Any]],
    *,
    scenario_index: int,
) -> dict[str, Any]:
    cue_present = variant in {"positive", "memory-included"}
    prompt = spec["prompt"]
    if variant == "memory-included":
        prompt = (
            f"Known personal memory:\n{spec['memory_summary']}\n\n{prompt}"
        )
    output_choice = spec["output_choice"]
    memory_choice = spec["memory_choice"] if cue_present else output_choice
    base_case_id = f"parm-personamem-{spec['slug']}"
    return {
        "case_id": f"{base_case_id}-{variant}",
        "base_case_id": base_case_id,
        "corpus_id": source["corpus_id"],
        "variant": variant,
        "prompt": prompt,
        "observation": {
            "kind": spec["observation_kind"],
            "content_path": f"contexts/{spec['slug']}.md",
            "replacements": (
                []
                if cue_present
                else [{"old": spec["cue"], "new": spec["replacement"]}]
            ),
        },
        "cue": {
            "present": cue_present,
            "text": spec["cue"],
            "type": spec["cue_type"],
            "query": spec["cue_query"],
        },
        "memory": {
            "corpus_id": source["corpus_id"],
            "text": spec["memory_summary"],
            "gold_source_ids": [source["gold_source_id"]],
            "sources": [
                {
                    "source_id": source["gold_source_id"],
                    "path": source["history_path"],
                    "sha256": source["history_sha256"],
                    "perturbations": [],
                    "turn_source_hash": source["gold_source_hash"],
                    "update_status": "current",
                    "sensitive_info": False,
                }
            ],
            "sensitive_terms": [],
        },
        "decisions": {
            "answer_type": "natural_language_choice",
            "output_only": {"choice": output_choice},
            "memory_conditioned": {"choice": memory_choice},
        },
        "distractors": {"sources": distractors},
        "provenance": {
            "approved": True,
            "evaluation_split": "development",
            "source_dataset": "bowen-upenn/PersonaMem-v2",
            "source_revision": DATASET_REVISION,
            "source_split": "train_text",
            "source_row_id": f"train_text:{spec['row_offset']}",
            "persona_id": source["persona_id"],
            "construction_record": (
                f"construction_records.jsonl#{base_case_id}"
            ),
            "generation_seed": scenario_index,
            "generation_model": None,
            "context_builder_version": CONTEXT_BUILDER_VERSION,
            "case_builder_version": CASE_BUILDER_VERSION,
            "prompt_version": PROMPT_VERSION,
            "manual_repairs": [],
        },
    }


def _short_text(text: str) -> str:
    normalized = " ".join(text.split())
    return normalized[:240]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
