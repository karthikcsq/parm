from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import tiktoken

from personamem_v0_specs import SPECS


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "personamem-v2-train-v0"
DATASET_ROOT = ROOT / "data" / "benchmark_personamem_mixed_v0"
CONTEXT_ROOT = DATASET_ROOT / "contexts"
CASES_PATH = DATASET_ROOT / "cases.jsonl"
CONSTRUCTION_PATH = DATASET_ROOT / "construction_records.jsonl"
DATASET_MANIFEST_PATH = DATASET_ROOT / "dataset_manifest.json"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "source_manifest.json"
PILOT_SOURCES_PATH = SOURCE_ROOT / "pilot_sources.jsonl"
RECORDS_PATH = SOURCE_ROOT / "records.jsonl"
DATASET_REVISION = "b7b42b78917157afed063527a1c959e98f6109f2"
CONTEXT_BUILDER_VERSION = "personamem_mixed_context_v1"
CASE_BUILDER_VERSION = "personamem_mixed_triplet_v1"
PROMPT_VERSION = "broad_mixed_context_selection_v1"
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
ENVELOPE_STYLES = (
    "research_notebook",
    "forwarded_thread",
    "meeting_dump",
    "web_clippings",
    "working_draft",
    "mixed_markdown",
)
MIXED_MEMORY_NAMES = (
    "Juniper Hall",
    "Northline",
    "Amber House",
    "Rookery Room",
    "Blue Lantern",
    "Westbridge",
    "Morrow House",
    "Kestrel Studio",
    "Fieldstone",
    "Mercer Room",
    "Eastbank",
    "Aster House",
    "Fable Room",
    "Harbor Nine",
    "Linden Hall",
    "Cobalt House",
    "Briar Room",
    "Alder House",
    "Orchard Room",
    "Willow Annex",
    "Northstar House",
    "Maple Room",
    "Mosaic Hall",
    "Beacon Studio",
    "Summit House",
    "Atlas Room",
    "Rainier House",
    "Crescent Room",
    "Meridian Hall",
    "Foundry Room",
)
NEUTRAL_CONTROL_DETAIL = (
    "The surviving note records standard logistics and ordinary terms, but "
    "does not describe a specialized format, subject, schedule, or personal "
    "fit."
)
NOISE_SENTENCES = (
    "The source export duplicated a few lines and the date column is not reliable.",
    "Someone marked this for follow-up, but the owner field was left blank.",
    "The surrounding comments praise the logistics without identifying a specialized fit.",
    "A copied footer interrupts the paragraph here, followed by an unrelated scheduling note.",
    "The confidence tag came from an older pass and should not be treated as a ranking.",
    "Two reviewers discussed accessibility and cancellation terms in a separate thread.",
    "The text below was pasted from a mobile view, so headings and paragraphs are uneven.",
    "A later edit questions the venue details but does not change the general assessment.",
)
GENERAL_DETAILS = (
    "It has ordinary booking terms and enough recent feedback to compare.",
    "The write-up is positive about execution but offers no unusual personal fit.",
    "Notes mention clear logistics, a standard cancellation window, and broad appeal.",
    "The reviewer calls it credible and convenient while leaving several details unresolved.",
    "It appears suitable for a general audience and carries no special eligibility requirement.",
    "The team liked the pacing and documentation but did not flag a distinctive feature.",
)


def main() -> None:
    if len(MIXED_MEMORY_NAMES) != len(SPECS):
        raise ValueError("mixed memory names must cover every scenario")
    source_manifest = json.loads(
        SOURCE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
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
        mixed_spec = {
            **spec,
            "memory_name": MIXED_MEMORY_NAMES[scenario_index - 1],
        }
        source_row_id = f"train_text:{spec['row_offset']}"
        source = pilot_sources[source_row_id]
        corpus_id = source["corpus_id"]
        envelope_style = ENVELOPE_STYLES[
            (scenario_index - 1) % len(ENVELOPE_STYLES)
        ]
        context, cue_text, control_text = build_context(
            mixed_spec,
            scenario_index=scenario_index,
            envelope_style=envelope_style,
        )
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
        distractors = [
            {
                "source_id": record["source_id"],
                "text": _short_text(record["text"]),
                "perturbations": record["perturbations"],
            }
            for record in records_by_corpus[corpus_id]
            if record["source_id"] != source["gold_source_id"]
        ][:3]
        cases.extend(
            make_case(
                mixed_spec,
                source,
                variant,
                distractors,
                cue_text=cue_text,
                control_text=control_text,
                scenario_index=scenario_index,
                envelope_style=envelope_style,
            )
            for variant in VARIANTS
        )
        control_context = context.replace(cue_text, control_text, 1)
        construction_records.append(
            {
                "base_case_id": (
                    f"parm-personamem-mixed-{spec['slug']}"
                ),
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
                "envelope_style": envelope_style,
                "positive_observation_sha256": _text_sha256(context),
                "control_observation_sha256": _text_sha256(control_context),
                "source_history_sha256": source["history_sha256"],
                "gold_source_id": source["gold_source_id"],
                "gold_source_hash": source["gold_source_hash"],
                "transformation": {
                    "user_query_use": "topic_seed_only",
                    "private_memory_origin": "related_conversation_snippet",
                    "cue_created_separately": True,
                    "gold_choice_created_separately": True,
                    "control_edits": 1,
                    "clean_listing_rows": False,
                    "fixed_target_position": False,
                },
                "acceptance": {
                    "status": "accepted_structurally",
                    "ordinary_prompt": True,
                    "credible_output_only_lead": True,
                    "one_decisive_affordance": True,
                    "matched_choice_detail": True,
                    "cue_control_symmetry": True,
                    "gold_source_provenance_verified": True,
                    "gold_source_semantic_support": (
                        "separate_audit_required"
                    ),
                    "fairness_status": "pending",
                },
                "manual_repairs": [],
            }
        )

    _write_jsonl(CASES_PATH, cases)
    _write_jsonl(CONSTRUCTION_PATH, construction_records)
    manifest = {
        "schema_version": 1,
        "dataset_id": "benchmark_personamem_mixed_v0",
        "validation_profile": "personamem_v2_mixed_v0",
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
            "envelope_styles": len(ENVELOPE_STYLES),
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
            "method": "deterministic_mixed_envelope",
            "envelope_styles": list(ENVELOPE_STYLES),
        },
    }
    DATASET_MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_readme()
    print(f"wrote {len(cases)} mixed-format cases")


def build_context(
    spec: dict[str, Any],
    *,
    scenario_index: int,
    envelope_style: str,
) -> tuple[str, str, str]:
    lead_text = _lead_text(spec, envelope_style)
    cue_text = _cue_text(
        spec["memory_name"],
        spec["positive_detail"],
        envelope_style,
    )
    control_text = _cue_text(
        spec["memory_name"],
        NEUTRAL_CONTROL_DETAIL,
        envelope_style,
    )
    lead_block = 3 + ((scenario_index * 5) % 9)
    cue_block = 18 + ((scenario_index * 11) % 29)
    if cue_block == lead_block:
        cue_block += 7

    blocks = [_opening(envelope_style, scenario_index)]
    block_index = 1
    while len(ENCODING.encode("\n\n".join(blocks))) < 9_000:
        if block_index == lead_block:
            block = lead_text
        elif block_index == cue_block:
            block = cue_text
        else:
            block = _noise_block(
                spec,
                block_index=block_index,
                scenario_index=scenario_index,
                envelope_style=envelope_style,
            )
        blocks.append(block)
        block_index += 1
    context = "\n\n".join(blocks).strip() + "\n"
    token_count = len(ENCODING.encode(context))
    if not 8_000 <= token_count <= 12_000:
        raise ValueError(f"{spec['slug']}: context has {token_count} tokens")
    for text, name in (
        (lead_text, "lead"),
        (cue_text, "cue"),
    ):
        if context.count(text) != 1:
            raise ValueError(
                f"{spec['slug']}: {name} must occur exactly once"
            )
    if context.count(spec["output_name"]) != 1:
        raise ValueError(f"{spec['slug']}: output name must be unique")
    if context.count(spec["memory_name"]) != 1:
        raise ValueError(f"{spec['slug']}: memory name must be unique")
    return context, cue_text, control_text


def _opening(envelope_style: str, scenario_index: int) -> str:
    openings = {
        "research_notebook": (
            "# working research notebook\n\n"
            "> export assembled from browser clips, voice notes, and an older "
            "comparison draft. Repeated confidence tags are not rankings."
        ),
        "forwarded_thread": (
            "Subject: Fwd: loose options / please read before replying\n\n"
            "This thread has collapsed quotes, missing attachments, and replies "
            "that are no longer in chronological order."
        ),
        "meeting_dump": (
            "# meeting dump (auto-transcript + edits)\n\n"
            "Several speakers were not identified. TODO markers survived from "
            "the live call and some lines belong to the previous agenda item."
        ),
        "web_clippings": (
            "# saved web fragments\n\n"
            "Reader mode failed on several pages. Navigation text, newsletter "
            "footers, and snippets from unrelated tabs remain in the export."
        ),
        "working_draft": (
            "# draft that nobody cleaned up\n\n"
            "Comments are inline; numbers came from different review passes; "
            "do not assume the newest paragraph is the most important."
        ),
        "mixed_markdown": (
            "# combined notes.md\n\n"
            "Sources: pasted table, chat excerpts, scratch notes, and partial "
            "JSON. Formatting is inconsistent by design."
        ),
    }
    return openings[envelope_style] + f"\n\nbatch_ref={scenario_index:02d}"


def _lead_text(spec: dict[str, Any], envelope_style: str) -> str:
    name = spec["output_name"]
    variants = {
        "research_notebook": (
            f"General recommendation from the first pass: **{name}** still "
            "has the best broad evidence. The combined review put it at 9.8 "
            "and described the execution as consistently polished."
        ),
        "forwarded_thread": (
            f"> Maya, earlier: “If we need one safe default, I would use "
            f"{name}. It has the strongest general feedback.”\n\n"
            "No one challenged that 9.8 assessment in the later replies."
        ),
        "meeting_dump": (
            f"[09:42] unresolved ranking discussion. Consensus was {name} for "
            "the default: broadest appeal, strongest delivery record, 9.8 in "
            "the combined sheet. action?? keep unless a personal constraint "
            "changes the call."
        ),
        "web_clippings": (
            f"…cached review text… {name} received the strongest overall "
            "assessment (9.8) and is presented as the broadly suitable choice. "
            "[cookie settings] [next article] The execution notes were positive."
        ),
        "working_draft": (
            f"The current draft recommends {name}. COMMENT: this is the general "
            "answer, not a claim of specialized fit. Score carried forward: "
            "9.8; reviewers repeatedly called it polished and dependable."
        ),
        "mixed_markdown": (
            f"| carried-forward decision | {name} | 9.8 | strongest general "
            "case |\n\nsomeone in chat: this is still the default unless the "
            "later material gives us a reason to personalize."
        ),
    }
    return variants[envelope_style]


def _cue_text(name: str, detail: str, envelope_style: str) -> str:
    variants = {
        "research_notebook": (
            f"Buried in the reviewer notebook: {name} was scored 8.8. {detail} "
            "The author called it a narrower choice and did not compare it "
            "directly with the general leader."
        ),
        "forwarded_thread": (
            f"> clipped from an attachment nobody summarized: {name} — 8.8. "
            f"{detail}\n\nThe next reply only says “specialized, probably not "
            "the default,” then changes the subject."
        ),
        "meeting_dump": (
            f"[speaker ? / audio recovered] {name}, around 8.8 in the old "
            f"sheet. {detail} [cross-talk] narrower than the default; no vote "
            "was taken after this came up."
        ),
        "web_clippings": (
            f"[save failed · text view] {name}. Community rating snapshot: "
            f"8.8. {detail} Related links and a newsletter signup split this "
            "paragraph from the main comparison."
        ),
        "working_draft": (
            f"INSERT LATER? {name} (8.8). {detail} Margin note: specialized "
            "option; leave below the general recommendation unless context "
            "makes this detail matter."
        ),
        "mixed_markdown": (
            f"chat_export[unknown time]: “I also found {name}; old score 8.8. "
            f"{detail}”\n\n<!-- pasted footer --> narrower case / not ranked "
            "first in the source document."
        ),
    }
    return variants[envelope_style]


def _noise_block(
    spec: dict[str, Any],
    *,
    block_index: int,
    scenario_index: int,
    envelope_style: str,
) -> str:
    first = _noise_name(spec, block_index, scenario_index)
    second = _noise_name(spec, block_index + 97, scenario_index)
    detail = GENERAL_DETAILS[
        (block_index * 3 + scenario_index) % len(GENERAL_DETAILS)
    ]
    noise = NOISE_SENTENCES[
        (block_index * 5 + scenario_index) % len(NOISE_SENTENCES)
    ]
    wrappers = {
        "research_notebook": (
            f"## fragment {block_index}\n{first} appears in a copied paragraph "
            f"with an obsolete reference tag. {detail} A margin note mentions "
            f"{second}, but only to question the date. {noise}"
        ),
        "forwarded_thread": (
            f"On an unknown date, Review Desk wrote:\n> We looked again at "
            f"{first}; the old sheet marks it as archived. {detail}\n\n"
            f"RE: {second} / attachment missing. {noise}"
        ),
        "meeting_dump": (
            f"[{8 + block_index % 10:02d}:{(block_index * 7) % 60:02d}] "
            f"{first} came up, then somebody corrected the timestamp. "
            f"{detail} [inaudible] {second}? probably a separate "
            f"topic. {noise}"
        ),
        "web_clippings": (
            f"Skip to content · {first} · cached archival copy. {detail} "
            f"Recommended link: {second}. Privacy choices / manage consent. "
            f"{noise}"
        ),
        "working_draft": (
            f"Paragraph {block_index} [move?]: {first} is a carried-over "
            f"reference. {detail} ~~Compare with {second}.~~ COMMENT unresolved. "
            f"{noise}"
        ),
        "mixed_markdown": (
            f"- [ ] verify {first} / not a current option\n\n"
            f"{{\"cross_ref\": \"{second}\", \"owner\": null}}\n\n"
            f"{detail} {noise}"
        ),
    }
    return wrappers[envelope_style]


def _noise_name(
    spec: dict[str, Any], block_index: int, scenario_index: int
) -> str:
    del spec
    label = (
        "archive fragment"
        if (block_index + scenario_index) % 2
        else "source attachment"
    )
    return f"{label} {block_index:03d}"


def make_case(
    spec: dict[str, Any],
    source: dict[str, Any],
    variant: str,
    distractors: list[dict[str, Any]],
    *,
    cue_text: str,
    control_text: str,
    scenario_index: int,
    envelope_style: str,
) -> dict[str, Any]:
    cue_present = variant in {"positive", "memory-included"}
    prompt = (
        "After reading the collected material, choose exactly one option you would "
        "move forward with. Reply using only its name exactly as written."
    )
    if variant == "memory-included":
        prompt = (
            f"Known personal memory:\n{spec['memory_summary']}\n\n"
            "Choose the visible option that best fits this memory, even when "
            "another option has the stronger general rating.\n\n"
            f"{prompt}"
        )
    output_choice = spec["output_name"]
    memory_choice = spec["memory_name"] if cue_present else output_choice
    base_case_id = f"parm-personamem-mixed-{spec['slug']}"
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
                else [{"old": cue_text, "new": control_text}]
            ),
        },
        "cue": {
            "present": cue_present,
            "text": cue_text,
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
            "evaluation_split": "development_mixed",
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
            "envelope_style": envelope_style,
            "manual_repairs": [],
        },
    }


def _write_readme() -> None:
    (DATASET_ROOT / "README.md").write_text(
        """# PersonaMem mixed-context development slice

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

This slice inherits the V0 source rows. A separate semantic audit found that
13/30 labeled memory claims are not supported by the raw conversation the
retriever is allowed to index. Treat mixed V0 as a retrieval-development slice,
report the 17 supported or inferable scenarios separately, and replace the
unsupported rows before promoting a sealed benchmark.

Rebuild and validate:

```powershell
$env:PYTHONPATH = 'src'
& 'C:\\Users\\karth\\anaconda3\\python.exe' `
  scripts\\build_personamem_mixed_v0_benchmark.py
& 'C:\\Users\\karth\\anaconda3\\python.exe' -m parm_bench.cli validate `
  data\\benchmark_personamem_mixed_v0
```
""",
        encoding="utf-8",
    )


def _short_text(text: str) -> str:
    return " ".join(text.split())[:240]


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
