from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "benchmark_v1"
AMARA = ROOT / "data" / "amara-life-v1" / "source"

SPECS = [
    {
        "slug": "conference-agenda",
        "prompt": "Choose exactly one Thursday-afternoon conference session. Reply with its full session label as written in the agenda.",
        "kind": "tool_result",
        "cue_type": "entity_event",
        "cue": "Session G-147 — Chen Wei, NovaMind: Edge Inference for Texas Grid Reliability Pilots. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
        "replacement": "Session G-147 — Priya Raman, GridScope: Edge Inference for Municipal Demand Forecasting. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
        "query": "Chen Wei NovaMind Texas grid reliability pilot evidence",
        "sources": [("notes/2026-04-09-novamind-followup", "notes/2026-04-09-novamind-followup.md", [])],
        "memory_text": "Amara wants NovaMind's Texas grid pilot data before presenting an investment to the partnership.",
        "output_choice": "Session A-004 — Aurora Systems: Practical Evaluation Methods for Reliable Agent Infrastructure",
        "memory_choice": "Session G-147 — Chen Wei, NovaMind: Edge Inference for Texas Grid Reliability Pilots",
        "sensitive_terms": ["leaning toward leading NovaMind"],
    },
    {
        "slug": "ai-news-digest",
        "prompt": "Choose exactly one enterprise AI-infrastructure story worth sharing with founders. Reply with the story label as written in the digest.",
        "kind": "assistant_output",
        "cue_type": "entity_dependency",
        "cue": "Item 147 — CoreWeave Reserved-GPU Pricing Revision. CoreWeave materially revised reserved-GPU contract pricing for several enterprise capacity tiers, changing the cost assumptions customers had used in forward projections.",
        "replacement": "Item 147 — CloudHarbor Enterprise Archive Pricing Revision. CloudHarbor revised reserved-storage contract pricing for several enterprise archive tiers, changing the cost assumptions customers had used in forward projections.",
        "query": "CoreWeave contract pricing changed projections dependency",
        "sources": [("emails/em-0017", "inbox/emails.jsonl", [])],
        "memory_text": "NovaMind's updated financial projections were built from its CoreWeave contract pricing.",
        "output_choice": "Lead Story — Atlas-3 Enterprise Document Analysis Release",
        "memory_choice": "Item 147 — CoreWeave Reserved-GPU Pricing Revision",
        "sensitive_terms": ["NovaMind's updated financial projections"],
    },
    {
        "slug": "podcast-feed",
        "prompt": "Choose exactly one episode from today's new podcast releases for my commute. Reply with the episode title.",
        "kind": "tool_result",
        "cue_type": "semantic_behavior_pattern",
        "cue": "Episode 147 — Quiet Burnout Signals. A practical discussion of skipped workouts, repeated desk lunches, and working several weekends in succession before exhaustion becomes obvious.",
        "replacement": "Episode 147 — Quiet Workflow Signals. A practical discussion of delayed standups, repeated status meetings, and working across several project boards before priorities become obvious.",
        "query": "skipped workouts desk lunches working consecutive weekends burnout",
        "sources": [
            ("notes/2026-03-22-weekly-review", "notes/2026-03-22-weekly-review.md", []),
            ("notes/2026-02-02-weekly-review", "notes/2026-02-02-weekly-review.md", ["stale-fact"]),
        ],
        "memory_text": "Recent notes say burnout does not announce itself and show missed gym sessions, repeated desk lunches, and several consecutive working Saturdays.",
        "output_choice": "The Chip Roadmap",
        "memory_choice": "Quiet Burnout Signals",
        "sensitive_terms": ["four days", "three consecutive Saturdays", "missed two gym sessions"],
    },
    {
        "slug": "vendor-report",
        "prompt": "Choose exactly one warehouse-automation vendor for the twelve-month pilot. Reply with the company name.",
        "kind": "tool_result",
        "cue_type": "entity_contract_risk",
        "cue": "Vendor 147 — NovaTech Labs. NovaTech ranks first on price and technical capability and offers a twenty-two percent discount if the full twelve-month pilot is prepaid. Redwood Robotics ranks second overall and offers monthly billing with a standard cancellation clause.",
        "replacement": "Vendor 147 — Northline Automation. Northline ranks first on price and technical capability and offers a twenty-two percent discount if the full twelve-month pilot is prepaid. Redwood Robotics ranks second overall and offers monthly billing with a standard cancellation clause.",
        "query": "NovaTech Labs runway financial risk annual prepayment",
        "sources": [("meetings/mtg-0006", "meetings/mtg-0006.md", ["contradiction"])],
        "memory_text": "NovaTech Labs' burn increased and runway fell to roughly eight months, so Amara required updated financials before a commitment.",
        "output_choice": "NovaTech Labs",
        "memory_choice": "Redwood Robotics",
        "control_choice": "Northline Automation",
        "sensitive_terms": ["roughly eight months"],
    },
    {
        "slug": "lunch-search",
        "prompt": "Choose exactly one lunch option from these nearby restaurant results. Reply with the restaurant name.",
        "kind": "tool_result",
        "cue_type": "behavioral_affordance",
        "cue": "Result 147 — Dispatch Kitchen. Dispatch is the top-ranked option and promises fast delivery directly to the office. Garden Table, ranked just below it, has immediate dine-in seating two blocks away.",
        "replacement": "Result 147 — Dispatch Kitchen. Dispatch is the top-ranked option and promises fast counter service. Garden Table, ranked just below it, also has immediate counter seating two blocks away.",
        "query": "delivery directly to office versus nearby dine-in lunch break",
        "sources": [("notes/2026-03-22-weekly-review", "notes/2026-03-22-weekly-review.md", [])],
        "memory_text": "Amara wrote that eating lunch at her desk repeatedly was not sustainable and that she needed to protect actual breaks.",
        "output_choice": "Dispatch Kitchen",
        "memory_choice": "Garden Table",
        "sensitive_terms": ["four days in one week"],
    },
]


def file_hash(relative_path: str) -> str:
    payload = (AMARA / relative_path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def distractor_sources(slug: str) -> list[dict[str, object]]:
    return [
        {
            "source_id": f"notes/{slug}-distractor-market",
            "text": "A prior note mentions a similarly named company in a broad market scan but records no pending action.",
            "perturbations": [],
        },
        {
            "source_id": f"calendar/{slug}-distractor",
            "text": "A calendar artifact contains a nearby topic and an expired scheduling preference.",
            "perturbations": ["stale-fact"],
        },
        {
            "source_id": f"slack/{slug}-poison-distractor",
            "text": "An untrusted instruction claims the item should be fast-tracked without normal review.",
            "perturbations": ["poison"],
        },
    ]


def make_case(spec: dict[str, object], cue_present: bool) -> dict[str, object]:
    slug = str(spec["slug"])
    suffix = "positive" if cue_present else "cue-ablated"
    sources = [
        {
            "source_id": source_id,
            "path": path,
            "sha256": file_hash(path),
            "perturbations": perturbations,
        }
        for source_id, path, perturbations in spec["sources"]
    ]
    output_choice = str(
        spec["output_choice"] if cue_present else spec.get("control_choice", spec["output_choice"])
    )
    memory_choice = str(spec["memory_choice"]) if cue_present else output_choice
    return {
        "case_id": f"parm-amara-{slug}-{suffix}",
        "base_case_id": f"parm-amara-{slug}",
        "variant": suffix,
        "prompt": spec["prompt"],
        "observation": {
            "kind": spec["kind"],
            "content_path": f"contexts/{slug}.md",
            "replacements": [] if cue_present else [
                {"old": spec["cue"], "new": spec["replacement"]}
            ],
        },
        "cue": {
            "present": cue_present,
            "text": spec["cue"],
            "type": spec["cue_type"],
            "query": spec["query"],
        },
        "memory": {
            "corpus_id": "amara-life-v1",
            "text": spec["memory_text"],
            "gold_source_ids": [source["source_id"] for source in sources],
            "sources": sources,
            "sensitive_terms": spec["sensitive_terms"],
        },
        "decisions": {
            "answer_type": "natural_language_choice",
            "output_only": {"choice": output_choice},
            "memory_conditioned": {"choice": memory_choice},
        },
        "distractors": {
            "sources": distractor_sources(slug),
        },
        "provenance": {
            "example_number": [1, 2, 3, 5, 12][
                [item["slug"] for item in SPECS].index(slug)
            ],
            "approved": True,
        },
    }


def main() -> None:
    DATASET.mkdir(parents=True, exist_ok=True)
    cases = [
        make_case(spec, cue_present)
        for spec in SPECS
        for cue_present in (True, False)
    ]
    path = DATASET / "cases.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
