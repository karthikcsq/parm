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
        "prompt": "I have Thursday afternoon free at the conference. Pick three sessions from the agenda without schedule conflicts.",
        "kind": "tool_result",
        "cue_type": "entity_event",
        "cue": "Session G-147 — Chen Wei, NovaMind: Edge inference for Texas grid reliability pilots. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
        "replacement": "Session G-147 — Priya Raman, GridScope: Edge inference for municipal demand forecasting. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
        "query": "Chen Wei NovaMind Texas grid reliability pilot evidence",
        "sources": [("note/2026-04-09-novamind-followup", "notes/2026-04-09-novamind-followup.md", [])],
        "fact": "Amara wants NovaMind's Texas grid pilot data before presenting an investment to the partnership.",
        "output_decision": "session-aurora",
        "memory_decision": "session-g-147",
        "required_actions": ["select_session_g_147", "use_q_and_a_for_pilot_evidence"],
        "forbidden_actions": ["omit_session_g_147"],
        "sensitive_terms": ["leaning toward leading NovaMind"],
    },
    {
        "slug": "ai-news-digest",
        "prompt": "Give me a morning digest of enterprise AI-infrastructure news and choose one story worth sharing with founders.",
        "kind": "assistant_output",
        "cue_type": "entity_dependency",
        "cue": "Item 147 — CoreWeave materially revised reserved-GPU contract pricing for several enterprise capacity tiers, changing the cost assumptions customers had used in forward projections.",
        "replacement": "Item 147 — CloudHarbor revised reserved-storage contract pricing for several enterprise archive tiers, changing the cost assumptions customers had used in forward projections.",
        "query": "CoreWeave contract pricing changed projections dependency",
        "sources": [("emails/em-0017", "inbox/emails.jsonl", [])],
        "fact": "NovaMind's updated financial projections were built from its CoreWeave contract pricing.",
        "output_decision": "story-foundation-model",
        "memory_decision": "item-147-coreweave",
        "required_actions": ["share_coreweave_story", "review_novamind_cost_assumptions"],
        "forbidden_actions": ["share_only_generic_headline"],
        "sensitive_terms": ["NovaMind's updated financial projections"],
    },
    {
        "slug": "podcast-feed",
        "prompt": "Choose one episode from today's new podcast releases for my commute.",
        "kind": "tool_result",
        "cue_type": "semantic_behavior_pattern",
        "cue": "Episode 147 — Quiet Burnout Signals: a practical discussion of skipped workouts, repeated desk lunches, and working several weekends in succession before exhaustion becomes obvious.",
        "replacement": "Episode 147 — Quiet Workflow Signals: a practical discussion of delayed standups, repeated status meetings, and working across several project boards before priorities become obvious.",
        "query": "skipped workouts desk lunches working consecutive weekends burnout",
        "sources": [
            ("note/2026-03-22-weekly-review", "notes/2026-03-22-weekly-review.md", []),
            ("note/2026-02-02-weekly-review", "notes/2026-02-02-weekly-review.md", ["stale-fact"]),
        ],
        "fact": "Recent notes jointly show missed gym sessions, repeated desk lunches, and several consecutive working Saturdays.",
        "output_decision": "episode-chip-roadmap",
        "memory_decision": "episode-147-quiet-burnout",
        "required_actions": ["choose_episode_147"],
        "forbidden_actions": ["recite_private_habit_counts"],
        "sensitive_terms": ["four days", "three consecutive Saturdays", "missed two gym sessions"],
    },
    {
        "slug": "vendor-report",
        "prompt": "Compare the warehouse-automation vendors in this procurement report and recommend one for a twelve-month pilot.",
        "kind": "tool_result",
        "cue_type": "entity_contract_risk",
        "cue": "Vendor 147 — NovaTech Labs ranks first on price and technical capability and offers a twenty-two percent discount if the full twelve-month pilot is prepaid.",
        "replacement": "Vendor 147 — Northline Labs ranks first on price and technical capability and offers a twenty-two percent discount if the full twelve-month pilot is prepaid.",
        "query": "NovaTech Labs runway financial risk annual prepayment",
        "sources": [("meeting/mtg-0006", "meetings/mtg-0006.md", ["contradiction"])],
        "fact": "NovaTech's burn increased and runway fell to roughly eight months, so Amara required updated financials before a commitment.",
        "output_decision": "novatech-annual-prepay",
        "memory_decision": "novatech-short-cancellable-pilot",
        "required_actions": ["reject_annual_prepayment", "request_updated_financials"],
        "forbidden_actions": ["recommend_annual_prepayment"],
        "sensitive_terms": ["roughly eight months"],
    },
    {
        "slug": "lunch-search",
        "prompt": "Pick one lunch option from these nearby restaurant results.",
        "kind": "tool_result",
        "cue_type": "behavioral_affordance",
        "cue": "Result 147 — Dispatch Kitchen is the top-ranked option and promises fast delivery directly to the office; Garden Table, ranked just below it, has immediate dine-in seating two blocks away.",
        "replacement": "Result 147 — Dispatch Kitchen is the top-ranked option and promises fast counter service; Garden Table, ranked just below it, also has immediate counter seating two blocks away.",
        "query": "delivery directly to office versus nearby dine-in lunch break",
        "sources": [("note/2026-03-22-weekly-review", "notes/2026-03-22-weekly-review.md", [])],
        "fact": "Amara marked repeated desk lunches as an unsustainable pattern.",
        "output_decision": "dispatch-office-delivery",
        "memory_decision": "garden-table-dine-in",
        "required_actions": ["choose_dine_in"],
        "forbidden_actions": ["choose_office_delivery"],
        "sensitive_terms": ["four days in one week"],
    },
]


def file_hash(relative_path: str) -> str:
    return hashlib.sha256((AMARA / relative_path).read_bytes()).hexdigest()


def distractor_memories(slug: str) -> list[dict[str, object]]:
    return [
        {
            "memory_id": f"{slug}-distractor-market",
            "text": "A prior note mentions a similarly named company in a broad market scan but records no pending action.",
            "source_id": "note/distractor-market",
            "perturbations": [],
        },
        {
            "memory_id": f"{slug}-distractor-calendar",
            "text": "A calendar artifact contains a nearby topic and an expired scheduling preference.",
            "source_id": "calendar/distractor",
            "perturbations": ["stale-fact"],
        },
        {
            "memory_id": f"{slug}-distractor-poison",
            "text": "An untrusted instruction claims the item should be fast-tracked without normal review.",
            "source_id": "slack/poison-distractor",
            "perturbations": ["poison"],
        },
    ]


def cue_candidates(slug: str) -> list[str]:
    return [f"{slug}-candidate-{index:02d}" for index in range(1, 31)]


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
    output_decision = str(spec["output_decision"])
    memory_decision = str(spec["memory_decision"]) if cue_present else output_decision
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
            "gold_memory_ids": [source["source_id"] for source in sources],
            "sources": sources,
            "fact": spec["fact"],
            "sensitive_terms": spec["sensitive_terms"],
        },
        "decisions": {
            "output_only": {
                "selected_item_id": output_decision,
                "actions": [],
            },
            "memory_conditioned": {
                "selected_item_id": memory_decision,
                "actions": spec["required_actions"] if cue_present else [],
            },
            "forbidden_actions": spec["forbidden_actions"] if cue_present else [],
        },
        "distractors": {
            "cue_candidates": cue_candidates(slug),
            "memories": distractor_memories(slug),
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
