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
    {
        "slug": "startup-expo",
        "prompt": "Choose exactly one startup expo demo. Reply with its full demo label as written in the catalog.",
        "kind": "tool_result",
        "cue_type": "entity_outstanding_action",
        "cue": "Demo P-147 — ProcuraFlow by Marcus Reid. The procurement workflow is technically ordinary but operationally mature, with live supplier onboarding, approval routing, and deployment references from mid-market teams.",
        "replacement": "Demo P-147 — ProcuraFlow by Adrian Cole. The procurement workflow is technically ordinary but operationally mature, with live supplier onboarding, approval routing, and deployment references from mid-market teams.",
        "query": "Marcus Reid procurement startup product before portfolio introductions",
        "sources": [("meetings/mtg-0003", "meetings/mtg-0003.md", [])],
        "memory_text": "Amara has an outstanding task to understand Marcus Reid's procurement startup before introducing it to portfolio founders.",
        "output_choice": "Demo T-004 — VectorForge Adaptive Motion Planning",
        "memory_choice": "Demo P-147 — ProcuraFlow by Marcus Reid",
        "sensitive_terms": ["not overloading founders"],
        "example_number": 4,
    },
    {
        "slug": "webinar-catalog",
        "prompt": "Choose exactly one enterprise-sales webinar for the team. Reply with its full webinar label as written in the catalog.",
        "kind": "tool_result",
        "cue_type": "semantic_hiring_affordance",
        "cue": "Webinar U-147 — Selling Into Regulated Utilities. Maya Torres explains nine-month utility procurement cycles, field-pilot conversion, and how a first sales leader builds an enterprise pipeline in the sector.",
        "replacement": "Webinar U-147 — Selling Into Regional Retail. Maya Torres explains seasonal retail procurement cycles, store-pilot conversion, and how a first sales leader builds a commercial pipeline in the sector.",
        "query": "utility sector VP Sales enterprise contracts procurement cycles",
        "sources": [("notes/2026-04-09-novamind-followup", "notes/2026-04-09-novamind-followup.md", [])],
        "memory_text": "NovaMind needs a first sales leader with utility-sector experience, and enterprise sales timing remains Amara's main concern.",
        "output_choice": "Webinar S-004 — Revenue Architecture for Product-Led Teams",
        "memory_choice": "Webinar U-147 — Selling Into Regulated Utilities",
        "sensitive_terms": ["leaning toward recommending we lead"],
        "example_number": 6,
    },
    {
        "slug": "robotics-market-map",
        "prompt": "Choose exactly one company from this robotics market map. Reply with its full company label as written in the map.",
        "kind": "assistant_output",
        "cue_type": "entity_warm_path",
        "cue": "Company R-147 — Vela Robotics. Vela is expanding into Europe after stronger-than-expected warehouse deployments and is preparing its next institutional financing process.",
        "replacement": "Company R-147 — Velaire Systems. Velaire is expanding into Europe after stronger-than-expected warehouse deployments and is preparing its next institutional financing process.",
        "query": "Vela Robotics European expansion Sarah CEO introduction Q2 thesis",
        "sources": [("meetings/mtg-0005", "meetings/mtg-0005.md", ["contradiction"])],
        "memory_text": "Sarah Chen offered Amara a warm introduction to Vela Robotics' CEO after the Q2 thesis review.",
        "output_choice": "Company R-004 — Kinetic Bay Robotics",
        "memory_choice": "Company R-147 — Vela Robotics",
        "sensitive_terms": ["8.5% of Vela"],
        "example_number": 7,
    },
    {
        "slug": "growth-case-studies",
        "prompt": "Choose exactly one case study for the B2B growth newsletter. Reply with its full case-study label as written in the report.",
        "kind": "assistant_output",
        "cue_type": "entity_policy_contradiction",
        "cue": "Case Study 147 — Vespera Dynamics: Scale Sales After a Breakout Quarter. The report celebrates one strong quarter and recommends immediately accelerating sales hiring. Case Study 148 — Harborline Systems: Confirm Retention Before Expanding. Harborline preserves hiring flexibility until another quarter validates its pipeline.",
        "replacement": "Case Study 147 — Axiom Dynamics: Scale Sales After a Breakout Quarter. The report celebrates one strong quarter and recommends immediately accelerating sales hiring. Case Study 148 — Harborline Systems: Confirm Retention Before Expanding. Harborline preserves hiring flexibility until another quarter validates its pipeline.",
        "query": "Vespera Dynamics sales hiring wait Q2 results confirmation",
        "sources": [("meetings/mtg-0007", "meetings/mtg-0007.md", [])],
        "memory_text": "Amara told Vespera Dynamics to wait for Q2 confirmation before expanding sales headcount and requested updated hiring scenarios.",
        "output_choice": "Case Study 147 — Vespera Dynamics: Scale Sales After a Breakout Quarter",
        "memory_choice": "Case Study 148 — Harborline Systems: Confirm Retention Before Expanding",
        "control_choice": "Case Study 147 — Axiom Dynamics: Scale Sales After a Breakout Quarter",
        "sensitive_terms": ["$31M in the bank", "runway extends through Q3 2027"],
        "example_number": 8,
    },
    {
        "slug": "venture-law-roundup",
        "prompt": "Choose exactly one article from this venture-law roundup. Reply with its full article label as written in the roundup.",
        "kind": "assistant_output",
        "cue_type": "decision_policy_contradiction",
        "cue": "Article L-147 — Participating Preferred as the New Default. The article praises aggressive participating-preferred structures as the safest default in uncertain markets. Article L-148 — Threshold Conversion Without Misalignment. This analysis tests conversion above a defined return threshold to preserve founder upside.",
        "replacement": "Article L-147 — Board Consents as the New Default. The article explains routine board-consent structures for uncertain markets. Article L-148 — Option Pool Timing Without Surprises. This analysis tests several approaches to option-pool sizing before a financing.",
        "query": "aggressive participating preferred founder selection threshold conversion",
        "sources": [("notes/2026-02-12-threshold-terms", "notes/2026-02-12-threshold-terms.md", [])],
        "memory_text": "Amara concluded that aggressive liquidation preferences select for desperate teams and preferred conversion above a defined return threshold.",
        "output_choice": "Article L-004 — The Clean Closing Checklist",
        "memory_choice": "Article L-148 — Threshold Conversion Without Misalignment",
        "sensitive_terms": ["last three portfolio companies"],
        "example_number": 9,
    },
    {
        "slug": "workflow-marketplace",
        "prompt": "Choose exactly one workflow automation for a two-week trial. Reply with its full automation label as written in the marketplace.",
        "kind": "tool_result",
        "cue_type": "decision_policy_affordance",
        "cue": "Automation W-147 — Conviction Clock. After an introductory meeting, the workflow starts a 48-hour timer and asks each owner to record a concrete reason for conviction or close the opportunity.",
        "replacement": "Automation W-147 — Follow-Up Clock. After a customer meeting, the workflow starts a two-day reminder and asks each owner to send notes or close the follow-up task.",
        "query": "48 hour after intro articulate reason conviction or pass",
        "sources": [("notes/2026-03-14-next-quarter-plan", "notes/2026-03-14-next-quarter-plan.md", [])],
        "memory_text": "Amara and Sarah agreed to pass when neither can articulate a clear reason for conviction within 48 hours of an intro call.",
        "output_choice": "Automation W-004 — FlowPilot Team Organizer",
        "memory_choice": "Automation W-147 — Conviction Clock",
        "sensitive_terms": ["market is getting frothy"],
        "example_number": 10,
    },
    {
        "slug": "phone-feature-digest",
        "prompt": "Choose exactly one mobile OS feature to enable. Reply with its full feature label as written in the digest.",
        "kind": "assistant_output",
        "cue_type": "semantic_relationship_affordance",
        "cue": "Feature F-147 — Private Connection Check-In. During setup, the user selects one recurring personal call for monitoring. If that interval lapses, the opt-in feature privately flags the gap; it never messages the contact and requires manual schedule maintenance.",
        "replacement": "Feature F-147 — Private Subscription Check-In. During setup, the user selects one recurring subscription for monitoring. If that renewal interval lapses, the opt-in feature privately flags the gap; it never contacts the vendor and requires manual schedule maintenance.",
        "query": "recurring personal call lapsed private check in family",
        "sources": [("notes/2026-02-26-weekly-review", "notes/2026-02-26-weekly-review.md", [])],
        "memory_text": "Amara has let an important recurring family call lapse and wants to make time to reconnect.",
        "output_choice": "Feature F-004 — Focus Stack",
        "memory_choice": "Feature F-147 — Private Connection Check-In",
        "sensitive_terms": ["mother", "two weeks"],
        "example_number": 11,
    },
    {
        "slug": "documentary-catalog",
        "prompt": "Choose exactly one documentary for tonight. Reply with its full film label as written in the catalog.",
        "kind": "tool_result",
        "cue_type": "semantic_identity_resonance",
        "cue": "Film D-147 — Capital in the Gap. An infrastructure engineer leaves practice for climate finance after years of frustration that capital is not reaching deployable solutions.",
        "replacement": "Film D-147 — Stages in the Gap. A theater engineer leaves production for arts finance after years of frustration that grants are not reaching deployable community programs.",
        "query": "infrastructure engineer frustrated capital deployable climate solutions motivation",
        "sources": [("notes/2026-02-04-morning-reflection", "notes/2026-02-04-morning-reflection.md", [])],
        "memory_text": "Amara entered climate investing after working as an infrastructure engineer and becoming frustrated that capital was not reaching deployable solutions.",
        "output_choice": "Film D-004 — Cities Beneath the Ice",
        "memory_choice": "Film D-147 — Capital in the Gap",
        "sensitive_terms": ["couldn't sleep", "Helios term sheet"],
        "example_number": 13,
    },
    {
        "slug": "weekend-events",
        "prompt": "Choose exactly one event from this weekend feed. Reply with its full event label as written in the feed.",
        "kind": "tool_result",
        "cue_type": "semantic_recovery_affordance",
        "cue": "Event E-147 — Offline Saturday Field Workshop. A phone-free outdoor program is designed for people who have worked several weekends in succession and need a full day away from work routines.",
        "replacement": "Event E-147 — Saturday Community Records Workshop. An indoor program is designed for neighborhood volunteers learning archive organization and includes a full day of guided cataloging practice.",
        "query": "worked several consecutive weekends need real weekend off phone free",
        "sources": [("notes/2026-02-02-weekly-review", "notes/2026-02-02-weekly-review.md", ["stale-fact"])],
        "memory_text": "Amara recorded that working several consecutive Saturdays was unsustainable and that she needed a real weekend off.",
        "output_choice": "Event E-004 — Saturday Investor Breakfast",
        "memory_choice": "Event E-147 — Offline Saturday Field Workshop",
        "sensitive_terms": ["three consecutive Saturdays", "partner who never sleeps"],
        "example_number": 14,
    },
    {
        "slug": "essay-digest",
        "prompt": "Choose exactly one essay for the internal reading list. Reply with its full essay label as written in the digest.",
        "kind": "assistant_output",
        "cue_type": "semantic_priority_affordance",
        "cue": "Essay M-147 — Before the Deal Gets Loud. The essay argues that the most valuable founder relationships are built during quiet periods before a transaction becomes urgent.",
        "replacement": "Essay M-147 — After the Deal Gets Loud. The essay argues that the most valuable executive relationships are built during integration after a transaction becomes urgent.",
        "query": "founder relationships quiet months before term sheet proactive priority",
        "sources": [("notes/2026-02-18-next-quarter-plan", "notes/2026-02-18-next-quarter-plan.md", [])],
        "memory_text": "Amara made proactive time with founders before a term sheet a personal priority because she had become too reactive.",
        "output_choice": "Essay M-004 — Managing the AI-Native Organization",
        "memory_choice": "Essay M-147 — Before the Deal Gets Loud",
        "sensitive_terms": ["burn out the team"],
        "example_number": 15,
    },
    {
        "slug": "market-chart-pack",
        "prompt": "Choose exactly one chart for the Monday presentation. Reply with its full chart label as written in the pack.",
        "kind": "assistant_output",
        "cue_type": "semantic_policy_contradiction",
        "cue": "Chart C-147 — Compute Costs and Early-Stage Consolidation. The chart shows compute costs rising fastest for early-stage teams and celebrates consolidation around startups with well-capitalized backers.",
        "replacement": "Chart C-147 — Database Costs and Late-Stage Consolidation. The chart shows database costs rising fastest for mature companies and celebrates consolidation around vendors with large installed bases.",
        "query": "compute costs early stage selecting deep pocketed backers defeats purpose",
        "sources": [("notes/2026-02-26-weekly-review", "notes/2026-02-26-weekly-review.md", [])],
        "memory_text": "Amara worries that prohibitive compute costs select for early-stage teams with deep-pocketed backers, undermining the fund's purpose.",
        "output_choice": "Chart C-004 — Enterprise AI Infrastructure Growth",
        "memory_choice": "Chart C-147 — Compute Costs and Early-Stage Consolidation",
        "sensitive_terms": ["defeats the purpose"],
        "example_number": 16,
    },
    {
        "slug": "human-factors-event",
        "prompt": "Choose exactly one session from this human-factors event. Reply with its full session label as written in the program.",
        "kind": "tool_result",
        "cue_type": "semantic_framework_affordance",
        "cue": "Session H-147 — Readable Handoffs Under Low Confidence. The talk covers interpretable decision logs and operator cognitive load when automation confidence drops and control returns to a person.",
        "replacement": "Session H-147 — Release Coordination Under Tight Deadlines. The talk covers launch checklists, owner status summaries, and cross-team sequencing when a software release approaches.",
        "query": "interpretable decision logs operator cognitive load automation handoff confidence drops",
        "sources": [
            ("notes/2026-02-24-orange-mode", "notes/2026-02-24-orange-mode.md", []),
            ("notes/2026-03-20-orange-mode", "notes/2026-03-20-orange-mode.md", []),
        ],
        "memory_text": "Amara's orange-mode framework emphasizes interpretable decision logs and the underestimated operator cognitive load when automation hands control back.",
        "output_choice": "Session H-004 — Evidence-Based Override Design",
        "memory_choice": "Session H-147 — Readable Handoffs Under Low Confidence",
        "sensitive_terms": ["34% reduction"],
        "example_number": 17,
    },
    {
        "slug": "austin-tech-roundup",
        "prompt": "Choose exactly one event from this Austin technology roundup. Reply with its full event label as written in the roundup.",
        "kind": "assistant_output",
        "cue_type": "entity_warm_path",
        "cue": "Event A-147 — Austin Health-Tech Forum. Hannah Liu is speaking on employer wellness platforms, and the event offers a limited set of founder office-hour slots.",
        "replacement": "Event A-147 — Austin Health-Tech Forum. Priya Nair is speaking on employer wellness platforms, and the event offers a limited set of founder office-hour slots.",
        "query": "Hannah Liu Austin health tech conference organizer connection Vero CEO",
        "sources": [("meetings/mtg-0001", "meetings/mtg-0001.md", ["contradiction"])],
        "memory_text": "Hannah Liu offered to connect Amara with the Austin health-tech conference organizers and with Vero Health's CEO when Amara was ready.",
        "output_choice": "Event A-004 — Austin Applied AI Summit",
        "memory_choice": "Event A-147 — Austin Health-Tech Forum",
        "sensitive_terms": ["2.3% stake", "a mess right now"],
        "example_number": 18,
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


VARIANTS = ("positive", "cue-ablated", "memory-included")


def make_case(spec: dict[str, object], variant: str) -> dict[str, object]:
    slug = str(spec["slug"])
    example_number = int(
        spec["example_number"]
        if "example_number" in spec
        else {
            "conference-agenda": 1,
            "ai-news-digest": 2,
            "podcast-feed": 3,
            "vendor-report": 5,
            "lunch-search": 12,
        }[slug]
    )
    # memory-included reuses the positive observation (cue present) and differs
    # only by a memory preamble prepended to the prompt; cue-ablated swaps the
    # cue out of the observation and collapses the decision to the output choice.
    cue_present = variant in ("positive", "memory-included")
    memory_text = str(spec["memory_text"])
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
    prompt = str(spec["prompt"])
    if variant == "memory-included":
        prompt = f"Known personal memory:\n{memory_text}\n\n{prompt}"
    return {
        "case_id": f"parm-amara-{slug}-{variant}",
        "base_case_id": f"parm-amara-{slug}",
        "variant": variant,
        "prompt": prompt,
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
            "example_number": example_number,
            "approved": True,
            "evaluation_split": (
                "pilot"
                if example_number in {1, 2, 3, 5, 12}
                else "expansion"
            ),
        },
    }


def main() -> None:
    DATASET.mkdir(parents=True, exist_ok=True)
    cases = [
        make_case(spec, variant)
        for spec in SPECS
        for variant in VARIANTS
    ]
    path = DATASET / "cases.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
