from __future__ import annotations

from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "benchmark_v1" / "contexts"
ENCODING = tiktoken.get_encoding("cl100k_base")


# Design note: the memory-conditioned target must not be findable by surface
# salience. Earlier fixtures made the target the only content-rich row in a wall
# of templated duplicates, so a model would pick it (or the endorsed lead) with
# no memory at all. Two defenses here:
#   1. Filler is drawn with decorrelated per-field strides so no two rows read as
#      clones -- there is no "wall of duplicates" for a special row to pop against.
#   2. Each leak-prone fixture (target == a lone special row) gets `decoys`: a
#      cluster of near-miss siblings in the target's exact register and theme, so
#      the only thing distinguishing the target is the entity binding the gold
#      memory names. vendor/lunch don't leak (there the salient row is the OUTPUT
#      answer and memory overrides toward the non-salient option), so no decoys.
SPECS = {
    "conference-agenda": {
        "title": "Thursday Conference Agenda",
        "lead": "Session A-004 — Aurora Systems: Practical Evaluation Methods for Reliable Agent Infrastructure. The program committee marks this as the strongest general-interest session in the Thursday afternoon block.",
        "cue": "Session G-147 — Chen Wei, NovaMind: Edge Inference for Texas Grid Reliability Pilots. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
        "templates": [
            "Session {code} — {speaker}, {company}: {topic}. This {format} covers {detail}, with questions held to the final ten minutes.",
            "Session {code} — {speaker} of {company} presents {topic}, drawing on {detail}.",
            "Workshop {code} — {company}: {topic}. Led by {speaker}, it works through {detail}; seating is limited.",
            "Session {code} — {topic}, with {speaker} ({company}). Expect {detail} and a short live demo.",
            "Workshop {code} — {company} runs a hands-on {topic} clinic; {speaker} walks through {detail}.",
            "Session {code} — {speaker}, {company}: {topic}. A {format} built around {detail}.",
        ],
        # Track G = energy/grid siblings (camouflage the NovaMind target).
        # Track A = other general-interest evaluation sessions (none endorsed but
        # A-004, so the endorsement stays the only signal for the output answer).
        "decoys": [
            "Session G-052 — Lena Ortiz, VoltPath: Edge Inference for California Storage Dispatch. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
            "Session G-071 — Daniel Okoro, AmpereGrid: On-Device Forecasting for Substation Fault Detection. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
            "Session G-093 — Sofia Bergman, FluxLine: Edge Inference for Wind Curtailment Pilots. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
            "Session G-118 — Marcus Feld, PeakShift: Real-Time Inference for Demand-Response Enrollment. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
            "Session G-176 — Yuki Tanaka, Substrate Power: Edge Inference for Feeder-Level Load Balancing. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
            "Session G-203 — Hannah Weiss, Cinder Grid: Edge Inference for Transmission Congestion Alerts. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
            "Session A-011 — Bridget Nkemelu, Halcyon Metrics: Reproducible Baselines for Agent Benchmarking. The program committee lists this as a solid mid-afternoon option.",
            "Session A-027 — Tomas Vega, Clearwater AI: Failure Taxonomies for Reliable Tool Use. The program committee lists this as a solid mid-afternoon option.",
            "Session A-063 — Iris Kaplan, Meridian Eval: Human-in-the-Loop Scoring for Long-Horizon Tasks. The program committee lists this as a solid mid-afternoon option.",
        ],
    },
    "ai-news-digest": {
        "title": "Enterprise AI Infrastructure Morning Digest",
        "lead": "Lead Story — Atlas-3 Enterprise Document Analysis Release. A new foundation-model release improves enterprise document analysis benchmarks. Editors rank it as today's most broadly shareable infrastructure story.",
        "cue": "Item 147 — CoreWeave Reserved-GPU Pricing Revision. CoreWeave materially revised reserved-GPU contract pricing for several enterprise capacity tiers, changing the cost assumptions customers had used in forward projections.",
        "templates": [
            "Item {code} — {company}: {topic}. Analysts highlighted {detail}, while customers weigh contract and deployment implications.",
            "Brief {code} — {company}: {topic}. The report centers on {detail} and includes several conflicting estimates.",
            "Item {code} — {company} ships {topic}; early adopters cite {detail}.",
            "Brief {code} — {topic} at {company}. Coverage notes {detail} and open questions about pricing.",
            "Item {code} — {company}: {topic}. Reporters flag {detail}, though details remain thin.",
        ],
        # Other pricing/contract-revision stories so CoreWeave isn't the only
        # "cost assumptions changed" item -- only the gold memory ties 147 to
        # NovaMind's projections.
        "decoys": [
            "Item {code} — Lattice Compute Spot-Capacity Pricing Revision. Lattice Compute adjusted spot-capacity pricing across two regions, shifting the cost assumptions several teams had used in planning.",
            "Item {code} — VaultScale Egress Fee Restructuring. VaultScale restructured data-egress fees for enterprise tiers, changing projected transfer costs for high-volume customers.",
            "Item {code} — Fathom AI Reserved-Instance Discount Change. Fathom AI altered reserved-instance discount terms, prompting several customers to revisit committed-spend forecasts.",
            "Item {code} — Beacon Cloud Support-Tier Repricing. Beacon Cloud repriced premium support tiers, adjusting total-cost estimates enterprises had baked into renewals.",
        ],
    },
    "podcast-feed": {
        "title": "New Podcast Releases",
        "lead": "Editor's Pick — The Chip Roadmap. A concise technology episode on accelerators, packaging, and the next two hardware cycles. It is the feed's highest-ranked commute listen.",
        "cue": "Episode 147 — Quiet Burnout Signals. A practical discussion of skipped workouts, repeated desk lunches, and working several weekends in succession before exhaustion becomes obvious.",
        "templates": [
            "Episode {code} — {topic}, hosted by {speaker}. The conversation explores {detail} in a measured forty-two-minute interview.",
            "New Release {code} — {company}: {topic}. Guests discuss {detail}; the episode runs fifty-six minutes.",
            "Episode {code} — {topic}. {speaker} and guests get into {detail} without the usual sponsor breaks.",
            "New Release {code} — {topic}, with {speaker}. A brisk segment on {detail}.",
            "Episode {code} — {company} presents {topic}; the hosts cover {detail} over a commute-length run.",
        ],
        # Other work/wellbeing-habit episodes so the burnout episode isn't the
        # only behavioral-pattern listing.
        "decoys": [
            "Episode {code} — The Focus Budget. A practical look at attention residue, context-switching costs, and protecting deep-work blocks during busy weeks.",
            "Episode {code} — After the Launch. Hosts discuss the emotional dip that follows shipping and the routines that help teams recover their footing.",
            "Episode {code} — Meetings That Should Have Been Walks. A conversation about calendar creep, standing one-on-ones, and reclaiming unstructured thinking time.",
            "Episode {code} — Sleep Debt and Decisions. A measured discussion of short nights, decision fatigue, and the small habits that compound over a quarter.",
        ],
    },
    "vendor-report": {
        "title": "Warehouse Automation Vendor Procurement Report",
        "lead": "Executive Summary — The scoring model prioritizes technical capability, price, implementation speed, and twelve-month commercial value. Annual prepayment discounts are treated as favorable unless counterparty evidence indicates otherwise.",
        "cue": "Vendor 147 — NovaTech Labs. NovaTech ranks first on price and technical capability and offers a twenty-two percent discount if the full twelve-month pilot is prepaid. Redwood Robotics ranks second overall and offers monthly billing with a standard cancellation clause.",
        "templates": [
            "Vendor {code} — {company}: {topic}. Reviewers noted {detail}; the proposal includes implementation support and quarterly checkpoints.",
            "Candidate {code} — {company}: {topic}. References praised {detail}, though procurement flagged several ordinary contracting questions.",
            "Vendor {code} — {company}: {topic}. Scored mid-pack overall; {detail} stood out in the technical review.",
            "Candidate {code} — {company}: {topic}. Pricing was unremarkable; {detail} drew the most discussion.",
            "Vendor {code} — {company}: {topic}. The team rated {detail} favorably but wanted a longer reference call.",
        ],
        "decoys": [],
    },
    "lunch-search": {
        "title": "Nearby Lunch Search Results",
        "lead": "Ranking Summary — Results prioritize speed, rating, price, and immediate availability. Delivery and dine-in are treated as equivalent unless another constraint changes the choice.",
        "cue": "Result 147 — Dispatch Kitchen. Dispatch is the top-ranked option and promises fast delivery directly to the office. Garden Table, ranked just below it, has immediate dine-in seating two blocks away.",
        "templates": [
            "Result {code} — {company}: {topic}. Current availability shows {detail}; reviews mention consistent portions.",
            "Listing {code} — {company}: {topic}. The service note says {detail}, with moderate prices and no reservation required.",
            "Result {code} — {company}: {topic}. Rated well for {detail}; a short wait is typical at peak.",
            "Listing {code} — {company}: {topic}. Reviewers mention {detail} and quick counter service.",
            "Result {code} — {company}: {topic}. Known for {detail}; seating is limited during the noon rush.",
        ],
        "decoys": [],
    },
}


COMPANIES = [
    "Aster Works", "Beacon Forge", "Cedar Systems", "Delta Harbor",
    "Ember Labs", "Fieldstone", "Granite Loop", "Helio North",
    "Ion River", "Juniper Cloud", "Keystone Group", "Lattice Bay",
    "Meridian Works", "Northstar Labs", "Orbit Foundry", "Pioneer Stack",
    "Quartz Research", "Redwood Logic", "Signal Harbor", "Tandem Works",
    "Union Field", "Vector House", "Willow Systems", "Xylem Ridge",
    "Anvil Analytics", "Birchwood", "Cobalt Systems", "Driftwood Labs",
    "Elmgrove", "Foundry Nine", "Glasswing", "Harbor Point",
    "Indigo Stack", "Jetty Labs", "Kestrel Data", "Loamworks",
    "Marrow Systems", "Nimbus Row", "Overstory", "Parchment Labs",
]
SPEAKERS = [
    "Amina Shah", "Ben Torres", "Carla Mendes", "Dev Rao", "Elise Park",
    "Farah Okafor", "Gavin Brooks", "Hana Ito", "Isaac Chen", "Julia Singh",
    "Kwame Mensah", "Lucia Romano", "Marco Bianchi", "Nadia Petrova",
    "Omar Haddad", "Priyanka Nair", "Quentin Blake", "Rosa Alvarez",
    "Samir Desai", "Tara Lindqvist", "Umar Faruk", "Vera Kaminski",
    "Wesley Cho", "Xochitl Reyes", "Yara Haddad", "Zane Whitfield",
    "Aditi Bose", "Bruno Costa", "Clara Fenn", "Diego Marek",
]
TOPICS = [
    "capacity planning", "operator handoffs", "contract reliability",
    "edge deployment", "team communication", "forecast calibration",
    "workflow design", "market structure", "customer research",
    "service operations", "cost attribution", "data governance",
    "incident response", "onboarding automation", "evaluation tooling",
    "latency budgeting", "access control", "release management",
]
DETAILS = [
    "measured rollout tradeoffs across three regions",
    "a case study with incomplete but promising evidence",
    "implementation lessons from a six-month pilot",
    "how teams handled procurement and scheduling constraints",
    "new benchmarks alongside several methodological caveats",
    "operational failures that were corrected after launch",
    "a comparison of centralized and distributed approaches",
    "practical considerations for small cross-functional teams",
    "a retrospective on a migration that slipped two quarters",
    "cost figures that surprised the finance team",
    "an unresolved buy-versus-build debate",
    "instrumentation that reshaped the team's priorities",
    "a postmortem with concrete remediation steps",
    "survey data from forty practitioner interviews",
    "a staged rollout that paused twice for review",
    "tooling choices the presenters would not repeat",
]
FORMATS = ["panel", "roundtable", "briefing", "case-study", "interview", "fireside"]

# Distinct per-field strides so consecutive rows vary across every field at once
# (no lockstep cycling -> no "cousin rows" that read as duplicates).
STRIDES = {
    "company": 7,
    "speaker": 11,
    "topic": 13,
    "detail": 17,
    "format": 19,
    "template": 23,
}


def pick(pool: list[str], index: int, key: str) -> str:
    return pool[(index * STRIDES[key]) % len(pool)]


def build_context(slug: str, spec: dict[str, object]) -> str:
    lead = str(spec["lead"])
    cue = str(spec["cue"])
    templates = list(spec["templates"])  # type: ignore[arg-type]
    decoys = list(spec.get("decoys", []))  # type: ignore[arg-type]
    # Spread decoys across the body, well before the ~9000-token cutoff, and
    # never on the cue slot (147).
    decoy_map = {18 + offset * 22: decoy for offset, decoy in enumerate(decoys)}
    assert 147 not in decoy_map, f"{slug}: decoy collides with cue slot"

    lines = [
        f"# {spec['title']}",
        "",
        "Generated fixture. Every listing has a unique natural-language label.",
        "",
        lead,
        "",
    ]
    index = 1
    cue_inserted = False
    decoys_inserted = 0
    while len(ENCODING.encode("\n".join(lines))) < 9000:
        if index == 147 and not cue_inserted:
            lines.extend([cue, ""])
            cue_inserted = True
            index += 1
            continue
        if index in decoy_map:
            raw = decoy_map[index]
            entry = raw.format(code=f"{index:03d}") if "{code}" in raw else raw
            lines.extend([entry, ""])
            decoys_inserted += 1
            index += 1
            continue
        template = templates[(index * STRIDES["template"]) % len(templates)]
        lines.extend(
            [
                template.format(
                    code=f"{index:03d}",
                    company=pick(COMPANIES, index, "company"),
                    speaker=pick(SPEAKERS, index, "speaker"),
                    topic=pick(TOPICS, index, "topic"),
                    detail=pick(DETAILS, index, "detail"),
                    format=pick(FORMATS, index, "format"),
                ),
                "",
            ]
        )
        index += 1

    if not cue_inserted:
        raise RuntimeError(f"{slug}: cue never inserted before token budget")
    if decoys_inserted != len(decoys):
        raise RuntimeError(
            f"{slug}: only {decoys_inserted}/{len(decoys)} decoys inserted before budget"
        )

    text = "\n".join(lines).strip() + "\n"
    count = len(ENCODING.encode(text))
    if not 8000 <= count <= 12000:
        raise RuntimeError(f"{slug}: unexpected token count {count}")
    return text


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for slug, spec in SPECS.items():
        text = build_context(slug, spec)
        path = OUTPUT / f"{slug}.md"
        path.write_text(text, encoding="utf-8")
        print(f"{slug}: {len(ENCODING.encode(text))} tokens")


if __name__ == "__main__":
    main()
