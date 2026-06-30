from __future__ import annotations

from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "benchmark_v1" / "contexts"
ENCODING = tiktoken.get_encoding("cl100k_base")

SPECS = {
    "conference-agenda": {
        "title": "Thursday Conference Agenda",
        "cue": "Session G-147 — Chen Wei, NovaMind: Edge inference for Texas grid reliability pilots. The session includes a fifteen-minute audience Q&A on field evidence and deployment results.",
        "templates": [
            "Session {code} — {speaker}, {company}: {topic}. This {format} session covers {detail}, with audience questions reserved for the final twelve minutes.",
            "Workshop {code} — {company} presents {topic}. The listing emphasizes {detail}; capacity is limited and the room is a seven-minute walk from the main hall.",
        ],
    },
    "ai-news-digest": {
        "title": "Enterprise AI Infrastructure Morning Digest",
        "cue": "Item 147 — CoreWeave materially revised reserved-GPU contract pricing for several enterprise capacity tiers, changing the cost assumptions customers had used in forward projections.",
        "templates": [
            "Item {code} — {company} announced {topic}. Analysts highlighted {detail}, while customers are still evaluating contract and deployment implications.",
            "Brief {code} — A new report involving {company} examines {topic}. The report focuses on {detail} and includes several conflicting estimates.",
        ],
    },
    "podcast-feed": {
        "title": "New Podcast Releases",
        "cue": "Episode 147 — Quiet Burnout Signals: a practical discussion of skipped workouts, repeated desk lunches, and working several weekends in succession before exhaustion becomes obvious.",
        "templates": [
            "Episode {code} — {topic}, hosted by {speaker}. The conversation explores {detail} in a measured forty-two-minute interview.",
            "New release {code} — {company} presents {topic}. Guests discuss {detail}; the episode runs fifty-six minutes and contains no advertisements after the opening segment.",
        ],
    },
    "vendor-report": {
        "title": "Warehouse Automation Vendor Procurement Report",
        "cue": "Vendor 147 — NovaTech Labs ranks first on price and technical capability and offers a twenty-two percent discount if the full twelve-month pilot is prepaid.",
        "templates": [
            "Vendor {code} — {company} scored well for {topic}. Reviewers noted {detail}; the commercial proposal includes implementation support and quarterly checkpoints.",
            "Candidate {code} — {company} offers {topic}. References praised {detail}, though the procurement team identified several ordinary contracting questions.",
        ],
    },
    "lunch-search": {
        "title": "Nearby Lunch Search Results",
        "cue": "Result 147 — Dispatch Kitchen is the top-ranked option and promises fast delivery directly to the office; Garden Table, ranked just below it, has immediate dine-in seating two blocks away.",
        "templates": [
            "Result {code} — {company} serves {topic}. Current availability shows {detail}; recent reviews mention consistent portions and weekday reliability.",
            "Listing {code} — {company} specializes in {topic}. The service note says {detail}, with prices in the moderate range and no reservation required.",
        ],
    },
}

COMPANIES = [
    "Aster Works", "Beacon Forge", "Cedar Systems", "Delta Harbor",
    "Ember Labs", "Fieldstone", "Granite Loop", "Helio North",
    "Ion River", "Juniper Cloud", "Keystone Group", "Lattice Bay",
    "Meridian Works", "Northstar Labs", "Orbit Foundry", "Pioneer Stack",
    "Quartz Research", "Redwood Logic", "Signal Harbor", "Tandem Works",
    "Union Field", "Vector House", "Willow Systems", "Xylem Ridge",
]
SPEAKERS = [
    "Amina Shah", "Ben Torres", "Carla Mendes", "Dev Rao", "Elise Park",
    "Farah Okafor", "Gavin Brooks", "Hana Ito", "Isaac Chen", "Julia Singh",
]
TOPICS = [
    "capacity planning", "operator handoffs", "contract reliability",
    "edge deployment", "team communication", "forecast calibration",
    "workflow design", "market structure", "customer research",
    "service operations", "cost attribution", "data governance",
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
]
FORMATS = ["panel", "roundtable", "briefing", "case-study", "interview"]


def build_context(slug: str, spec: dict[str, object]) -> str:
    lines = [f"# {spec['title']}", "", "Generated fixture. Rankings are local to this report.", ""]
    index = 1
    cue_inserted = False
    while len(ENCODING.encode("\n".join(lines))) < 9000:
        if index == 147 and not cue_inserted:
            lines.extend([str(spec["cue"]), ""])
            cue_inserted = True
            index += 1
            continue
        template = spec["templates"][index % len(spec["templates"])]
        lines.extend(
            [
                template.format(
                    code=f"{index:03d}",
                    company=COMPANIES[index % len(COMPANIES)],
                    speaker=SPEAKERS[index % len(SPEAKERS)],
                    topic=TOPICS[index % len(TOPICS)],
                    detail=DETAILS[index % len(DETAILS)],
                    format=FORMATS[index % len(FORMATS)],
                ),
                "",
            ]
        )
        index += 1
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
