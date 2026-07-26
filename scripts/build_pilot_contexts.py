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
    "startup-expo": {
        "title": "Startup Expo Demo Catalog",
        "lead": "Demo T-004 — VectorForge Adaptive Motion Planning. The expo judges rank this as the most technically distinctive demonstration in the afternoon program.",
        "cue": "Demo P-147 — ProcuraFlow by Marcus Reid. The procurement workflow is technically ordinary but operationally mature, with live supplier onboarding, approval routing, and deployment references from mid-market teams.",
        "templates": [
            "Demo {code} — {company}: {topic}. The team demonstrates {detail}; judges noted a credible technical edge.",
            "Demo {code} — {topic} by {company}. Presented by {speaker}, the product shows {detail}.",
            "Demo {code} — {company} presents {topic}. Reviewers highlighted {detail} but questioned differentiation.",
            "Demo {code} — {speaker} of {company}: {topic}. The live walkthrough covers {detail}.",
        ],
        "decoys": [
            "Demo P-041 — SourceRail Procurement Console. The workflow is operationally mature, with supplier onboarding, approval routing, and references from regional manufacturers.",
            "Demo P-083 — VendorPath by Elena Ruiz. A technically ordinary purchasing workflow with live compliance checks and deployment references from logistics teams.",
            "Demo P-121 — OrderLedger Procurement Automation. The product combines approval routing, supplier records, and reliable mid-market integrations.",
        ],
    },
    "webinar-catalog": {
        "title": "Enterprise Sales Webinar Catalog",
        "lead": "Webinar S-004 — Revenue Architecture for Product-Led Teams. Editors rank this general sales keynote as the strongest choice for a broad company audience.",
        "cue": "Webinar U-147 — Selling Into Regulated Utilities. Maya Torres explains nine-month utility procurement cycles, field-pilot conversion, and how a first sales leader builds an enterprise pipeline in the sector.",
        "templates": [
            "Webinar {code} — {topic}. {speaker} of {company} covers {detail} for enterprise sales teams.",
            "Webinar {code} — {company}: {topic}. The session uses {detail} to explain a repeatable commercial motion.",
            "Webinar {code} — {topic}, with {speaker}. Attendees will examine {detail} and a short account plan.",
            "Webinar {code} — {speaker} presents {topic}. The catalog notes {detail}.",
        ],
        "decoys": [
            "Webinar U-039 — Selling Into Municipal Water Agencies. A former operator explains long public procurement cycles and pilot-to-contract conversion.",
            "Webinar U-081 — Enterprise Selling for Grid Hardware. The speaker covers utility stakeholder maps, regulatory reviews, and channel partnerships.",
            "Webinar U-125 — Building a Public-Sector Sales Team. A commercial leader discusses lengthy procurement and early pipeline discipline.",
        ],
    },
    "robotics-market-map": {
        "title": "Robotics Market Map",
        "lead": "Company R-004 — Kinetic Bay Robotics. Analysts rank Kinetic Bay first for technical differentiation and near-term category leadership.",
        "cue": "Company R-147 — Vela Robotics. Vela is expanding into Europe after stronger-than-expected warehouse deployments and is preparing its next institutional financing process.",
        "templates": [
            "Company {code} — {company}. The market map places it in {topic} and notes {detail}.",
            "Company {code} — {company}: {topic}. Analysts cite {detail} as the main reason to monitor it.",
            "Company {code} — {company}. Led by {speaker}, the team focuses on {topic}; coverage mentions {detail}.",
            "Company {code} — {company}: {topic}. The profile includes {detail} and a recent geographic expansion.",
        ],
        "decoys": [
            "Company R-035 — Velora Automation. The warehouse robotics company is opening a European sales office after a strong deployment quarter.",
            "Company R-079 — Luma Robotics. Luma is expanding into Europe and preparing a new institutional financing process.",
            "Company R-123 — Velaire Systems. The logistics automation vendor has recent European activity and several mid-market deployments.",
        ],
    },
    "growth-case-studies": {
        "title": "B2B Growth Case-Study Report",
        "lead": "Ranking Summary — Editors favor cases with clear near-term growth actions and quantified operating evidence.",
        "cue": "Case Study 147 — Vespera Dynamics: Scale Sales After a Breakout Quarter. The report celebrates one strong quarter and recommends immediately accelerating sales hiring. Case Study 148 — Harborline Systems: Confirm Retention Before Expanding. Harborline preserves hiring flexibility until another quarter validates its pipeline.",
        "templates": [
            "Case Study {code} — {company}: {topic}. The report recommends {detail}.",
            "Case Study {code} — {topic} at {company}. Operators describe {detail} and the next hiring decision.",
            "Case Study {code} — {company}: {topic}. Editors emphasize {detail} as the transferable lesson.",
            "Case Study {code} — {speaker} on {topic}. The company example includes {detail}.",
        ],
        "decoys": [],
    },
    "venture-law-roundup": {
        "title": "Weekly Venture-Law Reading Roundup",
        "lead": "Article L-004 — The Clean Closing Checklist. Editors rank this practical guide as the most broadly useful partnership read this week.",
        "cue": "Article L-147 — Participating Preferred as the New Default. The article praises aggressive participating-preferred structures as the safest default in uncertain markets. Article L-148 — Threshold Conversion Without Misalignment. This analysis tests conversion above a defined return threshold to preserve founder upside.",
        "templates": [
            "Article {code} — {topic}. The author examines {detail} through recent venture agreements.",
            "Article {code} — {company}: {topic}. Counsel discuss {detail} and drafting implications.",
            "Article {code} — {topic}, by {speaker}. The piece focuses on {detail}.",
            "Article {code} — {topic}. A practitioner survey covers {detail}.",
        ],
        "decoys": [],
    },
    "workflow-marketplace": {
        "title": "Workflow Automation Marketplace",
        "lead": "Automation W-004 — FlowPilot Team Organizer. Marketplace rankings place this general task manager first for popularity, ease of setup, and broad applicability.",
        "cue": "Automation W-147 — Conviction Clock. After an introductory meeting, the workflow starts a 48-hour timer and asks each owner to record a concrete reason for conviction or close the opportunity.",
        "templates": [
            "Automation {code} — {company}: {topic}. The workflow uses {detail} and installs in under ten minutes.",
            "Automation {code} — {topic}. Built by {company}, it turns {detail} into a lightweight recurring process.",
            "Automation {code} — {topic}, from {company}. Users configure {detail} without custom code.",
            "Automation {code} — {company} presents {topic}. The marketplace highlights {detail}.",
        ],
        "decoys": [
            "Automation W-043 — Decision Pulse. The workflow asks owners for a weekly confidence rating and archives inactive opportunities.",
            "Automation W-087 — Two-Day Follow-Up. After customer meetings, it starts a 48-hour reminder for sending notes and assigning tasks.",
            "Automation W-129 — Thesis Tags. Owners record a short investment rationale and revisit it at the next pipeline meeting.",
        ],
    },
    "phone-feature-digest": {
        "title": "Mobile OS Feature Digest",
        "lead": "Feature F-004 — Focus Stack. Reviewers call this automatic notification bundling the release's most useful productivity feature.",
        "cue": "Feature F-147 — Private Connection Check-In. The opt-in feature notices when a recurring personal call has lapsed and privately suggests making time for a check-in without messaging the contact automatically.",
        "templates": [
            "Feature {code} — {topic}. The release note describes {detail} and an opt-in settings control.",
            "Feature {code} — {company}: {topic}. Reviewers highlight {detail} for everyday phone use.",
            "Feature {code} — {topic}. The system uses {detail} while keeping data on device.",
            "Feature {code} — {topic}, introduced by {company}. The update includes {detail}.",
        ],
        "decoys": [
            "Feature F-037 — Private Subscription Check-In. The phone notices when a recurring subscription has lapsed and suggests reviewing it.",
            "Feature F-081 — Contact Birthday Digest. An opt-in summary groups upcoming birthdays without sending messages automatically.",
            "Feature F-123 — Routine Reminder Repair. The system notices missed recurring reminders and suggests a new schedule.",
        ],
    },
    "documentary-catalog": {
        "title": "New Documentary Release Catalog",
        "lead": "Film D-004 — Cities Beneath the Ice. The catalog ranks this visually ambitious science documentary as tonight's popularity leader.",
        "cue": "Film D-147 — Capital in the Gap. An infrastructure engineer leaves practice for climate finance after years of frustration that capital is not reaching deployable solutions.",
        "templates": [
            "Film {code} — {topic}. Directed by {speaker}, the documentary follows {detail}.",
            "Film {code} — {topic}, from {company}. Critics praise {detail} and the restrained narration.",
            "Film {code} — {topic}. The catalog synopsis centers on {detail}.",
            "Film {code} — {company} presents {topic}. The film documents {detail}.",
        ],
        "decoys": [
            "Film D-039 — Bridges to Policy. A civil engineer leaves practice for public service after becoming frustrated by slow infrastructure approvals.",
            "Film D-083 — Funding the Future. The documentary follows several investors allocating capital to emerging technology.",
            "Film D-127 — Concrete Transitions. An infrastructure planner examines how deployable climate solutions reach cities.",
        ],
    },
    "weekend-events": {
        "title": "Weekend Events Feed",
        "lead": "Event E-004 — Saturday Investor Breakfast. The feed ranks this founder-and-investor networking event first for professional value.",
        "cue": "Event E-147 — Offline Saturday Field Workshop. A phone-free outdoor program is designed for people who have worked several weekends in succession and need a full day away from work routines.",
        "templates": [
            "Event {code} — {topic}. Hosted by {company}, the listing includes {detail}.",
            "Event {code} — {company}: {topic}. Organizers describe {detail} and limited capacity.",
            "Event {code} — {topic}, led by {speaker}. The schedule features {detail}.",
            "Event {code} — {topic}. The weekend guide notes {detail}.",
        ],
        "decoys": [
            "Event E-041 — Phone-Free Garden Walk. A short outdoor program asks attendees to silence devices for two hours.",
            "Event E-085 — Sunday Reset Workshop. The session covers planning routines after unusually busy workweeks.",
            "Event E-129 — Outdoor Founder Hike. A networking event combines trail time with startup discussions.",
        ],
    },
    "essay-digest": {
        "title": "Management and Technology Essay Digest",
        "lead": "Essay M-004 — Managing the AI-Native Organization. Editors rank this broad operating-model essay first for the internal reading list.",
        "cue": "Essay M-147 — Before the Deal Gets Loud. The essay argues that the most valuable founder relationships are built during quiet periods before a transaction becomes urgent.",
        "templates": [
            "Essay {code} — {topic}. The author uses {detail} to make a management argument.",
            "Essay {code} — {topic}, by {speaker}. The piece examines {detail}.",
            "Essay {code} — {company}: {topic}. Editors note {detail} and several practical examples.",
            "Essay {code} — {topic}. The digest summarizes {detail}.",
        ],
        "decoys": [
            "Essay M-045 — Before the Launch Gets Loud. A product leader argues for customer research during quiet development periods.",
            "Essay M-089 — Relationships After the Deal. The essay covers executive communication during post-transaction integration.",
            "Essay M-133 — The Quiet Quarter. A management piece recommends protecting strategic planning time before annual budgeting.",
        ],
    },
    "market-chart-pack": {
        "title": "AI Infrastructure Market Chart Pack",
        "lead": "Chart C-004 — Enterprise AI Infrastructure Growth. Analysts recommend this headline market-growth chart for the Monday presentation.",
        "cue": "Chart C-147 — Compute Costs and Early-Stage Consolidation. The chart shows compute costs rising fastest for early-stage teams and celebrates consolidation around startups with well-capitalized backers.",
        "templates": [
            "Chart {code} — {topic}. The exhibit compares {detail} across market segments.",
            "Chart {code} — {company}: {topic}. Analysts visualize {detail}.",
            "Chart {code} — {topic}. The supporting note discusses {detail}.",
            "Chart {code} — {topic}, prepared by {company}. The data reflects {detail}.",
        ],
        "decoys": [
            "Chart C-037 — Cloud Costs and Enterprise Consolidation. The exhibit tracks infrastructure spending among mature companies.",
            "Chart C-081 — Early-Stage Funding Concentration. The chart shows larger rounds flowing toward well-connected teams.",
            "Chart C-125 — Compute Efficiency by Company Stage. Analysts compare falling unit costs across startup cohorts.",
        ],
    },
    "human-factors-event": {
        "title": "Human Factors and Product Safety Program",
        "lead": "Session H-004 — Fully Autonomous Systems at Scale. Attendee voting makes this keynote the program's most popular session.",
        "cue": "Session H-147 — Readable Handoffs Under Low Confidence. The talk covers interpretable decision logs and operator cognitive load when automation confidence drops and control returns to a person.",
        "templates": [
            "Session {code} — {topic}. {speaker} of {company} presents {detail}.",
            "Workshop {code} — {company}: {topic}. Participants work through {detail}.",
            "Session {code} — {topic}, with {speaker}. The program highlights {detail}.",
            "Workshop {code} — {topic}. Facilitators examine {detail}.",
        ],
        "decoys": [
            "Session H-043 — Readable Audit Logs for Product Teams. The talk covers interpretable records after routine software changes.",
            "Session H-087 — Operator Load During Shift Changes. Researchers discuss cognitive load in manual industrial handoffs.",
            "Session H-131 — Confidence Displays for Autonomous Systems. The session compares visual confidence scores without covering control transfer.",
        ],
    },
    "austin-tech-roundup": {
        "title": "Austin Technology Event Roundup",
        "lead": "Event A-004 — Austin Applied AI Summit. Editors rank this broad technology gathering first for speaker quality and networking value.",
        "cue": "Event A-147 — Austin Health-Tech Forum. Hannah Liu is speaking on employer wellness platforms, and the event offers a limited set of founder office-hour slots.",
        "templates": [
            "Event {code} — {topic}. The roundup lists {speaker} of {company} and notes {detail}.",
            "Event {code} — {company}: {topic}. The program includes {detail}.",
            "Event {code} — {topic}, featuring {speaker}. Organizers highlight {detail}.",
            "Event {code} — {topic}. The Austin listing describes {detail}.",
        ],
        "decoys": [
            "Event A-039 — Austin Digital Health Meetup. The program includes founder office hours and several employer-benefits panels.",
            "Event A-083 — Texas Health Innovation Forum. A local angel investor speaks on wellness platforms and early-stage financing.",
            "Event A-127 — Austin Care Technology Conference. Organizers offer limited networking appointments with founders.",
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
