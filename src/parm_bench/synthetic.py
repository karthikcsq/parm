from __future__ import annotations

from typing import Any


FAMILIES = [
    {
        "goal_family": "introductions",
        "actionability": "intro",
        "target_type": "company",
        "bridge_type": "organization",
        "mid_type": "group",
        "goal": "I am looking for Midwest robotics startups that might be worth contacting for partnership conversations. Identify promising leads and draft practical next steps.",
        "targets": ["Acme Robotics", "Brightpath Labs", "Copperline AI", "Daybreak Systems", "Evergreen Works"],
        "bridges": ["High Alpha", "Northstar Labs", "Canal Ventures", "Signal Foundry", "Atlas Studio"],
        "trigger_templates": [
            "{target} is a strong lead because it was spun out of {bridge}.",
            "{target} is a strong lead because it lists {bridge} as its founding partner.",
        ],
        "memory_template": "{person}, your entrepreneurship professor, previously worked at {bridge}.",
        "suggestion_template": (
            "{target} connects to {person} through {bridge}; ask {person} for an intro before cold outreach."
        ),
        "action_keywords": ["intro", "ask"],
    },
    {
        "goal_family": "customer_discovery",
        "actionability": "prioritization",
        "target_type": "company",
        "bridge_type": "product",
        "mid_type": "customer_segment",
        "goal": "I need to choose which healthcare operations startups to interview first for customer discovery. Rank the targets and explain the outreach angle.",
        "targets": ["ClinicFlow", "OpsPilot", "Rosterly", "CareLedger", "Deskwise"],
        "bridges": ["Epic Connect", "Stripe Billing", "Salesforce Health Cloud", "Zendesk Suite", "HubSpot Ops"],
        "trigger_templates": [
            "{target} sells into clinics that recently adopted {bridge}.",
            "{target} is expanding around teams using {bridge}.",
        ],
        "memory_template": "{person} led a buying committee for {bridge} last year.",
        "suggestion_template": (
            "Prioritize {target}: {person}'s {bridge} buying experience can sharpen discovery questions."
        ),
        "action_keywords": ["prioritize", "discovery"],
    },
    {
        "goal_family": "opportunity_risk",
        "actionability": "warning",
        "target_type": "opportunity",
        "bridge_type": "platform",
        "mid_type": "dependency",
        "goal": "Review these implementation opportunities and recommend which one deserves deeper diligence this week.",
        "targets": ["Remote Scribe Rollout", "Campus Kiosk Pilot", "Inventory Sync Project", "Alumni Portal Migration", "Payments Rewrite"],
        "bridges": ["Okta", "Workday", "NetSuite", "ServiceNow", "Snowflake"],
        "trigger_templates": [
            "{target} depends heavily on integrations with {bridge}.",
            "{target}'s pitch emphasizes a partnership with {bridge}.",
        ],
        "memory_template": "{person} noted that {bridge} had procurement and reliability concerns.",
        "suggestion_template": (
            "Flag a warning for {target}: {person}'s note about {bridge} suggests asking about reliability risk."
        ),
        "action_keywords": ["warning", "risk", "asking"],
    },
    {
        "goal_family": "travel_planning",
        "actionability": "warning",
        "target_type": "place",
        "bridge_type": "venue",
        "mid_type": "neighborhood",
        "goal": "I am visiting Chicago next Thursday evening and need a complete dinner, hotel, and transit itinerary that keeps the hotel under $250.",
        "targets": ["Maple House Stay", "Lakeside Market Walk", "Old Town Dinner", "Riverfront Hotel", "Station Quarter Tour"],
        "bridges": ["River North", "Lincoln Park", "The Loop", "West Loop", "Wicker Park"],
        "trigger_templates": [
            "{target} is closest to {bridge}.",
            "The suggested itinerary centers on {bridge} near {target}.",
        ],
        "memory_template": "{person} told you {bridge} is difficult to navigate after 9 PM.",
        "suggestion_template": (
            "Avoid choosing {target} as-is: {person}'s note about {bridge} suggests changing timing or transit plans."
        ),
        "action_keywords": ["avoid", "timing", "transit"],
    },
    {
        "goal_family": "health_admin",
        "actionability": "follow_up_question",
        "target_type": "provider",
        "bridge_type": "policy",
        "mid_type": "coverage_group",
        "goal": "Compare local care options for a routine visit and recommend which provider to call first.",
        "targets": ["North Clinic", "Cedar Dental", "Riverbend PT", "Summit Vision", "Oak Family Care"],
        "bridges": ["BlueCare Silver", "HSA Basic", "Delta Preferred", "VisionPlus", "CareFirst PPO"],
        "trigger_templates": [
            "{target} accepts {bridge}.",
            "The directory marks {target} as in-network for {bridge}.",
        ],
        "memory_template": "{person} previously found that {bridge} required prior authorization for routine visits.",
        "suggestion_template": (
            "Ask a follow-up question before choosing {target}: {person}'s {bridge} experience suggests checking prior authorization."
        ),
        "action_keywords": ["follow-up", "checking", "authorization"],
    },
    {
        "goal_family": "learning_research",
        "actionability": "enrichment",
        "target_type": "paper",
        "bridge_type": "method",
        "mid_type": "research_area",
        "goal": "Help me choose which technical learning resource to read first and give me a useful summary plan.",
        "targets": ["Sparse Recall Notes", "Causal Graph Primer", "Agent Memory Survey", "Temporal KG Tutorial", "Retrieval Eval Paper"],
        "bridges": ["Personalized PageRank", "difference-in-differences", "contrastive learning", "Bayesian updating", "active retrieval"],
        "trigger_templates": [
            "{target} relies on {bridge}.",
            "A central technique in {target} is {bridge}.",
        ],
        "memory_template": "{person} once shared a clear walkthrough of {bridge}.",
        "suggestion_template": (
            "Enrich the review of {target}: revisit {person}'s walkthrough of {bridge} before summarizing it."
        ),
        "action_keywords": ["enrich", "revisit", "summarizing"],
    },
    {
        "goal_family": "event_planning",
        "actionability": "prioritization",
        "target_type": "vendor",
        "bridge_type": "venue",
        "mid_type": "event_constraint",
        "goal": "Plan a 60-person evening event and recommend the best vendor or venue package to pursue.",
        "targets": ["Marigold Catering", "North Hall Booking", "Bright AV Crew", "Stonebridge Florals", "Packet Print Shop"],
        "bridges": ["Founders Hall", "The Annex", "Union Ballroom", "Riverside Loft", "Studio B"],
        "trigger_templates": [
            "{target} is the preferred vendor for {bridge}.",
            "The event package pairs {target} with {bridge}.",
        ],
        "memory_template": "{person} said {bridge} has strict load-in rules that affected a past event.",
        "suggestion_template": (
            "Prioritize logistics checks for {target}: {person}'s {bridge} note means load-in constraints may matter."
        ),
        "action_keywords": ["prioritize", "logistics", "constraints"],
    },
    {
        "goal_family": "hiring",
        "actionability": "follow_up_question",
        "target_type": "candidate",
        "bridge_type": "project",
        "mid_type": "work_history",
        "goal": "Review candidates for a senior operator role and suggest which interview should happen first.",
        "targets": ["Avery Morgan", "Riley Stone", "Casey Nguyen", "Morgan Blake", "Taylor Singh"],
        "bridges": ["Project Lantern", "Atlas Migration", "Mercury Launch", "Northwind Rewrite", "Beacon Analytics"],
        "trigger_templates": [
            "{target} highlights work on {bridge}.",
            "The resume's strongest project is {bridge} by {target}.",
        ],
        "memory_template": "{person} managed a team adjacent to {bridge} and noted the hard parts were stakeholder alignment.",
        "suggestion_template": (
            "Ask {target} a follow-up question about stakeholder alignment on {bridge}; {person}'s memory points to that as the hard part."
        ),
        "action_keywords": ["follow-up", "ask", "stakeholder"],
    },
    {
        "goal_family": "personal_finance",
        "actionability": "warning",
        "target_type": "financial_product",
        "bridge_type": "institution",
        "mid_type": "account_feature",
        "goal": "Compare financial products for travel spending and recommend the best option to investigate further.",
        "targets": ["Summit Cash Account", "Everyday Rewards Card", "FlexSaver CD", "Travel Miles Plus", "Index Starter Plan"],
        "bridges": ["MetroBank", "Cobalt Credit", "Northfield Brokerage", "Pioneer CU", "Harbor Trust"],
        "trigger_templates": [
            "{target} is issued by {bridge}.",
            "The strongest offer comes from {bridge}'s {target}.",
        ],
        "memory_template": "{person} warned that {bridge} had slow dispute resolution.",
        "suggestion_template": (
            "Flag a warning on {target}: {person}'s experience with {bridge} suggests checking dispute-resolution terms."
        ),
        "action_keywords": ["warning", "checking", "terms"],
    },
    {
        "goal_family": "home_ops",
        "actionability": "prioritization",
        "target_type": "service",
        "bridge_type": "equipment",
        "mid_type": "home_system",
        "goal": "Compare household service quotes and recommend which provider to follow up with first.",
        "targets": ["ClearWater Service", "BrightHeat Repair", "Oakline Electric", "PorchLight Install", "BlueRoof Estimate"],
        "bridges": ["Navien NPE-240A", "Trane XR14", "Tesla Powerwall", "Moen Flo", "LiftMaster 87504"],
        "trigger_templates": [
            "{target} specializes in {bridge}.",
            "The quote recommends {bridge} for {target}.",
        ],
        "memory_template": "{person} mentioned that {bridge} needed a certified installer to preserve warranty coverage.",
        "suggestion_template": (
            "Prioritize certification checks for {target}: {person}'s {bridge} note suggests warranty coverage depends on it."
        ),
        "action_keywords": ["prioritize", "certification", "warranty"],
    },
    {
        "goal_family": "general_travel",
        "actionability": "warning",
        "context_requirement": "general_knowledge",
        "synthetic_suffix": False,
        "target_type": "itinerary_item",
        "bridge_type": "landmark",
        "mid_type": "travel_pattern",
        "goal": "Plan a 5-day first-time Paris itinerary focused on museums, food, and walkable neighborhoods.",
        "targets": ["Louvre highlights visit", "Montmartre walking afternoon", "Seine evening walk", "Marais food crawl", "Orsay museum morning"],
        "bridges": ["the Louvre", "Montmartre", "the Seine", "Le Marais", "Musee d'Orsay"],
        "trigger_templates": [
            "{bridge} is a natural anchor for a first-time Paris museum itinerary.",
            "A first-time Paris plan should usually include {bridge}.",
        ],
        "memory_template": "{person} previously said long museum visits over 90 minutes become draining.",
        "suggestion_template": (
            "Reconsider a long {target}: {person}'s museum-fatigue note suggests keeping {bridge} short and pairing it with an outdoor walk."
        ),
        "action_keywords": ["reconsider", "short", "walk"],
    },
    {
        "goal_family": "cooking_planning",
        "actionability": "warning",
        "context_requirement": "general_knowledge",
        "synthetic_suffix": False,
        "target_type": "dish",
        "bridge_type": "dish_style",
        "mid_type": "meal_constraint",
        "goal": "Plan a vegetarian Italian dinner-party menu for eight people.",
        "targets": ["mushroom risotto", "eggplant involtini", "baked ziti", "white bean ribollita", "spinach lasagna"],
        "bridges": ["risotto", "eggplant involtini", "baked pasta", "ribollita", "lasagna"],
        "trigger_templates": [
            "{bridge} is a classic vegetarian Italian main course.",
            "{target} would be a familiar Italian centerpiece for a vegetarian dinner.",
        ],
        "memory_template": "{person} mentioned that {bridge} sits too heavy at dinner parties.",
        "suggestion_template": (
            "Avoid making {target} the only main: {person}'s note about {bridge} suggests offering a lighter alternative."
        ),
        "action_keywords": ["avoid", "alternative", "lighter"],
    },
    {
        "goal_family": "fitness_planning",
        "actionability": "warning",
        "context_requirement": "general_knowledge",
        "synthetic_suffix": False,
        "target_type": "exercise",
        "bridge_type": "movement",
        "mid_type": "body_constraint",
        "goal": "Create a beginner-friendly 4-week strength plan for someone returning to exercise.",
        "targets": ["bodyweight squat progression", "reverse lunge progression", "step-up progression", "hinge practice", "split-squat progression"],
        "bridges": ["squats", "lunges", "step-ups", "hip hinges", "split squats"],
        "trigger_templates": [
            "{bridge} are a common beginner lower-body strength movement.",
            "{target} builds {bridge}, a standard beginner lower-body strength-plan component.",
        ],
        "memory_template": "{person} previously mentioned knee irritation from deep {bridge}.",
        "suggestion_template": (
            "Modify {target}: {person}'s note about knee irritation from {bridge} suggests box squats or a shallower range first."
        ),
        "action_keywords": ["modify", "knee", "range"],
    },
    {
        "goal_family": "writing_help",
        "actionability": "enrichment",
        "context_requirement": "general_knowledge",
        "synthetic_suffix": False,
        "target_type": "essay_section",
        "bridge_type": "argument",
        "mid_type": "writing_feedback",
        "goal": "Help me outline a persuasive essay about remote work for a college writing class.",
        "targets": ["work-life balance section", "productivity section", "collaboration section", "accessibility section", "office-culture counterargument"],
        "bridges": ["work-life balance", "productivity", "collaboration", "accessibility", "office culture"],
        "trigger_templates": [
            "{bridge} is a common argument in remote-work essays.",
            "A persuasive remote-work outline can include a section on {bridge}.",
        ],
        "memory_template": "{person} previously pushed back when your essay used generic claims about {bridge}.",
        "suggestion_template": (
            "Strengthen the {target}: {person}'s feedback about generic {bridge} claims suggests using a concrete mechanism or example."
        ),
        "action_keywords": ["strengthen", "concrete", "example"],
    },
    {
        "goal_family": "interview_prep",
        "actionability": "enrichment",
        "context_requirement": "general_knowledge",
        "synthetic_suffix": False,
        "target_type": "practice_block",
        "bridge_type": "interview_topic",
        "mid_type": "skill_gap",
        "goal": "Help me prepare for a product manager interview at a B2B SaaS company.",
        "targets": ["prioritization practice block", "customer-discovery practice block", "metrics practice block", "roadmap tradeoff drill", "stakeholder communication drill"],
        "bridges": ["prioritization", "customer discovery", "product metrics", "roadmap tradeoffs", "stakeholder communication"],
        "trigger_templates": [
            "{bridge} is a likely product-manager interview topic.",
            "A B2B SaaS product-manager interview should include practice on {bridge}.",
        ],
        "memory_template": "{person} noted that your {bridge} answers were strongest when you explicitly named tradeoffs.",
        "suggestion_template": (
            "Upgrade the {target}: {person}'s note about {bridge} suggests practicing with a rubric and explicitly naming what you would not build."
        ),
        "action_keywords": ["upgrade", "rubric", "tradeoffs"],
    },
]

DISTRACTOR_TYPES = [
    "semantic",
    "graph_proximity",
    "stale_invalid_edge",
    "goal_irrelevant",
]

JOIN_DEPTHS = [1, 2, 3]

PEOPLE = [
    "Professor Iyer",
    "Maya Chen",
    "Owen Patel",
    "Dr. Rivera",
    "Sam Okafor",
    "Leah Brooks",
    "Nina Shah",
    "Evan Miller",
    "Priya Raman",
    "Jordan Lee",
]


def generate_cases(count: int = 50) -> list[dict[str, Any]]:
    return [build_case(index) for index in range(count)]


def build_case(index: int) -> dict[str, Any]:
    family = FAMILIES[index % len(FAMILIES)]
    join_depth = JOIN_DEPTHS[index % len(JOIN_DEPTHS)]
    context_requirement = family.get("context_requirement", "tool")
    trigger_source = "generated_output" if context_requirement == "general_knowledge" else "tool_response"
    distractor_type = DISTRACTOR_TYPES[index % len(DISTRACTOR_TYPES)]

    target_base = family["targets"][index % len(family["targets"])]
    target = target_base if family.get("synthetic_suffix", True) is False else f"{target_base} {100 + index}"
    bridge = family["bridges"][index % len(family["bridges"])]
    person = PEOPLE[index % len(PEOPLE)]
    case_id = f"parm-v1-{index + 1:03d}"

    target_id = "n_item"
    bridge_id = "n_trigger"
    memory_id = "n_memory"
    person_id = "n_person"
    mid_id = "n_mid"

    nodes = [
        {"id": target_id, "label": target, "type": family["target_type"]},
        {"id": bridge_id, "label": bridge, "type": family["bridge_type"]},
        {"id": person_id, "label": person, "type": "person"},
        {"id": memory_id, "label": family["memory_template"].format(person=person, bridge=bridge), "type": "memory"},
    ]
    edges = [_edge("e_context", target_id, bridge_id, "output_mentions_cue")]
    if join_depth == 1:
        gold_path = [bridge_id, memory_id]
        edges.append(_edge("e_gold_1", bridge_id, memory_id, "direct_memory_link"))
    elif join_depth == 2:
        gold_path = [bridge_id, mid_id, memory_id]
        nodes.append({"id": mid_id, "label": f"{bridge} related context", "type": family["mid_type"]})
        edges.extend(
            [
                _edge("e_gold_1", bridge_id, mid_id, "has_related_context"),
                _edge("e_gold_2", mid_id, memory_id, "appears_in_memory"),
            ]
        )
    else:
        nodes.append({"id": mid_id, "label": f"{bridge} related context", "type": family["mid_type"]})
        gold_path = [bridge_id, mid_id, person_id, memory_id]
        edges.extend(
            [
                _edge("e_gold_1", bridge_id, mid_id, "has_related_context"),
                _edge("e_gold_2", mid_id, person_id, "mentions_person"),
                _edge("e_gold_3", person_id, memory_id, "appears_in_memory"),
            ]
        )

    nodes, edges, distractors = _add_distractors(nodes, edges, distractor_type, target_id, bridge_id, index)

    trigger_sentence = family["trigger_templates"][index % len(family["trigger_templates"])].format(
        target=target,
        bridge=bridge,
    )
    context_summary, objective_recommendation = _build_decision_context(
        family,
        index,
        target,
        bridge,
        trigger_sentence,
    )
    assistant_output = (
        objective_recommendation
        if trigger_source == "generated_output"
        else "I will compare the available options after checking current context."
    )
    tool_response = (
        {
            "summary": "No tool response required; cue is general model knowledge.",
            "source": "not_required",
        }
        if context_requirement == "general_knowledge"
        else {"summary": context_summary, "source": "synthetic_local_context"}
    )

    user_goal = family["goal"].format(target=target, bridge=bridge)
    suggestion = family["suggestion_template"].format(target=target, person=person, bridge=bridge)
    return {
        "case_id": case_id,
        "goal_family": family["goal_family"],
        "trigger_source": trigger_source,
        "join_depth": join_depth,
        "distractor_type": distractor_type,
        "actionability": family["actionability"],
        "context_requirement": context_requirement,
        "user_goal": user_goal,
        "assistant_output": assistant_output,
        "tool_response": tool_response,
        "graph": {"nodes": nodes, "edges": edges},
        "trigger_entity": {"node_id": bridge_id, "label": bridge},
        "distractors": distractors,
        "expected": {
            "suggestion": suggestion,
            "decision_effect": _decision_effect(family, target_id),
            "gold_path": gold_path,
            "memory_fact_node_id": memory_id,
            "action_keywords": family["action_keywords"],
        },
    }


def _build_decision_context(
    family: dict[str, Any],
    index: int,
    target: str,
    bridge: str,
    trigger_sentence: str,
) -> tuple[str, str]:
    alternatives = _alternative_options(family, index)
    goal_family = family["goal_family"]

    if goal_family == "travel_planning":
        context = (
            "Current options for next Thursday evening in Chicago:\n"
            f"- {target}: $229 hotel near {bridge}, easiest dinner-to-hotel routing, 18-minute ride from the station.\n"
            f"- {alternatives[0][0]}: $241 hotel near {alternatives[0][1]}, better museums but weaker dinner fit.\n"
            f"- {alternatives[1][0]}: $198 hotel near {alternatives[1][1]}, cheaper but adds two transfers after dinner.\n"
            f"Best objective fit: {target} because it best balances price, dinner options, and transit convenience."
        )
        decision = (
            f"Objective itinerary recommendation: choose {target}. {trigger_sentence} It is under the $250 budget "
            "and gives the cleanest dinner-to-hotel routing."
        )
    elif goal_family == "introductions":
        context = (
            "Partnership lead scan:\n"
            f"- {target}: strongest fit; {trigger_sentence}\n"
            f"- {alternatives[0][0]}: promising product signal, but weaker partnership evidence.\n"
            f"- {alternatives[1][0]}: interesting category fit, but outreach path is less clear.\n"
            f"Best objective fit: start with {target}."
        )
        decision = f"Objective outreach recommendation: start with {target}. {trigger_sentence}"
    elif goal_family == "customer_discovery":
        context = (
            "Customer discovery target scan:\n"
            f"- {target}: highest-priority interview; {trigger_sentence}\n"
            f"- {alternatives[0][0]}: relevant buyer segment, but less urgent signal.\n"
            f"- {alternatives[1][0]}: useful later, but current evidence is thinner.\n"
            f"Best objective fit: interview {target} first."
        )
        decision = f"Objective discovery recommendation: interview {target} first. {trigger_sentence}"
    elif goal_family == "opportunity_risk":
        context = (
            "Implementation opportunity scan:\n"
            f"- {target}: highest diligence priority; {trigger_sentence}\n"
            f"- {alternatives[0][0]}: smaller scope and lower near-term upside.\n"
            f"- {alternatives[1][0]}: interesting, but requirements are less concrete.\n"
            f"Best objective fit: diligence {target} first."
        )
        decision = f"Objective diligence recommendation: review {target} first. {trigger_sentence}"
    elif goal_family == "health_admin":
        context = (
            "Care option scan:\n"
            f"- {target}: best first call; {trigger_sentence}\n"
            f"- {alternatives[0][0]}: longer appointment wait.\n"
            f"- {alternatives[1][0]}: convenient location, but weaker coverage match.\n"
            f"Best objective fit: call {target} first."
        )
        decision = f"Objective care recommendation: call {target} first. {trigger_sentence}"
    elif goal_family == "learning_research":
        context = (
            "Learning resource scan:\n"
            f"- {target}: best first read; {trigger_sentence}\n"
            f"- {alternatives[0][0]}: good background, but less aligned with the current question.\n"
            f"- {alternatives[1][0]}: useful later as a follow-up.\n"
            f"Best objective fit: read {target} first."
        )
        decision = f"Objective learning recommendation: read {target} first. {trigger_sentence}"
    elif goal_family == "event_planning":
        context = (
            "Event package scan:\n"
            f"- {target}: best package for a 60-person evening event; {trigger_sentence}\n"
            f"- {alternatives[0][0]}: cheaper, but less complete staffing.\n"
            f"- {alternatives[1][0]}: polished, but less flexible on timing.\n"
            f"Best objective fit: pursue {target}."
        )
        decision = f"Objective event recommendation: pursue {target}. {trigger_sentence}"
    elif goal_family == "hiring":
        context = (
            "Candidate scan:\n"
            f"- {target}: strongest first interview; {trigger_sentence}\n"
            f"- {alternatives[0][0]}: strong execution background, but less relevant scope.\n"
            f"- {alternatives[1][0]}: promising, but fewer senior-operator signals.\n"
            f"Best objective fit: interview {target} first."
        )
        decision = f"Objective hiring recommendation: interview {target} first. {trigger_sentence}"
    elif goal_family == "personal_finance":
        context = (
            "Travel finance option scan:\n"
            f"- {target}: strongest offer; {trigger_sentence}\n"
            f"- {alternatives[0][0]}: lower rewards rate.\n"
            f"- {alternatives[1][0]}: useful benefits, but worse travel redemption fit.\n"
            f"Best objective fit: investigate {target} first."
        )
        decision = f"Objective finance recommendation: investigate {target} first. {trigger_sentence}"
    elif family.get("context_requirement") == "general_knowledge":
        context = "No tool response required; cue is general model knowledge."
        decision = (
            f"General-knowledge recommendation: include {target}. {trigger_sentence} "
            "Then tailor the final answer to the user's stated task."
        )
    else:
        context = (
            "Household service quote scan:\n"
            f"- {target}: best follow-up option; {trigger_sentence}\n"
            f"- {alternatives[0][0]}: lower price, but less complete scope.\n"
            f"- {alternatives[1][0]}: good availability, but weaker equipment fit.\n"
            f"Best objective fit: follow up with {target}."
        )
        decision = f"Objective home-service recommendation: follow up with {target}. {trigger_sentence}"

    return context, decision


def _decision_effect(family: dict[str, Any], target_id: str) -> dict[str, str]:
    if family["actionability"] == "warning":
        return {
            "type": "avoid_or_reconsider_item",
            "item_node_id": target_id,
        }
    return {
        "type": "improve_selected_item",
        "item_node_id": target_id,
    }


def _alternative_options(family: dict[str, Any], index: int) -> list[tuple[str, str]]:
    targets = family["targets"]
    bridges = family["bridges"]
    first = (index + 1) % len(targets)
    second = (index + 2) % len(targets)
    return [
        (f"{targets[first]} {200 + index}", bridges[first]),
        (f"{targets[second]} {300 + index}", bridges[second]),
    ]


def _add_distractors(
    nodes: list[dict[str, str]],
    edges: list[dict[str, Any]],
    distractor_type: str,
    target_id: str,
    bridge_id: str,
    index: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[str]]:
    distractor_id = "n_distractor"
    stale_id = "n_stale"
    irrelevant_id = "n_irrelevant"
    distractors = [distractor_id]

    if distractor_type == "semantic":
        nodes.append({"id": distractor_id, "label": f"Similar-sounding cue {index + 1}", "type": "distractor"})
        edges.append(_edge("e_distractor", bridge_id, distractor_id, "similar_name", valid=False))
    elif distractor_type == "graph_proximity":
        nodes.append({"id": distractor_id, "label": "Nearby but non-actionable memory note", "type": "memory"})
        edges.append(_edge("e_distractor", bridge_id, distractor_id, "nearby_note"))
    elif distractor_type == "stale_invalid_edge":
        nodes.append({"id": stale_id, "label": "Expired advisor relationship", "type": "memory"})
        edges.append(_edge("e_distractor", bridge_id, stale_id, "former_advisor", valid=False))
        distractors = [stale_id]
    else:
        nodes.append({"id": irrelevant_id, "label": "Interesting but unrelated personal note", "type": "memory"})
        edges.append(_edge("e_distractor", bridge_id, irrelevant_id, "same_industry"))
        distractors = [irrelevant_id]

    return nodes, edges, distractors


def _edge(
    edge_id: str,
    source: str,
    target: str,
    relation: str,
    valid: bool = True,
) -> dict[str, Any]:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "relation": relation,
        "valid": valid,
    }
