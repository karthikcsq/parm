"""Observation envelopes and filler vocabulary for the PARMBench v1 builder.

The legacy mixed_v0 generator drew every observation from six fixed templates
with one hard-coded opening line each, so a system could learn the wrapper
instead of the memory. This module supplies fourteen envelope styles, each with
several randomised opening, section, entry, and filler realisations.

Filler prose is assembled from composed slots rather than whole canned
sentences. Every slot value is at most four words and every slot is drawn
independently, so a six-word run of filler is pinned down by at least two
slots and its combination space runs to five figures. That keeps the
`construction_checks` phrase-reuse detector from seeing a template where it
should see a sample.

Nothing here talks to a model. Everything is a pure function of a seeded
`random.Random`, so a scenario replays byte-identically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


# --- name pools -----------------------------------------------------------

NAME_HEADS = (
    "Alder", "Amber", "Ashgrove", "Basalt", "Beacon", "Bramble", "Briar",
    "Cedar", "Clover", "Copper", "Cormorant", "Crescent", "Dovetail",
    "Elmfield", "Ember", "Falcon", "Fenwick", "Foxglove", "Garnet", "Gilder",
    "Harbour", "Hawthorn", "Heron", "Ironwood", "Juniper", "Kestrel",
    "Lantern", "Larkspur", "Linden", "Marlow", "Meridian", "Millrace",
    "Nightjar", "Northgate", "Oakhurst", "Orchard", "Pewter", "Quarry",
    "Redstart", "Rooksby", "Rosemead", "Saltmarsh", "Sandpiper", "Selkirk",
    "Sorrel", "Stonebridge", "Tamarisk", "Thistledown", "Vellum", "Wayfarer",
    "Westmoor", "Whitlow", "Willowbank", "Yardley",
)
NAME_TAILS = (
    "Annexe", "Arcade", "Atrium", "Bay", "Chambers", "Circuit", "Commons",
    "Concourse", "Court", "Crossing", "Cutting", "Dock", "Drift", "Exchange",
    "Foundry", "Gallery", "Gate", "Green", "Grove", "Hollow", "Junction",
    "Landing", "Lodge", "Loft", "Mews", "Mill", "Parade", "Passage",
    "Pavilion", "Quay", "Reach", "Rise", "Row", "Sidings", "Sound", "Steps",
    "Terrace", "Vault", "Verge", "Walk", "Wharf", "Wing", "Yard", "Rooms",
)

# --- composed sentence slots ---------------------------------------------

ROLE_ADJECTIVES = (
    "intake", "duty", "records", "rota", "district", "standards", "billing",
    "triage", "archive", "placement", "moderation", "fulfilment",
    "compliance", "roster", "counter", "outreach", "transport", "training",
)
ROLE_NOUNS = (
    "clerk", "officer", "coordinator", "planner", "lead", "analyst",
    "reviewer", "handler", "aide", "supervisor", "assessor", "liaison",
)
ACTION_VERBS = (
    "logged", "flagged", "cleared", "confirmed", "closed", "reopened",
    "amended", "retired", "escalated", "merged", "withdrew", "extended",
    "archived", "reconciled", "released", "annotated", "deferred", "rebuilt",
    "trimmed", "restored", "reassigned", "adjusted", "published", "paused",
)
ACTION_OBJECTS = (
    "revision", "mismatch", "backlog", "window", "query", "thread", "header",
    "batch", "ticket", "entry", "hold", "deadline", "correction", "draft",
    "count", "slot", "record", "review", "allocation", "summary", "reference",
    "shortlist", "handover", "receipt", "ordering", "notice", "reminder",
)
PLACE_PREPOSITIONS = (
    "on", "against", "in", "under", "across", "within", "beside", "through",
)
PLACE_NOUNS = (
    "routing sheet", "weekly ledger", "handover pack", "standing brief",
    "shared tracker", "intake folder", "original request", "amended schedule",
    "district summary", "shared inbox", "reconciliation tab", "earlier draft",
    "review packet", "counter copy", "field notebook", "temporary code",
    "printed roster", "correspondence log", "archived version",
    "standing agenda", "duplicate form", "referral queue", "prior estimate",
    "circulation list", "interim heading", "carbon copy", "wall planner",
    "second folder", "outgoing tray", "regional index", "paper diary",
    "returns sheet",
)
QUALIFIER_PREPOSITIONS = (
    "before", "after", "despite", "beyond", "without", "following", "pending",
    "during", "since", "alongside", "throughout", "notwithstanding",
)
QUALIFIER_NOUNS = (
    "cutoff", "second reading", "quarterly close", "later check",
    "short delay", "printed totals", "morning post", "weekly close",
    "courier window", "register reopening", "duplicate removal",
    "address correction", "shortened week", "interim review",
    "standing order", "next sitting", "final count", "archive sweep",
    "printed correction", "holiday closure", "counter handover",
    "quiet spell", "regional audit", "postal strike", "revised heading",
    "second courier",
)
# A connector between two composed slots. Every entry is a single word, so a
# connector never forms a low-entropy block of its own, and blanks keep the
# plain adjacency common enough to read naturally. Roughly a third are blank.
CONNECTORS = (
    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
    "again", "briefly", "promptly", "quietly", "twice", "separately",
    "formally", "informally", "provisionally", "partially", "immediately",
    "belatedly", "manually", "jointly", "locally", "centrally", "temporarily",
    "permanently", "quickly", "carefully", "unexpectedly", "additionally",
    "subsequently", "previously", "initially", "finally", "apparently",
    "reportedly", "accordingly", "otherwise", "likewise", "thereafter",
    "elsewhere", "overnight", "meanwhile", "already", "instead",
)

DETAIL_HEADS = (
    "Access", "Availability", "Capacity", "Contact", "Coverage", "Format",
    "Handover", "Hours", "Intake", "Language", "Location", "Notes",
    "Parking", "Payment", "Reference", "Setting", "Staffing", "Status",
    "Terms", "Transport", "Turnaround", "Waiting", "Facilities", "Seating",
    "Booking", "Entrance",
)
DETAIL_LEADS = (
    "step-free entry", "weekday mornings only", "shared with neighbours",
    "letter or telephone", "two eastern districts", "printed on request",
    "same desk as before", "closes early Friday", "walk-in first hour",
    "interpreter on notice", "lower concourse route", "kept with the file",
    "on-street bays", "invoice at month end", "quoted earlier",
    "one undivided room", "two on rotation", "unchanged since review",
    "six-monthly check", "short walk from stop", "within the week",
    "queue at opening", "kettle in the back", "benches, not chairs",
    "side gate open", "loading bay behind", "keys held locally",
    "no fixed rota", "shared stairwell", "counter at the front",
    "small yard attached", "notice board inside", "post collected daily",
    "signage in two scripts", "night bell fitted", "ramp on the left",
    "corner unit", "upper floor only", "gravel approach", "narrow doorway",
)
DETAIL_TAILS = (
    "since the refit", "for now", "as recorded", "per the schedule",
    "unless notified", "on the current form", "under review",
    "with one exception", "in most weeks", "by arrangement",
    "at the usual rate", "with no charge", "on request", "after hours",
    "for regular callers", "in the summer months", "during term",
    "outside peak times", "with prior notice", "when staffed",
    "if space allows", "subject to weather", "for the trial period",
    "until further notice", "on alternate weeks", "with a deposit",
    "under the old code", "beside the entrance", "through the side door",
    "at the shared desk", "in the annexe", "on the ground floor",
    "past the courtyard", "near the loading area", "behind reception",
    "off the main corridor", "by the notice board", "along the back wall",
    "at the far end", "close to the stairs",
)

SECTION_WORDS = (
    "intake", "returns", "amendments", "coverage", "backlog", "handover",
    "referrals", "shortlist", "queries", "logistics", "notices", "holdings",
    "allocations", "clearances", "revisions", "correspondence", "openings",
    "placements",
)
CODE_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"

SPEAKERS = (
    "AH", "BR", "CD", "DK", "EM", "FL", "GN", "HS", "JP", "KT", "LW", "MR",
    "NB", "PV", "RQ", "SD", "TE", "VK", "WC", "YO",
)
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)


@dataclass(frozen=True)
class Envelope:
    """One observation surface: how a document opens, sections, and lists."""

    name: str
    kinds: tuple[str, ...]
    register: str
    openings: tuple[str, ...]
    sections: tuple[str, ...]
    entry: str
    filler_item: str
    noise: str


ENVELOPES: tuple[Envelope, ...] = (
    Envelope(
        name="operations_log",
        kinds=("tool_result",),
        register="a terse internal operations log with dated entries",
        openings=(
            "operations log {code} — {section} desk, week ending {date}",
            "duty log {code}: {section} entries carried to {date}",
            "log extract {code} · {section} · compiled {date}",
            "shift log {code} for {section}, printed {date}",
        ),
        sections=("-- {section} --", "== {section} ==", "[{section}]"),
        entry="{label}\n    {body}",
        filler_item="{label}\n    {body}",
        noise="{sentences}",
    ),
    Envelope(
        name="vendor_sheet",
        kinds=("tool_result",),
        register="a supplier comparison sheet with short factual rows",
        openings=(
            "| supplier sheet {code} | compiled {date} | {section} |",
            "supplier sheet {code} — {section}, refreshed {date}",
            "| sourcing table {code} | {section} | {date} |",
            "sourcing sheet {code}: {section} responses to {date}",
        ),
        sections=("| {section} |", "### {section}"),
        entry="| {label} | {body} |",
        filler_item="| {label} | {body} |",
        noise="{sentences}",
    ),
    Envelope(
        name="forwarded_thread",
        kinds=("assistant_output", "tool_result"),
        register="a forwarded email thread with quoted replies",
        openings=(
            "Fwd: {section} — anything usable here? (thread {code}, {date})",
            "Fwd: re: {section}, pulled together {date} [{code}]",
            "Fwd: {section} shortlist as promised — {date}, ref {code}",
            "Fwd: notes on {section} from {date}, thread {code}",
        ),
        sections=("> --- {section} ---", ">> on {section}:"),
        entry="> {label}\n> {body}",
        filler_item="> {label} — {body}",
        noise="> {sentences}",
    ),
    Envelope(
        name="field_notebook",
        kinds=("assistant_output",),
        register="a working field notebook with unpolished observations",
        openings=(
            "field notebook, {section} visit {date} (page {code})",
            "notebook page {code} — {section}, written up {date}",
            "site notebook: {section}, {date}, entry {code}",
            "walking notes {code}, {section} round, {date}",
        ),
        sections=("~ {section} ~", "// {section}"),
        entry="{label}\n  {body}",
        filler_item="{label} — {body}",
        noise="{sentences}",
    ),
    Envelope(
        name="directory_export",
        kinds=("tool_result",),
        register="a plain directory export with key-value fields",
        openings=(
            "directory export {code} · {section} · generated {date}",
            "export {code}: {section} directory, snapshot {date}",
            "listing export {code} ({section}) taken {date}",
            "registry export {code} — {section} — {date}",
        ),
        sections=("--- {section} ---",),
        entry="- name: {label}\n  detail: {body}",
        filler_item="- name: {label}\n  detail: {body}",
        noise="# {sentences}",
    ),
    Envelope(
        name="meeting_minutes",
        kinds=("assistant_output",),
        register="minutes of a working meeting with named speakers",
        openings=(
            "minutes {code}: {section} working group, {date}",
            "{section} group minutes, {date}, ref {code}",
            "notes of the {section} sitting held {date} ({code})",
            "working minutes {code} — {section} — {date}",
        ),
        sections=("{section}:", "Item — {section}"),
        entry="{label}\n  {body}",
        filler_item="{label}: {body}",
        noise="{sentences}",
    ),
    Envelope(
        name="clipped_pages",
        kinds=("tool_result",),
        register="a scrapbook of clipped web pages with rough separators",
        openings=(
            "clipped pages {code} — searching {section}, saved {date}",
            "saved clips {code}: {section} ({date})",
            "clipboard dump {code} · {section} · {date}",
            "clips for {section}, gathered {date}, batch {code}",
        ),
        sections=("~~~ {section} ~~~", "* * * {section} * * *"),
        entry="[{label}]\n{body}",
        filler_item="[{label}] {body}",
        noise="{sentences}",
    ),
    Envelope(
        name="ticket_queue",
        kinds=("tool_result",),
        register="a support ticket queue with terse status lines",
        openings=(
            "queue {code} — {section}, exported {date}",
            "ticket queue {code}: {section} open items at {date}",
            "queue snapshot {code} ({section}) {date}",
            "open queue {code} for {section}, as at {date}",
        ),
        sections=("## {section}", "-- {section} --"),
        entry="{label}\n  {body}",
        filler_item="{label} · {body}",
        noise="{sentences}",
    ),
    Envelope(
        name="handbook_extract",
        kinds=("assistant_output",),
        register="an extract from a printed handbook with numbered notes",
        openings=(
            "handbook extract {code}, {section} chapter, edition of {date}",
            "extract {code} — {section} — handbook revised {date}",
            "from the {section} handbook ({date}), section {code}",
            "handbook {code}: {section}, reprinted {date}",
        ),
        sections=("{section}", "Note on {section}"),
        entry="{label}\n    {body}",
        filler_item="{label} — {body}",
        noise="{sentences}",
    ),
    Envelope(
        name="transcript_dump",
        kinds=("tool_result",),
        register="a rough call transcript with speaker turns",
        openings=(
            "transcript {code} — {section} call, {date}",
            "call transcript {code}: {section}, recorded {date}",
            "rough transcript {code} ({section}) from {date}",
            "transcribed {date}: {section} discussion, file {code}",
        ),
        sections=("[{section}]", "-- {section} --"),
        entry="{speaker}: {label} — {body}",
        filler_item="{speaker}: {label} — {body}",
        noise="{speaker}: {sentences}",
    ),
    Envelope(
        name="newsletter_roundup",
        kinds=("assistant_output",),
        register="a local newsletter round-up with short write-ups",
        openings=(
            "round-up {code} — what's on for {section}, {date}",
            "{section} round-up, issue {code}, {date}",
            "newsletter {code}: {section} notices for {date}",
            "this week in {section} — issue {code}, {date}",
        ),
        sections=("### {section}", "— {section} —"),
        entry="**{label}**\n{body}",
        filler_item="**{label}** — {body}",
        noise="{sentences}",
    ),
    Envelope(
        name="inspection_report",
        kinds=("tool_result",),
        register="an inspection report with short findings per site",
        openings=(
            "inspection report {code}: {section}, visits to {date}",
            "report {code} — {section} inspections closing {date}",
            "findings {code} for {section}, compiled {date}",
            "inspection round {code} ({section}) ending {date}",
        ),
        sections=("Finding group: {section}", "## {section}"),
        entry="{label}\n  finding: {body}",
        filler_item="{label} — finding: {body}",
        noise="{sentences}",
    ),
    Envelope(
        name="index_card_set",
        kinds=("assistant_output",),
        register="a set of typed index cards, one per item",
        openings=(
            "card set {code} — {section}, typed {date}",
            "index cards {code}: {section} ({date})",
            "card box {code}, {section} drawer, refiled {date}",
            "cards for {section}, batch {code}, {date}",
        ),
        sections=("card divider: {section}", "( {section} )"),
        entry="card {ordinal} — {label}\n  {body}",
        filler_item="card {ordinal} — {label}: {body}",
        noise="{sentences}",
    ),
    Envelope(
        name="planning_grid",
        kinds=("tool_result",),
        register="a planning grid with one paragraph per cell",
        openings=(
            "planning grid {code} · {section} · drawn up {date}",
            "grid {code} for {section}, version of {date}",
            "planning sheet {code}: {section}, {date}",
            "allocation grid {code} — {section} — {date}",
        ),
        sections=("row group: {section}", "<{section}>"),
        entry="{label}\n   >> {body}",
        filler_item="{label} >> {body}",
        noise="{sentences}",
    ),
)

ENVELOPE_NAMES = tuple(envelope.name for envelope in ENVELOPES)


def code(rng: random.Random) -> str:
    """A short reference token; keeps six-word runs from repeating."""

    return (
        f"{rng.choice(CODE_LETTERS)}{rng.choice(CODE_LETTERS)}"
        f"-{rng.randrange(100, 9999)}"
    )


def date_token(rng: random.Random) -> str:
    return f"{rng.randrange(1, 29)} {rng.choice(MONTHS)}"


def proper_name(rng: random.Random) -> str:
    return f"{rng.choice(NAME_HEADS)} {rng.choice(NAME_TAILS)}"


def _role(rng: random.Random) -> str:
    return f"the {rng.choice(ROLE_ADJECTIVES)} {rng.choice(ROLE_NOUNS)}"


def _action(rng: random.Random) -> str:
    return f"{rng.choice(ACTION_VERBS)} the {rng.choice(ACTION_OBJECTS)}"


def _place(rng: random.Random) -> str:
    return f"{rng.choice(PLACE_PREPOSITIONS)} the {rng.choice(PLACE_NOUNS)}"


def _qualifier(rng: random.Random) -> str:
    return (
        f"{rng.choice(QUALIFIER_PREPOSITIONS)} the "
        f"{rng.choice(QUALIFIER_NOUNS)}"
    )


def _join(rng: random.Random, parts: Sequence[str]) -> str:
    """Join composed slots, sometimes through one short randomised connector.

    At most one connector per sentence keeps the prose readable while still
    breaking up the word runs a repetition detector would otherwise see.
    """

    connector = rng.choice(CONNECTORS)
    junction = rng.randrange(1, len(parts)) if len(parts) > 1 else 0
    joined = [parts[0]]
    for index, part in enumerate(parts[1:], start=1):
        if connector and index == junction:
            joined.append(connector)
        joined.append(part)
    return " ".join(joined)


def sentence(rng: random.Random) -> str:
    """One filler sentence assembled from four independent composed slots."""

    role = _role(rng)
    action = _action(rng)
    place = _place(rng)
    qualifier = _qualifier(rng)
    shape = rng.randrange(6)
    if shape == 0:
        text = _join(rng, (role, action, place, qualifier))
    elif shape == 1:
        text = _join(rng, (qualifier + ",", role, action, place))
    elif shape == 2:
        text = _join(rng, (place + ",", role, action, qualifier))
    elif shape == 3:
        text = _join(rng, (role, action + ",", qualifier + ",", place))
    elif shape == 4:
        text = _join(rng, (qualifier, role, action))
    else:
        text = _join(rng, (place, role, action, qualifier))
    return text[0].upper() + text[1:] + "."


def sentences(rng: random.Random, count: int) -> str:
    return " ".join(sentence(rng) for _ in range(count))


def detail_line(rng: random.Random) -> str:
    body = _join(rng, (rng.choice(DETAIL_LEADS), rng.choice(DETAIL_TAILS)))
    return f"{rng.choice(DETAIL_HEADS)}: {body}"


def filler_body(rng: random.Random) -> str:
    parts = [sentences(rng, rng.randrange(1, 3))]
    if rng.random() < 0.35:
        parts.append(detail_line(rng))
    return " ".join(parts)


def render_entry(
    envelope: Envelope,
    rng: random.Random,
    label: str,
    body: str,
    ordinal: int,
    *,
    filler: bool = False,
) -> str:
    template = envelope.filler_item if filler else envelope.entry
    return template.format(
        label=label,
        body=body,
        ordinal=ordinal,
        speaker=rng.choice(SPEAKERS),
    )


def render_section(envelope: Envelope, rng: random.Random) -> str:
    return rng.choice(envelope.sections).format(section=rng.choice(SECTION_WORDS))


def render_opening(envelope: Envelope, rng: random.Random) -> str:
    return rng.choice(envelope.openings).format(
        code=code(rng),
        date=date_token(rng),
        section=rng.choice(SECTION_WORDS),
    )


def render_noise(envelope: Envelope, rng: random.Random) -> str:
    return envelope.noise.format(
        sentences=sentences(rng, rng.randrange(2, 6)),
        speaker=rng.choice(SPEAKERS),
    )


def balanced_assignment(
    options: Sequence[str],
    count: int,
    rng: random.Random,
) -> list[str]:
    """Spread `options` evenly across `count` slots, then shuffle the order."""

    pool: list[str] = []
    while len(pool) < count:
        block = list(options)
        rng.shuffle(block)
        pool.extend(block)
    pool = pool[:count]
    rng.shuffle(pool)
    return pool
