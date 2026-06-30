from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


AMARA_VERSION = "amara-life-v1"
ARTIFACT_FOLDERS = ("calendar", "doc", "emails", "meetings", "notes", "slack")
PREFIX_MAP = {
    "company": "companies",
    "doc": "source",
    "docs": "source",
    "events": "concepts",
    "fund": "deal",
    "org": "companies",
    "organizations": "companies",
    "orgs": "companies",
    "person": "people",
    "processes": "concepts",
    "tool": "concepts",
    "tools": "concepts",
    "user": "people",
}
SLUG_TYPES = {
    "companies": "company",
    "concepts": "concept",
    "deal": "deal",
    "entities": "concept",
    "people": "person",
    "source": "source",
}
MARKDOWN_LINK = re.compile(
    r"(\[([^\]]+)\]\()([a-z][a-z0-9_-]*/[a-z0-9][a-z0-9/-]*)(\))"
)
ATTENDEES_LINE = re.compile(r"^(attendees\s*:\s*)(.+)$", re.MULTILINE)


def sha256_file(path: str | Path) -> str:
    # Corpus files are text. Canonicalize line endings so provenance survives
    # Git's LF/CRLF checkout policy across operating systems.
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def load_manifest(source: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(
        (Path(source) / "corpus-manifest.json").read_text(encoding="utf-8")
    )
    artifacts = payload.get("artifacts", [])
    return {
        str(item["slug"]): item
        for item in artifacts
        if isinstance(item, dict) and item.get("slug")
    }


def normalize_amara(source: str | Path, output: str | Path) -> int:
    source_path = Path(source)
    output_path = Path(output)
    manifest = load_manifest(source_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    for folder in ("doc", "meetings", "notes"):
        for item in (source_path / folder).glob("*.md"):
            target = output_path / folder / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            text = item.read_text(encoding="utf-8")
            slug_prefix = "meeting" if folder == "meetings" else folder.rstrip("s")
            slug = f"{slug_prefix}/{item.stem}"
            metadata = manifest.get(slug, {})
            text = _add_frontmatter_value(text, "source", AMARA_VERSION)
            text = _add_frontmatter_value(
                text, "source_sha256", sha256_file(item)
            )
            perturbations = metadata.get("perturbations", [])
            if perturbations:
                text = _add_frontmatter_value(text, "perturbations", perturbations)
            target.write_text(text, encoding="utf-8")

    _normalize_jsonl(source_path, output_path, manifest)
    _normalize_calendar(source_path, output_path, manifest)
    _prepare_graph(output_path)
    return sum(1 for _ in output_path.rglob("*.md"))


def _write_page(
    output: Path,
    relative_path: str,
    frontmatter: dict[str, Any],
    body: str,
) -> None:
    path = output / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    lines.extend(
        f"{key}: {json.dumps(value, ensure_ascii=False)}"
        for key, value in frontmatter.items()
    )
    lines.extend(["---", body.strip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _normalize_jsonl(
    source: Path,
    output: Path,
    manifest: dict[str, dict[str, Any]],
) -> None:
    streams = [
        ("inbox/emails.jsonl", "email"),
        ("slack/messages.jsonl", "slack"),
    ]
    for relative_path, kind in streams:
        raw_path = source / relative_path
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                perturbation = item.get("perturbation", {})
                frontmatter: dict[str, Any] = {
                    "id": item["id"],
                    "type": kind,
                    "date": item["ts"],
                    "source": AMARA_VERSION,
                    "source_item": item["slug"],
                    "source_sha256": hashlib.sha256(
                        line.encode("utf-8")
                    ).hexdigest(),
                }
                if perturbation:
                    frontmatter["perturbations"] = [perturbation["kind"]]
                    frontmatter["perturbation_fixture_id"] = perturbation.get(
                        "fixture_id"
                    )
                elif manifest.get(item["slug"], {}).get("perturbations"):
                    frontmatter["perturbations"] = manifest[item["slug"]][
                        "perturbations"
                    ]

                if kind == "email":
                    recipients = ", ".join(
                        f"{person['name']} <{person['email']}>"
                        for person in item["to"]
                    )
                    body = "\n".join(
                        [
                            f"# {item['subject']}",
                            "",
                            f"From: {item['from']['name']} <{item['from']['email']}>",
                            f"To: {recipients}",
                            f"Date: {item['ts']}",
                            f"Thread: {item['thread_id']}",
                            "",
                            item["body_text"],
                        ]
                    )
                else:
                    body = "\n".join(
                        [
                            f"# Slack message in {item['channel']}",
                            "",
                            f"Author: {item['user']['name']} (@{item['user']['handle']})",
                            f"Date: {item['ts']}",
                            "",
                            item["text"],
                        ]
                    )
                _write_page(output, f"{item['slug']}.md", frontmatter, body)


def _normalize_calendar(
    source: Path,
    output: Path,
    manifest: dict[str, dict[str, Any]],
) -> None:
    raw_path = source / "calendar.ics"
    raw = raw_path.read_text(encoding="utf-8")
    unfolded = re.sub(r"\r?\n[ \t]", "", raw)
    for block in re.findall(r"BEGIN:VEVENT\r?\n(.*?)\r?\nEND:VEVENT", unfolded, re.S):
        fields: dict[str, list[str]] = {}
        for line in block.splitlines():
            key, _, value = line.partition(":")
            fields.setdefault(key, []).append(value)
        uid = fields["UID"][0].split("@", 1)[0]
        start = fields.get("DTSTART", [""])[0]
        end = fields.get("DTEND", [""])[0]
        summary = fields.get("SUMMARY", [uid])[0]
        slug = f"calendar/{uid}"
        frontmatter: dict[str, Any] = {
            "id": uid,
            "type": "calendar-event",
            "date": start,
            "source": AMARA_VERSION,
            "source_item": slug,
            "source_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
        }
        if manifest.get(slug, {}).get("perturbations"):
            frontmatter["perturbations"] = manifest[slug]["perturbations"]
        _write_page(
            output,
            f"calendar/{uid}.md",
            frontmatter,
            "\n".join(
                [
                    f"# {summary}",
                    "",
                    f"Start: {start}",
                    f"End: {end}",
                    f"Location: {fields.get('LOCATION', [''])[0]}",
                ]
            ),
        )


def _add_frontmatter_value(text: str, key: str, value: Any) -> str:
    if not text.startswith("---\n"):
        return f"---\n{key}: {json.dumps(value)}\n---\n{text}"
    end = text.find("\n---", 4)
    if end == -1:
        return text
    frontmatter = text[4:end]
    if re.search(rf"^{re.escape(key)}\s*:", frontmatter, re.MULTILINE):
        return text
    return text[:4] + f"{key}: {json.dumps(value)}\n" + text[4:]


def _canonicalize_slug(slug: str) -> str:
    prefix, tail = slug.split("/", 1)
    return f"{PREFIX_MAP.get(prefix, prefix)}/{tail}"


def _prepare_graph(corpus: Path) -> None:
    labels: dict[str, set[str]] = defaultdict(set)
    folder_types = {"doc": "source", "meetings": "meeting", "notes": "note"}
    for folder in ARTIFACT_FOLDERS:
        for path in (corpus / folder).glob("*.md"):
            text = path.read_text(encoding="utf-8")
            if folder in folder_types:
                text = _add_frontmatter_value(text, "type", folder_types[folder])
            if folder == "meetings":
                heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                if heading:
                    text = _add_frontmatter_value(
                        text, "title", heading.group(1).strip()
                    )

            def replace(match: re.Match[str]) -> str:
                slug = _canonicalize_slug(match.group(3))
                labels[slug].add(match.group(2).strip())
                return f"{match.group(1)}{slug}{match.group(4)}"

            text = MARKDOWN_LINK.sub(replace, text)
            text = ATTENDEES_LINE.sub(
                lambda match: match.group(1)
                + re.sub(
                    r"\buser/([a-z0-9][a-z0-9/-]*)",
                    r"people/\1",
                    match.group(2),
                ),
                text,
            )
            path.write_text(text, encoding="utf-8")

    for slug, names in sorted(labels.items()):
        prefix, tail = slug.split("/", 1)
        if prefix not in SLUG_TYPES:
            continue
        title = sorted(names, key=lambda value: (-len(value), value.casefold()))[0]
        _write_page(
            corpus,
            f"{slug}.md",
            {
                "id": tail,
                "type": SLUG_TYPES[prefix],
                "source": AMARA_VERSION,
                "generated_stub": True,
                "aliases": sorted(names, key=str.casefold),
            },
            f"# {title}\n\nEntity page generated from references in the Amara Life fixture.",
        )
