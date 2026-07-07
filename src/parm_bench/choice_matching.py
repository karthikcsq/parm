from __future__ import annotations

import re


_LABEL_SEPARATOR = re.compile(r"\s+(?:—|–|-)\s+", re.UNICODE)


def matches_choice(response_text: str, choice: str) -> bool:
    response = f" {normalize_choice(response_text)} "
    return any(f" {alias} " in response for alias in choice_aliases(choice))


def choice_aliases(choice: str) -> tuple[str, ...]:
    full = normalize_choice(choice)
    aliases = [full]
    parts = _LABEL_SEPARATOR.split(choice, maxsplit=1)
    if len(parts) != 2:
        return tuple(aliases)

    locator, title = (normalize_choice(part) for part in parts)
    if _is_descriptive_title(title):
        aliases.append(title)
    if any(character.isdigit() for character in locator):
        aliases.append(locator)
    return tuple(dict.fromkeys(aliases))


def normalize_choice(text: str) -> str:
    return " ".join(
        "".join(
            character.casefold() if character.isalnum() else " "
            for character in text
        ).split()
    )


def _is_descriptive_title(value: str) -> bool:
    return len(value) >= 8 and len(value.split()) >= 2
