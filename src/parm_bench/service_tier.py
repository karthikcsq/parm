"""Opt-in OpenAI service tier for cost control.

Set OPENAI_SERVICE_TIER=flex in the environment (the ignored root .env is the
right place) to send eligible Responses API calls at the discounted flex tier
during development. The tier changes price and latency, not output, so it must
stay out of every cache key: a response cached under flex replays identically
under the default tier. Unset or empty means the API default. Embedding calls
do not accept a service tier and are unaffected.
"""

from __future__ import annotations

import os

_ALLOWED_TIERS = {"auto", "default", "flex", "priority"}


def service_tier_kwargs() -> dict[str, str]:
    tier = os.environ.get("OPENAI_SERVICE_TIER", "").strip().lower()
    if not tier:
        return {}
    if tier not in _ALLOWED_TIERS:
        raise ValueError(
            f"unsupported OPENAI_SERVICE_TIER {tier!r}; "
            f"expected one of {sorted(_ALLOWED_TIERS)}"
        )
    return {"service_tier": tier}
