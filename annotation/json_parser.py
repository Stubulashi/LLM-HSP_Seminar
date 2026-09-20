"""Extract JSON from LLM output (scaf.md 5.4): find first { / find last } → json.loads.

The retry policy is a supplementary convention (docs/decisions.md): a parse failure is
retried up to 2 times; if it still fails, the item is flagged for human review.
"""

from __future__ import annotations

import json

MAX_PARSE_RETRIES = 2


class JsonParseError(ValueError):
    """No valid JSON could be extracted from the LLM output."""


def parse_json(response: str) -> dict:
    """Extract a JSON object from the model output (first { to last }; scaf.md L558-574)."""
    if not isinstance(response, str) or not response.strip():
        raise JsonParseError("empty model response")
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JsonParseError(f"no JSON object found in response: {response[:80]!r}...")
    try:
        data = json.loads(response[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JsonParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JsonParseError("extracted JSON is not an object")
    return data
