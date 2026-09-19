"""LLM 输出 JSON 提取（scaf.md 5.4）：find first { / find last } → json.loads。

重试策略为补充约定（docs/decisions.md）：parse 失败最多重试 2 次，
仍失败则标记待人工 review。
"""

from __future__ import annotations

import json

MAX_PARSE_RETRIES = 2


class JsonParseError(ValueError):
    """LLM 输出中无法提取合法 JSON。"""


def parse_json(response: str) -> dict:
    """从模型输出提取 JSON 对象（首个 { 到末个 }，scaf.md L558-574）。"""
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
