"""响应处理（scaf.md 10 节 L1016-1066，裁决 C1：processing/ → analysis/）。

extract_confidence：从原始 response 稳健提取置信度（0-100）。修正 P0-2：
- 旧版只认字面 `Confidence:<num>`（CONFIDENCE_RE），对真实输出中的
  "Confidence score: 85/100"、"My confidence score is 85/100." 等别名零宽容，
  导致解析率掉到 0-8%；改为按置信关键字定位 + 行尾就近取整数的策略。
- 规避旧版取 matches[-1] 在整个 response 里抓最后任意数字（易把 prose 尾数
  或分子误当置信）的陷阱：改为自底向上扫行、取最后一个含置信关键字行内第一个
  <=100 的整数。无有效匹配返回 None（不静默错值）。

extract_interpretation：去除含置信自报告/编号格式噪声，保留解释文本（P1-5
归一化，避免 prompt 回显混杂进 embedding 语义）。
"""

from __future__ import annotations

import re

# 兼容低容量模型常见别名：Confidence / confidence score / confidence level
CONF_KEYWORD = re.compile(r"confidence\s*\b", re.IGNORECASE)
# 行内靠近置信关键字的整数（0-100，天然排除 "85/100" 的分母 100）
_INT = re.compile(r"(\d{1,3})")
# 旧格式专名（兼容历史测试：整行 `Confidence: 65`）
CONFIDENCE_RE = re.compile(r"Confidence\s*:\s*(\d{1,3})", re.IGNORECASE)

# prompt 结构性回显关键词：模型常把 prompt 里的章节名/引导句原样复述成独立行，
# 这些不携带新语义，却会污染 embedding（P1-5 归一化）。仅当整行**无后续正文**时剔除。
_ECHO_HEADINGS = (
    "explanation", "final answer", "answer to the question",
    "current situation interpretation", "character intention",
    "possible intentions or beliefs", "potential intentions or beliefs",
    "current understanding", "confidence",
)
# 匹配：可选中英文数字序数前缀 + 可选冒号/破折号结尾，行内无其它正文
_ECHO_LINE = re.compile(
    r"^(?:\d+[\.\\)]\s*)?(?:[A-Za-z]+\.)?\s*"
    r"([a-z \-]+?)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
# 低信息退化填充：0.5B 常循环“就绪/下一句”填充，无解释语义
_FILLER = re.compile(r"^(?:i'?m )?ready for the next (?:sentence|line)", re.IGNORECASE)


def _is_structural_echo(line: str) -> bool:
    """该行是否为无正文的 prompt 章节/头回显，应剔除。"""
    if _FILLER.search(line):
        return True
    m = _ECHO_LINE.match(line)
    if not m or not m.group(1):
        return False
    head = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(":.")
    return head.lower() in _ECHO_HEADINGS


def _line_confidence(line: str) -> int | None:
    """取含置信关键字行内第一个 <=100 的整数；无则 None。"""
    kw = CONF_KEYWORD.search(line)
    if not kw:
        return None
    tail = line[kw.end():]
    for n in _INT.findall(tail):
        val = int(n)
        if 0 <= val <= 100:
            return val
    return None


def extract_confidence(text: str) -> int | None:
    """稳健提取置信度：自底向上扫行，最后一个含置信关键字行确定值。

    - 命中含置信关键词的行，取其行内（关键字之后）第一个 0-100 整数。
    - 整段无任何置信关键字时，回退旧正则从整 text 取最后一个 `<num>`
      （保留历史行为；单元测试 test_extract_confidence_takes_last 覆盖）。
    """
    text = text or ""
    lines = text.splitlines() or [text]
    for line in reversed(lines):
        val = _line_confidence(line)
        if val is not None:
            return max(0, min(100, val))
    # 回退：旧款整文本末匹配（仅当不存在可定位的置信行时触发）
    matches = CONFIDENCE_RE.findall(text)
    if not matches:
        return None
    return max(0, min(100, int(matches[-1])))


def _is_confidence_line(stripped: str) -> bool:
    """行首（可带编号）直接以置信开头且无正文前缀 → 纯置信行。"""
    return bool(re.match(r"^(?:\d+[\.\)]\s*)?confidence\b", stripped, re.IGNORECASE))


def _strip_trailing_confidence(stripped: str) -> str:
    """行内剥离置信自报段：保留 confidence 关键字之前的正文（如“Sarah said… Confidence: 85”）。

    修复：v1.2 直答模板下模型常把答案与置信写在同一行，旧逻辑整行删除导致答案丢失。
    仅当行首即 confidence（纯置信行）时整行丢弃（调用方先行判断）。
    """
    m = CONF_KEYWORD.search(stripped)
    if not m:
        return stripped
    left = stripped[:m.start()].strip()
    return left


def extract_interpretation(text: str) -> str:
    """去除置信度段与格式噪声，返回解释文本（scaf.md L1051-1065）。

    处理顺序：
    ① 纯置信行（行首 confidence，无正文前缀）整行剔除；
    ② 其余含置信自报的行，只剥离 confidence 及其后段落，保留前缀正文
       （修复“单行答案+Confidence”被整行误删）；
    ③ 无正文的 prompt 章节/头回显（如 “5. Explanation”）与低信息退化填充剔除。
    其余保留拼接，保证 embedding/judge 只对解释正文建模。
    """
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _is_confidence_line(stripped) or _is_structural_echo(stripped):
            continue
        stripped = _strip_trailing_confidence(stripped)
        if stripped:
            lines.append(stripped)
    return " ".join(lines)
