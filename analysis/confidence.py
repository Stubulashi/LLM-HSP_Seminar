"""Response processing (scaf.md section 10 L1016-1066; ruling C1: processing/ → analysis/).

extract_confidence: robustly extract the confidence (0-100) from a raw response. Fix P0-2:
- The old version only accepted the literal `Confidence:<num>` (CONFIDENCE_RE) and had zero
  tolerance for aliases common in real output such as "Confidence score: 85/100" or
  "My confidence score is 85/100.", which dropped the parse rate to 0-8%; the new strategy
  locates the confidence keyword and takes the nearest integer at the end of the line.
- It also avoids the old trap of matches[-1] grabbing the last arbitrary number in the whole
  response (easily a number at the end of prose or a numerator): instead it scans lines from
  the bottom and takes the first integer <=100 after the keyword on the last keyword line.
  With no valid match it returns None (never a silently wrong value).

extract_interpretation: removes confidence self-reports and numbered-format noise, keeping
the interpretation text (P1-5 normalisation, so prompt echoes do not pollute the embedding).
"""

from __future__ import annotations

import re

# tolerate aliases common in low-capacity models: Confidence / confidence score / confidence level
CONF_KEYWORD = re.compile(r"confidence\s*\b", re.IGNORECASE)
# integers near the confidence keyword within the line (0-100; the denominator 100 of "85/100" is naturally excluded)
_INT = re.compile(r"(\d{1,3})")
# legacy format pattern (kept for historical tests: a whole line `Confidence: 65`)
CONFIDENCE_RE = re.compile(r"Confidence\s*:\s*(\d{1,3})", re.IGNORECASE)

# Structural prompt-echo keywords: models often repeat section names / lead-in sentences from the prompt
# as standalone lines. They carry no new semantics but pollute the embedding (P1-5 normalisation),
# so they are dropped only when the line has no following body text.
_ECHO_HEADINGS = (
    "explanation", "final answer", "answer to the question",
    "current situation interpretation", "character intention",
    "possible intentions or beliefs", "potential intentions or beliefs",
    "current understanding", "confidence",
)
# Matches: optional numeric ordinal prefix + optional colon/dash ending, with no other text on the line
_ECHO_LINE = re.compile(
    r"^(?:\d+[\.\\)]\s*)?(?:[A-Za-z]+\.)?\s*"
    r"([a-z \-]+?)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
# Low-information degenerate filler: 0.5B models often loop "ready for the next..." lines with no interpretative content
_FILLER = re.compile(r"^(?:i'?m )?ready for the next (?:sentence|line)", re.IGNORECASE)


def _is_structural_echo(line: str) -> bool:
    """Whether the line is a body-less prompt section/heading echo that should be dropped."""
    if _FILLER.search(line):
        return True
    m = _ECHO_LINE.match(line)
    if not m or not m.group(1):
        return False
    head = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(":.")
    return head.lower() in _ECHO_HEADINGS


def _line_confidence(line: str) -> int | None:
    """Take the first integer <=100 after the confidence keyword on the line; None if there is none."""
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
    """Robustly extract the confidence: scan lines bottom-up; the last line containing a confidence keyword decides.

    - On a matching line, take the first 0-100 integer after the keyword.
    - When the whole text has no confidence keyword, fall back to the old regex taking the
      last `<num>` of the text (historical behaviour; covered by the unit test test_extract_confidence_takes_last).
    """
    text = text or ""
    lines = text.splitlines() or [text]
    for line in reversed(lines):
        val = _line_confidence(line)
        if val is not None:
            return max(0, min(100, val))
    # fallback: legacy whole-text last-match (only when no locatable confidence line exists)
    matches = CONFIDENCE_RE.findall(text)
    if not matches:
        return None
    return max(0, min(100, int(matches[-1])))


def _is_confidence_line(stripped: str) -> bool:
    """The line starts (optionally after a number) directly with confidence and has no body prefix → a pure confidence line."""
    return bool(re.match(r"^(?:\d+[\.\)]\s*)?confidence\b", stripped, re.IGNORECASE))


def _strip_trailing_confidence(stripped: str) -> str:
    """Strip the confidence self-report from within the line, keeping the body before the confidence keyword (e.g. "Sarah said… Confidence: 85").

    Fix: with the v1.2 direct-answer template the model often writes the answer and the
    confidence on the same line, and the old whole-line deletion lost the answer. A line is
    dropped entirely only when it starts with confidence (a pure confidence line; the caller checks this first).
    """
    m = CONF_KEYWORD.search(stripped)
    if not m:
        return stripped
    left = stripped[:m.start()].strip()
    return left


def extract_interpretation(text: str) -> str:
    """Remove confidence segments and format noise and return the interpretation text (scaf.md L1051-1065).

    Order of operations:
    1) a pure confidence line (starts with confidence, no body prefix) is dropped entirely;
    2) on other lines with a confidence self-report, only the confidence part and everything
       after it is stripped, keeping the body prefix (fixes "single-line answer + Confidence"
       being deleted as a whole);
    3) body-less prompt section/heading echoes (e.g. "5. Explanation") and low-information
       degenerate filler are dropped.
    The rest is kept and joined, so the embedding/judge only model the interpretation body.
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
