"""Story preprocessing (scaf.md 5.1): clean_text + split_sentence.

The basis for a tokenizer choice is missing (pipeline.md L399-411), so a deterministic
punctuation/newline split is used for now, with a replaceable splitter interface kept
(supplementary convention; see docs/decisions.md).
"""

from __future__ import annotations

import re

# sentence-ending punctuation (including a possible closing quote/bracket)
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*")


class StoryPreprocessor:
    def clean_text(self, text: str) -> str:
        """Collapse redundant whitespace and normalise punctuation (scaf.md L415-424)."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r" *([.,!?])", r"\1", text)
        return text.strip()

    def split_sentence(self, text: str, splitter=None):
        """Split on sentence-ending punctuation; the splitter is replaceable (interface reserved for an NLP tokenizer)."""
        if splitter is not None:
            return splitter(text)
        text = self.clean_text(text)
        parts = [p for p in _SENT_END.split(text) if p.strip()]
        return [p.strip() for p in parts]
