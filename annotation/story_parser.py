"""Story 预处理（scaf.md 5.1）：clean_text + split_sentence。

tokenizer 选型依据缺失（pipeline.md L399-411），先采用标点/换行的确定性切分，
预留 replace_splitter 接口（补充约定，见 docs/decisions.md）。
"""

from __future__ import annotations

import re

# 句子结束标点（含可能的闭合引号/括号）
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*")


class StoryPreprocessor:
    def clean_text(self, text: str) -> str:
        """移除多余空白并规范标点（scaf.md L415-424）。"""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r" *([.,!?])", r"\1", text)
        return text.strip()

    def split_sentence(self, text: str, splitter=None):
        """按句子结束标点切分；splitter 可替换（预留 NLP tokenizer 接口）。"""
        if splitter is not None:
            return splitter(text)
        text = self.clean_text(text)
        parts = [p for p in _SENT_END.split(text) if p.strip()]
        return [p.strip() for p in parts]
