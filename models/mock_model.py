"""确定性 Mock 模型（补充约定：用于冒烟测试与复现性抽查）。

generate 基于 prompt 内容哈希生成稳定响应：
- 标注 prompt（含 "linguistic annotation assistant"）：返回合法标注 JSON
- 实验 prompt：返回解释 + "Confidence: XX"（兼容 analysis.confidence 提取）
同一 prompt 结果恒等——支撑"同 seed 复跑一致"断言。
"""

from __future__ import annotations

import hashlib
import json
import re

from models.base_model import BaseModel

_ANNOTATION_MARK = "linguistic annotation assistant"


def _parse_prompt_sentences(prompt: str) -> list[str]:
    """从标注 prompt 的编号列表提取句子（'1. xxx' 行）。

    只解析 "Sentences:" 标记之后的行，避免把指令中的编号项
    （如 "1. Sentence function: ..."）误当句子。
    """
    tail = prompt
    for marker in ("Sentences:", "sentences:"):
        idx = prompt.rfind(marker)
        if idx != -1:
            tail = prompt[idx + len(marker):]
            break
    sentences = []
    for line in tail.splitlines():
        m = re.match(r"^\s*\d+\.\s+(.+)$", line)
        if m:
            sentences.append(m.group(1).strip())
    return sentences


class MockModel(BaseModel):
    def __init__(self, config: dict | None = None, seed: int = 42):
        self.config = config or {}
        self.seed = seed
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def generate(self, prompt: str) -> str:
        if not self.loaded:
            raise RuntimeError("model not loaded; call load() first")
        if _ANNOTATION_MARK in prompt:
            return self._generate_annotation(prompt)
        return self._generate_experiment(prompt)

    def _generate_annotation(self, prompt: str) -> str:
        """标注模式：返回结构合法的标注 JSON（pipeline 会覆盖 id/task）。"""
        sentences = _parse_prompt_sentences(prompt)
        if not sentences:
            sentences = ["mock sentence one.", "mock sentence two."]
        data = {
            "id": "mock",
            "task": "faux_pas",
            "sentences": [
                {"id": i, "text": s, "function": "background"}
                for i, s in enumerate(sentences, start=1)
            ],
            "critical_sentence": len(sentences),
            "gold_answer": "mock gold answer for smoke test",
        }
        return "Here is JSON:\n" + json.dumps(data)

    def _generate_experiment(self, prompt: str) -> str:
        h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        confidence = 40 + (h % 55)  # 40-94
        snippet = re.sub(r"\s+", " ", prompt)[-120:]
        return (
            f"Current interpretation: mock interpretation of {snippet!r}\n"
            f"Character intention: mock intention\n"
            f"Confidence: {confidence}"
        )
