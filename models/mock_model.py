"""Deterministic mock model (supplementary convention: used for smoke tests and reproducibility spot checks).

generate derives a stable response from a hash of the prompt content:
- annotation prompts (containing "linguistic annotation assistant") return valid annotation JSON;
- experiment prompts return an interpretation plus "Confidence: XX" (compatible with the analysis.confidence extraction).
The same prompt always yields the same result, which supports the "same seed, same output" assertion.
"""

from __future__ import annotations

import hashlib
import json
import re

from models.base_model import BaseModel

_ANNOTATION_MARK = "linguistic annotation assistant"


def _parse_prompt_sentences(prompt: str) -> list[str]:
    """Extract sentences from the numbered list in an annotation prompt ('1. xxx' lines).

    Only lines after the "Sentences:" marker are parsed, so numbered items inside the
    instructions (such as "1. Sentence function: ...") are not mistaken for sentences.
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
        """Annotation mode: return structurally valid annotation JSON (the pipeline overrides id/task)."""
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
