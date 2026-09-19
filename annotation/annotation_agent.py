"""Annotation Agent（pipeline.md L415-510 / scaf.md 5.2-5.4）。

特殊 LLM 调用——它是数据工具而非实验对象（scaf.md L461-467）。
流程：build_annotation_prompt → model.generate → parse_json → validate；
失败重试最多 2 次（补充约定），仍失败抛 AnnotationError 标记待人工 review。
"""

from __future__ import annotations

from annotation.json_parser import JsonParseError, parse_json
from annotation.schema import FUNCTIONS
from annotation.validator import AnnotationValidator


class AnnotationError(RuntimeError):
    """标注失败（含重试后仍失败）。"""


def build_annotation_prompt(sentence_list: list[str], template: str) -> str:
    """按标注模板组装 prompt（pipeline.md L434-460 措辞，裁决 C8）。"""
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentence_list, start=1))
    return template.format(sentences=numbered)


class AnnotationAgent:
    def __init__(self, model, prompt_template: str, max_attempts: int = 3):
        """model 仅需实现 generate(prompt) -> str（低耦合，scaf.md Principle 1）。"""
        self.model = model
        self.prompt_template = prompt_template
        self.max_attempts = max_attempts  # 首次 + 重试 2 次

    def annotate(self, sentence_list: list[str]) -> dict:
        prompt = build_annotation_prompt(sentence_list, self.prompt_template)
        validator = AnnotationValidator()
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.model.generate(prompt)
                data = parse_json(response)
                ok, errors = validator.validate(data)
                if ok:
                    return data
                last_error = AnnotationError(f"validation failed: {'; '.join(errors)}")
            except (JsonParseError, AnnotationError) as exc:
                last_error = exc
            if attempt < self.max_attempts:
                # 重试时提示只返回合法 JSON（scaf.md L546-553 场景的应对）
                prompt = self.prompt_template.format(
                    sentences="\n".join(
                        f"{i}. {s}" for i, s in enumerate(sentence_list, start=1)
                    )
                ) + "\n\nImportant: return valid JSON only, with all required fields."
        raise AnnotationError(
            f"annotation failed after {self.max_attempts} attempts: {last_error}"
        ) from last_error


# 供人类审查展示的 function 枚举（pipeline.md L441-449）
FUNCTION_OPTIONS = FUNCTIONS
