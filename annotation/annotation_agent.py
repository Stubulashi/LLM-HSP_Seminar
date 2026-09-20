"""Annotation Agent (pipeline.md L415-510 / scaf.md 5.2-5.4).

A special LLM call: this is a data tool, not an experiment subject (scaf.md L461-467).
Flow: build_annotation_prompt → model.generate → parse_json → validate;
failures retry up to 2 extra times (supplementary convention); if it still fails, raise
AnnotationError so the item can be flagged for human review.
"""

from __future__ import annotations

from annotation.json_parser import JsonParseError, parse_json
from annotation.schema import FUNCTIONS
from annotation.validator import AnnotationValidator


class AnnotationError(RuntimeError):
    """Annotation failed (including after retries)."""


def build_annotation_prompt(sentence_list: list[str], template: str) -> str:
    """Build the prompt from the annotation template (wording per pipeline.md L434-460; ruling C8)."""
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentence_list, start=1))
    return template.format(sentences=numbered)


class AnnotationAgent:
    def __init__(self, model, prompt_template: str, max_attempts: int = 3):
        """model only needs to implement generate(prompt) -> str (low coupling; scaf.md Principle 1)."""
        self.model = model
        self.prompt_template = prompt_template
        self.max_attempts = max_attempts  # first attempt + 2 retries

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
                # on retries, remind the model to return valid JSON only (response to the scenario in scaf.md L546-553)
                prompt = self.prompt_template.format(
                    sentences="\n".join(
                        f"{i}. {s}" for i, s in enumerate(sentence_list, start=1)
                    )
                ) + "\n\nImportant: return valid JSON only, with all required fields."
        raise AnnotationError(
            f"annotation failed after {self.max_attempts} attempts: {last_error}"
        ) from last_error


# function enum presented in human review (pipeline.md L441-449)
FUNCTION_OPTIONS = FUNCTIONS
