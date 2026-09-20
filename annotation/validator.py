"""Annotation Validator (pipeline.md L514-543 / scaf.md 5.5; extended by rulings C8/C11).

Checks: required fields (id/task/sentences/critical_sentence/gold_answer), the task enum,
the function enum, continuous sentence ids (starting at 1), and critical_sentence in range.
"""

from __future__ import annotations

from annotation.schema import FUNCTIONS, REQUIRED_FIELDS, SENTENCE_FIELDS, TASKS


class AnnotationValidator:
    def validate(self, data: dict) -> tuple[bool, list[str]]:
        """Returns (passed, list of errors)."""
        errors: list[str] = []

        for field in REQUIRED_FIELDS:
            if field not in data:
                errors.append(f"missing required field: {field}")
        if errors:
            return False, errors

        if data["task"] not in TASKS:
            errors.append(f"invalid task {data['task']!r}; expected one of {TASKS}")

        sentences = data["sentences"]
        if not isinstance(sentences, list) or not sentences:
            errors.append("sentences must be a non-empty list")
        else:
            for i, sent in enumerate(sentences, start=1):
                for f in SENTENCE_FIELDS:
                    if f not in sent:
                        errors.append(f"sentence[{i}] missing field: {f}")
                if sent.get("id") != i:
                    errors.append(f"sentence[{i}] id must be {i} (continuous from 1)")
                if sent.get("function") not in FUNCTIONS:
                    errors.append(
                        f"sentence[{i}] invalid function {sent.get('function')!r}; "
                        f"expected one of {FUNCTIONS}"
                    )

        cs = data.get("critical_sentence")
        valid_ids = {s.get("id") for s in sentences if isinstance(s, dict)}
        if cs not in valid_ids:
            errors.append(f"critical_sentence {cs!r} out of range {sorted(valid_ids)}")

        if not isinstance(data.get("gold_answer"), str) or not data["gold_answer"].strip():
            errors.append("gold_answer must be a non-empty string")

        if data.get("question") is not None and not isinstance(data["question"], str):
            errors.append("question must be a string when present")

        return not errors, errors
