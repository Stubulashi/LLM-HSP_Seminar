"""Prompt Builder (pipeline.md section 8 L704-735 / proj.md section 6 L519-538; rulings C9/C13).

One shared entry point for all experiments; prompt content comes from prompts.yaml
(versioned; scaf.md Principle 2).
- condition_a: the natural incremental interpretation template for all models (proj.md L519-538);
- condition_b: a separate template (ruling C13; used later, replaces the base template rather than concatenating);
- task-level question: if the story provides a question (all false_belief stories do;
  faux_pas/implicature currently none) and the task template contains {question}, it is
  injected; otherwise, when a question exists, an "answer the question directly first" hint
  is appended after the condition template (P1-5 calibration: gives non-false_belief tasks
  a single target question too), so that accuracy/emergence stay comparable.
"""

from __future__ import annotations


class PromptBuilder:
    def __init__(self, prompts_config: dict):
        """prompts_config is the expanded structure of ConfigManager.get("prompts")."""
        self.config = prompts_config

    def _task_template(self, task: str) -> str | None:
        """Return the task template containing a {question} slot, or None."""
        t = self.config.get("tasks", {}).get(task, {}).get("template")
        if t and "{question}" in t:
            return t
        return None

    def build(
        self,
        context: str,
        task: str = "default",
        condition: str = "condition_a",
        question: str | None = None,
    ) -> str:
        """Assemble the full prompt from the condition and task (system + template)."""
        try:
            cond = self.config["conditions"][condition]
        except KeyError as exc:
            raise KeyError(f"unknown condition {condition!r}") from exc

        system = cond.get("system", "")
        task_tpl = self._task_template(task)
        if task_tpl is not None and question:
            body = task_tpl.format(context=context, question=question)
        else:
            body = cond["template"].format(context=context)
            if question:  # no {question} slot in the condition template, but still ask for a direct answer first
                body = (body.rstrip()
                        + f"\n\nQuestion: {question}\n"
                          "Give a direct answer to the question first, then the rest.")

        return f"{system}\n\n{body}" if system else body
