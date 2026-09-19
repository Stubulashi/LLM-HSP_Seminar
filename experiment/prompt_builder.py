"""Prompt Builder（pipeline.md 八节 L704-735 / proj.md 六节 L519-538，裁决 C9/C13）。

所有实验统一入口；prompt 内容来自 prompts.yaml（版本化，scaf.md Principle 2）。
- condition_a：所有模型的自然增量解释模板（proj.md L519-538）
- condition_b：独立模板（裁决 C13，延后使用，覆盖式替换不拼接）
- 任务级 question：若 story 提供 question（false_belief 全会提供；faux_pas/implicature
  目前零个），且任务模板含 {question}，则注入；否则若 question 存在，在条件模板后附“先直接回答该问题”提示（P1-5 校准：让非 false_belief 任务也能拿到单一目标问），
  使 accuracy/emergence 可比对。
"""

from __future__ import annotations


class PromptBuilder:
    def __init__(self, prompts_config: dict):
        """prompts_config = ConfigManager.get("prompts") 的展开结构。"""
        self.config = prompts_config

    def _task_template(self, task: str) -> str | None:
        """返回含 {question} 插槽的任务模板；无则 None。"""
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
        """按条件与任务组装完整 prompt（system + template）。"""
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
            if question:  # 无条件模板 {question} 槽，但仍请求它先直接回答该问题
                body = (body.rstrip()
                        + f"\n\nQuestion: {question}\n"
                          "Give a direct answer to the question first, then the rest.")

        return f"{system}\n\n{body}" if system else body
