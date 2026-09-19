"""Annotation schema 常量（裁决 C8/C9/C11，见 docs/decisions.md）。

- REQUIRED_FIELDS：pipeline.md L519-543 与 scaf.md L585-593 一致
- TASKS：裁决 C11 冻结的任务枚举
- FUNCTIONS：pipeline.md L441-449 的 function 枚举
- question 为可选字段（裁决 C9）；reviewed 由人工审查写入（裁决 C15）
"""

TASKS = ("faux_pas", "false_belief", "implicature")

FUNCTIONS = ("background", "context_update", "critical_event", "resolution")

REQUIRED_FIELDS = ("id", "task", "sentences", "critical_sentence", "gold_answer")

SENTENCE_FIELDS = ("id", "text", "function")

OPTIONAL_FIELDS = ("question", "reviewed")
