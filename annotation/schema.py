"""Annotation schema constants (rulings C8/C9/C11; see docs/decisions.md).

- REQUIRED_FIELDS: consistent between pipeline.md L519-543 and scaf.md L585-593
- TASKS: the frozen task enum (ruling C11)
- FUNCTIONS: the function enum from pipeline.md L441-449
- question is optional (ruling C9); reviewed is written by the human review step (ruling C15)
"""

TASKS = ("faux_pas", "false_belief", "implicature")

FUNCTIONS = ("background", "context_update", "critical_event", "resolution")

REQUIRED_FIELDS = ("id", "task", "sentences", "critical_sentence", "gold_answer")

SENTENCE_FIELDS = ("id", "text", "function")

OPTIONAL_FIELDS = ("question", "reviewed")
