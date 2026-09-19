"""Human Review System（pipeline.md L547-575 / scaf.md 5.6，裁决 C15）。

可修改字段明确为 function / gold_answer / critical_sentence（裁决 C15）；
审查通过后写入 reviewed: true（裁决 C15 状态字段）。
"""

from __future__ import annotations

from annotation.schema import FUNCTIONS
from annotation.validator import AnnotationValidator


def _ask(prompt: str, input_fn) -> str:
    """包装 input，便于测试注入。"""
    return input_fn(prompt).strip().lower()


def review_annotation(story_id: str, data_manager, input_fn=input, logger=None) -> dict:
    """交互式审查标注；返回审查后数据（已写回 data/annotated/{story_id}.json）。"""
    data = data_manager.load_annotation(story_id)

    print(f"=== Review {story_id} ===")
    for sent in data["sentences"]:
        print(f"Sentence {sent['id']}: {sent['text']}")
        print(f"Function: {sent['function']}")
        if _ask("Change? (y/n) ", input_fn) == "y":
            new_fn = _ask(f"New function {FUNCTIONS}: ", input_fn)
            while new_fn not in FUNCTIONS:
                print(f"Invalid function; choose from {FUNCTIONS}")
                new_fn = _ask(f"New function {FUNCTIONS}: ", input_fn)
            sent["function"] = new_fn
            print("Updated.")

    print(f"Critical sentence: {data['critical_sentence']}")
    print(f"Gold answer: {data['gold_answer']}")
    if data.get("question"):
        print(f"Question: {data['question']}")
    if _ask("Modify critical sentence or gold answer? (y/n) ", input_fn) == "y":
        if _ask("Change critical_sentence? (y/n) ", input_fn) == "y":
            data["critical_sentence"] = int(input_fn("New critical_sentence: ").strip())
        if _ask("Change gold_answer? (y/n) ", input_fn) == "y":
            data["gold_answer"] = input_fn("New gold_answer: ").strip()

    ok, errors = AnnotationValidator().validate(data)
    if not ok:
        raise ValueError(f"reviewed data invalid: {'; '.join(errors)}")

    data["reviewed"] = True  # 裁决 C15
    data_manager.save_annotation(story_id, data)
    if logger is not None:
        logger.info(f"reviewed {story_id}: accepted after human review")
    return data
