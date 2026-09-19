"""标注管线（pipeline.md 六节 Step 1-5 / scaf.md 5 节）。

run_annotation_pipeline：load → split → annotate → validate → save。
task 归属（裁决 C11）：由调用方显式传入（CLI --task 或目录名推断）。
"""

from __future__ import annotations

from pathlib import Path

from annotation.annotation_agent import AnnotationAgent, AnnotationError
from annotation.schema import TASKS
from annotation.story_parser import StoryPreprocessor
from annotation.validator import AnnotationValidator


def infer_task(input_dir: str | Path) -> str | None:
    """尝试从目录名推断任务（裁决 C11 的目录推断路径）。"""
    name = Path(input_dir).name
    return name if name in TASKS else None


def run_annotation_pipeline(
    input_dir: str | Path,
    model,
    prompt_template: str,
    data_manager,
    task: str | None = None,
    logger=None,
    limit: int | None = None,
) -> list[dict]:
    """对 input_dir 下全部 *.txt 运行标注管线，返回成功标注列表。

    model 仅需实现 generate(prompt) -> str（AnnotationAgent 的低耦合要求）。
    """
    if task is None:
        task = infer_task(input_dir)
    if task not in TASKS:
        raise AnnotationError(
            f"cannot infer task from {input_dir!r}; pass --task in {TASKS}"
        )

    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.txt"))
    if limit is not None:
        files = files[:limit]

    preprocessor = StoryPreprocessor()
    agent = AnnotationAgent(model, prompt_template)
    validator = AnnotationValidator()
    results: list[dict] = []

    for path in files:
        story_id = path.stem
        raw = data_manager.load_raw_story(path)
        sentences = preprocessor.split_sentence(raw)
        if logger is not None:
            logger.info(f"annotating {story_id}: {len(sentences)} sentences")

        data = agent.annotate(sentences)
        data["id"] = story_id  # 以文件名覆盖 LLM 返回的 id
        data["task"] = task  # 裁决 C11：task 字段
        data["reviewed"] = False  # 裁决 C15：初始未审查

        ok, errors = validator.validate(data)  # 防御性二次校验
        if not ok:
            raise AnnotationError(f"{story_id}: {'; '.join(errors)}")

        data_manager.save_annotation(story_id, data)
        if logger is not None:
            logger.info(f"saved annotation {story_id} -> data/annotated/{story_id}.json")
        results.append(data)

    return results
