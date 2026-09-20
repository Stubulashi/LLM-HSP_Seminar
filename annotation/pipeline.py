"""Annotation pipeline (pipeline.md section 6, steps 1-5 / scaf.md section 5).

run_annotation_pipeline: load → split → annotate → validate → save.
Task assignment (ruling C11): passed in explicitly by the caller (CLI --task, or inferred from the directory name).
"""

from __future__ import annotations

from pathlib import Path

from annotation.annotation_agent import AnnotationAgent, AnnotationError
from annotation.schema import TASKS
from annotation.story_parser import StoryPreprocessor
from annotation.validator import AnnotationValidator


def infer_task(input_dir: str | Path) -> str | None:
    """Try to infer the task from the directory name (the directory-inference path of ruling C11)."""
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
    """Run the annotation pipeline over every *.txt in input_dir; returns the list of successful annotations.

    model only needs to implement generate(prompt) -> str (the low-coupling requirement of AnnotationAgent).
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
        data["id"] = story_id  # override the LLM-returned id with the file name
        data["task"] = task  # ruling C11: task field
        data["reviewed"] = False  # ruling C15: not reviewed initially

        ok, errors = validator.validate(data)  # defensive second validation
        if not ok:
            raise AnnotationError(f"{story_id}: {'; '.join(errors)}")

        data_manager.save_annotation(story_id, data)
        if logger is not None:
            logger.info(f"saved annotation {story_id} -> data/annotated/{story_id}.json")
        results.append(data)

    return results
