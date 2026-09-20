"""annotation package: the annotation pipeline (data entry point; pipeline.md section 6 / scaf.md section 5)."""

from annotation.annotation_agent import AnnotationAgent, AnnotationError
from annotation.human_review import review_annotation
from annotation.json_parser import JsonParseError, parse_json
from annotation.pipeline import run_annotation_pipeline
from annotation.schema import FUNCTIONS, TASKS
from annotation.story_parser import StoryPreprocessor
from annotation.validator import AnnotationValidator

__all__ = [
    "AnnotationAgent",
    "AnnotationError",
    "AnnotationValidator",
    "FUNCTIONS",
    "JsonParseError",
    "StoryPreprocessor",
    "TASKS",
    "parse_json",
    "review_annotation",
    "run_annotation_pipeline",
]
