"""annotation 包：标注管线（数据入口，pipeline.md 六节 / scaf.md 5 节）。"""

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
