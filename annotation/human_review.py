"""Human Review System (pipeline.md L547-575 / scaf.md 5.6; ruling C15).

The editable fields are exactly function / gold_answer / critical_sentence (ruling C15);
after review the record is marked with reviewed: true (ruling C15 status field).
"""

from __future__ import annotations

from annotation.schema import FUNCTIONS
from annotation.validator import AnnotationValidator


def _ask(prompt: str, input_fn) -> str:
    """Wrap input so tests can inject a fake one."""
    return input_fn(prompt).strip().lower()


def review_annotation(story_id: str, data_manager, input_fn=input, logger=None) -> dict:
    """Interactively review an annotation; returns the reviewed data (already written back to data/annotated/{story_id}.json)."""
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

    data["reviewed"] = True  # ruling C15
    data_manager.save_annotation(story_id, data)
    if logger is not None:
        logger.info(f"reviewed {story_id}: accepted after human review")
    return data
