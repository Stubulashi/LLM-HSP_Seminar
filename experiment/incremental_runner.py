"""Incremental Runner (pipeline.md section 7 L609-700 / scaf.md section 7 L839-898).

Accumulate context sentence by sentence → build prompt → model.generate → record; every
step is persisted immediately, so a crash never loses earlier steps. Record fields follow
ruling C3: model/story_id/task/step/context/response/confidence/repetition/metadata.
confidence is None at the raw stage and filled in by the analysis pipeline (scaf.md Principle 3: raw → processed).
"""

from __future__ import annotations

import sys

from datetime import datetime

from tqdm import tqdm


class IncrementalRunner:
    def __init__(self, prompt_builder, recorder, logger=None):
        self.prompt_builder = prompt_builder
        self.recorder = recorder
        self.logger = logger

    def run_story(
        self,
        story: dict,
        model,
        model_name: str,
        repetition: int,
        temperature: float,
        seed: int,
    ) -> list[dict]:
        """Run the incremental experiment on one story and return the records; every step is already on disk."""
        context_parts: list[str] = []
        records: list[dict] = []
        n_total = len(story["sentences"])
        bar = tqdm(total=n_total, unit="step", file=sys.stdout,
                   desc=f"{model_name}/{story['id']}", dynamic_ncols=True)

        for sentence in story["sentences"]:
            context_parts.append(sentence["text"])
            context = " ".join(context_parts)

            prompt = self.prompt_builder.build(
                context=context,
                task=story.get("task", "default"),
                question=story.get("question"),
            )
            response = model.generate(prompt)

            record = {
                "model": model_name,
                "story_id": story["id"],
                "task": story.get("task", "default"),
                "step": sentence["id"],
                "context": context,
                "response": response,
                "confidence": None,  # filled in by analysis (scaf.md Principle 3)
                "repetition": repetition,
                "metadata": {
                    "model_version": getattr(model, "path", model_name),
                    "prompt_version": self.prompt_builder.config.get("prompt_version", "?"),
                    "dataset_version": self.recorder.dataset_version(story["id"]),
                    "temperature": temperature,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            }
            self.recorder.save_step(record, model_name, story["id"], repetition)
            bar.update(1)
            bar.set_description(
                f"{model_name}/{story['id']} step {sentence['id']}/{n_total}"
            )
            if self.logger is not None:
                self.logger.info(
                    f"{model_name}/{story['id']} step {sentence['id']} saved"
                )
            records.append(record)

        bar.close()
        return records
