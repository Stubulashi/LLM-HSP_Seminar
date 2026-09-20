"""Model base class (pipeline.md section 9 L738-793 / scaf.md section 6 L677-701).

Every model must implement load / generate / unload; the experiment system only depends
on generate(prompt) -> str (low coupling; scaf.md Principle 1).
Optional extension: generate_batch(prompts) lets engines such as vLLM run batches
concurrently (to make full use of the GPU); the default implementation loops over
generate one by one, so every backend works.
"""

from __future__ import annotations


class BaseModel:
    def load(self) -> None:
        """Load the model and tokenizer."""

    def generate(self, prompt: str) -> str:
        """Take a prompt and return the text response (pipeline.md L740-744)."""
        raise NotImplementedError

    def generate_batch(self, prompts: list[str], seeds: list[int] | None = None) -> list[str]:
        """Batch generation; by default falls back to generate one by one. Batch engines (vLLM) override this to use the GPU fully."""
        return [self.generate(p) for p in prompts]

    def unload(self) -> None:
        """Release GPU memory (scaf.md L748-754)."""
