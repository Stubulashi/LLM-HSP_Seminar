"""Local HuggingFace model (scaf.md 6.1 L705-755; ruling C1: one HFModel covers all models).

config is the single-model dict from config/models.yaml (path/architecture/backend/size/training).
Optional fields (basis missing; see docs/decisions.md):
    dtype: bfloat16 (default) | float32
    quantization: none (default) | 8bit | 4bit (needs bitsandbytes)
    max_memory: device memory budget (dict form; defaults to device_map auto)
"""

from __future__ import annotations

import gc

from models.base_model import BaseModel


def _model_load_kwargs(config: dict):
    """Build the from_pretrained keyword arguments from a single-model config (pure function, easy to unit-test).

    - quantization none uses torch_dtype; 4bit/8bit uses BitsAndBytesConfig
      (transformers does not accept top-level load_in_4bit/load_in_8bit for Qwen2; it must go through quantization_config).
    """
    import torch
    from transformers import BitsAndBytesConfig

    kwargs: dict = {"device_map": "auto"}
    quantization = config.get("quantization", "none")
    if quantization in ("4bit", "8bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(quantization == "4bit"),
            load_in_8bit=(quantization == "8bit"),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # let bnb decide torch_dtype on its own; passing it here would conflict
    else:
        dtype = config.get("dtype", "bfloat16")
        if dtype:
            kwargs["torch_dtype"] = getattr(torch, dtype)
    if config.get("max_memory"):
        kwargs["max_memory"] = config["max_memory"]
    return kwargs


class HFModel(BaseModel):
    def __init__(self, config: dict):
        self.config = config
        self.path = config["path"]
        self.tokenizer = None
        self.model = None
        self.device = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = self.path
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        kwargs = _model_load_kwargs(self.config)

        self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and not self.config.get("cpu_ok", True):
            raise RuntimeError(f"model {path} requires GPU but CUDA unavailable")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """tokenize → inference → decode (pipeline.md L771-779 / scaf.md L725-744).

        temperature <= 0 means greedy decoding (do_sample=False), which keeps runs reproducible and the judge deterministic;
        otherwise sampling is used (do_sample=True). transformers does not allow temperature=0 together with do_sample=True.
        """
        import torch

        if self.model is None:
            raise RuntimeError("model not loaded; call load() first")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def unload(self) -> None:
        """Delete the model and empty the CUDA cache (an enhanced version of scaf.md L748-754)."""
        import torch

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
