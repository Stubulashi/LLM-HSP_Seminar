"""本地 HuggingFace 模型（scaf.md 6.1 L705-755，裁决 C1：统一 HFModel 覆盖全部模型）。

config 为 config/models.yaml 中单模型 dict（含 path/architecture/backend/size/training）。
可选字段（依据缺失，见 docs/decisions.md）：
    dtype: bfloat16（默认）| float32
    quantization: none（默认）| 8bit | 4bit（需 bitsandbytes）
    max_memory: 设备显存预算（dict 形式，缺省按 device_map auto）
"""

from __future__ import annotations

import gc

from models.base_model import BaseModel


def _model_load_kwargs(config: dict):
    """按单模型配置生成 from_pretrained 的关键字参数（纯函数，便于单测）。

    - quantization none 走 torch_dtype；4bit/8bit 走 BitsAndBytesConfig
      （transformers Qwen2 不认顶层 load_in_4bit/load_in_8bit，须经 quantization_config）。
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
        # bnb 自行决定 torch_dtype，避免同时传冲突
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
        """tokenize → inference → decode（pipeline.md L771-779 / scaf.md L725-744）。

        temperature <= 0 时按贪心解码（do_sample=False），保证可复现/用作 judge 的确定性；
        否则按采样（do_sample=True）。transformers 不允许 temperature=0 与 do_sample=True 并用。
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
        """删除模型并清空 CUDA 缓存（scaf.md L748-754 增强版）。"""
        import torch

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
