"""Model factory (scaf.md 6.2 L758-796; rulings C4/C14).

Dispatches on the backend field of the config (huggingface/api); hard-coded checks like
`if model=="qwen"` are forbidden. In-process model cache: each model is loaded only once,
and cleared on unload (so the scheduler can reuse it).
APIModel is deferred (forbidden item 4).
"""

from __future__ import annotations

from config.config_manager import ConfigError, ConfigManager
from models.base_model import BaseModel
from models.hf_model import HFModel
from models.vllm_model import VLLMModel


class ModelFactory:
    def __init__(self, config: ConfigManager):
        self.config = config
        self._instances: dict[str, BaseModel] = {}

    def create(self, model_name: str) -> BaseModel:
        """Create (or reuse a cached) model instance; does not load automatically."""
        if model_name in self._instances:
            return self._instances[model_name]

        model_cfg = self.config.get("models", model_name)
        backend = model_cfg.get("backend", "huggingface")
        if backend == "huggingface":
            instance: BaseModel = HFModel(model_cfg)
        elif backend == "vllm":
            instance = VLLMModel(model_cfg)
        elif backend == "api":
            raise NotImplementedError(
                f"backend 'api' for {model_name} is deferred (forbidden item 4 in docs/decisions.md)"
            )
        else:
            raise ConfigError(f"unknown backend {backend!r} for model {model_name!r}")

        self._instances[model_name] = instance
        return instance

    def unload_all(self) -> None:
        """Unload every loaded model and clear the cache."""
        for instance in self._instances.values():
            instance.unload()
        self._instances.clear()
