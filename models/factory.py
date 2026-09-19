"""模型工厂（scaf.md 6.2 L758-796，裁决 C4/C14）。

按 config 的 backend 字段分发（huggingface/api），禁止 `if model=="qwen"` 式硬编码；
同进程模型缓存：同一模型只 load 一次，unload 后清除（供 scheduler 复用）。
APIModel 延后实现（禁止事项 4）。
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
        """创建（或复用缓存中的）模型实例，不自动 load。"""
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
                f"backend 'api' for {model_name} 延后实现（docs/decisions.md 禁止事项 4）"
            )
        else:
            raise ConfigError(f"unknown backend {backend!r} for model {model_name!r}")

        self._instances[model_name] = instance
        return instance

    def unload_all(self) -> None:
        """卸载全部已加载模型并清空缓存。"""
        for instance in self._instances.values():
            instance.unload()
        self._instances.clear()
