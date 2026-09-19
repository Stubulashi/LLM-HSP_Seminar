"""配置管理系统（scaf.md 2 节；裁决 C1：core/config_manager.py → config/config_manager.py）。

接口：load / get / update（scaf.md L204-239）；缺失键抛 ConfigError 而非静默返回 None。
"""

from __future__ import annotations

from pathlib import Path

import yaml


class ConfigError(KeyError):
    """配置缺失或加载失败。"""


class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config: dict = {}
        self.load(self.config_dir / "models.yaml")
        self.load(self.config_dir / "experiment.yaml")
        self.load(self.config_dir / "prompts.yaml")

    def load(self, path: str | Path) -> None:
        """读取并解析单个 yaml 文件，合并入全局配置（scaf.md L212-222）。"""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ConfigError(f"config file must contain a mapping: {path}")
        self.config.update(data)

    def get(self, section: str, key: str | None = None):
        """取值；缺失键报错而非静默 None（裁决：ConfigManager 缺失键报错）。

        get("models")           -> 全部模型 dict
        get("models", "qwen7b") -> 单个模型 dict
        """
        if section not in self.config:
            raise ConfigError(f"config section not found: {section!r}")
        value = self.config[section]
        if key is not None:
            if key not in value:
                raise ConfigError(f"config key not found: {section}.{key}")
            value = value[key]
        return value

    def update(self, key: str, value) -> None:
        """修改配置（scaf.md L232-237）；不落盘，仅内存态。"""
        self.config[key] = value
