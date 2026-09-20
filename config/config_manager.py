"""Configuration management (scaf.md section 2; ruling C1: core/config_manager.py → config/config_manager.py).

Interface: load / get / update (scaf.md L204-239); a missing key raises ConfigError instead of silently returning None.
"""

from __future__ import annotations

from pathlib import Path

import yaml


class ConfigError(KeyError):
    """Configuration missing or failed to load."""


class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config: dict = {}
        self.load(self.config_dir / "models.yaml")
        self.load(self.config_dir / "experiment.yaml")
        self.load(self.config_dir / "prompts.yaml")

    def load(self, path: str | Path) -> None:
        """Read and parse a single YAML file, then merge it into the global config (scaf.md L212-222)."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ConfigError(f"config file must contain a mapping: {path}")
        self.config.update(data)

    def get(self, section: str, key: str | None = None):
        """Get a value; a missing key raises instead of returning None silently (ruling: ConfigManager raises on missing keys).

        get("models")           -> dict of all models
        get("models", "qwen7b") -> dict of a single model
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
        """Update the in-memory config (scaf.md L232-237); nothing is written to disk."""
        self.config[key] = value
