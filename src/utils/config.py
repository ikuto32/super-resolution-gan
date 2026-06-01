"""Configuration loading utilities for YAML experiment files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ConfigDict = dict[str, Any]

DEFAULT_REQUIRED_KEYS: tuple[str, ...] = (
    "project.name",
    "project.output_dir",
    "seed",
    "data.train_dir",
    "data.val_dir",
    "data.image_size_hr",
    "data.image_size_lr",
    "data.batch_size",
    "data.num_workers",
    "degradation.downsample.method",
    "degradation.downsample.scale",
    "model.generator",
    "model.discriminator",
    "loss",
    "training.epochs",
    "optimizer.generator",
    "optimizer.discriminator",
)


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


def read_yaml(path: str | Path) -> ConfigDict:
    """Read a YAML mapping from ``path``.

    Empty YAML files are treated as empty dictionaries. Non-mapping top-level
    YAML values are rejected because experiment configuration files are expected
    to be key/value documents.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"Configuration at {config_path} must contain a YAML mapping."
        raise ConfigError(msg)
    return data


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge two dictionaries without mutating either input.

    Values from ``override`` replace values from ``base``. If both values for a
    key are dictionaries, they are recursively merged.
    """

    merged: ConfigDict = deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_dicts(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def get_nested(config: Mapping[str, Any], key_path: str) -> Any:
    """Return the value at a dot-separated key path.

    Raises ``KeyError`` if any path component is missing or if traversal reaches
    a non-mapping value before the final key.
    """

    current: Any = config
    for key in key_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(key_path)
        current = current[key]
    return current


def mapping_section(
    config: Mapping[str, Any] | None,
    key: str,
    default: Mapping[str, Any] | None = None,
) -> ConfigDict:
    """Return a config subsection as a plain dictionary.

    Missing sections use ``default`` when it is a mapping, otherwise an empty
    dictionary. Non-mapping section values are treated as empty dictionaries so
    callers can safely chain ``.get(...)`` calls. For nested sections, pass the
    result of :func:`get_nested` as ``config`` or ``default``.
    """

    fallback: Mapping[str, Any] = default or {}
    value: Any = config.get(key, fallback) if isinstance(config, Mapping) else fallback
    return dict(value) if isinstance(value, Mapping) else {}


def validate_required_keys(
    config: Mapping[str, Any],
    required_keys: Sequence[str] = DEFAULT_REQUIRED_KEYS,
) -> None:
    """Validate that all dot-separated ``required_keys`` exist in ``config``."""

    missing: list[str] = []
    for key_path in required_keys:
        try:
            get_nested(config, key_path)
        except KeyError:
            missing.append(key_path)

    if missing:
        joined = ", ".join(missing)
        msg = f"Missing required configuration keys: {joined}"
        raise ConfigError(msg)


def load_config(
    path: str | Path,
    *,
    default_path: str | Path | None = None,
    required_keys: Sequence[str] = DEFAULT_REQUIRED_KEYS,
) -> ConfigDict:
    """Load and validate a YAML configuration file.

    When ``default_path`` is provided, the default configuration is loaded first
    and the configuration at ``path`` is recursively merged on top of it.
    """

    config = read_yaml(path)
    if default_path is not None:
        defaults = read_yaml(default_path)
        config = merge_dicts(defaults, config)

    validate_required_keys(config, required_keys)
    return config
