"""Load experiment settings."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seeds": [0],
    "embedding": {"embed_dim": 256, "deg": 64},
    "model": {
        "hidden_dim": 256,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "dropout": 0.5,
        "branch_dim": None,
        "spectral_threshold": 1.0,
    },
    "train": {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "epochs": 100,
        "warmup_epochs": 0,
        "grad_clip": 0.1,
        "batch_size": 1,
        "eval_freq": 1,
        "patience": None,
        "final_training": False,
        "selection_metric": None,
        "compile": True,
        "num_workers": 0,
    },
    "root": "data",
    "output_dir": "results",
    "tensorboard_dir": None,
}
_SECTIONS = tuple(key for key, value in DEFAULTS.items() if isinstance(value, dict))


class _Loader(yaml.SafeLoader):
    """YAML loader that also reads exponent floats without a decimal point, such as ``2e-5``."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float", re.compile(r"^[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)[eE][-+]?[0-9]+$"), list("-+.0123456789")
)


def _invalid(key: str, reason: str) -> ValueError:
    return ValueError(f"{key} {reason}")


def _merge(base: dict[str, Any], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if len(parts) == 1:
        config[key] = value
    elif len(parts) == 2 and isinstance(config.get(parts[0]), dict):
        config[parts[0]][parts[1]] = value
    else:
        raise _invalid(key, "is not a configuration key")


def _validate_keys(config: Mapping[str, Any]) -> None:
    for key in config:
        if key not in DEFAULTS and key != "dataset":
            raise _invalid(key, "is not a configuration key")
    for section in _SECTIONS:
        values = config[section]
        if not isinstance(values, Mapping):
            raise _invalid(section, "must be a mapping")
        for key in values:
            if key not in DEFAULTS[section]:
                raise _invalid(f"{section}.{key}", "is not a configuration key")


def _validate(config: Mapping[str, Any]) -> None:
    _validate_keys(config)
    if not isinstance(config.get("dataset"), str) or not config["dataset"]:
        raise _invalid("dataset", "must be a nonempty string")
    if (
        not isinstance(config["seeds"], list)
        or not config["seeds"]
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in config["seeds"])
    ):
        raise _invalid("seeds", "must be a nonempty list of integers")
    width = config["model"]["branch_dim"]
    if width is not None and (
        not isinstance(width, int) or isinstance(width, bool) or width <= 0 or width % config["model"]["num_heads"]
    ):
        raise _invalid("model.branch_dim", "must be a positive multiple of model.num_heads")
    threshold = config["model"]["spectral_threshold"]
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold):
        raise _invalid("model.spectral_threshold", "must be a finite number")


def _resolve(config: Mapping[str, Any], overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """Apply defaults and overrides, then validate the resulting settings."""
    resolved = deepcopy(DEFAULTS)
    _merge(resolved, config)
    for override in overrides or ():
        if "=" not in override:
            raise _invalid(override, "must have key=value form")
        key, raw_value = override.split("=", 1)
        _set_dotted(resolved, key, yaml.load(raw_value, Loader=_Loader))
    _validate(resolved)
    return resolved


def resolve_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply defaults to a configuration and validate its keys.

    Args:
        config: Experiment settings to merge with the built-in defaults.

    Returns:
        A complete, independent configuration dictionary.
    """
    return _resolve(config)


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """Load YAML settings and apply dotted command-line overrides.

    Args:
        path: YAML configuration path.
        overrides: Entries of the form ``section.key=value``.

    Returns:
        A validated configuration with defaults applied.
    """
    loaded = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_Loader)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise _invalid(str(path), "must contain a mapping")
    return _resolve(loaded, overrides)
