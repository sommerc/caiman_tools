"""Loading and merging of CNMF-E pipeline parameters."""

from __future__ import annotations

import copy
import importlib.resources
import json
from pathlib import Path
from typing import Any


def _load_default_params() -> dict[str, Any]:
    data = importlib.resources.files("caiman_tools").joinpath("default_cnmfe_params.json")
    with data.open("r") as f:
        return json.load(f)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_params(user_params_path: Path | None) -> dict[str, Any]:
    """Load default CNMF-E params, deep-merging a user-supplied JSON on top."""
    params = _load_default_params()
    if user_params_path is not None:
        with open(user_params_path) as f:
            user_params = json.load(f)
        params = _deep_merge(params, user_params)
    return params
