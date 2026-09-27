"""Load a model yaml into a ``HybridConfig`` -- the one path every script uses (FM5).

A name resolves to ``configs/model/<name>.yaml`` in the repository; an existing path is used as is.
The yaml goes through ``HybridConfig.from_hydra``, never a hand-written kwarg list.
"""

from pathlib import Path
from typing import Any

import yaml

from lexhybrid.config.hybrid_config import HybridConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_DIR = REPO_ROOT / "configs" / "model"


def resolve_model_yaml(name_or_path: str | Path) -> Path:
    """``configs/model/<name>.yaml`` for a bare name, else the given path; raises if missing."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
        return candidate
    path = MODEL_CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        known = sorted(p.stem for p in MODEL_CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no model config {name_or_path!r}; known: {known}")
    return path


def available_model_configs() -> list[str]:
    """Names of every yaml under ``configs/model``."""
    return sorted(p.stem for p in MODEL_CONFIG_DIR.glob("*.yaml"))


def load_model_yaml(name_or_path: str | Path) -> dict[str, Any]:
    """The raw yaml mapping (model fields plus the training keys a model yaml carries)."""
    with open(resolve_model_yaml(name_or_path)) as f:
        return yaml.safe_load(f) or {}


def load_model_config(name_or_path: str | Path, **overrides) -> HybridConfig:
    """``HybridConfig.from_hydra`` over the yaml, with ``overrides`` applied last."""
    return HybridConfig.from_hydra(load_model_yaml(name_or_path), **overrides)
