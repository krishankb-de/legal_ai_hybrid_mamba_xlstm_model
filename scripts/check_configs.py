#!/usr/bin/env python3
"""validate.sh gate 3: every model yaml composes through Hydra and pins what it must.

Checks, for each ``configs/model/*.yaml``:
  * ``compose("config", ["model=<name>", "dataset=synthetic", "trainer=cpu_debug"])`` succeeds and
    ``HybridConfig.from_hydra(cfg.model)`` builds (every key a mixer reads reaches the dataclass);
  * a non-empty ``layer_pattern``;
  * operators pinned explicitly in the yaml itself (R8): ``tfla_impl`` when the pattern has an
    ``mlstm``, ``scan_impl`` when it has a ``mamba`` -- inheriting the dataclass default is how the
    reference trained an arm with the wrong operator;
  * legacy operators only where allowed: ``hybrid_legal_legacy`` and the ``ref_*`` replicas;
  * vocabulary: 151,936 for the legal yamls (Qwen3, decision 6), 50,257 for the ``ref_*`` replicas.

Exit code 0 when every yaml passes.
"""

import sys
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from lexhybrid.config.hybrid_config import HybridConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
LEGACY_ALLOWED = ("hybrid_legal_legacy", "ref_")
LEGAL_VOCAB = 151_936
REFERENCE_VOCAB = 50_257


def model_names() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "model").glob("*.yaml"))


def compose_model(name: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[f"model={name}", "dataset=synthetic", "trainer=cpu_debug", "experiment_name=ci_check"],
        )


def check_model(name: str) -> list[str]:
    """Problems with ``configs/model/<name>.yaml``; empty when it passes."""
    problems = []
    raw = yaml.safe_load((CONFIG_DIR / "model" / f"{name}.yaml").read_text()) or {}
    try:
        cfg = compose_model(name)
        config = HybridConfig.from_hydra(cfg.model)
    except Exception as exc:  # noqa: BLE001 -- report every failure, not just the first class
        return [f"{name}: does not compose/build: {exc}"]
    pattern = list(config.layer_pattern)
    if not pattern:
        problems.append(f"{name}: empty layer_pattern")
    if "mlstm" in pattern and "tfla_impl" not in raw:
        problems.append(f"{name}: has mlstm layers but does not pin tfla_impl")
    if "mamba" in pattern and "scan_impl" not in raw:
        problems.append(f"{name}: has mamba layers but does not pin scan_impl")
    legacy_ok = name.startswith(LEGACY_ALLOWED)
    if not legacy_ok:
        if "mlstm" in pattern and config.tfla_impl == "legacy":
            problems.append(f"{name}: tfla_impl legacy outside hybrid_legal_legacy / ref_* (R8)")
        if "mamba" in pattern and config.scan_impl == "legacy":
            problems.append(f"{name}: scan_impl legacy outside hybrid_legal_legacy / ref_* (R8)")
    if name.startswith("ref_") and config.vocab_size != REFERENCE_VOCAB:
        problems.append(
            f"{name}: reference replica must keep vocab {REFERENCE_VOCAB}, has {config.vocab_size}"
        )
    if not name.startswith("ref_") and config.vocab_size != LEGAL_VOCAB:
        problems.append(
            f"{name}: legal config must use vocab {LEGAL_VOCAB} (decision 6), has {config.vocab_size}"
        )
    return problems


def main(argv=None) -> int:
    names = model_names()
    problems = [p for name in names for p in check_model(name)]
    for name in names:
        print(f"  {'FAIL' if any(p.startswith(name + ':') for p in problems) else 'ok  '} {name}")
    for p in problems:
        print(f"  - {p}")
    print(f"{len(names)} model configs, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
