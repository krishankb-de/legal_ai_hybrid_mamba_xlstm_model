"""Shared test configuration (plan §8.2).

Markers are declared in pyproject.toml under --strict-markers. Here every marker that depends on
the machine is turned into a skip WITH ITS REASON, never a silent pass:

    linux_only  skipped unless sys.platform == "linux"           (runs in CI)
    cuda        skipped unless a CUDA device is available        (runs on the cluster)
    multigpu    skipped unless >= 2 CUDA devices                 (runs on the cluster)
    network     skipped unless LEXHYBRID_NETWORK_TESTS=1         (weekly CI job)
    reference   skipped when Reference/hybrid_model_mamba_xlstm is absent (after P1-Z)

Tests never reach the Hugging Face Hub unless marked `network`.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = REPO_ROOT / "Reference" / "hybrid_model_mamba_xlstm"

if os.environ.get("LEXHYBRID_NETWORK_TESTS") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _skip_reasons() -> dict:
    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return {
        "linux_only": None
        if sys.platform.startswith("linux")
        else f"needs Linux, running on {sys.platform} (runs in CI)",
        "cuda": None if n_cuda >= 1 else "needs a CUDA GPU (runs on the cluster: scripts/slurm/gpu_tests.sh)",
        "multigpu": None if n_cuda >= 2 else f"needs >= 2 CUDA GPUs, found {n_cuda} (runs on the cluster)",
        "network": None
        if os.environ.get("LEXHYBRID_NETWORK_TESTS") == "1"
        else "live network test; set LEXHYBRID_NETWORK_TESTS=1 (weekly CI job)",
        "reference": None
        if REFERENCE_ROOT.exists()
        else "Reference/ has been deleted (P1-Z); fixtures are the record",
    }


def pytest_collection_modifyitems(config, items):
    reasons = _skip_reasons()
    for item in items:
        for marker, reason in reasons.items():
            if reason is not None and marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _seed_everything():
    """Every test starts from the same RNG state, so failures reproduce."""
    torch.manual_seed(0)
    yield


def load_script(name: str):
    """Import ``scripts/<name>.py`` as a module without touching sys.path."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    """Fixture form of ``load_script``: ``script("evaluate_lm").main([...])``."""
    return load_script
