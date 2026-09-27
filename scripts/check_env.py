#!/usr/bin/env python3
"""Verify that this Python environment IS the locked environment (plan §8.5, decision 18).

Checks, in order:
  1. interpreter: major.minor equals ``.python-version`` (or ``--expect-python``);
  2. lock parity: every distribution ``uv.lock`` pins for this platform is installed at exactly
     the pinned version (read through ``uv export --frozen``, markers evaluated here);
  3. key imports: torch, pytorch_lightning, hydra, omegaconf, transformers, datasets, einops, yaml;
  4. ``lexhybrid`` is installed editable from this repository (no ``sys.path`` shims);
  5. torch build: CUDA 12.8 on Linux (the cluster's driver; decision 18), CPU/MPS on macOS.

Run it everywhere the code runs: the Mac (validate.sh gate 1), every CI job, every cluster job.
Exit code 1 on any failure.

    .venv/bin/python scripts/check_env.py
    .venv/bin/python scripts/check_env.py --expect-python 3.12      # CI forward-compat matrix entry
"""

import argparse
import importlib
import importlib.metadata as md
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
KEY_IMPORTS = (
    "torch",
    "pytorch_lightning",
    "hydra",
    "omegaconf",
    "transformers",
    "datasets",
    "einops",
    "yaml",
)
EXPECTED_CUDA = {"linux": "12.8", "darwin": None}


def expected_python(root: Path = REPO_ROOT) -> str:
    """major.minor from ``.python-version``."""
    return ".".join((root / ".python-version").read_text().strip().split(".")[:2])


def check_python(expected: str, actual=sys.version_info) -> list[str]:
    got = f"{actual[0]}.{actual[1]}"
    return [] if got == expected else [f"python {got} but the environment pins {expected}"]


# CSI escape sequences. uv colours its output when FORCE_COLOR is set, even into a pipe, and the CI
# workflow sets FORCE_COLOR=1: the "# via" annotations then arrived as "\x1b[32m    # via x\x1b[39m"
# and failed to parse (CI run 36322586655).
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_export(text: str) -> list[Requirement]:
    """Requirements from ``uv export --format requirements-txt --no-hashes`` output."""
    reqs = []
    for raw in text.splitlines():
        line = _ANSI.sub("", raw).strip()
        if not line or line.startswith(("#", "-")):
            continue
        reqs.append(Requirement(line))
    return reqs


def compare_installed(reqs: list[Requirement], installed: dict[str, str], env: dict | None = None):
    """(mismatches, missing) for the requirements whose markers apply in ``env``."""
    env = env if env is not None else default_environment()
    mismatches, missing = [], []
    for req in reqs:
        if req.marker is not None and not req.marker.evaluate(env):
            continue
        pins = [s for s in req.specifier if s.operator == "=="]
        if not pins:
            continue
        want = pins[0].version
        name = canonicalize_name(req.name)
        have = installed.get(name)
        if have is None:
            missing.append(f"{req.name}=={want}")
        elif Version(have) != Version(want):
            mismatches.append(f"{req.name}: installed {have}, lock pins {want}")
    return mismatches, missing


def installed_distributions() -> dict[str, str]:
    return {
        canonicalize_name(d.metadata["Name"]): d.version for d in md.distributions() if d.metadata["Name"]
    }


def uv_export(root: Path = REPO_ROOT) -> str:
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        raise FileNotFoundError("uv not found on PATH (the environment is defined by uv.lock; install uv)")
    cmd = [
        uv,
        "export",
        "--frozen",
        "--no-hashes",
        "--no-emit-project",
        "--no-header",
        "--no-annotate",
        "--color",
        "never",
        "--format",
        "requirements-txt",
    ]
    env = {k: v for k, v in os.environ.items() if k not in ("FORCE_COLOR", "CLICOLOR_FORCE")}
    env["NO_COLOR"] = "1"
    return subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True, env=env).stdout


def check_cuda_build(platform: str, cuda_version: str | None) -> list[str]:
    key = "linux" if platform.startswith("linux") else ("darwin" if platform == "darwin" else platform)
    if key not in EXPECTED_CUDA:
        return [f"unsupported platform {platform}"]
    want = EXPECTED_CUDA[key]
    return [] if cuda_version == want else [f"torch CUDA build {cuda_version!r}, expected {want!r} on {key}"]


def check_editable(root: Path = REPO_ROOT) -> list[str]:
    try:
        dist = md.distribution("lexhybrid")
    except md.PackageNotFoundError:
        return ["lexhybrid is not installed in this environment (run `uv sync --locked`)"]
    direct = dist.read_text("direct_url.json")
    info = json.loads(direct) if direct else {}
    if not info.get("dir_info", {}).get("editable", False):
        return ["lexhybrid is installed but not editable (uv sync installs the project editable)"]
    spec = importlib.util.find_spec("lexhybrid")
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    if origin is None or root.resolve() not in origin.parents:
        return [f"lexhybrid resolves to {origin}, not to this repository ({root})"]
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expect-python", default=os.environ.get("LEXHYBRID_EXPECT_PYTHON"))
    ap.add_argument("--no-lock", action="store_true", help="skip the uv.lock parity check")
    ap.add_argument("--allow-non-editable", action="store_true", help="for the wheel-install CI job")
    args = ap.parse_args(argv)

    problems: list[str] = []
    expected = args.expect_python or expected_python()
    problems += check_python(expected)
    print(f"  python       {sys.version.split()[0]} (expected {expected}.x) at {sys.executable}")

    if not args.no_lock:
        try:
            mismatches, missing = compare_installed(parse_export(uv_export()), installed_distributions())
            problems += mismatches + [f"missing: {m}" for m in missing]
            print(f"  uv.lock      {len(mismatches)} version mismatch(es), {len(missing)} missing")
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            problems.append(f"could not read the lock: {exc}")

    for mod in KEY_IMPORTS:
        try:
            m = importlib.import_module(mod)
            print(f"  import       {mod:<18} {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"import {mod} failed: {exc}")

    if not args.allow_non_editable:
        problems += check_editable()

    try:
        import torch

        problems += check_cuda_build(sys.platform, torch.version.cuda)
        print(f"  torch build  cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
    except ImportError:
        pass

    if problems:
        print("ENVIRONMENT CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("ENVIRONMENT OK: interpreter, locked versions, imports, editable install and torch build all match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
