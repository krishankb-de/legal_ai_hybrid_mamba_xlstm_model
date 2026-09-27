"""`scripts/sync_to_cluster.sh` and `scripts/sync_from_cluster.sh` (plan P4-D, rule f).

Each script runs from a copy inside a temporary tree, with fake `ssh` and `rsync` on PATH that
record their arguments, so nothing reaches the network or the real repository.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

TO = REPO_ROOT / "scripts" / "sync_to_cluster.sh"
FROM = REPO_ROOT / "scripts" / "sync_from_cluster.sh"
ENV = "CLUSTER_HOST=hpc\nCLUSTER_REPO=/sc/home/u/lexhybrid\nSCRATCH_ROOT=/sc/scratch/u/lexhybrid\n"


def _load_tree_hash():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_metadata", REPO_ROOT / "lexhybrid/utils/run_metadata.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.tree_hash


def make_tree(tmp_path: Path, env: str | None = ENV) -> tuple[Path, Path, Path]:
    """A repo copy with both scripts, run_metadata.py and .rsync-exclude; fake ssh/rsync in bin/."""
    root = tmp_path / "repo"
    for rel in (
        "scripts/sync_to_cluster.sh",
        "scripts/sync_from_cluster.sh",
        "lexhybrid/utils/run_metadata.py",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, root / rel)
    shutil.copy(REPO_ROOT / ".rsync-exclude", root / ".rsync-exclude")
    (root / "configs" / "x.yaml").parent.mkdir(parents=True)
    (root / "configs" / "x.yaml").write_text("a: 1\n")
    if env is not None:
        (root / "scripts" / "slurm").mkdir(parents=True)
        (root / "scripts" / "slurm" / "cluster.env").write_text(env)
    bin_dir, calls = tmp_path / "bin", tmp_path / "calls.log"
    bin_dir.mkdir()
    for tool in ("ssh", "rsync"):
        fake = bin_dir / tool
        fake.write_text(f'#!/usr/bin/env bash\nprintf \'%s\\n\' "{tool} $*" >> "{calls}"\n')
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return root, bin_dir, calls


def run(root: Path, bin_dir: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "PYTHON": sys.executable}
    env.pop("CLUSTER_ENV", None)
    return subprocess.run(
        ["bash", str(root / "scripts" / script), *args], env=env, capture_output=True, text=True, cwd=root
    )


def calls(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def test_sync_to_writes_stamp_makes_dirs_and_rsyncs_with_delete(tmp_path):
    root, bin_dir, log = make_tree(tmp_path)
    res = run(root, bin_dir, "sync_to_cluster.sh")
    assert res.returncode == 0, res.stderr
    stamp = (root / ".sync_stamp").read_text().strip()
    m = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) ([0-9a-f]{64})", stamp)
    assert m, f"stamp is not '<ISO date> <sha256>': {stamp!r}"
    assert m.group(2) == _load_tree_hash()(root), "stamp hash differs from run_metadata.tree_hash"
    got = calls(log)
    assert got[0] == "ssh hpc mkdir -p '/sc/home/u/lexhybrid' '/sc/scratch/u/lexhybrid'"
    assert got[1] == f"rsync -az --delete --exclude-from={root}/.rsync-exclude ./ hpc:/sc/home/u/lexhybrid/"
    assert len(got) == 2


def test_sync_to_dry_run_touches_nothing_remote_and_writes_no_stamp(tmp_path):
    root, bin_dir, log = make_tree(tmp_path)
    res = run(root, bin_dir, "sync_to_cluster.sh", "--dry-run")
    assert res.returncode == 0, res.stderr
    assert not (root / ".sync_stamp").exists()
    got = calls(log)
    assert len(got) == 1 and got[0].startswith("rsync ") and "--dry-run" in got[0]


@pytest.mark.parametrize("script", ["sync_to_cluster.sh", "sync_from_cluster.sh"])
def test_missing_cluster_env_fails_before_any_remote_call(tmp_path, script):
    root, bin_dir, log = make_tree(tmp_path, env=None)
    args = ["logs"] if script == "sync_from_cluster.sh" else []
    res = run(root, bin_dir, script, *args)
    assert res.returncode != 0 and "cluster.env" in res.stderr
    assert calls(log) == []


def test_sync_to_refuses_an_incomplete_env(tmp_path):
    root, bin_dir, log = make_tree(tmp_path, env="CLUSTER_HOST=hpc\nCLUSTER_REPO=/r\n")
    res = run(root, bin_dir, "sync_to_cluster.sh")
    assert res.returncode != 0 and "SCRATCH_ROOT" in res.stderr
    assert calls(log) == []


@pytest.mark.parametrize("subdir", ["logs", "analysis"])
def test_sync_from_pulls_into_cluster_dir_without_delete(tmp_path, subdir):
    root, bin_dir, log = make_tree(tmp_path)
    res = run(root, bin_dir, "sync_from_cluster.sh", subdir)
    assert res.returncode == 0, res.stderr
    assert calls(log) == [f"rsync -az hpc:/sc/home/u/lexhybrid/{subdir}/ cluster/{subdir}/"]
    assert (root / "cluster" / subdir).is_dir()


@pytest.mark.parametrize("args", [[], ["outputs"], ["logs/../outputs"], ["data"], ["logs", "analysis"]])
def test_sync_from_refuses_anything_but_logs_or_analysis(tmp_path, args):
    root, bin_dir, log = make_tree(tmp_path)
    res = run(root, bin_dir, "sync_from_cluster.sh", *args)
    assert res.returncode == 2
    assert calls(log) == []


def test_scripts_run_no_git():
    for script in (TO, FROM):
        code = "\n".join(ln for ln in script.read_text().splitlines() if not ln.lstrip().startswith("#"))
        assert not re.search(r"\bgit\b", code), f"{script.name} calls git (rule d)"
