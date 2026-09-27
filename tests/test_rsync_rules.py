"""`.rsync-exclude` does what `scripts/sync_to_cluster.sh` needs (plan P1-A, P4-D).

rsync is first-match-wins and has no `!` negation: an include for the test fixtures must be a
`+ ` line placed before the `*.pt` exclude, or the parity fixtures never reach the cluster and its
preflight fails. The static tests read the file; the behavioural test runs the local rsync
(openrsync on macOS, GNU rsync on the Linux CI runner) on a temporary tree.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

RULES = REPO_ROOT / ".rsync-exclude"
MUST_EXCLUDE = (".venv/", "Reference/", "outputs/", "logs/", "cluster/", ".git/", "scripts/slurm/cluster.env")


def rules() -> list[str]:
    return [
        ln.rstrip("\n")
        for ln in RULES.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith(("#", ";"))
    ]


def test_no_gitignore_style_negation():
    bad = [ln for ln in rules() if ln.startswith("!") and ln.strip() != "!"]
    assert not bad, f"'!' is not a negation in rsync (a bare '!' clears the list): {bad}"


def test_fixture_include_precedes_the_weight_excludes():
    lines = rules()
    include = lines.index("+ tests/fixtures/**")
    for pattern in ("*.pt", "*.ckpt", "*.parquet"):
        assert include < lines.index(pattern), f"'+ tests/fixtures/**' must come before {pattern}"


@pytest.mark.parametrize("pattern", MUST_EXCLUDE)
def test_private_and_bulky_paths_are_excluded(pattern):
    assert pattern in rules()


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed on this machine")
def test_rsync_copies_fixtures_and_nothing_private(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    files = {
        "lexhybrid/__init__.py": "x",
        "tests/fixtures/reference_parity/model.pt": "fixture",
        "tests/fixtures/deep/a/b.parquet": "fixture",
        "stray_weights.pt": "w",
        "outputs/run/last.ckpt": "c",
        "Reference/hybrid_model_mamba_xlstm/README.md": "r",
        ".venv/bin/python": "p",
        ".git/HEAD": "ref",
        "scripts/slurm/cluster.env": "CLUSTER_HOST=x",
        "logs/job_1.log": "l",
    }
    for rel, text in files.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text)
    dst.mkdir()
    subprocess.run(
        ["rsync", "-a", f"--exclude-from={RULES}", f"{src}/", f"{dst}/"],
        check=True,
        capture_output=True,
    )
    copied = {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()}
    assert copied == {
        "lexhybrid/__init__.py",
        "tests/fixtures/reference_parity/model.pt",
        "tests/fixtures/deep/a/b.parquet",
    }, f"unexpected sync result: {sorted(copied)}"


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed on this machine")
def test_delete_keeps_what_the_cluster_owns(tmp_path):
    """`sync_to_cluster.sh` runs `rsync --delete`; excluded paths are protected on the receiver,
    so the cluster's venv, logs and checkpoints survive while stale code is removed."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "lexhybrid").mkdir(parents=True)
    (src / "lexhybrid" / "__init__.py").write_text("x")
    kept = ["logs/train_42.log", "outputs/run/last.ckpt", ".venv/bin/python", "cluster/logs/a.log"]
    for rel in [*kept, "lexhybrid/stale.py"]:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        (dst / rel).write_text("cluster side")
    subprocess.run(
        ["rsync", "-a", "--delete", f"--exclude-from={RULES}", f"{src}/", f"{dst}/"],
        check=True,
        capture_output=True,
    )
    assert all((dst / rel).exists() for rel in kept), "rsync --delete removed cluster-owned files"
    assert not (dst / "lexhybrid" / "stale.py").exists(), "stale code should be deleted"


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed on this machine")
def test_every_source_file_reaches_the_cluster(tmp_path):
    """Job 2588703: unanchored `data/probes/` also dropped `lexhybrid/data/probes/`. Mirror every
    source file of the repo (empty copies) and check rsync keeps all of them."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    sources = [
        p.relative_to(REPO_ROOT).as_posix()
        for d in ("lexhybrid", "scripts", "configs", "tests")
        for p in (REPO_ROOT / d).rglob("*")
        if p.is_file() and p.suffix in {".py", ".sh", ".yaml", ".yml"} and "__pycache__" not in p.parts
    ]
    sources = [s for s in sources if s != "scripts/slurm/cluster.env"]
    assert any(s.startswith("lexhybrid/data/probes/") for s in sources)
    for rel in sources:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).touch()
    dst.mkdir()
    subprocess.run(
        ["rsync", "-a", f"--exclude-from={RULES}", f"{src}/", f"{dst}/"], check=True, capture_output=True
    )
    missing = sorted(s for s in sources if not (dst / s).exists())
    assert not missing, f"excluded by .rsync-exclude but needed on the cluster: {missing[:10]}"


def test_data_excludes_are_anchored_to_the_root():
    unanchored = [ln for ln in rules() if ln.startswith("data/")]
    assert not unanchored, f"anchor with a leading '/', else lexhybrid/data/... is excluded too: {unanchored}"


def test_rules_file_is_where_the_sync_script_will_look():
    assert RULES == Path(REPO_ROOT, ".rsync-exclude") and RULES.is_file()
