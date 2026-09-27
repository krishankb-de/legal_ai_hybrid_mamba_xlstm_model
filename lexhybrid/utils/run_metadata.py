"""``run_metadata.json``: what produced a checkpoint, written at the start of every run.

Ported from the reference without its ``git`` subprocess calls (repository rule: the model runs
no git command). Provenance is instead the ``.sync_stamp`` written by ``scripts/sync_to_cluster.sh``,
a content hash of the source tree, and the commit id read from ``.git/HEAD`` as a plain file when
a ``.git`` directory is present.
"""

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
TREE_DIRS = ("lexhybrid", "configs", "scripts")
TREE_SUFFIXES = (".py", ".yaml", ".yml", ".sh", ".toml")


def tree_hash(root: Path = REPO_ROOT, dirs: tuple = TREE_DIRS) -> str:
    """sha256 over the sorted (relative path, file sha256) pairs of the source tree."""
    h = hashlib.sha256()
    for d in dirs:
        base = root / d
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file() and p.suffix in TREE_SUFFIXES):
            if "__pycache__" in path.parts:
                continue
            h.update(str(path.relative_to(root)).encode())
            h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def read_sync_stamp(root: Path = REPO_ROOT) -> str:
    """Contents of ``.sync_stamp`` (date + tree hash of the last sync to the cluster), or ""."""
    path = root / ".sync_stamp"
    return path.read_text().strip() if path.exists() else ""


def read_git_head(root: Path = REPO_ROOT) -> str:
    """The checked-out commit id, read from ``.git/HEAD`` and its ref file; "" if unavailable."""
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[len("ref: ") :]
            ref_file = git / ref
            if ref_file.exists():
                return ref_file.read_text().strip()
            packed = git / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split(" ", 1)[0]
            return ""
        return head
    except OSError:
        return ""


def write_run_metadata(cfg: DictConfig, output_dir: str, extra: dict[str, Any] | None = None) -> Path:
    """Persist ``<output_dir>/run_metadata.json`` and return its path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "sync_stamp": read_sync_stamp(),
        "tree_hash": tree_hash(),
        "git_head": read_git_head(),
        "python_version": sys.version.split()[0],
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    if extra:
        metadata.update(extra)
    path = out_dir / "run_metadata.json"
    with path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[run_metadata] wrote {path}")
    return path
