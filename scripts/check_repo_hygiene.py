#!/usr/bin/env python3
"""Repository hygiene: the tree that would be (or was) pushed is clean (plan §8.1, P1-W2).

Two modes, same checks:
  default  walk the working tree minus what ``.gitignore`` excludes -- what a commit would contain.
           ``validate.sh`` uses this in every mode (it runs after the venv and caches exist).
  --ci     walk the checkout minus ``.git/``. The CI ``hygiene`` job runs this on a pristine
           checkout, before anything is installed, so the walked tree IS the pushed tree. Needs no
           third-party package (``pathspec`` is imported only in the default mode).

Fails on: a file over the size limit; checkpoints, tensors or data files outside
``tests/fixtures/``; anything under ``Reference/``, ``.venv/``, ``outputs/``, ``logs/``,
``cluster/``, ``data/raw/``, ``data/shards/``; CRLF line endings in text files; two paths that
differ only by case (they break checkouts on macOS); secret-shaped strings; shell scripts without a
shebang; and critical files that are absent or that ``.gitignore`` would silently exclude (an
analysis document once reported as committed was in fact ignored in the reference).
"""

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_MB = 5.0
BINARY_DATA_SUFFIXES = {
    ".ckpt",
    ".pt",
    ".pth",
    ".safetensors",
    ".parquet",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".bin",
    ".arrow",
}
DATA_ALLOWED_UNDER = ("tests/fixtures/",)
FORBIDDEN_PREFIXES = (
    "Reference/",
    ".venv/",
    ".venv-vllm/",
    "outputs/",
    "logs/",
    "cluster/",
    "data/raw/",
    "data/shards/",
    "data/hf/",
)
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".md",
    ".toml",
    ".json",
    ".jsonl",
    ".txt",
    ".cfg",
    ".ini",
    ".env",
}
CRITICAL_FILES = (
    "README.md",
    "CLAUDE.md",
    "LEGAL_BUILD_PLAN.md",
    "legal_build_state.json",
    "uv.lock",
    "pyproject.toml",
    ".python-version",
    ".gitignore",
    ".github/workflows/ci.yml",
)
MUST_NOT_BE_IGNORED_GLOBS = ("Docs/*.md", "analysis/*.md", "analysis/**/*.md", "tests/fixtures/**/*")
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
}


def load_ignore(root: Path):
    """``.gitignore`` as a ``pathspec`` matcher (git's own wildmatch semantics)."""
    import pathspec  # dev dependency; not needed for --ci on a pristine checkout

    gi = root / ".gitignore"
    lines = gi.read_text().splitlines() if gi.exists() else []
    return pathspec.GitIgnoreSpec.from_lines(lines)


def list_files(root: Path, ci: bool) -> list[str]:
    """Relative POSIX paths of the files a push would contain."""
    spec = None if ci else load_ignore(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        keep = []
        for d in dirnames:
            if d == ".git":
                continue
            if spec is not None and spec.match_file(rel_dir + d + "/"):
                continue
            keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            rel = rel_dir + f
            if spec is not None and spec.match_file(rel):
                continue
            out.append(rel)
    return sorted(out)


def check_files(root: Path, files: list[str], max_mb: float = MAX_MB) -> list[str]:
    problems = []
    lowered: dict[str, str] = {}
    for rel in files:
        path = root / rel
        low = rel.lower()
        if low in lowered and lowered[low] != rel:
            problems.append(f"case collision: {lowered[low]} and {rel}")
        lowered[low] = rel
        if rel.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"forbidden path in the tree: {rel}")
            continue
        size_mb = path.stat().st_size / 1e6
        if size_mb > max_mb:
            problems.append(f"{rel} is {size_mb:.1f} MB (limit {max_mb} MB)")
        if path.suffix in BINARY_DATA_SUFFIXES and not rel.startswith(DATA_ALLOWED_UNDER):
            problems.append(f"checkpoint/data file outside tests/fixtures/: {rel}")
        if path.suffix in TEXT_SUFFIXES or path.name in (".gitignore", ".python-version", ".rsync-exclude"):
            try:
                data = path.read_bytes()
            except OSError as exc:
                problems.append(f"cannot read {rel}: {exc}")
                continue
            if b"\r\n" in data:
                problems.append(f"CRLF line endings: {rel}")
            text = data.decode("utf-8", errors="replace")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    problems.append(f"{label}-shaped string in {rel}")
            if path.suffix == ".sh" and not text.startswith("#!"):
                problems.append(f"shell script without a shebang: {rel}")
    return problems


def check_critical(root: Path, files: list[str], ci: bool) -> list[str]:
    """Critical files must exist; locally they must also survive ``.gitignore``.

    In --ci mode the checkout contains exactly what was pushed, so "present" is the whole check
    (a missing ``uv.lock`` means it was never committed).
    """
    problems = []
    present = set(files)
    for rel in CRITICAL_FILES:
        if not (root / rel).exists():
            problems.append(f"critical file missing: {rel}")
        elif rel not in present:
            problems.append(f"critical file excluded by .gitignore: {rel}")
    if ci:
        return problems
    spec = load_ignore(root)
    for pattern in MUST_NOT_BE_IGNORED_GLOBS:
        for path in root.glob(pattern):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                if spec.match_file(rel):
                    problems.append(f"{rel} exists but .gitignore excludes it (it would never be committed)")
    return problems


def run(root: Path, ci: bool, max_mb: float = MAX_MB) -> list[str]:
    files = list_files(root, ci)
    return check_files(root, files, max_mb) + check_critical(root, files, ci)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ci", action="store_true", help="the checkout is the pushed tree; do not apply .gitignore"
    )
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--max-mb", type=float, default=MAX_MB)
    args = ap.parse_args(argv)
    root = Path(args.root)
    problems = run(root, args.ci, args.max_mb)
    n = len(list_files(root, args.ci))
    if problems:
        print(f"HYGIENE FAILED ({'pushed tree' if args.ci else 'working tree minus .gitignore'}, {n} files):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"HYGIENE OK: {n} files checked ({'pushed tree' if args.ci else 'working tree minus .gitignore'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
