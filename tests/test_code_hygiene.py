"""Code hygiene: the source tree stays free of the upstream reference's names, path shims and git calls.

* no reference package name or reference argument name under `lexhybrid/ scripts/ tests/ configs/`,
  except `scripts/dump_reference_parity.py`, the one file allowed to import the reference;
* no `sys.path` edits in `lexhybrid/` or `scripts/` (the package is installed editable);
* no git call in `lexhybrid/` or `scripts/` (provenance is read from `.git/HEAD` as a file or from
  `.sync_stamp`).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

# Built by concatenation so this file does not contain the strings it forbids.
REF_PACKAGE = "hybrid" + "_xmamba"
REF_DOC_ARG = "cu" + "_seqlens"
PARITY_DUMPER = "scripts/dump_reference_parity.py"

SCANNED_DIRS = ("lexhybrid", "scripts", "tests", "configs")
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".json", ".txt", ".cfg", ".ini"}


def _code_files() -> list[Path]:
    out = []
    for top in SCANNED_DIRS:
        for p in sorted((REPO_ROOT / top).rglob("*")):
            rel = p.relative_to(REPO_ROOT).as_posix()
            if not p.is_file() or "__pycache__" in p.parts or rel.startswith("tests/fixtures/"):
                continue
            if p.suffix in TEXT_SUFFIXES:
                out.append(p)
    return out


def test_scan_sees_the_code():
    rels = {p.relative_to(REPO_ROOT).as_posix() for p in _code_files()}
    for expected in ("lexhybrid/models/hybrid_lm.py", "scripts/validate.sh", "configs/config.yaml"):
        assert expected in rels
    assert not any(r.startswith("tests/fixtures/") for r in rels)


@pytest.mark.parametrize("needle", [REF_PACKAGE, REF_DOC_ARG])
def test_no_reference_names_in_the_code(needle):
    hits = []
    for p in _code_files():
        rel = p.relative_to(REPO_ROOT).as_posix()
        if rel == PARITY_DUMPER:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if needle in line:
                hits.append(f"{rel}:{n}: {line.strip()}")
    assert not hits, f"{needle!r} outside {PARITY_DUMPER}:\n  " + "\n  ".join(hits)


def test_the_parity_dumper_is_the_one_reference_importer():
    text = (REPO_ROOT / PARITY_DUMPER).read_text(encoding="utf-8")
    assert REF_PACKAGE in text, "the dumper should name the reference package it imports"


def _python_files(*tops: str) -> list[Path]:
    return [p for p in _code_files() if p.suffix == ".py" and p.relative_to(REPO_ROOT).parts[0] in tops]


def _is_sys_path_edit(node: ast.AST) -> bool:
    """`sys.path.insert(...)`, `sys.path.append(...)`, `sys.path.extend(...)`, `sys.path += ...`."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        target = node.func.value
        return (
            node.func.attr in {"insert", "append", "extend"}
            and isinstance(target, ast.Attribute)
            and target.attr == "path"
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
        )
    if isinstance(node, ast.AugAssign):
        t = node.target
        return isinstance(t, ast.Attribute) and t.attr == "path" and getattr(t.value, "id", None) == "sys"
    return False


def test_no_sys_path_edits():
    hits = []
    for p in _python_files("lexhybrid", "scripts"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if _is_sys_path_edit(node):
                hits.append(f"{p.relative_to(REPO_ROOT).as_posix()}:{node.lineno}")
    assert not hits, f"sys.path edits (use the editable install instead): {hits}"


_PROCESS_CALLS = {"run", "call", "check_call", "check_output", "Popen", "system", "popen", "getoutput"}


def _names_git(arg: ast.AST) -> bool:
    if isinstance(arg, (ast.List, ast.Tuple)) and arg.elts:
        first = arg.elts[0]
        return (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and (first.value == "git" or first.value.endswith("/git"))
        )
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return re.match(r"\s*(\S*/)?git(\s|$)", arg.value) is not None
    return False


def git_calls_in_python(source: str) -> list[int]:
    """Line numbers of process calls whose command is git."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name in _PROCESS_CALLS and node.args and _names_git(node.args[0]):
            lines.append(node.lineno)
    return lines


_SHELL_GIT = re.compile(r"(^|[\s;&|(`$])git\s")


def git_calls_in_shell(source: str) -> list[int]:
    return [
        n
        for n, line in enumerate(source.splitlines(), 1)
        if not line.lstrip().startswith("#") and _SHELL_GIT.search(line.split(" #")[0])
    ]


def test_git_call_detectors():
    assert git_calls_in_python("import subprocess\nsubprocess.run(['git', 'rev-parse', 'HEAD'])\n") == [2]
    assert git_calls_in_python("import os\nos.system('git status')\n") == [2]
    assert git_calls_in_python("import subprocess\nsubprocess.run(['/usr/bin/git', 'log'])\n") == [2]
    assert git_calls_in_python("import subprocess\nsubprocess.run(['uv', 'export'])\n") == []
    assert git_calls_in_python("head = open('.git/HEAD').read()\n") == []
    assert git_calls_in_shell("#!/bin/bash\nsha=$(git rev-parse HEAD)\n") == [2]
    assert git_calls_in_shell("#!/bin/bash\n# never call git here\necho digit \n") == []


def test_no_git_calls():
    hits = []
    for top in ("lexhybrid", "scripts"):
        for p in sorted((REPO_ROOT / top).rglob("*")):
            if "__pycache__" in p.parts or not p.is_file():
                continue
            rel = p.relative_to(REPO_ROOT).as_posix()
            if p.suffix == ".py":
                hits += [f"{rel}:{n}" for n in git_calls_in_python(p.read_text(encoding="utf-8"))]
            elif p.suffix == ".sh":
                hits += [f"{rel}:{n}" for n in git_calls_in_shell(p.read_text(encoding="utf-8"))]
    assert not hits, f"git calls: {hits}"
