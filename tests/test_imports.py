"""Every module imports, and every internal import names the file with its exact on-disk case.

macOS filesystems are case-insensitive, so `from lexhybrid.Layers import ...` works on the Mac and
fails on Linux (CI, the cluster). Checking the case here catches it before the push.
"""

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

import lexhybrid
from tests.conftest import REPO_ROOT

MODULES = sorted(m.name for m in pkgutil.walk_packages(lexhybrid.__path__, "lexhybrid."))
SOURCES = sorted(p for d in ("lexhybrid", "scripts", "tests") for p in (REPO_ROOT / d).rglob("*.py"))


@pytest.mark.parametrize("name", MODULES)
def test_every_module_imports(name):
    importlib.import_module(name)


def _exists_with_exact_case(path: Path) -> bool:
    current = REPO_ROOT
    for part in path.relative_to(REPO_ROOT).parts:
        if part not in {p.name for p in current.iterdir()}:
            return False
        current = current / part
    return True


def _internal_targets(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "lexhybrid":
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "lexhybrid":
                    yield alias.name


def test_internal_imports_match_the_on_disk_case():
    problems = []
    for source in SOURCES:
        for target in _internal_targets(ast.parse(source.read_text(), filename=str(source))):
            base = REPO_ROOT.joinpath(*target.split("."))
            candidates = [base.with_suffix(".py"), base / "__init__.py"]
            if not any(c.exists() and _exists_with_exact_case(c) for c in candidates):
                problems.append(f"{source.relative_to(REPO_ROOT)}: import {target}")
    assert problems == []
