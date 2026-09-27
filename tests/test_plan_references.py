"""Every test a plan box names exists (plan rule R13).

A box's "Test:" line is the evidence a reader goes looking for. For every P1 box, and for any later
box once it is ticked, each `tests/<file>.py::<name>` it mentions must be a test function in that
file. A bare `::<name>` refers to the file named last in the same box. Bracketed parameter lists
(`[impl,chunk]`) describe the parametrization and are not part of the name.
"""

from __future__ import annotations

import ast
import json
import re

import pytest

from tests.conftest import REPO_ROOT
from tests.test_port_map import box_texts

PLAN = REPO_ROOT / "LEGAL_BUILD_PLAN.md"
STATE = REPO_ROOT / "legal_build_state.json"


def test_files_named(box_text: str) -> set[str]:
    """Every `tests/<file>.py` path a box mentions, with or without a test name."""
    return set(re.findall(r"`(tests/[\w/]+\.py)", box_text))


test_files_named.__test__ = False  # a helper, despite its name


def test_references(box_text: str) -> list[tuple[str, str]]:
    """(file, test name) pairs a box names; `::name` inherits the last file named before it."""
    out, current = [], None
    for m in re.finditer(
        r"`(tests/[\w/]+\.py)(?:::(\w+))?(?:\[[^\]`]*\])?`|`::(\w+)(?:\[[^\]`]*\])?`", box_text
    ):
        path, name_with_path, bare = m.groups()
        if path:
            current = path
            if name_with_path:
                out.append((path, name_with_path))
        elif bare and current:
            out.append((current, bare))
    return out


test_references.__test__ = False  # a helper, despite its name


def defined_tests(path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            names.add(node.name)
    return names


def _boxes_to_check() -> dict[str, str]:
    texts = box_texts(PLAN.read_text(encoding="utf-8"))
    state = json.loads(STATE.read_text(encoding="utf-8"))
    ticked = {
        cid
        for ph in state.get("phases", {}).values()
        for cid, box in ph.get("checkboxes", {}).items()
        if box.get("done")
    }
    return {cid: t for cid, t in texts.items() if cid.startswith("P1-") or cid in ticked}


def test_reference_parser():
    text = (
        "Test: `tests/test_a.py::test_one`, `::test_two[x,y]` and `tests/test_b.py` "
        "(`::test_three`); also `tests/test_c.py::test_four[p-q]`."
    )
    assert test_references(text) == [
        ("tests/test_a.py", "test_one"),
        ("tests/test_a.py", "test_two"),
        ("tests/test_b.py", "test_three"),
        ("tests/test_c.py", "test_four"),
    ]


def test_p1_names_its_tests():
    refs = [r for t in _boxes_to_check().values() for r in test_references(t)]
    assert len(refs) >= 25, f"P1 should name its tests; found only {len(refs)} references"


@pytest.mark.parametrize("box", sorted(_boxes_to_check()))
def test_every_named_test_exists(box):
    texts = _boxes_to_check()
    missing = [
        f"{path} (file missing)" for path in test_files_named(texts[box]) if not (REPO_ROOT / path).exists()
    ]
    for path, name in test_references(texts[box]):
        file = REPO_ROOT / path
        if file.exists() and name not in defined_tests(file):
            missing.append(f"{path}::{name}")
    assert not missing, f"{box} names tests that do not exist: {missing}"
