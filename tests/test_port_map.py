"""The port map is a contract, not prose (plan P1-Y).

`Docs/REFERENCE_PORT_MAP.md` §0 names, for every reference module, the file that replaces it and the
plan box that creates it. These tests keep the table and the tree in agreement, and keep the new
code free of the reference's names, path shims and git calls:

* every destination of a P1 box exists, and every later destination exists once its box is ticked;
* no destination is excluded by `.gitignore` (CI would not see it);
* every box named in the table is a real plan checkbox;
* while `Reference/` exists, every source it names exists there (after P1-Z the fixtures are the record);
* no reference package name or reference argument name under `lexhybrid/ scripts/ tests/ configs/`,
  except `scripts/dump_reference_parity.py`, the one file allowed to import the reference;
* no `sys.path` edits in `lexhybrid/` or `scripts/` (the package is installed editable);
* no git call in `lexhybrid/` or `scripts/` (rule (d): git belongs to the user).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from tests.conftest import REFERENCE_ROOT, REPO_ROOT, load_script

PORT_MAP = REPO_ROOT / "Docs" / "REFERENCE_PORT_MAP.md"
STATE = REPO_ROOT / "legal_build_state.json"
PLAN = REPO_ROOT / "LEGAL_BUILD_PLAN.md"

# Built by concatenation so this file does not contain the strings it forbids.
REF_PACKAGE = "hybrid" + "_xmamba"
REF_DOC_ARG = "cu" + "_seqlens"
PARITY_DUMPER = "scripts/dump_reference_parity.py"

SCANNED_DIRS = ("lexhybrid", "scripts", "tests", "configs")
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".json", ".txt", ".cfg", ".ini"}
BOX_RE = re.compile(r"^P\d+-[A-Z]\d*$")


# ---------------------------------------------------------------- table parsing


def expand_braces(path: str) -> list[str]:
    """`a/{b,c}.py` -> [`a/b.py`, `a/c.py`]; groups expand left to right, recursively."""
    m = re.search(r"\{([^{}]*)\}", path)
    if m is None:
        return [path]
    head, tail = path[: m.start()], path[m.end() :]
    out = []
    for alt in m.group(1).split(","):
        out.extend(expand_braces(head + alt + tail))
    return out


def _cell_paths(cell: str) -> list[str]:
    paths = []
    for span in re.findall(r"`([^`]+)`", cell):
        paths.extend(expand_braces(span.strip()))
    return paths


def section_zero_rows(text: str) -> list[dict]:
    """Rows of the §0 table as {source: [...], destination: [...], box: str, raw: str}."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("## 0."))
    except StopIteration as exc:
        raise AssertionError("port map has no '## 0.' section") from exc
    rows, header_seen = [], False
    for ln in lines[start + 1 :]:
        if ln.startswith("## "):
            break
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if not header_seen:
            assert cells[:3] == ["Source", "Destination", "Box"], f"unexpected §0 header: {cells}"
            header_seen = True
            continue
        if set("".join(cells)) <= set("-: "):
            continue  # the |---|---| separator
        assert len(cells) == 4, f"§0 row does not have 4 cells: {ln}"
        rows.append(
            {
                "source": _cell_paths(cells[0]),
                "destination": _cell_paths(cells[1]),
                "box": cells[2],
                "raw": ln,
            }
        )
    return rows


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    found = section_zero_rows(PORT_MAP.read_text(encoding="utf-8"))
    assert len(found) >= 30, f"§0 table looks truncated: {len(found)} rows"
    return found


def _ticked_boxes() -> set[str]:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    return {
        cid
        for ph in state.get("phases", {}).values()
        for cid, box in ph.get("checkboxes", {}).items()
        if box.get("done")
    }


def _required(row: dict, ticked: set[str]) -> bool:
    return row["box"].startswith("P1-") or row["box"] in ticked


# ---------------------------------------------------------------- the table itself


def test_brace_expansion():
    assert expand_braces("a/{b,c}.py") == ["a/b.py", "a/c.py"]
    assert expand_braces("x/{p,q}/{r,s}.pt") == ["x/p/r.pt", "x/p/s.pt", "x/q/r.pt", "x/q/s.pt"]
    assert expand_braces("plain.py") == ["plain.py"]


def test_every_row_names_a_real_plan_box(rows):
    plan_state = load_script("plan_state")
    _, phases = plan_state.parse_plan()
    known = {cid for ph in phases.values() for cid in ph["checkboxes"]}
    bad = [r["box"] for r in rows if not BOX_RE.match(r["box"]) or r["box"] not in known]
    assert not bad, f"§0 names boxes the plan does not have: {bad}"
    assert all(r["destination"] for r in rows), "every §0 row needs at least one destination"


def box_texts(plan_text: str) -> dict[str, str]:
    """Full text of every checkbox in the plan, continuation lines included.

    The state file truncates descriptions, so the plan itself is the source here."""
    out, cur = {}, None
    for line in plan_text.splitlines():
        m = re.match(r"^- \[[ x]\] \*\*(P\d+-[A-Z]\d*)\*\*(.*)$", line)
        if m:
            cur = m.group(1)
            out[cur] = m.group(2)
        elif cur and line.startswith("  ") and line.strip():
            out[cur] += " " + line.strip()
        else:
            cur = None
    return out


def test_every_box_names_its_destination(rows):
    """The Box column is checkable: the box text names at least one of the row's files."""
    texts = box_texts(PLAN.read_text(encoding="utf-8"))
    unnamed = []
    for r in rows:
        named = {Path(p.split("::")[0]).name for p in _cell_paths(texts.get(r["box"], ""))}
        if not named & {Path(d).name for d in r["destination"]}:
            unnamed.append(f"{r['box']}: {r['destination']}")
    assert not unnamed, "§0 rows whose plan box does not name any destination:\n  " + "\n  ".join(unnamed)


def test_destinations_exist(rows):
    ticked = _ticked_boxes()
    missing = [
        f"{r['box']}: {d}"
        for r in rows
        if _required(r, ticked)
        for d in r["destination"]
        if not (REPO_ROOT / d).exists()
    ]
    assert not missing, "port-map destinations missing from the tree:\n  " + "\n  ".join(missing)


def test_destinations_are_not_gitignored(rows):
    """A destination that `.gitignore` excludes passes locally and is absent in CI."""
    spec = load_script("check_repo_hygiene").load_ignore(REPO_ROOT)
    ticked = _ticked_boxes()
    ignored = [d for r in rows if _required(r, ticked) for d in r["destination"] if spec.match_file(d)]
    assert not ignored, f"port-map destinations excluded by .gitignore: {ignored}"


@pytest.mark.reference
def test_sources_exist_in_the_reference(rows):
    missing = [f"{r['box']}: {s}" for r in rows for s in r["source"] if not (REFERENCE_ROOT / s).exists()]
    assert not missing, "port-map sources missing from Reference/:\n  " + "\n  ".join(missing)


# ---------------------------------------------------------------- the code


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
    assert not hits, f"git calls (rule (d): git belongs to the user): {hits}"
