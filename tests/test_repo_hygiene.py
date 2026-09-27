"""scripts/check_repo_hygiene.py on synthetic trees (plan P1-W2)."""

from pathlib import Path

import pytest

from tests.conftest import load_script

H = load_script("check_repo_hygiene")

CRITICAL = H.CRITICAL_FILES


def make_tree(root: Path, extra: dict[str, bytes] | None = None, gitignore: str = "Reference/\n*.ckpt\n"):
    for rel in CRITICAL:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    (root / ".gitignore").write_text(gitignore)
    for rel, data in (extra or {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


def test_clean_tree_passes(tmp_path):
    make_tree(tmp_path, {"lexhybrid/a.py": b"x = 1\n", "scripts/run.sh": b"#!/bin/bash\necho ok\n"})
    assert H.run(tmp_path, ci=False) == []
    assert H.run(tmp_path, ci=True) == []


def test_ignored_reference_is_fine_locally_but_fails_in_ci(tmp_path):
    make_tree(tmp_path, {"Reference/hybrid/x.py": b"y = 2\n"})
    assert H.run(tmp_path, ci=False) == []
    assert any("forbidden path" in p for p in H.run(tmp_path, ci=True))


@pytest.mark.parametrize(
    "rel,data,needle",
    [
        ("big.txt", b"a" * 6_000_001, "MB"),
        ("model.safetensors", b"\x00\x01", "outside tests/fixtures"),
        ("notes.md", b"line\r\n", "CRLF"),
        ("scripts/nos hebang.sh", b"echo hi\n", "shebang"),
        ("leak.py", b"token = 'hf_" + b"A" * 34 + b"'\n", "Hugging Face token"),
        # Built by concatenation so this test file itself contains no key-shaped string.
        ("key.txt", b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----\n", "private key"),
        ("outputs/run/last.json", b"{}\n", "forbidden path"),
    ],
)
def test_each_problem_is_caught(tmp_path, rel, data, needle):
    make_tree(tmp_path, {rel: data}, gitignore="")
    problems = H.run(tmp_path, ci=True)
    assert any(needle in p for p in problems), problems


def test_fixture_checkpoints_are_allowed(tmp_path):
    make_tree(tmp_path, {"tests/fixtures/reference_parity/mamba3.pt": b"\x00" * 100})
    assert H.run(tmp_path, ci=True) == []


def test_case_collisions_are_caught(tmp_path):
    make_tree(tmp_path, {"docs/A.md": b"a\n", "docs/a.md": b"b\n"}, gitignore="")
    try:
        problems = H.run(tmp_path, ci=True)
    except FileNotFoundError:
        pytest.skip("case-insensitive filesystem cannot hold both names")
    if len(list((tmp_path / "docs").iterdir())) < 2:
        pytest.skip("case-insensitive filesystem merged the two names (the CI Linux runner checks this)")
    assert any("case collision" in p for p in problems)


def test_an_ignored_analysis_document_is_reported(tmp_path):
    make_tree(
        tmp_path,
        {"analysis/results.md": b"# r\n"},
        gitignore="*.md\n!README.md\n!CLAUDE.md\n!LEGAL_BUILD_PLAN.md\n",
    )
    problems = H.run(tmp_path, ci=False)
    assert any("analysis/results.md exists but .gitignore excludes it" in p for p in problems)


def test_a_missing_critical_file_is_reported(tmp_path):
    make_tree(tmp_path)
    (tmp_path / "uv.lock").unlink()
    assert any("critical file missing: uv.lock" in p for p in H.run(tmp_path, ci=True))


def test_the_real_repository_passes_locally():
    assert H.run(H.REPO_ROOT, ci=False) == []
