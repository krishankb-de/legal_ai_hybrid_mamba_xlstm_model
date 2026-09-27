"""scripts/plan_state.py, always against a temporary copy -- never the live plan or state (P1-W)."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

MINI_PLAN = """# Mini plan

## 7. Phases

### P0 — Plan of record

- [ ] **P0-A** Write the plan.
- [ ] **P0-Z** Gate box.

**Gate:** done.

### P1 — Bootstrap and faithful port

- [ ] **P1-A** Create `pyproject.toml`.
      continuation line is ignored
- [ ] **P1-B** USER ACTION — delete `Reference/`.
- [ ] **P1-C** Submit `sbatch scripts/slurm/x.sh`.
- [ ] **P1-Z** Gate box.

**Gate:** all green.

### P5 — Pre-registered screen

*Pre-registered prediction (R4): something.*

- [ ] **P5-A** thing.
- [ ] **P5-Z** gate.

**Gate:** rule.

## 8. After the phases
"""


def make_repo(tmp_path: Path, plan_text: str = MINI_PLAN) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copy(REPO_ROOT / "scripts" / "plan_state.py", tmp_path / "scripts" / "plan_state.py")
    (tmp_path / "LEGAL_BUILD_PLAN.md").write_text(plan_text, encoding="utf-8")
    (tmp_path / "README.md").write_text("# x\n\n**Status: none.**\n", encoding="utf-8")
    return tmp_path


def helper(repo: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "plan_state.py"), *args],
        capture_output=True,
        text=True,
        cwd=repo,
    )


def state(repo: Path) -> dict:
    return json.loads((repo / "legal_build_state.json").read_text())


def test_init_creates_state_and_refuses_to_overwrite(tmp_path):
    repo = make_repo(tmp_path)
    assert helper(repo, "init").returncode == 0
    st = state(repo)
    assert st["phase_order"] == ["P0", "P1", "P5"] and st["current_phase"] == "P0"
    assert helper(repo, "init").returncode == 1


def test_check_passes_on_a_well_formed_plan(tmp_path):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    r = helper(repo, "check")
    assert r.returncode == 0, r.stdout + r.stderr


def test_tick_round_trips_plan_and_state(tmp_path):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    assert helper(repo, "tick", "P0-A", "--note", "wrote it").returncode == 0
    assert "- [x] **P0-A**" in (repo / "LEGAL_BUILD_PLAN.md").read_text()
    st = state(repo)
    assert (
        st["phases"]["P0"]["checkboxes"]["P0-A"]["done"] is True
        and st["phases"]["P0"]["status"] == "in_progress"
    )
    assert helper(repo, "tick", "P0-A", "--undo").returncode == 0
    assert "- [ ] **P0-A**" in (repo / "LEGAL_BUILD_PLAN.md").read_text()


def test_guards(tmp_path):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    helper(repo, "phase", "P1")
    r = helper(repo, "tick", "P1-B")
    assert r.returncode == 1 and "USER ACTION" in r.stderr
    r = helper(repo, "tick", "P1-C")
    assert r.returncode == 1 and "--evidence" in r.stderr
    r = helper(repo, "tick", "P5-A")
    assert r.returncode == 1 and "current_phase" in r.stderr
    assert helper(repo, "tick", "NOPE-A").returncode == 1
    assert helper(repo, "tick", "P1-B", "--user-confirmed", "delete reference").returncode == 0
    assert helper(repo, "tick", "P1-C", "--evidence", "job=123").returncode == 0
    ev = state(repo)["phases"]["P1"]["evidence"]
    assert ev["job"] == "123" and ev["P1-B_user_confirmed"] == "delete reference"


def test_job_bound_box_may_be_ticked_from_a_later_phase(tmp_path):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    helper(repo, "job", "add", "42", "--phase", "P5", "--box", "P5-A", "--arm", "S1")
    assert helper(repo, "tick", "P5-A").returncode == 0


def test_next_requires_the_gate_box(tmp_path):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    assert helper(repo, "next").returncode == 1
    helper(repo, "tick", "P0-A")
    helper(repo, "tick", "P0-Z")
    assert helper(repo, "next").returncode == 0 and state(repo)["current_phase"] == "P1"


def test_note_does_not_double_the_date(tmp_path):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    helper(repo, "note", "2026-01-02: already dated")
    helper(repo, "note", "plain")
    notes = state(repo)["notes"]
    assert notes[-2] == "2026-01-02: already dated"
    assert re.match(r"^\d{4}-\d{2}-\d{2}: plain$", notes[-1])


def test_resume_prints_only_the_current_phase(tmp_path):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    helper(repo, "phase", "P1")
    out = helper(repo, "resume").stdout
    assert "### P1 — Bootstrap" in out and "P1-A" in out
    assert "### P5" not in out and "P0-A" not in out


def test_block_decision_prereg_set_and_verdict(tmp_path):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    helper(repo, "block", "P1-B", "waiting for the user")
    assert state(repo)["blocked_on"]["box"] == "P1-B"
    helper(repo, "unblock")
    helper(repo, "decision", "teacher", "Qwen3-1.7B-Base")
    helper(repo, "prereg", "P5", "bar_rule", "2x SD")
    helper(repo, "set", "status", "busy")
    helper(repo, "verdict", "P0", "fine")
    st = state(repo)
    assert st["blocked_on"] is None and st["decisions"]["teacher"] == "Qwen3-1.7B-Base"
    assert st["pre_registered"]["P5"]["bar_rule"] == "2x SD" and st["status"] == "busy"
    assert st["phases"]["P0"]["verdict"] == "fine"
    assert helper(repo, "set", "phase_order", "x").returncode == 1


def test_ticking_the_blocked_box_clears_the_block(tmp_path):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    helper(repo, "block", "P0-Z", "waiting for the user")
    assert helper(repo, "tick", "P0-A").returncode == 0
    assert state(repo)["blocked_on"]["box"] == "P0-Z", "another box's tick must not clear the block"
    assert helper(repo, "tick", "P0-Z").returncode == 0
    assert state(repo)["blocked_on"] is None


@pytest.mark.parametrize(
    "mutation,needle",
    [
        (
            lambda s: s.replace("- [ ] **P5-A** thing.", "- [ ] **P5-A** thing.\n- [ ] **P5-A** again."),
            "duplicate",
        ),
        (lambda s: s.replace("**Gate:** rule.", ""), "Gate"),
        (lambda s: s.replace("- [ ] **P5-Z** gate.", ""), "P5-Z"),
        (lambda s: s.replace("*Pre-registered prediction (R4): something.*", ""), "prediction"),
        (lambda s: s.replace("- [ ] **P5-A** thing.", "- [ ] **P6-A** thing."), "sits under"),
    ],
)
def test_check_catches_malformed_plans(tmp_path, mutation, needle):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    plan = repo / "LEGAL_BUILD_PLAN.md"
    plan.write_text(mutation(plan.read_text()))
    r = helper(repo, "check")
    assert r.returncode == 1 and needle in r.stdout, r.stdout


def test_check_catches_a_hand_edited_checkbox(tmp_path):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    plan = repo / "LEGAL_BUILD_PLAN.md"
    plan.write_text(plan.read_text().replace("- [ ] **P0-A**", "- [x] **P0-A**"))
    r = helper(repo, "check")
    assert r.returncode == 1 and "run `sync`" in r.stdout


def test_readme_table(tmp_path):
    repo = make_repo(tmp_path)
    helper(repo, "init")
    assert helper(repo, "readme").returncode == 0
    text = (repo / "README.md").read_text()
    assert "| P1 | Bootstrap and faithful port |" in text and "**Status:" in text


def test_the_live_plan_passes_check():
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "plan_state.py"), "check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert r.returncode == 0, r.stdout
