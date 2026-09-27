"""The CI workflow and the local gate stay in step (plan §8.3-8.4, P1-W4).

PyYAML parses the workflow's `on:` key as the boolean True (YAML 1.1), hence `doc.get(True)`.
"""

import re

import pytest
import yaml

from tests.conftest import REPO_ROOT

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
VALIDATE = REPO_ROOT / "scripts" / "validate.sh"


@pytest.fixture(scope="module")
def wf():
    return yaml.safe_load(WORKFLOW.read_text())


def steps_text(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) + " " + str(step.get("uses", "")) for step in job["steps"])


def test_triggers_and_permissions(wf):
    on = wf.get("on", wf.get(True))
    assert {"push", "pull_request", "workflow_dispatch", "schedule"} <= set(on)
    assert wf["permissions"] == {"contents": "read"}
    assert wf["concurrency"]["cancel-in-progress"] is True
    assert wf["env"]["HF_HUB_OFFLINE"] == "1"


def test_expected_jobs_exist(wf):
    assert {"hygiene", "lint", "test", "package", "macos", "network"} <= set(wf["jobs"])


def test_every_installing_job_uses_the_lock_and_never_pip(wf):
    for name, job in wf["jobs"].items():
        text = steps_text(job)
        assert "pip install" not in text.replace("uv pip install", ""), f"{name} uses pip directly"
        if "setup-uv" in text:
            assert "uv sync --locked" in text, f"{name} installs without --locked"
        for step in job["steps"]:
            if "run" in step and "uv sync" in step["run"]:
                assert "--locked" in step["run"], f"{name}: {step['run']}"


def test_test_matrix_contains_the_pinned_interpreter(wf):
    pinned = (REPO_ROOT / ".python-version").read_text().strip()
    versions = wf["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    assert pinned in versions
    assert wf["jobs"]["test"]["env"]["LEXHYBRID_EXPECT_PYTHON"] == "${{ matrix.python-version }}"


def test_test_job_runs_the_local_gate_in_ci_mode(wf):
    text = steps_text(wf["jobs"]["test"])
    assert "bash scripts/validate.sh --ci" in text and "scripts/check_env.py" in text


def test_hygiene_runs_on_a_pristine_checkout(wf):
    job = wf["jobs"]["hygiene"]
    text = steps_text(job)
    assert "check_repo_hygiene.py --ci" in text and "plan_state.py check" in text
    assert "uv sync" not in text and "setup-uv" not in text, "the hygiene job must not install anything first"


def test_lint_runs_every_static_tool(wf):
    text = steps_text(wf["jobs"]["lint"])
    for tool in (
        "uv lock --check",
        "uv lock --check --project envs/scrub",
        "ruff check",
        "ruff format --check",
        "shellcheck",
        "actionlint",
        "bash -n",
    ):
        assert tool in text, f"lint job does not run {tool}"


def test_package_job_imports_the_wheel_outside_the_checkout(wf):
    text = steps_text(wf["jobs"]["package"])
    assert "uv build" in text and "--no-deps dist/lexhybrid-" in text and "site-packages" in text


def test_scheduled_jobs_only_run_on_schedule_or_dispatch(wf):
    for name in ("macos", "network"):
        cond = wf["jobs"][name]["if"]
        assert "schedule" in cond and "workflow_dispatch" in cond
    assert wf["jobs"]["network"]["env"]["LEXHYBRID_NETWORK_TESTS"] == "1"
    assert "-m network" in steps_text(wf["jobs"]["network"])
    for name in ("hygiene", "lint", "test", "package"):
        assert "if" not in wf["jobs"][name], f"{name} must run on every push"


def test_validate_marker_selections_match_the_plan():
    """§8.3: fast excludes slow/cuda/multigpu/network; --full and --ci exclude cuda/multigpu/network."""
    text = VALIDATE.read_text()
    fast = re.search(r'MARKERS_FAST="([^"]+)"', text).group(1)
    full = re.search(r'MARKERS_FULL="([^"]+)"', text).group(1)
    assert fast == "not slow and not cuda and not multigpu and not network"
    assert full == "not cuda and not multigpu and not network"
    plan = (REPO_ROOT / "LEGAL_BUILD_PLAN.md").read_text()
    assert f"`{fast}`" in plan and f"`{full}`" in plan


def test_markers_are_declared_once_in_pyproject():
    import tomllib

    markers = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"][
        "markers"
    ]
    names = {m.split(":")[0] for m in markers}
    assert names == {"slow", "cuda", "multigpu", "linux_only", "network", "reference"}
