"""SLURM wrappers under `scripts/slurm/` (plan P4-E onward; P4-V completes the list).

Static checks on every `*.sh` there: the header SLURM parses (port map §11.1), the facts learnt on
the HPI cluster (global scratch only on GLB_SCRATCH nodes; ga03 is an ARM node), and the
repository rules (no git, `cd` to the submit dir, `--gpus` never `--gres`).
"""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from tests.conftest import REPO_ROOT

SLURM_DIR = REPO_ROOT / "scripts" / "slurm"
WRAPPERS = sorted(SLURM_DIR.glob("*.sh"))


def directives(path) -> dict[str, str]:
    """`#SBATCH --key=value` lines before the first command, trailing comments dropped."""
    out = {}
    for ln in path.read_text().splitlines():
        m = re.match(r"#SBATCH\s+--([\w-]+)(?:=(\S+))?", ln)
        if m:
            out[m.group(1)] = m.group(2) or ""
    return out


def code(path) -> str:
    return "\n".join(ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#"))


def test_there_are_wrappers():
    assert WRAPPERS, "no scripts/slurm/*.sh found"


@pytest.mark.parametrize("path", WRAPPERS, ids=lambda p: p.name)
def test_header(path):
    d = directives(path)
    assert d.get("partition") == "aisc-batch"
    assert d.get("account") == "aisc"
    assert "ga03" in d.get("exclude", "").split(","), "ga03 is an ARM node; the x86 .venv cannot run there"
    assert d.get("constraint") == "GLB_SCRATCH", "/sc/scratch is mounted only on GLB_SCRATCH nodes"
    assert "gres" not in d, "use --gpus, never --gres (port map §11.1)"
    assert d.get("output", "").startswith("logs/") and d.get("error") == d.get("output")
    assert "time" in d and "job-name" in d


@pytest.mark.parametrize("path", WRAPPERS, ids=lambda p: p.name)
def test_body(path):
    body = code(path)
    assert "set -euo pipefail" in body
    assert 'cd "${SLURM_SUBMIT_DIR' in body
    assert not re.search(r"\bgit\b", body), "wrappers run no git (rule d); provenance is .sync_stamp"
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_setup_env_builds_from_the_lock_and_checks_it():
    body = code(SLURM_DIR / "setup_env.sh")
    assert "uv sync --locked" in body
    assert ".venv/bin/python scripts/check_env.py" in body
    assert body.index("uv sync --locked") < body.index("scripts/check_env.py")
    assert 'UV_CACHE_DIR="${SCRATCH_ROOT}' in body, "the uv cache must not fill the 200 GiB home quota"
    assert "gpus" not in directives(SLURM_DIR / "setup_env.sh"), "env setup is a CPU job"


FETCH_MODELS = (
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-0.6B-Base",
    "flair/ner-german-legal",
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",  # decision 14
)


def test_fetch_hf_downloads_every_model_the_plan_names():
    body = code(SLURM_DIR / "fetch_hf.sh")
    listed = re.findall(r'^\s+"([\w.-]+/[\w.-]+)\|', body, flags=re.M)
    assert sorted(listed) == sorted(FETCH_MODELS)
    assert "export HF_HUB_OFFLINE=0" in body and 'HF_HOME="${SCRATCH_ROOT}/.hf"' in body
    assert "gpus" not in directives(SLURM_DIR / "fetch_hf.sh"), "downloads are a CPU job"
    assert "echo $HF_TOKEN" not in body and 'echo "$HF_TOKEN' not in body, "never print the token"


def test_fetch_hf_revision_lookups_return_the_pinned_revisions():
    """The two shell lookups in fetch_hf.sh must find the constants the loaders use."""
    import importlib.util

    from lexhybrid.data.tokenizer import QWEN3_REVISION

    body = code(SLURM_DIR / "fetch_hf.sh")
    qwen_cmd = re.search(r"QWEN_REV=\"\$\(\.venv/bin/python -c '([^']+)'\)\"", body).group(1)
    qwen = subprocess.run([sys.executable, "-c", qwen_cmd], capture_output=True, text=True, cwd=REPO_ROOT)
    assert qwen.stdout.strip() == QWEN3_REVISION

    sed_expr = re.search(r"FLAIR_REV=\"\$\(sed -n '([^']+)' scripts/scrub_ner_worker.py\)\"", body).group(1)
    flair = subprocess.run(
        ["sed", "-n", sed_expr, "scripts/scrub_ner_worker.py"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    spec = importlib.util.spec_from_file_location(
        "scrub_ner_worker", REPO_ROOT / "scripts/scrub_ner_worker.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    assert flair.stdout.strip() == worker.REVISION and len(worker.REVISION) == 40


def test_preflight_runs_the_planned_checks_and_gates_the_verdict_line():
    body = code(SLURM_DIR / "preflight.sh")
    assert "scripts/screen_arms.py verify --full" in body
    assert '-m "not cuda and not slow and not reference"' in body
    assert "export HF_HUB_OFFLINE=1" in body, "preflight reads the fetched cache offline"
    assert "gpus" not in directives(SLURM_DIR / "preflight.sh"), "preflight is a CPU job"
    fail_exit = body.index("exit 1", body.index("PRE-FLIGHT FAILED"))
    assert fail_exit < body.index('echo "PRE-FLIGHT PASSED"'), "PASSED must only print after the failure exit"


@pytest.mark.parametrize("path", WRAPPERS, ids=lambda p: p.name)
def test_wrappers_that_check_the_env_can_find_uv(path):
    """check_env.py reads uv.lock through `uv`; job 2588703 failed with 'uv not found on PATH'."""
    body = code(path)
    if "scripts/check_env.py" not in body:
        pytest.skip("does not run check_env.py")
    assert "$HOME/.local/bin" in body.split("scripts/check_env.py")[0], "put ~/.local/bin on PATH first"
