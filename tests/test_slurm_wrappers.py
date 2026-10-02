"""SLURM wrappers under `scripts/slurm/` (plan P4-E onward; P4-V completes the list).

Static checks on every `*.sh` there: the header SLURM parses (port map §11.1), the facts learnt on
the HPI cluster (global scratch only on GLB_SCRATCH nodes; ga03 is an ARM node; gx13v1 has a faulty
GPU), and the repository rules (no git, `cd` to the submit dir, `--gpus` never `--gres`). Then the
wrappers that decide something at run time are RUN, in a sandbox copy of the repository whose
`.venv/bin/python` is a fake that records the entry points it is asked to start: the resume path
(FL5), the ARCH check before step 0 (FL1), ARM resolved on the compute node, the probes, the
profiler protocol and the corpus array.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

SLURM_DIR = REPO_ROOT / "scripts" / "slurm"
WRAPPERS = sorted(SLURM_DIR.glob("*.sh"))
EXCLUDED_NODES = {"ga03", "gx17v1", "gx13v1"}
# `set -uo pipefail` without -e: a status report whose every section must run (port map §11.7)
REPORTS = {"watch.sh"}


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


def runs_training(path) -> bool:
    body = code(path)
    return "scripts/train_pretrain.py" in body or "scripts/slurm/train_pretrain_1gpu.sh" in body


def test_there_are_wrappers():
    assert WRAPPERS, "no scripts/slurm/*.sh found"


def test_every_planned_wrapper_exists():
    """P4-V: the wrappers the plan's P4-P6 boxes name are all here."""
    planned = {
        "setup_env.sh", "fetch_hf.sh", "preflight.sh", "gpu_tests.sh", "multigpu_tests.sh",
        "build_corpus_array.sh", "probe_ckpt_size.sh", "profile.sh", "equivalence.sh",
        "kd_memory_probe.sh", "watch.sh", "screen_array.sh", "train_pretrain_1gpu.sh",
        "train_pretrain_4gpu.sh", "scrub_array.sh", "pack_corpus.sh",
    }  # fmt: skip
    assert planned <= {p.name for p in WRAPPERS}, sorted(planned - {p.name for p in WRAPPERS})


@pytest.mark.parametrize("path", WRAPPERS, ids=lambda p: p.name)
def test_header(path):
    d = directives(path)
    assert d.get("partition") == "pot-hpi-aisc-batch"
    assert d.get("account") == "aisc"
    assert EXCLUDED_NODES <= set(d.get("exclude", "").split(",")), (
        "ga03 is an ARM node (the x86 .venv cannot run there); gx13v1 has a faulty GPU"
    )
    assert d.get("constraint") == "GLB_SCRATCH", "/sc/scratch is mounted only on GLB_SCRATCH nodes"
    assert "gres" not in d, "use --gpus, never --gres (port map §11.1)"
    assert d.get("output", "").startswith("logs/") and d.get("error") == d.get("output")
    assert "time" in d and "job-name" in d


@pytest.mark.parametrize("path", WRAPPERS, ids=lambda p: p.name)
def test_body(path):
    body = code(path)
    assert ("set -uo pipefail" if path.name in REPORTS else "set -euo pipefail") in body
    assert 'cd "${SLURM_SUBMIT_DIR' in body
    assert not re.search(r"\bgit\b", body), "wrappers run no git (rule d); provenance is .sync_stamp"
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


@pytest.mark.parametrize("path", [p for p in WRAPPERS if runs_training(p)], ids=lambda p: p.name)
def test_training_wrappers_requeue_and_append(path):
    """FL5: pot-hpi-aisc-batch is preemptible. A training job is requeued, and its log is appended to (a
    requeue with the default open mode truncates the log it is resuming)."""
    d = directives(path)
    assert "requeue" in d and d.get("open-mode") == "append"


ARRAY_WRAPPERS = [p for p in WRAPPERS if re.search(r'IDX="\$\{SLURM_ARRAY_TASK_ID', code(p))]
MULTI_GPU_WRAPPERS = [p for p in WRAPPERS if int(directives(p).get("gpus", "0") or 0) >= 2]


def test_the_array_and_multi_gpu_wrappers_are_found():
    assert {p.name for p in ARRAY_WRAPPERS} == {"screen_array.sh", "build_corpus_array.sh", "scrub_array.sh"}
    assert {p.name for p in MULTI_GPU_WRAPPERS} == {
        "multigpu_tests.sh", "train_pretrain_4gpu.sh", "kd_memory_probe.sh",
    }  # fmt: skip


@pytest.mark.parametrize("path", ARRAY_WRAPPERS, ids=lambda p: p.name)
def test_array_wrappers_log_per_task(path):
    """An array wrapper picks its work by SLURM_ARRAY_TASK_ID and logs each task on its own."""
    assert "%A_%a" in directives(path)["output"], "one log per array task: logs/%x_%A_%a.log"


@pytest.mark.parametrize("path", MULTI_GPU_WRAPPERS, ids=lambda p: p.name)
def test_multi_gpu_wrappers_check_the_device_count(path):
    """A `--gpus` directive is not overridden by an environment variable (port map §11.1): a
    multi-GPU wrapper checks torch.cuda.device_count() against NUM_GPUS and its usage line puts
    `--gpus=N` on the sbatch command too."""
    gpus = int(directives(path)["gpus"])
    text = path.read_text()
    assert f"sbatch --gpus={gpus}" in text, "the usage line must carry --gpus on the sbatch line"
    body = code(path)
    assert re.search(rf'NUM_GPUS=("\$\{{NUM_GPUS:-{gpus}\}}"|{gpus}\b)', body), (
        "NUM_GPUS defaults to the directive"
    )
    checks = body if "device_count()" in body else code(SLURM_DIR / "train_pretrain_1gpu.sh")
    assert "torch.cuda.device_count()" in checks and "NUM_GPUS" in checks


def test_setup_env_builds_from_the_lock_and_checks_it():
    body = code(SLURM_DIR / "setup_env.sh")
    assert "uv sync --locked" in body
    assert ".venv/bin/python scripts/check_env.py" in body
    assert body.index("uv sync --locked") < body.index("scripts/check_env.py")
    # decision 22: the scrub environment from its own lock, checked by its own interpreter (P4-L)
    assert body.index("uv sync --locked --project envs/scrub") < body.index(
        "envs/scrub/.venv/bin/python scripts/check_env.py --scrub"
    )
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


@pytest.mark.parametrize(
    "path", [p for p in WRAPPERS if "scripts/check_env.py" in code(p)], ids=lambda p: p.name
)
def test_wrappers_that_check_the_env_can_find_uv(path):
    """check_env.py reads uv.lock through `uv`; job 2588703 failed with 'uv not found on PATH'."""
    body = code(path)
    assert "$HOME/.local/bin" in body.split("scripts/check_env.py")[0], "put ~/.local/bin on PATH first"


@pytest.mark.parametrize(
    "name, marker, gpus, verdict",
    [
        ("gpu_tests.sh", "-m cuda", "1", "GPU TESTS"),
        ("multigpu_tests.sh", "-m multigpu", "2", "MULTIGPU TESTS"),
    ],
)
def test_gpu_test_layers(name, marker, gpus, verdict):
    path = SLURM_DIR / name
    d, body = directives(path), code(path)
    assert d.get("gpus") == gpus
    assert marker in body and "scripts/check_env.py" in body
    assert "SLURM_JOB_ID" in body.split("TORCHINDUCTOR_CACHE_DIR=")[1].splitlines()[0], "fresh cache per job"
    fail_exit = body.index("exit 1", body.index(f"{verdict} FAILED"))
    assert fail_exit < body.index(f'echo "{verdict} PASSED"')


def test_gpu_tests_fails_fast_without_a_gpu():
    assert "torch.cuda.is_available()" in code(SLURM_DIR / "gpu_tests.sh")


def test_equivalence_runs_the_r1_gate_at_64_and_128_compiled():
    """P4-R: the box's command as written, the same gate on the legal base, P2-Y's CUDA smoke."""
    body = code(SLURM_DIR / "equivalence.sh").replace("\\\n", " ")  # join continued lines
    runs = re.findall(r"scripts/check_operator_equivalence\.py([^\n]*)", body)
    assert len(runs) == 2 and all("--chunk-sizes 64 128" in r and "--compile" in r for r in runs)
    assert "--model hybrid_legal_base" in runs[1]
    assert "tests/test_gpu.py -m cuda -k compile" in body
    fail_exit = body.index("exit 1", body.index("EQUIVALENCE FAILED"))
    assert fail_exit < body.index('echo "EQUIVALENCE PASSED"')


def test_pretrain_wrapper_resumes_from_last():
    """P6-A: the 4-GPU header, and the body it delegates to passes last.ckpt as
    +resume_from_checkpoint= when the file exists (defect 12), prints `du` of the outputs first
    and checks the GPUs it was given."""
    four = SLURM_DIR / "train_pretrain_4gpu.sh"
    d = directives(four)
    assert (d["gpus"], d["mem"], d["cpus-per-task"], d["time"]) == ("4", "256G", "32", "3-00:00:00")
    assert "requeue" in d and d["open-mode"] == "append"
    assert 'TRAINER_CFG="${TRAINER_CFG:-h100_multi_ddp}"' in code(four)
    assert "bash scripts/slurm/train_pretrain_1gpu.sh" in code(four)
    body = code(SLURM_DIR / "train_pretrain_1gpu.sh")
    assert 'CKPT="${OUT}/checkpoints/last.ckpt"' in body
    assert re.search(
        r'if \[\[ -f "\$CKPT" \]\]; then\s+echo "resuming from \$CKPT"\s+ARGS\+=\("\+resume_from_checkpoint=\$\{CKPT\}"\)',
        body,
    )
    assert body.index("du -sh") < body.index("torch.cuda.device_count()") < body.index("train_pretrain.py")


def test_screen_array_resolves_the_arm_on_the_node():
    """Port map §11.5: ARM is resolved inside the job by `screen_arms.py env`, EXPERIMENT from the
    submitting shell is dropped, and the run goes through the training body (ARCH check)."""
    body = code(SLURM_DIR / "screen_array.sh")
    assert 'scripts/screen_arms.py env "$ARM"' in body and 'eval "$ARM_ENV"' in body
    assert body.index("unset EXPERIMENT") < body.index("screen_arms.py env")
    assert "SAVE_TOP_K" in body and "bash scripts/slurm/train_pretrain_1gpu.sh" in body


def test_watch_runs_no_python():
    assert not re.search(r"\bpython", code(SLURM_DIR / "watch.sh")), "the watch script runs no python"


# -- the wrappers run in a sandbox ------------------------------------------------------------

FAKE_PYTHON = r"""#!/usr/bin/env bash
# Fake .venv/bin/python for tests/test_slurm_wrappers.py: records the entry points it is asked to
# start (one line per call, fields separated by \x1f) and plays their part.
record() { local IFS=$'\x1f'; printf '%s\n' "$*" >> "$CALLS"; }
arg_value() {  # prefix, args...: the value of the last argument starting with prefix
  local prefix="$1" hit=""; shift
  for a in "$@"; do [[ "$a" == "$prefix"* ]] && hit="${a#"$prefix"}"; done
  printf '%s' "$hit"
}
case "$1" in
  -c)  # a GPU check: `-c "<code>" [NUM_GPUS]`
    want="${!#}"; [[ "$want" =~ ^[0-9]+$ ]] || want=1
    [[ "${FAKE_GPUS:-8}" -ge "$want" ]]; exit $? ;;
  scripts/train_pretrain.py)
    record "INDUCTOR=${TORCHINDUCTOR_CACHE_DIR:-}" "$@"
    echo "${FAKE_ARCH}"
    if [[ " $* " != *" +arch_only=true "* && "${FAKE_WRITE_CKPT:-0}" == 1 ]]; then
      out="$(arg_value output_dir= "$@")"
      mkdir -p "$out/checkpoints" && head -c 4321 /dev/zero > "$out/checkpoints/last.ckpt"
    fi
    if [[ " $* " != *" +arch_only=true "* ]]; then exit "${FAKE_TRAIN_EXIT:-0}"; fi
    exit 0 ;;
  scripts/performance_profile.py | scripts/check_operator_equivalence.py | scripts/check_env.py)
    record "INDUCTOR=${TORCHINDUCTOR_CACHE_DIR:-}" "$@"; exit "${FAKE_TOOL_EXIT:-0}" ;;
  scripts/build_shards.py)
    record "$@"
    root=""; prev=""
    for a in "$@"; do [[ "$prev" == --root ]] && root="$a"; prev="$a"; done
    mkdir -p "$root/4096" "$root/8192" && echo '{"documents": 3}' > "$root/build_summary.json"
    exit 0 ;;
  -m)
    if [[ "$2" == lexhybrid.data.corpus.scrub_ler ]]; then
      record "KEY=${LEXHYBRID_SCRUB_KEY:-}" "$@"
      out=""; log=""; prev=""
      for a in "$@"; do
        [[ "$prev" == --out ]] && out="$a"; [[ "$prev" == --log ]] && log="$a"; prev="$a"
      done
      printf '{"id": 1}\n{"id": 2}\n' > "$out"; printf '{"doc_id": 1}\n' > "$log"
      exit 0
    fi
    if [[ "$2" == lexhybrid.data.corpus.collectors.* ]]; then
      record "$@"
      name="${2##*.}"; out=""; man=""; prev=""
      for a in "$@"; do
        [[ "$prev" == --out ]] && out="$a"; [[ "$prev" == --manifest-dir ]] && man="$a"; prev="$a"
      done
      mkdir -p "$out" "$man"
      printf '{"id": 1}\n{"id": 2}\n{"id": 3}\n' > "$out/$name.jsonl"
      printf '{"id": 1}\n{"id": 2}\n{"id": 3}\n' > "$man/$name.jsonl"
      exit 0
    fi ;;
esac
exec "$REAL_PYTHON" "$@"
"""


def _s1_fingerprint() -> str:
    """The ARCH line of the real base model (S1), built on meta from the screen's own table."""
    import importlib.util

    import torch

    from lexhybrid import HybridLanguageModel

    spec = importlib.util.spec_from_file_location("screen_arms", REPO_ROOT / "scripts" / "screen_arms.py")
    arms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arms)
    with torch.device("meta"):
        return HybridLanguageModel(arms.build_config("S1")).architecture_fingerprint()


def _executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def sandbox(tmp_path):
    """A repository copy with the wrappers, screen_arms.py and a fake interpreter; fake nvidia-smi,
    squeue and sacct on PATH. Returns ``run(script, **env) -> (CompletedProcess, calls)``."""
    repo, bin_dir = tmp_path / "repo", tmp_path / "bin"
    shutil.copytree(SLURM_DIR, repo / "scripts" / "slurm")
    (repo / "envs" / "scrub" / ".venv" / "bin").mkdir(parents=True)
    _executable(repo / "envs" / "scrub" / ".venv" / "bin" / "python", "#!/usr/bin/env bash\necho GPU: fake\n")
    shutil.copy(REPO_ROOT / "scripts" / "screen_arms.py", repo / "scripts" / "screen_arms.py")
    (repo / ".venv" / "bin").mkdir(parents=True)
    _executable(repo / ".venv" / "bin" / "python", FAKE_PYTHON)
    bin_dir.mkdir()
    _executable(bin_dir / "nvidia-smi", "#!/usr/bin/env bash\necho fake-gpu\n")
    _executable(bin_dir / "squeue", "#!/usr/bin/env bash\necho '  JOBID PARTITION NODELIST ST TIME NAME'\n")
    _executable(
        bin_dir / "sacct",
        "#!/usr/bin/env bash\necho '2600001    screen_S1   PREEMPTED   0:0'\necho '2600001.batch  batch  PREEMPTED 0:0'\n",
    )
    calls_path = tmp_path / "calls.log"
    s1_arch = _s1_fingerprint()

    def run(script: str, **env) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
        calls_path.unlink(missing_ok=True)
        full = {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(tmp_path / "home"),
            "USER": "tester",
            "SLURM_SUBMIT_DIR": str(repo),
            "SCRATCH_ROOT": str(tmp_path / "scratch"),
            "CALLS": str(calls_path),
            "REAL_PYTHON": sys.executable,
            "FAKE_ARCH": s1_arch,  # the model "built" is the base, S1
            **{k: str(v) for k, v in env.items()},
        }
        res = subprocess.run(
            ["bash", str(repo / "scripts" / "slurm" / script)],
            env=full,
            cwd=repo,
            capture_output=True,
            text=True,
        )
        lines = calls_path.read_text().splitlines() if calls_path.exists() else []
        return res, [ln.split("\x1f") for ln in lines]

    run.scratch = tmp_path / "scratch"
    run.repo = repo
    return run


def _train_calls(calls):
    arch = [c for c in calls if "scripts/train_pretrain.py" in c and "+arch_only=true" in c]
    train = [c for c in calls if "scripts/train_pretrain.py" in c and "+arch_only=true" not in c]
    return arch, train


def test_train_wrapper_defaults_and_the_arch_check_before_training(sandbox):
    res, calls = sandbox("train_pretrain_1gpu.sh")
    assert res.returncode == 0, res.stdout + res.stderr
    arch, train = _train_calls(calls)
    assert len(arch) == 1 and len(train) == 1 and calls.index(arch[0]) < calls.index(train[0])
    out = sandbox.scratch / "outputs" / "pretrain_hybrid_legal_base_s42"
    for want in (
        "model=hybrid_legal_base", "seed=42", "dataset=mixture_pretrain", "trainer=h100_single_gpu",
        "trainer.devices=1", "distill=qwen3_8b", "model.use_gradient_checkpointing=true",
        "callbacks.checkpoint.save_top_k=0",
        "experiment_name=pretrain_hybrid_legal_base_s42", f"output_dir={out}",
    ):  # fmt: skip
        assert want in train[0], want
    assert not any(a.startswith("+resume_from_checkpoint=") for a in train[0])
    assert "ARCH CHECK OK" in res.stdout and "TRAIN DONE pretrain_hybrid_legal_base_s42" in res.stdout
    assert res.stdout.index("du") < res.stdout.index("ARCH CHECK OK")


@pytest.mark.parametrize(
    "env, want",
    [
        ({}, "true"),  # the 8B teacher: checkpointing on (OOM without it, job 2589362)
        ({"GRAD_CKPT": "false"}, "false"),  # an explicit setting wins
        ({"DISTILL_CFG": "qwen3_1p7b"}, None),  # the 1.7B: the model's own default
        ({"DISTILL_CFG": "none"}, None),
    ],
)
def test_train_wrapper_checkpoints_the_8b_teacher_run(sandbox, env, want):
    """P4-U chose Qwen3-8B-Base, which fits only with gradient checkpointing (job 2589417)."""
    res, calls = sandbox("train_pretrain_1gpu.sh", **env)
    assert res.returncode == 0, res.stdout + res.stderr
    (train,) = _train_calls(calls)[1]
    got = next((a.split("=", 1)[1] for a in train if a.startswith("model.use_gradient_checkpointing=")), None)
    assert got == want


def test_train_wrapper_resumes_from_last_ckpt(sandbox):
    """FL5 / defect 12: a requeued run finds its last.ckpt and passes it on."""
    ckpt = sandbox.scratch / "outputs" / "screen_S1_s42" / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"0")
    res, calls = sandbox("train_pretrain_1gpu.sh", EXPERIMENT="screen_S1_s42", SLURM_RESTART_COUNT=1)
    assert res.returncode == 0, res.stdout + res.stderr
    arch, train = _train_calls(calls)
    assert f"+resume_from_checkpoint={ckpt}" in train[0]
    assert not any(a.startswith("+resume_from_checkpoint=") for a in arch[0])
    assert f"resuming from {ckpt}" in res.stdout and "restart: 1" in res.stdout


@pytest.mark.parametrize(
    "env, message",
    [
        ({"ARM_EXPECT": "mamba3x7|mamba3(d_state=64"}, "lacks 'mamba3(d_state=64'"),
        ({"ARM_FORBID": "attention"}, "forbidden 'attention'"),
        ({"NUM_GPUS": 4, "FAKE_GPUS": 2}, "fewer than NUM_GPUS=4"),
    ],
)
def test_train_wrapper_stops_before_step_0(sandbox, env, message):
    """FL1: an ARCH line without an expected token, or with a forbidden one, stops the job before
    training starts; so does a node with fewer GPUs than the run needs."""
    res, calls = sandbox("train_pretrain_1gpu.sh", **env)
    assert res.returncode == 1 and message in res.stdout, res.stdout + res.stderr
    assert _train_calls(calls)[1] == [], "training must not start"


def test_train_wrapper_levers_and_extra_overrides_come_last(sandbox):
    res, calls = sandbox(
        "train_pretrain_1gpu.sh",
        MAX_STEPS=5, ROW_LEN=2048, DISTILL_CFG="none",
        EXTRA_OVERRIDES="trainer.max_steps=7 dataset.batch_size=8",
    )  # fmt: skip
    assert res.returncode == 0, res.stdout + res.stderr
    (train,) = _train_calls(calls)[1]
    assert not any(a.startswith("distill=") for a in train), "none: the config default (no teacher)"
    assert "dataset.row_len=2048" in train
    assert train.index("trainer.max_steps=5") < train.index("trainer.max_steps=7"), "the arm's value wins"
    assert train[-1] == "dataset.batch_size=8"
    assert not any(a.startswith(("trainer.accumulate_grad_batches=", "model.learning_rate=")) for a in train)


def test_four_gpu_wrapper_runs_ddp_through_the_same_body(sandbox):
    res, calls = sandbox("train_pretrain_4gpu.sh", FAKE_GPUS=4, EXPERIMENT="p6_hybrid_s42_A")
    assert res.returncode == 0, res.stdout + res.stderr
    (train,) = _train_calls(calls)[1]
    assert {"trainer=h100_multi_ddp", "trainer.devices=4", "callbacks.checkpoint.save_top_k=1"} <= set(train)
    res, _ = sandbox("train_pretrain_4gpu.sh", FAKE_GPUS=3)
    assert res.returncode == 1 and "fewer than NUM_GPUS=4" in res.stdout


def test_screen_array_task_runs_its_arm(sandbox):
    """The array index picks the arm; screen_arms.py resolves it ON THE NODE; the arm's lever must
    show in the ARCH line (here it does not: S3 needs d_state 64 and the fake model has 128)."""
    arms = "S1-s42 S3-s43"
    res, calls = sandbox("screen_array.sh", ARMS=arms, SLURM_ARRAY_TASK_ID=0, EXPERIMENT="stray")
    assert res.returncode == 0, res.stdout + res.stderr
    (train,) = _train_calls(calls)[1]
    assert {"model=hybrid_legal_base", "seed=42", "experiment_name=screen_S1_s42"} <= set(train)
    assert {"trainer.max_steps=12000", "dataset.row_len=2048", "callbacks.checkpoint.save_top_k=0"} <= set(
        train
    )
    res, calls = sandbox("screen_array.sh", ARMS=arms, SLURM_ARRAY_TASK_ID=1)
    assert res.returncode == 1 and "lacks 'mamba3(d_state=64'" in res.stdout
    assert "model=hybrid_legal_ds64" in _train_calls(calls)[0][0] and _train_calls(calls)[1] == []
    res, _ = sandbox("screen_array.sh", ARMS=arms, SLURM_ARRAY_TASK_ID=2)
    assert res.returncode == 1 and "only 2 arms" in res.stdout


def test_probe_ckpt_size_reports_and_deletes_its_outputs(sandbox):
    res, calls = sandbox("probe_ckpt_size.sh", SLURM_JOB_ID=77, FAKE_WRITE_CKPT=1)
    assert res.returncode == 0, res.stdout + res.stderr
    (train,) = _train_calls(calls)[1]
    assert {"trainer.max_steps=10", "dataset=synthetic", "model=hybrid_legal_base"} <= set(train)
    assert not any(a.startswith("distill=") for a in train)
    assert "CKPT_BYTES=4321 CKPT_GB=0.000" in res.stdout and "PROBE_CKPT DONE" in res.stdout
    assert not (sandbox.scratch / "outputs" / "probe_ckpt_77").exists(), "the probe cleans up after itself"
    res, _ = sandbox("probe_ckpt_size.sh", SLURM_JOB_ID=78)  # training "ran" but wrote nothing
    assert res.returncode == 1 and "was not written" in res.stdout


def _compose(overrides):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


@pytest.mark.parametrize(
    "script, env",
    [
        ("train_pretrain_1gpu.sh", {}),
        ("train_pretrain_1gpu.sh", {"DISTILL_CFG": "none", "MAX_STEPS": 5, "ROW_LEN": 2048}),
        ("train_pretrain_4gpu.sh", {"FAKE_GPUS": 4}),
        ("screen_array.sh", {"ARMS": "S1-s42", "SLURM_ARRAY_TASK_ID": 0}),
        ("probe_ckpt_size.sh", {"SLURM_JOB_ID": 77, "FAKE_WRITE_CKPT": 1}),
        ("kd_memory_probe.sh", {"FAKE_GPUS": 4, "SLURM_JOB_ID": 91}),
    ],
    ids=lambda v: v if isinstance(v, str) else ",".join(f"{k}={x}" for k, x in v.items()) or "defaults",
)
def test_every_train_argument_list_composes_under_real_hydra(sandbox, script, env):
    """Job 2589359: the fake interpreter recorded `distill=null` happily; real Hydra refused it
    ("Config group override must be a string or a list") before step 0. Every train_pretrain.py
    argument list a wrapper builds must compose against configs/."""
    res, calls = sandbox(script, **env)
    assert res.returncode == 0, res.stdout + res.stderr
    lists = [c[c.index("scripts/train_pretrain.py") + 1 :] for c in calls if "scripts/train_pretrain.py" in c]
    assert lists
    for args in lists:
        cfg = _compose(args)
        distill = next((a.split("=", 1)[1] for a in args if a.startswith("distill=")), None)
        assert (cfg.get("distill") is None) if distill is None else cfg.distill.teacher.startswith("Qwen/")


def test_every_teacher_loads_the_revision_fetch_hf_puts_in_the_cache():
    """Job 2589362: fetch_hf.sh fetched Qwen3-1.7B-Base at the tokenizer's pinned commit, which
    writes no refs/main; the teacher was loaded without a revision, so offline it resolved `main`
    and found nothing. Each distill config names the revision its teacher was fetched at."""
    import yaml

    from lexhybrid.data.tokenizer import QWEN3_REVISION

    body = code(SLURM_DIR / "fetch_hf.sh")
    fetched = dict(re.findall(r'^\s*"([\w.-]+/[\w.-]+)\|([^"]+)"', body, flags=re.M))
    assert fetched["Qwen/Qwen3-1.7B-Base"] == "$QWEN_REV"
    fetched["Qwen/Qwen3-1.7B-Base"] = QWEN3_REVISION
    configs = sorted((REPO_ROOT / "configs" / "distill").glob("*.yaml"))
    assert configs
    for path in configs:
        cfg = yaml.safe_load(path.read_text())
        assert cfg["revision"] == fetched[cfg["teacher"]], path.name


def test_kd_probe_runs_four_shapes_and_survives_an_oom(sandbox):
    res, calls = sandbox("kd_memory_probe.sh", FAKE_GPUS=4, SLURM_JOB_ID=91, FAKE_TRAIN_EXIT=1)
    assert res.returncode == 0, res.stdout + res.stderr
    train = _train_calls(calls)[1]
    shapes = [
        (
            next(a for a in t if a.startswith("distill=")),
            next(a for a in t if a.startswith("dataset.row_len=")),
        )
        for t in train
    ]
    assert shapes == [
        ("distill=qwen3_1p7b", "dataset.row_len=4096"), ("distill=qwen3_1p7b", "dataset.row_len=8192"),
        ("distill=qwen3_8b", "dataset.row_len=4096"), ("distill=qwen3_8b", "dataset.row_len=8192"),
    ]  # fmt: skip
    assert all(
        {"trainer=h100_multi_ddp", "trainer.devices=4", "trainer.enable_checkpointing=false"} <= set(t)
        for t in train
    )
    assert all("model.use_gradient_checkpointing=false" in t for t in train), "the probe's own setting"
    assert "KD PROBE RUNS THAT DID NOT FINISH" in res.stdout and "KD PROBE DONE" in res.stdout


def test_kd_probe_reruns_only_the_named_shapes(sandbox):
    """Job 2589362: the 1.7B shapes failed on infrastructure and are rerun; the 8B OOMs are results
    and must not be measured again."""
    res, calls = sandbox("kd_memory_probe.sh", FAKE_GPUS=4, SLURM_JOB_ID=92, PROBES="t1p7b_L4096 t1p7b_L8192")
    assert res.returncode == 0, res.stdout + res.stderr
    train = _train_calls(calls)[1]
    assert [next(a for a in t if a.startswith("distill=")) for t in train] == ["distill=qwen3_1p7b"] * 2
    assert "t8b_" not in res.stdout.replace("RUNS THAT DID NOT FINISH", "")


def test_kd_probe_summary_reads_its_own_log(sandbox):
    """Jobs 2589362, 2589379, 2589417: the log is the job's own stdout, and GNU grep refuses to read
    its output file ("input file is also the output"), so the summary said "log not found". Run the
    wrapper the way Slurm does, stdout into logs/kd_probe_<id>.log (CI's GNU grep catches a relapse)."""
    (sandbox.repo / "logs").mkdir(exist_ok=True)
    _executable(
        sandbox.repo / "scripts" / "slurm" / "as_slurm.sh",
        '#!/usr/bin/env bash\nbash scripts/slurm/kd_memory_probe.sh > "logs/kd_probe_${SLURM_JOB_ID}.log" 2>&1\n',
    )
    res, _ = sandbox("as_slurm.sh", FAKE_GPUS=4, SLURM_JOB_ID=93, PROBES="t1p7b_L4096")
    assert res.returncode == 0, res.stdout + res.stderr
    log = (sandbox.repo / "logs" / "kd_probe_93.log").read_text()
    summary = log[log.index("== summary") :]
    assert "== probe t1p7b_L4096" in summary and "no probe lines" not in summary


def test_profile_measures_one_length_per_process_with_its_own_cache(sandbox):
    """Port map §11.8 / R7: one sequence length per process, a fresh Inductor cache per point, the
    Transformer's sanity row before the hybrid's; the P4-Q chunk-64 row at 4,096 is there."""
    res, calls = sandbox("profile.sh", SLURM_JOB_ID=55)
    assert res.returncode == 0 and "PROFILE DONE" in res.stdout, res.stdout + res.stderr
    points = [c for c in calls if "scripts/performance_profile.py" in c]
    assert len(points) == 22
    for p in points:
        lengths = p[p.index("--seq-lengths") + 1 :]
        assert lengths[0].isdigit() and not lengths[1].isdigit(), "exactly one length per process"
    caches = [p[0] for p in points]
    assert len(set(caches)) == len(caches), "every point has its own TORCHINDUCTOR_CACHE_DIR"
    models = [p[p.index("--models") + 1] for p in points]
    assert models.index("ref_transformer") < models.index("ref_hybrid_m3")
    assert models.index("transformer_legal_base") < models.index("hybrid_legal_base")
    chunk64 = [
        p
        for p in points
        if "--chunk-size" in p and p[p.index("--chunk-size") + 1] == "64" and "hybrid_legal_base" in p
    ]
    assert len(chunk64) == 1 and "4096" in chunk64[0] and "--doc-len" in chunk64[0]
    res, _ = sandbox("profile.sh", SLURM_JOB_ID=56, FAKE_TOOL_EXIT=1)
    assert res.returncode == 1 and "PROFILE FAILED POINTS" in res.stdout


@pytest.mark.parametrize("index, source, limit", [(0, "gii", "all"), (8, "fineweb2_de", "2000000")])
def test_corpus_array_task_collects_its_source(sandbox, index, source, limit):
    res, calls = sandbox("build_corpus_array.sh", SLURM_ARRAY_TASK_ID=index, SLURM_ARRAY_JOB_ID=3)
    assert res.returncode == 0, res.stdout + res.stderr
    (call,) = calls
    assert call[:2] == ["-m", f"lexhybrid.data.corpus.collectors.{source}"]
    flags = dict(zip(call[2::2], call[3::2]))
    assert flags["--limit"] == limit
    assert flags["--out"] == str(sandbox.scratch / "data" / "raw" / source)
    assert flags["--http-cache"] == str(sandbox.scratch / "data" / "http_cache" / source)
    assert f"CORPUS {source} DONE: 3 documents" in res.stdout
    res, _ = sandbox("build_corpus_array.sh", SLURM_ARRAY_TASK_ID=10)
    assert res.returncode == 1 and "only 10 sources" in res.stdout


def test_watch_reports_every_section_and_flags_a_requeue_that_did_not_resume(sandbox):
    logs = sandbox.repo / "logs"
    logs.mkdir()
    (logs / "screen_1_0.log").write_text("host: gx01  restart: 0\nTRAIN DONE\n")
    (logs / "screen_1_1.log").write_text("host: gx02  restart: 0\nhost: gx02  restart: 1\nTraceback x\n")
    (logs / "screen_1_2.log").write_text("host: gx03  restart: 0\nhost: gx03  restart: 1\nresuming from /x\n")
    ckpts = sandbox.scratch / "outputs" / "screen_S1_s42" / "checkpoints"
    ckpts.mkdir(parents=True)
    (ckpts / "last.ckpt").write_bytes(b"0" * 10)
    res, calls = sandbox("watch.sh", SLURM_JOB_ID=5)
    assert res.returncode == 0 and "== watch done" in res.stdout, res.stdout + res.stderr
    assert calls == [], "no python"
    assert "screen_S1_s42: last.ckpt" in res.stdout and "PREEMPTED" in res.stdout
    assert ".batch" not in res.stdout.split("== jobs of the last 3 days")[1].split("== preempted")[0]
    flagged = [ln for ln in res.stdout.splitlines() if "restarted from step 0?" in ln]
    assert len(flagged) == 1 and flagged[0].startswith("screen_1_1.log")


def test_scrub_array_task_scrubs_its_part_with_one_key(sandbox):
    """P4-L: task i scrubs part i % PARTS of source i / PARTS; every task uses the same HMAC key
    (created once, 32 hex bytes, private); a part already written is skipped."""
    for src in ("gii", "rii"):
        raw = sandbox.scratch / "data" / "raw" / src / f"{src}.jsonl"
        raw.parent.mkdir(parents=True)
        raw.write_text('{"id": 1}\n')
    env = dict(SOURCES="gii rii", PARTS=3, SLURM_ARRAY_JOB_ID=9)
    res, calls = sandbox("scrub_array.sh", SLURM_ARRAY_TASK_ID=4, **env)
    assert res.returncode == 0, res.stdout + res.stderr
    (call,) = calls
    flags = dict(zip(call[3::2], call[4::2]))
    assert call[1:3] == ["-m", "lexhybrid.data.corpus.scrub_ler"]
    assert (flags["--source"], flags["--part"], flags["--parts"]) == ("rii", "1", "3")
    assert flags["--out"] == str(sandbox.scratch / "data" / "scrubbed" / "rii.part-1-of-3.jsonl")
    assert flags["--log"] == str(sandbox.scratch / "data" / "manifests" / "scrub_rii.part-1-of-3.jsonl")
    key_file = sandbox.scratch / ".secrets" / "scrub_hmac.key"
    key = key_file.read_text()
    assert re.fullmatch(r"[0-9a-f]{64}", key) and call[0] == f"KEY={key}"
    assert oct(key_file.stat().st_mode & 0o777) == "0o600" and key not in res.stdout, "private, never printed"
    res, calls = sandbox("scrub_array.sh", SLURM_ARRAY_TASK_ID=0, **env)
    assert res.returncode == 0 and calls[0][0] == f"KEY={key}", "the second task reuses the key"
    res, calls = sandbox("scrub_array.sh", SLURM_ARRAY_TASK_ID=4, **env)
    assert res.returncode == 0 and calls == [] and "already scrubbed" in res.stdout
    res, _ = sandbox("scrub_array.sh", SLURM_ARRAY_TASK_ID=6, **env)
    assert res.returncode == 1 and "only 6 tasks" in res.stdout
    res, _ = sandbox("scrub_array.sh", SLURM_ARRAY_TASK_ID=0, SOURCES="dip", PARTS=1)
    assert res.returncode == 1 and "is missing" in res.stdout


def test_pack_corpus_packs_both_row_lengths_from_the_scrubbed_parts(sandbox):
    res, calls = sandbox("pack_corpus.sh", SLURM_CPUS_PER_TASK=32)
    assert res.returncode == 0 and "PACK CORPUS DONE" in res.stdout, res.stdout + res.stderr
    (call,) = calls
    assert call[0] == "scripts/build_shards.py"
    i = call.index("--row-len")
    assert call[i + 1 : i + 3] == ["4096", "8192"] and call[call.index("--workers") + 1] == "32"
    assert call[call.index("--scrubbed") + 1] == str(sandbox.scratch / "data" / "scrubbed")
    assert "--allow-unscrubbed" not in call, "R11: only scrubbed documents are packed"
