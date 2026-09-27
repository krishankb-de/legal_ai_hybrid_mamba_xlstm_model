# CLAUDE.md — instruction card for this repository

**Start every session with exactly this, and nothing else first:**

```
python3 scripts/plan_state.py resume
```

It prints the current phase id, `blocked_on`, `next_action`, open cluster jobs, the last five notes, and the **current phase section** of `LEGAL_BUILD_PLAN.md`. Work only inside that section. If the helper is broken, the fallback is:

```
awk -v h="### $(python3 -c "import json;print(json.load(open('legal_build_state.json'))['current_phase'])") —" 'index($0,h)==1{f=1} f&&/^(### |## )/&&index($0,h)!=1{exit} f' LEGAL_BUILD_PLAN.md
```

## What this project is

A DACH legal, citation-faithful, retrieval-gated language model: a corrected Mamba-3 + mLSTM + sparse-attention decoder (`lexhybrid`) that cites by pointer into retrieved passages and is gated by a verifier. Plan of record: `LEGAL_BUILD_PLAN.md` (phases P0–P11). State: `legal_build_state.json`. Port spec: `Docs/REFERENCE_PORT_MAP.md`. Blueprint: `Docs/Hybrid Mamba–xLSTM Codebase Review and DACH Legal AI Blueprint.md` (do not re-read it wholesale; the port map cites the sections that matter). Results go under `analysis/` using `Docs/analysis_TEMPLATE.md`.

## Rules

- **(a) Read only the current phase.** Do not read other phases, `Reference/` after P1-Z, or the blueprint unless the port map cites a section. `python3 scripts/plan_state.py section P3` prints a phase when the plan sends you there.
- **(b) The helper is the only writer.** Never open `legal_build_state.json` in an editor; never hand-edit a `- [ ]` line. If `scripts/plan_state.py` errors, fix the helper and its test (`tests/test_plan_state.py`); do not work around it.
- **(c) Validate before ticking code.** A code checkbox is ticked only in the session where `bash scripts/validate.sh` exited 0 *after* the change: `tick P2-D --evidence validate='exit 0, <N> passed, <date>'`. A `Pn-Z` gate needs `bash scripts/validate.sh --full`. Every code box ships its tests (plan §8, rule R13).
- **(d) Git belongs to the user.** The repository exists (remote `origin` → `github.com/krishankb-de/legal_ai_hybrid_mamba_xlstm_model`), and the user decides every commit and push. Run no `git` command unless the user asks for that specific command in this session; never push on your own. Wrappers and `run_metadata` use `.sync_stamp` and read `.git/HEAD` as a file, never `git rev-parse`.
- **(e) The cluster login node executes nothing scripted.** Allowed there, and only as `ssh $CLUSTER_HOST "cd $CLUSTER_REPO && <cmd>"` from the Mac: `sbatch`, `squeue --me`, `sacct`, `scancel`, `cat`, `tail`, `grep`, `ls`, `du`, `df`, `mkdir`. Never `python`, never `bash script.sh`, never `source .venv/bin/activate`, never a heredoc. Anything scripted is a `scripts/slurm/*.sh` with an `#SBATCH` header. `ARM` is resolved inside the job. `#SBATCH --gpus=1` in a file is not overridden by an env var: a 4-GPU run uses its own wrapper and `sbatch --gpus=4`. Never `--gres`.
- **(f) Sync without git.** `bash scripts/sync_to_cluster.sh` (rsync with `.rsync-exclude`; writes `.sync_stamp`). Pull only `logs/` and `analysis/` with `bash scripts/sync_from_cluster.sh <subdir>` into `cluster/<subdir>/`; never `--delete` on pull; never pull `outputs/`.
- **(g) A submission ends the session.** After `sbatch`: `python3 scripts/plan_state.py job add <id> --phase Pn --box Pn-K --arm <arm> --log logs/<name>_<id>.log`, then `next-action "sacct -j <id>; if COMPLETED do Pn-L"`, then stop. Never `sleep`, never poll. Next session: `ssh $CLUSTER_HOST "sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode -P"`. `COMPLETED` → pull the log → verdict box. `PENDING`/`RUNNING` → `job update <id> --status RUNNING`, stop. `FAILED`/`TIMEOUT`/`OUT_OF_MEMORY`/`REQUEUED` → pull the log tail into evidence; an infrastructure error (path, import, OOM) → fix the cause, add the test that would have caught it, resubmit once; a measurement → it is a result (rule i).
- **(h) RESULTS block.** Before ticking a verdict box, append under the phase's last checkbox:
  ```
  **RESULTS — measured <YYYY-MM-DD>, jobs <id>[, <id>].** <node>, <N>×H100, bf16, <rows/steps>.
  | arm | seed | val PPL | late-25% PPL | MQAR | statute-recall | multi-hop | s/step | peak GB |
  *Prediction:* <copy the italic line> → *Outcome:* CONFIRMED | REFUTED | PARTIAL — <one sentence>.
  *Gate:* PASS | FAIL — <rule as written>, <number>, <which side>.
  ```
  then `tick Pn-L --evidence job=<id> log=<path> <metric>=<value>` and `verdict Pn "<one line>"`.
- **(i) A failed gate is a result.** Write `*Gate:* FAIL`, `verdict Pn 'FAIL: …'`, `block Pn-Z 'gate failed; user decision'`, stop. Never change the bar, the seed, the arm, the config or the tolerance to make it pass. Never resubmit a measurement to get a different number.
- **(j) Session budget.** Stop after any `sbatch`, any `USER ACTION —` box, any `Pn-Z` gate, or about six ticks. `resume` restores everything. A `USER ACTION` box is ticked only with `--user-confirmed "<the user's exact words>"` after the user acted; until then `block Pn-X "<what>"` and stop.
- **(k) Honesty.** Never write a number without a job id and a log path. Never describe a one-seed difference as a win. "Tooling is not a tick — the gate is the measurement."
- **(l) CI.** `.github/workflows/ci.yml` runs on every push the user makes (you never push). It runs `validate.sh --ci` on Linux plus lint, hygiene and packaging jobs (plan §8.4). If the user reports a red run, or asks you to look (`gh run list`, `gh run view <id> --log-failed`), fixing it is the next box: reproduce with `bash scripts/validate.sh --ci` where you can, fix it, add a local test when the failure was reproducible locally, and note the run id. Checks the Mac cannot run are listed in plan §8.3.
- **(m) Environment.** Build or repair the venv only with `uv sync --locked`; change dependencies only by editing `pyproject.toml` and running `uv lock` in a box of its own. `scripts/check_env.py` must pass (Python 3.11, torch 2.11.0, every package equal to `uv.lock`).

## Commands

| Task | Command |
|---|---|
| Resume | `python3 scripts/plan_state.py resume` |
| Tick (code) | `python3 scripts/plan_state.py tick P1-C --evidence validate='exit 0, 41 passed, 2026-10-01'` |
| Tick (job) | `python3 scripts/plan_state.py tick P5-D --evidence job=2601234 log=cluster/logs/screen_2601234_0.log s42=9.87` |
| Note / next action | `python3 scripts/plan_state.py note "…"` · `python3 scripts/plan_state.py next-action "…"` |
| Block on the user | `python3 scripts/plan_state.py block P4-A "USER ACTION: fill scripts/slurm/cluster.env"` |
| Advance a phase | `python3 scripts/plan_state.py next` (needs `Pn-Z` ticked) |
| Validate | `bash scripts/validate.sh` (box) · `--full` (phase gate) · `--ci` (what CI runs) |
| Tests | `.venv/bin/python -m pytest tests/ -m "not slow and not cuda and not multigpu and not network" -q` |
| Environment | `uv sync --locked` · `.venv/bin/python scripts/check_env.py` · `uv lock --check` |
| CI lint locally | `actionlint .github/workflows/ci.yml` · `shellcheck scripts/*.sh scripts/slurm/*.sh` |
| Sync | `bash scripts/sync_to_cluster.sh` · `bash scripts/sync_from_cluster.sh logs` |
| Submit | `ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/<wrapper>.sh"` |
| Job state | `ssh $CLUSTER_HOST "sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode -P"` |

## Where Claude runs

Claude runs on the Mac; the cluster is reached only through the `ssh` one-liners above with `scripts/slurm/cluster.env` sourced (`CLUSTER_HOST`, `CLUSTER_REPO`, `SCRATCH_ROOT`). If a session is ever started on the cluster itself (hostname `lx*`), the same rules hold without the `ssh` prefix, and nothing scripted runs outside `sbatch`.

## Environment

Local: `.venv` from `uv sync --locked` (Python 3.11 from `.python-version`, torch 2.11.0 CPU/MPS wheel). CI: the same command on Linux (torch 2.11.0+cu128, CPU execution). Cluster: `.venv` built by `scripts/slurm/setup_env.sh` with the same command (torch 2.11.0+cu128), `outputs/` on `$SCRATCH_ROOT`, `HF_HUB_OFFLINE=1` in training jobs, `SAVE_TOP_K=0` on arms. `Reference/` exists only until P1-Z.
