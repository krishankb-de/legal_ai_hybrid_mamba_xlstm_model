# Contributing

Contributions are accepted from authorised collaborators only (see [LICENSE](LICENSE)).

## Setup

```bash
uv sync --locked                       # Python 3.11, torch 2.11.0, everything pinned by uv.lock
.venv/bin/python scripts/check_env.py  # the environment must equal the lock
```

The named-entity scrub has its own environment because its NER library needs an older
`transformers`: `uv sync --locked --project envs/scrub`, then
`envs/scrub/.venv/bin/python scripts/check_env.py --scrub`.

## Dependencies

Change dependencies only by editing `pyproject.toml` and running `uv lock`, in a change of its own.
If the scrub environment is affected, edit `envs/scrub/pyproject.toml` and run
`uv lock --project envs/scrub`. CI fails when either lock is out of date (`uv lock --check`).
Keep one torch version across both environments.

## Validation

```bash
bash scripts/validate.sh          # fast: environment, lint, configs, tests, CPU smoke, hygiene
bash scripts/validate.sh --full   # adds the slow tests
bash scripts/validate.sh --ci     # what CI runs: --full plus coverage floor and JUnit output
```

Run `validate.sh` before opening a pull request; run `--full` when you touch models, kernels,
decoding or training. An optional, gitignored `scripts/validate.local.sh` runs as a pre-check hook when present.

## Tests

```bash
.venv/bin/python -m pytest tests/ -m "not slow and not cuda and not multigpu and not network" -q
```

Markers (declared in `pyproject.toml`, `--strict-markers` is on):

| Marker | Meaning |
|---|---|
| `slow` | CPU test slower than about 10 s; run by `--full` and `--ci` |
| `cuda` | needs a CUDA GPU; run on the cluster (`scripts/slurm/gpu_tests.sh`) |
| `multigpu` | needs two or more GPUs; run on the cluster (`scripts/slurm/multigpu_tests.sh`) |
| `linux_only` | needs Linux; run in CI |
| `network` | needs live external services; set `LEXHYBRID_NETWORK_TESTS=1` (weekly CI job) |
| `reference` | needs a checkout of the upstream reference implementation; skipped when absent |

New code ships with tests. Tests must not reach the network or the Hugging Face Hub unless marked
`network`.

## Lint and style

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
shellcheck -x scripts/*.sh scripts/slurm/*.sh
actionlint
```

Ruff is configured in `pyproject.toml`: line length 110, target Python 3.11, rules `E F W I UP B`
(with `E501`, `E741` and `B905` ignored). Imports are sorted with `lexhybrid` as first-party.
Shell scripts must pass `bash -n` and shellcheck. Comments explain why, not what.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

| Job | Checks |
|---|---|
| `hygiene` | pushed tree: file sizes, data and checkpoints, secrets, case collisions, CRLF, critical files |
| `lint` | `uv lock --check` (both environments), ruff, `bash -n` and shellcheck, actionlint |
| `test` | `uv sync --locked`, environment check, `validate.sh --ci` on Python 3.11 and 3.12 |
| `package` | sdist and wheel build, install without the source tree, every module imports, CPU smoke |

A weekly run adds a clean macOS install and the `network` tests against the live sources.

## Cluster jobs

Login nodes are shared, so follow the usual site etiquette: run anything heavier than `sbatch`,
`squeue`, `sacct`, `scancel` and reading files as a batch job. Put it in a wrapper in
`scripts/slurm/` with an `#SBATCH` header and submit it with `sbatch`. Use
`scripts/sync_to_cluster.sh` and `scripts/sync_from_cluster.sh` (`logs` or `analysis` only; never pull
checkpoints). Configure the host in `scripts/slurm/cluster.env` (copy of `cluster.env.example`, not
committed). The wrappers' partition and account target one site and are asserted by
`tests/test_slurm_wrappers.py`; adapting them to another cluster means changing both.

## Repository hygiene

Do not commit data, corpora, shards, indexes, checkpoints, tensors, logs or large files. Collected
data (`data/raw`, `data/scrubbed`, `data/shards`, `data/index`, `data/cache`) and cluster outputs (`outputs/`) are ignored. Only small test
fixtures under `tests/fixtures/` and manifests under `data/manifests/` are tracked.
`scripts/check_repo_hygiene.py` enforces size limits, forbidden paths and secret-shaped strings; it
runs inside `validate.sh` and in CI.
