# lexhybrid — a DACH legal, citation-faithful, retrieval-gated hybrid Mamba-3 / xLSTM decoder

**Status: P0–P2 complete, 65/192 checkboxes; current phase P3.**

This repository is being built phase by phase from `LEGAL_BUILD_PLAN.md`. It ports the backbone, kernels, decode cache, packing, tests and measurement discipline of a finished medical research codebase (hybrid Mamba-3 + mLSTM + attention, PyTorch Lightning + Hydra, measured on the HPI aisc H100 cluster) into a production-grade package for German / Austrian / Swiss / EU law, and replaces everything domain-specific: tokenizer, teacher, corpus, targets, decode loop, sequence length.

What the finished system does, by construction:
- never generates statute text — it emits pointers (`<|q|><|c17|><|s3|>`) into a numbered retrieved context, and a renderer inserts the exact sentence and its label (`BGB §573 Abs. 2 Nr. 1`);
- never lets an unverified sentence reach the user — a verifier checks that every pointer resolves, every quote string-matches its source and every claim is entailed by the passage it cites;
- appends "Hinweis: Dies ist eine rechtliche Information, keine Rechtsberatung." to every answer.

What it does not claim (see the blueprint and the results documents under `analysis/` once they exist): better legal reasoning than frontier models; that the hybrid mixer improves quality over attention; anything from a single seed.

## Layout

| Path | Purpose |
|---|---|
| `LEGAL_BUILD_PLAN.md` | plan of record: phases P0–P11, rules, gates, testing strategy, risks, runbook |
| `legal_build_state.json` | resumable state (written only by the helper) |
| `scripts/plan_state.py` | the helper: `resume`, `tick`, `job`, `check`, … |
| `CLAUDE.md` | instruction card for the implementing model |
| `pyproject.toml`, `uv.lock`, `.python-version` | the environment: one lock for macOS arm64 and Linux x86_64, Python 3.11, torch 2.11.0 (cu128 on Linux) |
| `lexhybrid/` | the package: `config`, `layers`, `kernels` (SSD, TFLA, selective scan), `models` (hybrid LM, MTP head, slab loss), `decoding` (cached and uncached greedy/sample/beam, pointer constraints, best-of-n), `training` (pretraining module, distillation), `eval`, `utils`, `data`; `retrieval`, `verifier` and `renderer` arrive in P7 |
| `configs/` | Hydra configs: `model/` (the legal base, its matched Transformer, the legacy ablation, the screen variants, the reference replicas), `trainer/`, `dataset/`, `callbacks/`, `distill/` |
| `scripts/` | `validate.sh` and its gate scripts; `train_pretrain.py` (Hydra entry point), `screen_arms.py` (the P5 arm table), `param_counts.py`; evaluation, profiling and comparison scripts; `dump_reference_parity.py`; `scripts/slurm/` arrives in P4 |
| `tests/` | the pytest suite; `tests/fixtures/reference_parity/` holds what the medical reference computed, the record since `Reference/` was deleted at P1-Z |
| `.github/workflows/ci.yml` | lint, tests on Python 3.11 and 3.12, hygiene and package jobs on every push and pull request |
| `Docs/REFERENCE_PORT_MAP.md` | module-by-module port spec and the 18 recorded defects |
| `Docs/analysis_TEMPLATE.md` | results write-up skeleton |
| `Docs/Hybrid Mamba–xLSTM Codebase Review and DACH Legal AI Blueprint.md` | the design blueprint |
| `analysis/` | `param_counts.md` (computed from the configs); measured results from P4 onward |

## Working on it

```
uv sync --locked                          # build .venv exactly as uv.lock says
python3 scripts/plan_state.py resume      # every session starts here
bash scripts/validate.sh                  # before any code checkbox is ticked
bash scripts/validate.sh --full           # before any phase gate
```

Commits and pushes are the maintainer's decision; every push runs `.github/workflows/ci.yml`. Nothing runs on the cluster login node; everything cluster-side goes through `sbatch` wrappers in `scripts/slurm/`.

## Progress

| Phase | Title | Status |
|---|---|---|
| P0 | Plan of record | ✅ 7/7 |
| P1 | Bootstrap and faithful port | ✅ 31/31 |
| P2 | Correctness fixes, decoder deltas, legal configs | ✅ 27/27 |
| P3 | Data pipeline at smoke scale | ⬜ 0/26 |
| P4 | Cluster bring-up, corpus at scale, profiling | ⬜ 0/27 |
| P5 | Pre-registered screen | ⬜ 0/10 |
| P6 | Full backbone pretraining | ⬜ 0/15 |
| P7 | Retrieval, renderer, verifier | ⬜ 0/13 |
| P8 | Post-training | ⬜ 0/15 |
| P9 | Evaluation, benchmarks, writeup | ⬜ 0/10 |
| P10 | Release hygiene | ⬜ 0/8 |
| P11 | Optional: Qwen3 conversion road | ⬜ 0/3 |
