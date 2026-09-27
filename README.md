# lexhybrid — a DACH legal, citation-faithful, retrieval-gated hybrid Mamba-3 / xLSTM decoder

**Status: P0 complete, 7/180 checkboxes; current phase P1.**

This repository is being built phase by phase from `LEGAL_BUILD_PLAN.md`. It ports the backbone, kernels, decode cache, packing, tests and measurement discipline of a finished medical research codebase (hybrid Mamba-3 + mLSTM + attention, PyTorch Lightning + Hydra, measured on the HPI aisc H100 cluster) into a production-grade package for German / Austrian / Swiss / EU law, and replaces everything domain-specific: tokenizer, teacher, corpus, targets, decode loop, sequence length.

What the finished system does, by construction:
- never generates statute text — it emits pointers (`<|q|><|c17|><|s3|>`) into a numbered retrieved context, and a renderer inserts the exact sentence and its label (`BGB §573 Abs. 2 Nr. 1`);
- never lets an unverified sentence reach the user — a verifier checks that every pointer resolves, every quote string-matches its source and every claim is entailed by the passage it cites;
- appends "Hinweis: Dies ist eine rechtliche Information, keine Rechtsberatung." to every answer.

What it does not claim (see the blueprint and the results documents under `analysis/` once they exist): better legal reasoning than frontier models; that the hybrid mixer improves quality over attention; anything from a single seed.

## Layout

| Path | Purpose |
|---|---|
| `LEGAL_BUILD_PLAN.md` | plan of record: phases P0–P11, rules, gates, risks, runbook |
| `legal_build_state.json` | resumable state (written only by the helper) |
| `scripts/plan_state.py` | the helper: `resume`, `tick`, `job`, `check`, … |
| `CLAUDE.md` | instruction card for the implementing model |
| `Docs/REFERENCE_PORT_MAP.md` | module-by-module port spec and the 18 recorded defects |
| `Docs/analysis_TEMPLATE.md` | results write-up skeleton |
| `Docs/Hybrid Mamba–xLSTM Codebase Review and DACH Legal AI Blueprint.md` | the design blueprint |
| `lexhybrid/`, `configs/`, `scripts/`, `tests/`, `analysis/` | built from P1 onward |
| `Reference/` | the medical codebase, read-only, deleted at P1-Z |

## Working on it

```
python3 scripts/plan_state.py resume      # every session starts here
bash scripts/validate.sh                  # before any code checkbox is ticked (from P1)
```

No `git` until the user says so. Nothing runs on the cluster login node; everything cluster-side goes through `sbatch` wrappers in `scripts/slurm/`.

## Progress

| Phase | Title | Status |
|---|---|---|
| P0 | Plan of record | ✅ 7/7 |
| P1 | Bootstrap and faithful port | ⬜ 0/26 |
| P2 | Correctness fixes, decoder deltas, legal configs | ⬜ 0/26 |
| P3 | Data pipeline at smoke scale | ⬜ 0/26 |
| P4 | Cluster bring-up, corpus at scale, profiling | ⬜ 0/23 |
| P5 | Pre-registered screen | ⬜ 0/10 |
| P6 | Full backbone pretraining | ⬜ 0/15 |
| P7 | Retrieval, renderer, verifier | ⬜ 0/13 |
| P8 | Post-training | ⬜ 0/15 |
| P9 | Evaluation, benchmarks, writeup | ⬜ 0/10 |
| P10 | Release hygiene | ⬜ 0/6 |
| P11 | Optional: Qwen3 conversion road | ⬜ 0/3 |