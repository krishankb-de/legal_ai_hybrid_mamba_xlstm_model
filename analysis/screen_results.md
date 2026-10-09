# P5 results — pre-registered architecture screen

> **Written:** 2026-09-29, plan item P5-A (prediction and bar rule registered before any P5 submission; no number below exists yet). **Plan:** `LEGAL_BUILD_PLAN.md` · **State:** `legal_build_state.json` (`pre_registered.P5`).
> **Measured:** pending — aisc, 1×H100 80 GB per arm, bf16, 12,000 steps, 2,048-token rows, micro-batch 8 × accum 2 (32,768 tokens/step), warmup 500, LR 4e-4, clip 1.0, WSD, `SAVE_TOP_K=0`, `VAL_EVERY` 4,000, seeds 42 and 43, teacher per `decisions.teacher` (Qwen3-8B-Base, gradient checkpointing), commercial-safe shards of pack 2599735 (gii, rii, oldp, dip, fineweb2_de; 2,048-token rows cut from the 4,096 shards) (`scripts/slurm/screen_array.sh`, arms from `scripts/screen_arms.py`). Jobs: 2599890 (S1, FAILED — §7); no arm has a result yet. Logs under `cluster/logs/screen_<array>_<task>.log`.
> **Precedence:** where a summary line elsewhere disagrees with this document, the seed tables below are the record. Where an earlier number in this repo disagrees, the reason is stated here rather than the older number quietly replaced.

## 1. Summary

Pending: no arm has run. This section is written after P5-I applies the decision rule.

## 2. What was pre-registered, and what happened

| Prediction (written 2026-09-29, before submission) | Outcome | Verdict |
|---|---|---|
| S1 (attn-2) beats S0 (attn-0) on statute-recall and multi-hop by more than the bar and ties on val PPL | pending | — |
| S2 (attn-4) ties S1 on everything and is not adopted (simplest-arm rule) | pending | — |
| S3 (d_state 64) is within the bar on PPL and worse on late-position PPL | pending | — |
| S4 (MTP) is within the bar | pending | — |
| S5 (legacy recipe) is worse on val PPL by more than 1.0 | pending | — |
| S6 (α=0, no KD) is worse on val PPL by more than the bar | pending | — |

Bar as written before the numbers: **bar = 2 × the two-seed SD of S1's val PPL, floor 0.10 PPL** (`bar_value = max(2 × SD, 0.10)`, set in P5-D from the S1 seeds alone; if the floor binds, P5-D says so).

Decision rule as written before the numbers: the **simplest** arm (S1 unless a lever is needed) advances unless another arm beats it on the two-seed mean val PPL by more than the bar **and** does not lose statute-recall or multi-hop by more than the bar; a lever that wins PPL but loses recall is recorded and not adopted. Every arm is scored on val PPL, late-25% PPL, MQAR, statute-recall, multi-hop, s/step, peak GB. Unexpected results are recorded, not re-run (rule i).

## 3. Results

### 3.1 Seed table

Arms that are not parameter-matched carry their non-embedding delta against S1 in every table (R3; counts from `analysis/param_counts.md`, S1 = 104,066,872 non-embedding / 220,753,720 total). S5 is PPL/probe-only (decision 17; no cached decode) and its +2.36% is a stated difference (decision 20).

| arm | lever vs S1 | Δ non-emb | seed | val PPL | late-25% PPL | MQAR | statute-recall | multi-hop | s/step | peak GB | job |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| S1 `hybrid_legal_base` | control | — | 42 | | | | | | | | |
| S1 `hybrid_legal_base` | control | — | 43 | | | | | | | | |
| **S1 mean ± SD** | | | | | | | | | | | |
| S0 `hybrid_legal_attn0` | no attention | +2.82% | 42 | | | | | | | | |
| S0 `hybrid_legal_attn0` | no attention | +2.82% | 43 | | | | | | | | |
| S2 `hybrid_legal_attn4` | four attention layers | −2.82% | 42 | | | | | | | | |
| S2 `hybrid_legal_attn4` | four attention layers | −2.82% | 43 | | | | | | | | |
| S3 `hybrid_legal_ds64` | `mamba3_d_state 64` | −0.75% | 42 | | | | | | | | |
| S3 `hybrid_legal_ds64` | `mamba3_d_state 64` | −0.75% | 43 | | | | | | | | |
| S4 `hybrid_legal_mtp` | MTP head on | +0.00% (+9.73M training-only head) | 42 | | | | | | | | |
| S4 `hybrid_legal_mtp` | MTP head on | +0.00% (+9.73M training-only head) | 43 | | | | | | | | |
| S5 `hybrid_legal_legacy` | legacy operators (PPL/probe-only) | +2.36% | 42 | | | | | | | | |
| S5 `hybrid_legal_legacy` | legacy operators (PPL/probe-only) | +2.36% | 43 | | | | | | | | |
| S6 `hybrid_legal_base` + `distill.alpha=0` | no KD | 0 | 42 | | | | | | | | |
| S6 `hybrid_legal_base` + `distill.alpha=0` | no KD | 0 | 43 | | | | | | | | |

### 3.2 Paired comparisons

Decision rule (decision 16): paired mean > one baseline seed SD **and** sign at ≥ 2/3 seeds; bootstrap CIs beside, never instead. The screen itself decides by the P5 rule in §2 (two seeds; no claim is made from it).

| metric | A | B | diff | 95% CI (1,000 resamples by question) | seed SD | verdict |
|---|---|---|---|---|---|---|

### 3.3 Per-seed calls (to show where seeds disagree)

| metric | seed 42 | seed 43 |
|---|---|---|

## 4. Efficiency (if measured)

s/step and peak GB come from the `STEPSTATS … final=1` line of each arm's log (median seconds per optimizer step after 5 warm-up steps; peak GB allocated on the GPU).

| arm | s/step | peak GB | job |
|---|---|---|---|

## 5. What this licenses

- Pending.

## 6. Not licensed

- Any claim from the screen's two seeds: decision 16 needs three seeds for a claim; the screen only chooses the backbone by the pre-registered rule.

## 7. Open limitations

- **S1 attempt 1 failed, job 2599890 (2026-10-03), no result.** `sacct`: `2599890_0` (S1-s42) FAILED 00:22:49 exit 1, `2599890_1` (S1-s43) FAILED 00:12:25 exit 1, both on gx07. Both raised `TFLAFallbackError` in `mlstm_block` (exact TFLA, chunk 128, `fallback=error`): a chunk's log-decay half-range reached 43.2 (s42, step 838) and 54.6 (s43, step 419), above the factorised path's fp32 limit of 40. Training was healthy until then at 0.70 it/s (about 1.4 s/step). No validation ran, so no number from it enters any table. The operator change that resolves it (kernel fix at chunk 128, chunk 64, or the sequential fallback) is the user's decision and applies to every arm; logs `cluster/logs/screen_2599890_{0,1}.log`.
- Validation reads every validation row once (36,640 rows of 2,048 tokens); dip's long papers are about 61% of those tokens, so the pooled val PPL leans on dip. It is the same for every arm.

## 8. Reproduction

Key job ids: pending. Wrappers: `scripts/slurm/screen_array.sh` (`ARMS="S1-s42 S1-s43"`, then the other arms; S5 with `--time=24:00:00`) delegating to `scripts/slurm/train_pretrain_1gpu.sh`; arms and screen settings from `scripts/screen_arms.py`. Configs: `configs/model/hybrid_legal_*.yaml`. Checkpoints (cluster-only): `SAVE_TOP_K=0`, `last.ckpt` per arm under `$SCRATCH_ROOT/outputs/screen_<arm>_s<seed>/`. Every number above will carry a job id and a log path (R12).
