# Architecture screen: ablations of the hybrid backbone

Seven architecture and recipe variants (S0-S6) of the 220M-parameter hybrid, each trained for 12,000 steps on two seeds and compared on validation perplexity, late-position perplexity and three probes.

**Setup.** One H100 80 GB per arm, bf16, 12,000 steps, 2,048-token rows, micro-batch 8 × accum 2 (32,768 tokens/step), warmup 500, LR 4e-4, clip 1.0, WSD schedule, `SAVE_TOP_K=0`, `VAL_EVERY` 4,000, seeds 42 and 43. Teacher: Qwen3-8B-Base with gradient checkpointing (online logit KD), except S6. Data: the commercial-safe shards of the pretraining corpus (gii, rii, oldp, dip, fineweb2_de; 2,048-token rows cut from the 4,096 shards; see `analysis/corpus_manifest.md`). Launcher `scripts/slurm/screen_array.sh`; arms from `scripts/screen_arms.py`.

The screen was run twice. The **first run** is superseded: two of its runs collapsed (see Limitations), and an earlier launch failed before producing a number. The **second run** is the measurement. It changed one recipe setting for every arm (AdamW β2 0.95), and S6 loads no teacher.

## Summary

**S1, `hybrid_legal_base` with AdamW β2 0.95, advances by the selection rule.** No arm beats its two-seed mean val PPL of 5.3582 by more than the bar of 0.10 PPL. The closest are S3 (`d_state 64`) at 0.0575 lower and S2 (four attention layers) at 0.0148 lower. S4 (MTP) ties, at 0.0004 lower. S6 (no KD) is worse by 0.0488, within the bar. S0 (no attention) is worse by 0.675 and S5 (legacy recipe) by 1.281.

The rule's second condition, not losing statute recall or multi-hop by more than the bar, decided nothing. **Statute-recall exact match and multi-hop exact match are 0.000 in all 14 runs**, a floor at this scale (220M parameters, 393M training tokens). Only MQAR and statute-recall char-F1 separate the arms, and the rule uses neither. Of the six hypotheses, three are confirmed (S2, S4, S5), two refuted (S1 against S0, S6) and one partial (S3).

## Hypotheses and selection rule

| Hypothesis (stated before the run) | Result | Outcome |
|---|---|---|
| S1 (attn-2) beats S0 (attn-0) on statute-recall and multi-hop by more than the bar and ties on val PPL | statute-recall EM 0.000 vs 0.000 and multi-hop EM 0.000 vs 0.000, a floor in both; val PPL S0 6.0336 vs S1 5.3582, so S0 is worse by 0.675 and there is no tie. (MQAR, outside the hypothesis: S1 0.137, S0 0.007.) | **Refuted** |
| S2 (attn-4) ties S1 on everything and is not adopted (simplest-arm rule) | val PPL −0.0148, late-25% −0.0175, MQAR −0.024, char-F1 −0.010, EM ties at 0: every difference within the bar; not adopted | **Confirmed** |
| S3 (d_state 64) is within the bar on PPL and worse on late-position PPL | val PPL −0.0575, within the bar; late-25% PPL 4.9496 vs 5.0014, **better** by 0.052, not worse | **Partial** |
| S4 (MTP) is within the bar | val PPL 5.3578 vs 5.3582 (−0.0004); the seeds disagree on the sign | **Confirmed** |
| S5 (legacy recipe) is worse on val PPL by more than 1.0 | 6.6394 vs 5.3582: worse by 1.281 | **Confirmed** |
| S6 (α=0, no KD) is worse on val PPL by more than the bar | 5.4070 vs 5.3582: worse by 0.0488, inside the bar (its late-25% PPL is worse by 0.105, outside it) | **Refuted** |

**Bar (fixed before the numbers):** 2 × the two-seed SD of S1's val PPL, with a floor of 0.10 PPL (`bar_value = max(2 × SD, 0.10)`), computed from the S1 seeds alone. **The bar is 0.10 PPL; the floor binds:** S1's two seeds give 5.3634 and 5.3530, an SD of 0.0074, so 2 × SD = 0.0147.

**Selection rule (fixed before the numbers):** the **simplest** arm (S1 unless a lever is needed) advances unless another arm beats it on the two-seed mean val PPL by more than the bar **and** does not lose statute-recall or multi-hop by more than the bar; a lever that wins PPL but loses recall is recorded and not adopted. Every arm is scored on val PPL, late-25% PPL, MQAR, statute-recall, multi-hop, s/step, peak GB. Unexpected results are recorded, not re-run. Fixed before any probe result: statute recall is scored by exact match, with char-F1 beside it, and multi-hop by exact match; "lose by more than the bar" is S1's mean minus the arm's mean > 0.10 on the 0-1 scale.

## Results

### Seed table, second run (the measurement)

Every arm ran with AdamW β2 0.95; S6 loads no teacher. Values are assembled by `scripts/screen_table.py` (none copied by hand):
- **val PPL:** exp(`val/loss`) at step 11,999.
- **late-25% PPL:** predicted tokens with at least 1,536 tokens of their own document before them in the row, 16,061,503 of 74,979,957 validation tokens (21.4%). The row-position variant is in brackets. Computed by `scripts/evaluate_val_ppl.py`. Each run's overall PPL there equals the reader's within 0.0001.
- **Probes:** `scripts/evaluate_probes.py` at 2,048. MQAR has 256 items (1,024 queries), statute recall 100, multi-hop 139 (the built items that fit 2,048 tokens).
- **s/step and peak GB:** the `STEPSTATS` lines of each run's log. For S5 they are step-weighted over its two legs.

Every log shows `ARCH CHECK OK` and `OPTIM adamw lr=0.0004 betas=(0.9, 0.95)`; S6's also show `teacher: none (distill.alpha=0)`. No run collapsed: in every run the highest 250-step mean CE after step 2,000 is 2.86 or less. Gradient spikes still occur (largest per run 6.1-50.8) and all recovered.

S5 hit its 24 h walltime at about step 8,500 (s42) and 9,000 (s43). The continuations resumed from `last.ckpt` at steps 8,000 and 9,000 and finished: both logs show `resuming from …/screen2_S5_s4x/checkpoints/last.ckpt` and `TRAIN DONE`.

| arm | lever vs S1 | Δ non-emb | seed | val PPL | late-25% PPL (row) | MQAR | statute-recall EM (char-F1) | multi-hop EM | s/step | peak GB |
|---|---|---:|---|---|---|---|---|---|---|---|
| S1 `hybrid_legal_base` | control | — | 42 | 5.3634 | 5.0056 (4.9936) | 0.124 | 0.000 (0.166) | 0.000 | 2.8101 | 41.46 |
| S1 `hybrid_legal_base` | control | — | 43 | 5.3530 | 4.9972 (4.9842) | 0.149 | 0.000 (0.169) | 0.000 | 2.8276 | 41.46 |
| S0 `hybrid_legal_attn0` | no attention | +2.82% | 42 | 6.0241 | 5.8771 (5.8005) | 0.004 | 0.000 (0.146) | 0.000 | 2.9254 | 41.50 |
| S0 `hybrid_legal_attn0` | no attention | +2.82% | 43 | 6.0431 | 5.8925 (5.8143) | 0.011 | 0.000 (0.157) | 0.000 | 2.9354 | 41.50 |
| S2 `hybrid_legal_attn4` | four attention layers | −2.82% | 42 | 5.3611 | 5.0059 (4.9918) | 0.113 | 0.000 (0.145) | 0.000 | 2.6857 | 41.42 |
| S2 `hybrid_legal_attn4` | four attention layers | −2.82% | 43 | 5.3257 | 4.9619 (4.9501) | 0.112 | 0.000 (0.170) | 0.000 | 2.6898 | 41.42 |
| S3 `hybrid_legal_ds64` | `mamba3_d_state 64` | −0.75% | 42 | 5.2920 | 4.9460 (4.9313) | 0.105 | 0.000 (0.196) | 0.000 | 2.7437 | 41.45 |
| S3 `hybrid_legal_ds64` | `mamba3_d_state 64` | −0.75% | 43 | 5.3094 | 4.9533 (4.9416) | 0.085 | 0.000 (0.183) | 0.000 | 2.7607 | 41.45 |
| S4 `hybrid_legal_mtp` | MTP head on | +0.00% (+9.73M training-only head) | 42 | 5.3585 | 5.0003 (4.9883) | 0.092 | 0.000 (0.193) | 0.000 | 3.0599 | 41.67 |
| S4 `hybrid_legal_mtp` | MTP head on | +0.00% (+9.73M training-only head) | 43 | 5.3571 | 5.0027 (4.9895) | 0.098 | 0.000 (0.171) | 0.000 | 3.0688 | 41.67 |
| S5 `hybrid_legal_legacy` | legacy operators (PPL/probe-only) | +2.36% | 42 | 6.6649 | 6.6346 (6.5169) | 0.000 | 0.000 (0.151) | 0.000 | 9.3058 | 41.50 |
| S5 `hybrid_legal_legacy` | legacy operators (PPL/probe-only) | +2.36% | 43 | 6.6138 | 6.5769 (6.4596) | 0.000 | 0.000 (0.132) | 0.000 | 9.1475 | 41.50 |
| S6 `hybrid_legal_base` + `distill.alpha=0` | no KD (no teacher loaded) | 0 | 42 | 5.4175 | 5.1165 (5.0737) | 0.196 | 0.000 (0.153) | 0.000 | 1.1956 | 12.50 |
| S6 `hybrid_legal_base` + `distill.alpha=0` | no KD (no teacher loaded) | 0 | 43 | 5.3966 | 5.0957 (5.0529) | 0.189 | 0.000 (0.184) | 0.000 | 1.2141 | 12.50 |

Arms that are not parameter-matched carry their non-embedding delta against S1 (counts from `analysis/param_counts.md`; S1 = 104,066,872 non-embedding / 220,753,720 total). S5 is PPL/probe-only (no cached decode) and its +2.36% is a stated difference.

Two-seed mean ± sample SD:

| arm | val PPL | late-25% PPL | MQAR | statute-recall EM | char-F1 | multi-hop EM | s/step | peak GB |
|---|---|---|---|---|---|---|---|---|
| S1 | 5.3582 ± 0.0074 | 5.0014 ± 0.0059 | 0.137 ± 0.018 | 0.000 ± 0.000 | 0.168 ± 0.002 | 0.000 ± 0.000 | 2.819 ± 0.012 | 41.46 |
| S0 | 6.0336 ± 0.0134 | 5.8848 ± 0.0109 | 0.007 ± 0.005 | 0.000 ± 0.000 | 0.151 ± 0.008 | 0.000 ± 0.000 | 2.930 ± 0.007 | 41.50 |
| S2 | 5.3434 ± 0.0250 | 4.9839 ± 0.0311 | 0.113 ± 0.001 | 0.000 ± 0.000 | 0.157 ± 0.018 | 0.000 ± 0.000 | 2.688 ± 0.003 | 41.42 |
| S3 | 5.3007 ± 0.0123 | 4.9496 ± 0.0051 | 0.095 ± 0.015 | 0.000 ± 0.000 | 0.190 ± 0.009 | 0.000 ± 0.000 | 2.752 ± 0.012 | 41.45 |
| S4 | 5.3578 ± 0.0010 | 5.0015 ± 0.0017 | 0.095 ± 0.004 | 0.000 ± 0.000 | 0.182 ± 0.015 | 0.000 ± 0.000 | 3.064 ± 0.006 | 41.67 |
| S5 | 6.6394 ± 0.0361 | 6.6057 ± 0.0408 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.142 ± 0.013 | 0.000 ± 0.000 | 9.227 ± 0.112 | 41.50 |
| S6 | 5.4070 ± 0.0148 | 5.1061 ± 0.0147 | 0.193 ± 0.005 | 0.000 ± 0.000 | 0.169 ± 0.022 | 0.000 ± 0.000 | 1.205 ± 0.013 | 12.50 |

The rule applied (`scripts/screen_table.py`, bar 0.10):

```
RULE S0: val PPL S1-S0 = -0.6754 (bar 0.1); statute-recall loss +0.0000, multi-hop loss +0.0000 -> does not beat S1 by more than the bar
RULE S2: val PPL S1-S2 = +0.0148 (bar 0.1); statute-recall loss +0.0000, multi-hop loss +0.0000 -> does not beat S1 by more than the bar
RULE S3: val PPL S1-S3 = +0.0575 (bar 0.1); statute-recall loss +0.0000, multi-hop loss +0.0000 -> does not beat S1 by more than the bar
RULE S4: val PPL S1-S4 = +0.0004 (bar 0.1); statute-recall loss +0.0000, multi-hop loss +0.0000 -> does not beat S1 by more than the bar
RULE S5: val PPL S1-S5 = -1.2812 (bar 0.1); statute-recall loss +0.0000, multi-hop loss +0.0000 -> does not beat S1 by more than the bar
RULE S6: val PPL S1-S6 = -0.0488 (bar 0.1); statute-recall loss +0.0000, multi-hop loss +0.0000 -> does not beat S1 by more than the bar
RULE winner: S1 (no lever beats it as the rule requires)
```

### Seed table, first run (superseded: two runs collapsed)

val PPL = exp(`val/loss`) at step 11,999, the token-weighted validation CE over every validation row, read from each run's TensorBoard events by `scripts/screen_metrics.py`. The logged `val/perplexity` is a batch-mean of exp(CE) and is not used (see Limitations). s/step and peak GB (allocated) from each log's `STEPSTATS … final=1` line. S5 was cancelled seven minutes after it started, at step 24, when the screen was set to be re-run: no result. Late-25% PPL and probes were not computed for this run.

| arm | lever vs S1 | Δ non-emb | seed | val PPL | s/step | peak GB |
|---|---|---:|---|---|---|---|
| S1 `hybrid_legal_base` | control | — | 42 | **1184.77** (diverged, see Limitations) | 2.9569 | 41.46 |
| S1 `hybrid_legal_base` | control | — | 43 | 5.563 | 2.8894 | 41.46 |
| S0 `hybrid_legal_attn0` | no attention | +2.82% | 42 | 6.196 | 3.0236 | 41.50 |
| S0 `hybrid_legal_attn0` | no attention | +2.82% | 43 | 6.226 | 3.0253 | 41.50 |
| S2 `hybrid_legal_attn4` | four attention layers | −2.82% | 42 | 5.571 | 2.7372 | 41.42 |
| S2 `hybrid_legal_attn4` | four attention layers | −2.82% | 43 | 5.481 | 2.7314 | 41.42 |
| S3 `hybrid_legal_ds64` | `mamba3_d_state 64` | −0.75% | 42 | 5.464 | 2.8154 | 41.45 |
| S3 `hybrid_legal_ds64` | `mamba3_d_state 64` | −0.75% | 43 | **1186.43** (diverged, see Limitations) | 2.8749 | 41.45 |
| S4 `hybrid_legal_mtp` | MTP head on | +0.00% (+9.73M training-only head) | 42 | 5.473 | 3.1231 | 41.67 |
| S4 `hybrid_legal_mtp` | MTP head on | +0.00% (+9.73M training-only head) | 43 | 5.494 | 3.1389 | 41.67 |
| S5 `hybrid_legal_legacy` | legacy operators (PPL/probe-only) | +2.36% | 42 | – (cancelled) | | |
| S5 `hybrid_legal_legacy` | legacy operators (PPL/probe-only) | +2.36% | 43 | – (cancelled) | | |
| S6 `hybrid_legal_base` + `distill.alpha=0` | no KD | 0 | 42 | 5.270 | 2.8928 | 41.46 |
| S6 `hybrid_legal_base` + `distill.alpha=0` | no KD | 0 | 43 | 5.280 | 2.8621 | 41.46 |

### Paired comparisons

The screen decides by the selection rule on two-seed means; it makes no claim (a claim needs three seeds). No bootstrap CIs were computed for it. The differences against S1, with S1's seed SD beside them:

| metric | arm | arm mean | S1 mean | arm − S1 | S1 seed SD | beyond the bar (0.10)? |
|---|---|---|---|---|---|---|
| val PPL | S0 | 6.0336 | 5.3582 | +0.6754 | 0.0074 | yes |
| late-25% PPL | S0 | 5.8848 | 5.0014 | +0.8834 | 0.0059 | yes |
| MQAR | S0 | 0.0073 | 0.1367 | -0.1294 | 0.0180 | yes |
| statute-recall char-F1 | S0 | 0.1512 | 0.1678 | -0.0166 | 0.0018 | no |
| val PPL | S2 | 5.3434 | 5.3582 | -0.0148 | 0.0074 | no |
| late-25% PPL | S2 | 4.9839 | 5.0014 | -0.0175 | 0.0059 | no |
| MQAR | S2 | 0.1128 | 0.1367 | -0.0239 | 0.0180 | no |
| statute-recall char-F1 | S2 | 0.1575 | 0.1678 | -0.0103 | 0.0018 | no |
| val PPL | S3 | 5.3007 | 5.3582 | -0.0575 | 0.0074 | no |
| late-25% PPL | S3 | 4.9496 | 5.0014 | -0.0518 | 0.0059 | no |
| MQAR | S3 | 0.0952 | 0.1367 | -0.0415 | 0.0180 | no |
| statute-recall char-F1 | S3 | 0.1899 | 0.1678 | +0.0221 | 0.0018 | no |
| val PPL | S4 | 5.3578 | 5.3582 | -0.0004 | 0.0074 | no |
| late-25% PPL | S4 | 5.0015 | 5.0014 | +0.0001 | 0.0059 | no |
| MQAR | S4 | 0.0947 | 0.1367 | -0.0420 | 0.0180 | no |
| statute-recall char-F1 | S4 | 0.1817 | 0.1678 | +0.0140 | 0.0018 | no |
| val PPL | S5 | 6.6394 | 5.3582 | +1.2812 | 0.0074 | yes |
| late-25% PPL | S5 | 6.6057 | 5.0014 | +1.6044 | 0.0059 | yes |
| MQAR | S5 | 0.0000 | 0.1367 | -0.1367 | 0.0180 | yes |
| statute-recall char-F1 | S5 | 0.1417 | 0.1678 | -0.0261 | 0.0018 | no |
| val PPL | S6 | 5.4070 | 5.3582 | +0.0488 | 0.0074 | no |
| late-25% PPL | S6 | 5.1061 | 5.0014 | +0.1047 | 0.0059 | yes |
| MQAR | S6 | 0.1929 | 0.1367 | +0.0562 | 0.0180 | no |
| statute-recall char-F1 | S6 | 0.1686 | 0.1678 | +0.0008 | 0.0018 | no |

### Per-seed calls (where the seeds disagree)

Both seeds see the same batches (`dataset.seed: 0`), so each pair below differs in architecture and initialisation only.

| arm | val PPL arm − S1, seed 42 | seed 43 | seeds agree on the sign? |
|---|---|---|---|
| S0 | +0.6607 | +0.6901 | yes |
| S2 | -0.0023 | -0.0273 | yes |
| S3 | -0.0714 | -0.0436 | yes |
| S4 | -0.0049 | +0.0041 | **no** |
| S5 | +1.3015 | +1.2608 | yes |
| S6 | +0.0541 | +0.0436 | yes |

### Efficiency

s/step and peak GB come from the `STEPSTATS` lines of each arm's log: the median seconds per optimizer step after 5 warm-up steps, and the peak GB allocated on the GPU. S5's is step-weighted over its two legs; the first leg was killed at its walltime and so left no `final=1` line, and its last periodic line is used instead. S6's cost has no teacher in it, while every other arm runs the 8B teacher forward each step. S5's legacy operators run 3.3× slower than S1.

| arm | s/step (mean of 2 seeds) | peak GB allocated |
|---|---|---|
| S1 | 2.819 | 41.46 |
| S0 | 2.930 | 41.50 |
| S2 | 2.688 | 41.42 |
| S3 | 2.752 | 41.45 |
| S4 | 3.064 | 41.67 |
| S5 | 9.227 | 41.50 |
| S6 | 1.205 | 12.50 |

## Discussion

- **S1 is the backbone carried forward**: `hybrid_legal_base` (`attentionx2, mamba3x7, mlstmx3`, chunk 128, exact operators), with **AdamW β2 0.95**. The rule names it, and no lever beat it by the bar.
- Keeping two attention layers rather than none: S0 is worse by 0.675 val PPL and 0.88 late-25% PPL, and its MQAR is near 0 (0.007 against 0.137).
- Keeping the corrected operators over the legacy recipe: S5 is worse by 1.28 val PPL and is 3.3× slower per step.

## Limitations

- **Two seeds do not support claims.** The screen only chooses the backbone by the pre-specified rule; a claim would need three seeds.
- **S3 (`d_state 64`) and S2 (four attention layers) are not shown to be better:** both are lower on val PPL by less than the bar (0.058, 0.015), on two seeds that share data.
- **Statute recall and multi-hop do not separate the arms:** exact match is 0.000 everywhere, a floor (100 and 139 items), so the rule's recall clause compares zeros. A 220M model after 393M tokens does not copy a whole Absatz verbatim, nor answer the two-hop title question exactly. The probes need a larger model, more tokens or a softer score before they can steer a decision; the full-length pretraining run is the next test. MQAR (0.000-0.196) and char-F1 (0.13-0.20) do move, and char-F1 differences are 0.026 or less.
- **KD:** S6 is worse on val PPL by 0.049 (inside the bar) and on late-25% PPL by 0.105 (outside it). Its MQAR is the highest (0.193 against 0.137), which is unexplained. So the screen does not show that KD fails to help.
- **Both seeds see the same batches.** `configs/dataset/mixture_pretrain.yaml` fixes the data schedule at `seed: 0`, so seeds 42 and 43 differ in initialisation only (steps 1,774, 2,949 and 3,474 spike in both). The two-seed SD behind the bar therefore measures initialisation variance alone, not data-order variance.
- Validation reads every validation row once (36,640 rows of 2,048 tokens); dip's long papers are about 61% of those tokens, so the pooled val PPL leans on dip. It is the same for every arm.
- **An earlier S1 launch failed with no result.** Both seeds (S1-s42 after 00:22:49, S1-s43 after 00:12:25) raised `TFLAFallbackError` in `mlstm_block` (exact TFLA, chunk 128, `fallback=error`): a chunk's log-decay half-range reached 43.2 (s42, step 838) and 54.6 (s43, step 419), above the factorised path's fp32 limit of 40. Training was healthy until then at 0.70 it/s on the progress bar, which counts micro-batches (accum 2; the retry measured 2.89-2.96 s per optimizer step). No validation ran, so no number from it enters any table. The operator change that resolved it applies to every arm.
- **Two of the twelve finished first-run runs diverged: S1-s42 and S3-s43.** Both completed 12,000 steps without an error, then ended at val loss 7.077 and 7.079 (PPL about 1,185), against 1.66-1.83 for the other ten. From the 250-step `CURVE` rows: S1-s42 trained normally to step 3,499 (CE 2.43-2.65 per 250-step window, grad norm mean 0.38-0.61 outside one spike), spiked in steps 3,500-3,749 (grad norm mean 10.0, max 93.3) and sat at CE about 7.4 from step 3,750 on, with grad norm back to 0.6-0.8. S3-s43 trained normally to step 6,249 (val loss 2.038 at step 3,999) and spiked in steps 6,250-6,499 (grad norm max 541.6), then the same plateau. Both spikes came at the constant 4e-4 of the WSD stable phase with clip 1.0. Spikes that recovered appear in most runs: the largest grad norm per run is 74.6 (S0-s42), 58.2 (S1-s42, steps 2,750-2,999, before its collapse), 33.9 (S2-s43), 31.1 (S1-s43), 28.4 (S4-s42), 17.8 (S0-s43), 11.1 (S4-s43), 8.2 (S2-s42), 4.2 (S3-s42), against 3.9 and 3.1 for the two runs without KD (S6). Ten finished KD runs against two without is a hint that the KD term feeds the spikes, not a finding. The plateau CE of about 7.2-7.4 is roughly what a unigram model scores: the model collapsed and did not recover in 6,000-8,000 more steps. At the 25-step resolution TensorBoard keeps, neither collapse was announced: S1-s42's grad norm was 0.33-0.40 (one 2.52 at step 3,474, recovered) up to step 3,624, then 93.33 at step 3,649, when the loss was already 9.32. S3-s43's was 0.29-0.41 up to step 6,349, then 541.58 at step 6,374 with the loss still 2.23, and 9.76 at step 6,399. One step, or a few, inside 25 took the model from CE about 2.4 to about 7.4. The cause is not established; `last.ckpt` is from step 12,000, so no checkpoint from before either spike exists. With S1-s42 diverged, the bar as specified would be max(2 × 833.82, 0.10) = 1,667.64 PPL (S1 PPL 1,184.765 and 5.563), which no arm difference can exceed, so the screen was re-run instead.
- **What the collapse left behind** (`scripts/inspect_checkpoint.py` on `last.ckpt`). The RMS of AdamW's next update, `exp_avg / (sqrt(exp_avg_sq) + eps)`, measures per block how much gradient still arrives:
  - **S1-s42 (collapsed):** about 1e-8 in L0-L3, rising through L4 5e-6, L6 6e-4, L8 0.03 to L11 0.09.
  - **S3-s43 (collapsed):** 1e-4 to 1e-3 in L0-L8, then 0.008, 0.035 and 0.089 in L9-L11.
  - **The healthy twins** (S1-s43, S3-s42): 0.14-0.22 in every block.

  After the collapse almost no gradient reached the lower blocks, falling roughly tenfold per block going down. Their weights decayed to 0.6-0.85× the healthy norm. Nothing blew up: norm weights, the mLSTM gate biases (still at their init values in every run), `A_log` and `dt_bias` all look like the healthy run's. With `norm_topology: hybrid`, every block after the first ends in `norm2(x + mlp(x))`, so the stream is re-normalised after each block and no identity path runs through the stack. That fits gradient starvation that does not heal; it is not proven to be the cause.
- **The spikes are tied to batches, and KD amplifies them** (`train/grad_norm` of all runs at every logged step, one in 25). Arms with the same seed spike on the same logged steps: at step 2,949 (s42) S1 reached 58.2, S2 8.2, S6 2.9; at step 3,474 (s43) all six arms showed a spike. The two runs without KD (S6) never exceed 3.9, while the KD runs reach 10-75 on those batches. The two fatal steps (S1-s42 at 3,649, S3-s43 at 6,374) appear in no other run at the logged resolution.
- **A diagnostic re-run of S1-s42 ran with β2 0.95, not the first run's 0.999.** Its log shows `OPTIM … betas=(0.9, 0.95)` and experiment `diag_screen2_S1_s42`. SLURM had copied the batch script at submission, before the edit that pinned 0.999, while `screen_arms.py` and the yamls it reads were taken from disk when it started. So it is a third S1-s42 run under the fix, with per-block diagnostics, and it does not show the first run's failure. It stopped at step 7,000 as planned with val loss 2.0158 (step 3,999) and 1.8803 (step 6,999), against the first run's collapsed 7.09 at step 3,999. It printed 108 `DIAG SPIKE` steps (more than 5× the running median grad norm). The largest was 69.7 at step 2,948, the batch that spiked the first run (step 2,949); its per-block norms were L4_mlstm 47.0, L0_mamba3 30.9, L2_mamba3 23.3, L3_attention 22.0. Every spike recovered. The big spikes sit mostly in the lower stack (L0-L4) and the first mLSTM block.
- **With β2 0.95 the spikes stay, but they no longer kill a run.** In the second run the largest grad norm per run is 6.1-50.8, and S6 (no KD) reaches 27.7 (`screen2_S6_s43`, step 7,000 window). That weakens the first run's hint that KD feeds the spikes. What changed is that every run recovers.
- **The first probe evaluation crashed before scoring anything** (all 12 tasks, exit 1, after the val-PPL step had finished). `multihop.build_items` called `parse_citation` on gii's `1-DM-GoldmünzG §4`, and the statute grammar refused it: it only took an act code that is one capitalised token. 11,423 of gii's 90,255 citation ids (12.7%, 899 acts, among them every book of the Sozialgesetzbuch, `SGB 5`) failed the same way. Another 111 (`DBAbkG ARM Art. 3`) were mis-read as decisions, because the decision grammar's court pattern takes any word ending in G. The statute grammar now takes the code as everything before the provision, matched lazily. On the 90,255 gii ids collected from the scrubbed parts, all parse; the 78,671 that parsed as statutes before parse identically; only the 111 move from decision to statute. The probes were re-run (`VALPPL=0`).
- **The logged `val/perplexity` is not the validation PPL.** `pretrain_module.validation_step` logged `compute_perplexity(ce)` per batch with `on_epoch=True`, so the epoch value is the token-weighted mean of exp(batch CE). Jensen's inequality makes it at least exp(mean CE), and a few hard batches can dominate it: S0-s43 at step 7,999 logged 31.69 against exp(2.1031) = 8.19. Every number here uses exp(`val/loss`). This was fixed (`_accumulate_val`): the second run's logged `val/perplexity` equals exp(`val/loss`).
- **The first run's S6 ran the teacher forward** although `distill.alpha=0` (`pretrain_module.py` checked `self.teacher`, not alpha), so that S6's s/step and peak GB include the 8B teacher. The second run loads no teacher when alpha is 0 (`build_teacher`).
- **S5 ran in two legs.** Each first leg's timeout lost the steps after its last checkpoint, at least 8,000-8,500 for s42 and whatever ran past 9,000 for s43. The second leg re-ran them from that checkpoint. **The second leg did not continue the data schedule.** The training loader is not resumable, so Lightning restarted it at row 0, and the continuation logs show its `PossibleUserWarning` ("resuming from a checkpoint that ended before the epoch ended and your dataloader is not resumable"). S5's last 4,000 (s42) and 3,000 (s43) optimizer steps therefore re-read the schedule's first 64,000 and 48,000 rows, which the run had already seen, instead of rows 128,000-192,000. The other 12 second-run runs never resumed. S5 is PPL/probe-only and lost by 1.28 PPL; with fresh rows it might have been somewhat better, but not by the 1.18 PPL it would need to reach the bar. The conclusion does not change, and S5 was not re-run. The loader was made resumable afterwards. The val PPL is from the second leg's TensorBoard (`version_1`); the readers take the highest version.

## Reproduction

- **Training:** `scripts/slurm/screen_array.sh` (`ARMS="S1-s42 S1-s43"`, then the other arms; S5 with `--time=24:00:00`, plus a continuation) delegates to `scripts/slurm/train_pretrain_1gpu.sh`. Arms and screen settings come from `scripts/screen_arms.py` (`ATTEMPT = 2`).
- **Metrics:** validation PPL readers via `scripts/slurm/screen_metrics.sh`; late-25% PPL via `scripts/evaluate_val_ppl.py`; probes via `scripts/slurm/eval_probes.sh`.
- **Configs and checkpoints:** configs `configs/model/hybrid_legal_*.yaml` (`adam_beta2: 0.95`). Checkpoints (cluster-only): `last.ckpt` per arm under `$SCRATCH_ROOT/outputs/screen2_<arm>_s<seed>/`.
- **The table:** `.venv/bin/python scripts/screen_table.py --prefix screen2_ --bar 0.10 --val-logs <reader logs> --train-logs <training logs> --eval-dirs <evaluation output directories>`
