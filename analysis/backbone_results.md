# Backbone results: the hybrid and a Transformer baseline at 8,192 tokens

Two seeds of the 220M-parameter hybrid and one seed of a matching Transformer, each trained in two stages on the commercial-safe German legal corpus, then compared on validation perplexity, late-position perplexity and three probes.

**Setup.** Stage A: 4×H100 DDP, bf16-mixed, 17,000 steps × 262,144 tokens (4,456,448,000 tokens) on 4,096-token rows, AdamW β2 0.95, Qwen3-8B-Base teacher (α 0.5, T 2). Stage B: 1,037 steps × 524,288 tokens on 8,192-token rows, to step 18,037. Evaluation: one H100, bf16, 75,007,430 validation tokens in 9,161 rows of 8,192, on the final checkpoint of each run.

## Summary

The hybrid's stage-B validation perplexity is 3.7831 ± 0.0042 (two seeds). The single Transformer seed is 3.8337. **This is descriptive and supports no claim:** there is one Transformer seed, and the gap of 0.051 is inside the 0.10 PPL bar the architecture screen used.

Of the four predictions, two held and two did not. The seeds agree, and stage B improved late-position perplexity by at least 3%. Parity with the Transformer did not hold as written, because the hybrid is lower by far more than one seed SD. Statute recall **fell** at the longer context on all three checkpoints.

**No model copies a statute verbatim.** Exact match on statute recall is 0.00 to 0.01, and on multi-hop it is 0.000, at both context lengths. These probes remain near their floor.

## Predictions

| Prediction (stated before the run) | Result | Outcome |
|---|---|---|
| The hybrid reaches a lower val PPL than the Transformer by less than one seed SD (parity, not a win) | Lower by 0.0506; the two-seed SD is 0.0042, so about 12 SDs | **Refuted** as written; one Transformer seed, no claim |
| Late-25% PPL at 8,192 improves after stage B by at least 3% | −3.57% and −4.38% on the two hybrid seeds | **Confirmed** on the document-position definition; the row-position variant is −2.85% and −3.36% |
| Statute recall at 8K is at least its 2K value | char-F1 0.1732 against 0.2210 and 0.1870 against 0.2282 (hybrid), 0.1732 against 0.2011 (Transformer) | **Refuted** |
| Both hybrid seeds finish within 10% of each other on val PPL | 3.7861 and 3.7801, 0.16% apart | **Confirmed** |

## Results

### Validation

| arm | seed | val PPL | late-25% PPL | late-25%, stage A → B |
|---|---|---|---|---|
| hybrid `hybrid_legal_base` | 42 | 3.7861 | 3.6533 | 3.7886 → 3.6533 (−3.57%) |
| hybrid `hybrid_legal_base` | 43 | 3.7801 | 3.6438 | 3.8107 → 3.6438 (−4.38%) |
| **hybrid mean ± SD** (n = 2) | | **3.7831 ± 0.0042** | **3.6486 ± 0.0067** | |
| Transformer `transformer_legal_base` | 42 | 3.8337 | 3.7156 | not comparable (below) |
| research-corpus hybrid (descriptive) | 42 | 3.8382 | 3.7200 | 3.8682 → 3.7200 (−3.83%) |

- **late-25% PPL:** predicted tokens with at least 1,536 tokens of their own document before them, 11,607,727 tokens. The row-position variant, stage A → B: seed 42 3.7566 → 3.6494 (−2.85%), seed 43 3.7682 → 3.6417 (−3.36%), research-corpus 3.8130 → 3.6967 (−3.05%).
- **Transformer stage A:** its PPL at 8,192 is 119.93 because it was trained only to 4,096 and does not extrapolate, so no stage A → B improvement is reported for it.

### Probes on the stage-B checkpoints, by context length

| arm | seed | context | MQAR | statute EM | statute char-F1 | statute gold NLL | multi-hop EM | multi-hop gold NLL |
|---|---|---|---|---|---|---|---|---|
| hybrid | 42 | 2,048 | 0.1191 | 0.00 | 0.2210 | 0.2578 | 0.000 | 1.0878 |
| hybrid | 42 | 8,192 | 0.0977 | 0.00 | 0.1732 | 0.8573 | 0.000 | 1.0840 |
| hybrid | 43 | 2,048 | 0.1445 | 0.01 | 0.2282 | 0.2219 | 0.000 | 1.1165 |
| hybrid | 43 | 8,192 | 0.1123 | 0.00 | 0.1870 | 0.7425 | 0.000 | 1.1174 |
| Transformer | 42 | 2,048 | 0.0957 | 0.00 | 0.2011 | 0.2791 | 0.000 | 1.0782 |
| Transformer | 42 | 8,192 | 0.0908 | 0.00 | 0.1732 | 0.8620 | 0.000 | 1.0816 |

MQAR has 256 items (1,024 queries) and statute recall 100, at both lengths. Multi-hop has 139 items at 2,048 and 200 at 8,192, because more items fit.

### Handoff

All stage-B checkpoints load into the model their config builds with 0 missing and 0 unexpected keys.

### Learning-rate sensitivity at screen scale

Every run above used LR 4e-4. To see how much that choice matters, the hybrid (`hybrid_legal_base`) and its no-attention variant (`hybrid_legal_attn0`) were retrained at 2e-4 and 8e-4, with the architecture screen's recipe otherwise unchanged: 12,000 steps of 2,048-token rows on one H100 each, AdamW β2 0.95, seed 42, the same data order. The 4e-4 rows are the screen's seed-42 runs. Val PPL is exp of the final `val/loss`. Late-25% PPL uses the screen's definition at 2,048 tokens (at least 1,536 tokens of the token's own document before it; the row-position variant is in brackets).

| arm | LR | val PPL | late-25% PPL | s/step | peak GB |
|---|---|---|---|---|---|
| hybrid | 2e-4 | 5.4840 | 5.1068 (5.1088) | 2.8117 | 41.46 |
| hybrid | **4e-4** | **5.3634** | **5.0056 (4.9936)** | 2.8101 | 41.46 |
| hybrid | 8e-4 | 5.6989 | 5.3273 (5.3219) | 2.9744 | 41.46 |
| no attention | 2e-4 | 6.4192 | 6.2966 (6.2212) | 2.9475 | 41.50 |
| no attention | **4e-4** | **6.0241** | **5.8771 (5.8005)** | 2.9254 | 41.50 |
| no attention | 8e-4 | 1,181.75 (diverged) | 1,211.69 (1,203.11) | 2.9891 | 41.50 |

- **4e-4 is the best of the three for both architectures**, on val PPL and on late-25% PPL. On the hybrid, half the LR costs 0.121 val PPL and twice the LR 0.336. Without attention, half the LR costs 0.395.
- **The hybrid leads at every learning rate.** The no-attention variant at its best LR (6.0241) trails the hybrid at its best (5.3634) by 0.661 PPL, more than six times the 0.10 bar the screen used. The choice of the hybrid does not rest on the LR.
- **The no-attention variant diverged at 8e-4.** The gradient norm spiked to 14.5 between steps 1,250 and 1,500. Between steps 1,750 and 2,250 the training loss rose from about 2.6 to 9.3, and it never recovered (training CE about 7.2 to the end). The hybrid at 8e-4 trained through gradient-norm spikes of up to 29 (steps 6,500 to 6,750) and finished 0.336 PPL behind its 4e-4 run.
- **Each LR has one seed.** At 4e-4 the two-seed SD is 0.0074 for the hybrid and 0.0134 without attention. Every difference above is many times that, but these are single runs.

## Discussion

Statute gold NLL roughly triples between 2,048 and 8,192 on all three checkpoints. The probe places the provision at the start of the prompt and fills the remainder with other law, so at 8,192 the answer sits about four times further back. Both architectures got worse by a similar amount. Stage B was short (1,037 steps), so this does not show what a longer stage at 8,192 would reach.

Recall does not separate the architectures at 8,192: the hybrid seeds differ by 0.0138 in char-F1, twice the 0.0069 between their mean and the Transformer, and hybrid seed 42 ties the Transformer at 0.1732. At 2,048 the order reverses (seeds 0.0072 apart, mean 0.0235 above the Transformer), on one Transformer seed.

## Limitations

- **One Transformer seed and two hybrid seeds do not support a claim** in either direction, on any row above.
- **The two-seed SD is a small sample,** so "about 12 SDs" is a rough scale, not a test.
- **The 2,048 and 8,192 statute items are built separately.** Each takes the first 100 statutes that fit its context, and the distractors differ, so the comparison is not item by item. More statutes fit at 8,192.
- **Exact match is near its floor,** so the statute comparison rests on char-F1 and gold NLL.
- **The document-position late-25% is the pre-stated definition.** On the row-position variant, seed 42 misses 3% (−2.85%).
- **The research-corpus arm** is evaluated on the commercial-safe validation rows and has no probes. It is descriptive only.
- **Learning-rate sensitivity was measured at screen scale only** (2,048-token rows, 12,000 steps, one seed per LR). The full backbone runs used LR 4e-4 throughout.

## Reproduction

- **Training:** `scripts/slurm/train_pretrain_4gpu.sh` with `configs/model/hybrid_legal_base.yaml` and `configs/model/transformer_legal_base.yaml`.
- **Metrics:** validation and late-25% PPL via `scripts/evaluate_val_ppl.py`; probes via `scripts/slurm/eval_probes.sh`, with `CONTEXT=2048` and `CONTEXT=8192`.
- **Learning-rate sensitivity:** `scripts/slurm/screen_array.sh` with the arms `S0-lr2-s42 S0-lr8-s42 S1-lr2-s42 S1-lr8-s42` (`scripts/screen_arms.py`); val PPL via `scripts/slurm/screen_metrics.sh`, late-25% PPL via `scripts/slurm/eval_probes.sh` with `PROBES=0`.
- **Checkpoints (cluster-only):** `$SCRATCH_ROOT/outputs/<run>/checkpoints/last.ckpt`, one run per row of the validation table.
