# P4 results — profile ladder, chunk size, KD memory

> **Written:** 2026-09-30, plan items P4-Q and P4-U. **Plan:** `LEGAL_BUILD_PLAN.md` · **State:** `legal_build_state.json`.
> **Measured:** gx07, 1×H100 80 GB (profile, equivalence) and 4×H100 (KD probe), bf16, batch 4, 10 timed iterations per point, random weights; jobs 2589361 (profile), 2589360 and 2589378 (equivalence), 2589362, 2589379 and 2589417 (KD probe). Logs under `cluster/logs/<name>_<id>.log`; per-point CSVs under `cluster/logs/profile_2589361/<point>/efficiency_curves.csv`.
> **Precedence:** where a summary line elsewhere disagrees with this document, the tables below are the record.

## 1. Summary

1. The ported `ref_hybrid_m3` reproduces the reference's compiled chunk-128 point at L=16,384: 121.06 ms, 7.17 GB (reference ≈ 123 ms, 7.17 GB). Uncompiled at chunk 64 it runs at 571.43 ms, 16% *faster* than the reference's ≈ 680 ms, so outside the ±10% band. (job 2589361)
2. Chunk size stays **128**. It passes R1 on `hybrid_legal_base` (chunk 64 vs 128: 1.68e-6 rel-max-err; compiled vs eager: 1.85e-6; tolerance 1e-4). At 4,096 packed training it is faster than chunk 64: 444.20 ms vs 648.59 ms median. (jobs 2589361, 2589360)
3. The packed (flex) path costs `hybrid_legal_base` ×1.14 / ×1.10 / ×1.10 over the unpacked SDPA step at 2,048 / 4,096 / 8,192 tokens, within the ×1.2 prediction. On the all-attention `transformer_legal_base` it costs ×1.83 / ×1.34 / ×1.08, outside the prediction at 2,048 and 4,096 tokens. (job 2589361)
4. KD probe, Qwen3-8B-Base teacher: out of memory at 4,096-token rows (micro-batch 8) and at 8,192 (micro-batch 4). At the OOM each rank held 77.0 / 76.1 GiB (82.7 / 81.7 GB) allocated of 79.2 GiB (85.0 GB), failing in the slab KL. (job 2589362)
5. KD probe, Qwen3-1.7B-Base teacher (rerun after the cache fix). At 4,096 rows (8 × 2 × 4 GPUs, 262,144 tokens/step): 4.144 s/step, peak 80.60 GB allocated / 83.29 GB reserved per rank. At 8,192 rows (4 × 4 × 4, 524,288 tokens/step): 9.186 s/step, 70.74 / 80.26 GB. (job 2589379)
6. **Teacher: Qwen/Qwen3-8B-Base with gradient checkpointing** (decided 2026-10-01 after job 2589417; the provisional 1.7B came from the unchecked probes).
7. With gradient checkpointing on (P6's recipe; job 2589417), both teachers fit at both shapes. 1.7B: 4.224 s/step at 30.11 GB (4,096 rows) and 10.891 s at 20.14 GB (8,192). 8B: 6.104 s at 43.18 GB and 15.174 s at 33.22 GB. Applied to these numbers, the P4-U rule would pick the 8B: 43.18 GB ≤ 70 and 6.104 / 4.224 = 1.44× ≤ 1.6×. The user decides between the two (§4). By the P4-U rule as written, the 8B teacher's peak exceeds 70 GB at 4,096 rows (it does not fit at all). Both probes ran *without* model gradient checkpointing, which P6's recipe lists (§7).

## 2. What was pre-registered, and what happened

| Prediction (written before submission) | Outcome | Verdict |
|---|---|---|
| `ref_hybrid_m3` reproduces the reference's L=16,384 numbers within 10%: compiled chunk 128 ≈ 123 ms, 7.17 GB | 121.06 ms (−1.6%), 7.169 GB | CONFIRMED |
| … uncompiled chunk 64 ≈ 680 ms | 571.43 ms (−16%) | REFUTED (faster than the band) |
| `hybrid_legal_base` at 4,096 rows trains uncompiled within 35 GB per GPU with the 1.7B teacher | 80.60 GB allocated per rank (job 2589379; no gradient checkpointing) | REFUTED |
| … within 60 GB with the 8B teacher | OOM at micro-batch 8 (> 79 GB per rank) | REFUTED |
| the flex-attention packed path is within 1.2× of the unpacked SDPA time | hybrid 1.14 / 1.10 / 1.10; transformer 1.83 / 1.34 / 1.08 | PARTIAL: holds for the hybrid, not for the all-attention model below 8,192 |

Decision rule as written before the numbers (P4-U): "8B if peak ≤ 70 GB at 4,096 rows and step time ≤ 1.6× the 1.7B step; else 1.7B". 8B peak > 82.7 GB (OOM) without gradient checkpointing, so the first, provisional decision was Qwen/Qwen3-1.7B-Base. With checkpointing on (P6's recipe, job 2589417), the same rule passes the 8B: 43.18 GB ≤ 70 and 1.44× ≤ 1.6×. **Decision (2026-10-01): `decisions.teacher` = Qwen/Qwen3-8B-Base, with gradient checkpointing on** (the user: "choose whichever would give the best result, I am thinking of 8B"). The training wrappers default to it: `DISTILL_CFG=qwen3_8b`, and `GRAD_CKPT=true` for the 8B.

Decision rule as written before the numbers (P4-Q): "128 unless it fails R1 or loses > 5% at 4,096". Chunk 128 passes R1 and wins by 31.5% at 4,096. **Decision: `mamba3_chunk_size` = `mlstm_chunk_size` = 128** (unchanged).

## 3. Results

### 3.1 Sanity points (L=16,384, forward, bf16, batch 4)

| point | chunk (Mamba-3 / mLSTM, effective) | compiled | median ms | SD ms | peak GB | job |
|---|---|---|---|---|---|---|
| `ref_transformer` | – | no | 175.99 | 0.71 | 7.25 | 2589361 |
| `ref_hybrid_m3` | 64 / 128 | no | 571.43 | 10.90 | 7.19 | 2589361 |
| `ref_hybrid_m3` | 128 / 128 | yes | 121.06 | 2.61 | 7.17 | 2589361 |

### 3.2 Chunk size (training, forward+backward, slab loss, packed, doc_len 1,000, L=4,096)

| chunk (effective) | median ms | SD ms | peak GB | R1 on `hybrid_legal_base` (L=2,048, fp32) | job |
|---|---|---|---|---|---|
| 128 / 128 | 444.20 | 48.97 | 28.21 | baseline | 2589361, 2589360 |
| 64 / 64 | 648.59 | 73.82 | 26.86 | 1.68e-6 PASS | 2589361, 2589360 |

## 4. Efficiency

Uncompiled, bf16, batch 4, 10 iterations; median ms (SD) / peak GB. `effective_chunk_size` is 128/128 on every hybrid row (printed by the profiler from the built module).

Inference (forward, causal SDPA):

| L | `hybrid_legal_base` | `transformer_legal_base` | ratio (hybrid / transformer) | job |
|---|---|---|---|---|
| 2,048 | 47.13 (4.68) / 2.99 | 14.06 (1.16) / 3.03 | 3.35 | 2589361 |
| 8,192 | 176.85 (10.39) / 10.54 | 63.70 (7.81) / 10.57 | 2.78 | 2589361 |
| 16,384 | 364.94 (48.76) / 20.60 | 153.14 (3.38) / 20.66 | 2.38 | 2589361 |

Training (forward+backward, slab loss), unpacked SDPA vs packed flex (doc_len 1,000):

| L | hybrid unpacked | hybrid packed | packed / unpacked | transformer unpacked | transformer packed | packed / unpacked | job |
|---|---|---|---|---|---|---|---|
| 2,048 | 191.94 (22.05) / 15.19 | 219.37 (35.45) / 16.30 | 1.14 | 83.43 (22.91) / 9.09 | 153.02 (10.96) / 9.12 | 1.83 | 2589361 |
| 4,096 | 402.82 (44.47) / 25.98 | 444.20 (48.97) / 28.21 | 1.10 | 169.74 (31.36) / 13.65 | 226.66 (14.52) / 13.72 | 1.34 | 2589361 |
| 8,192 | 917.26 (134.53) / 47.56 | 1005.94 (153.21) / 52.01 | 1.10 | 353.96 (14.37) / 22.78 | 381.69 (20.48) / 22.91 | 1.08 | 2589361 |

One length per process, one Inductor cache per point.

KD memory probe (P4-T/U): `hybrid_legal_base`, online logit KD (α 0.5, T 2, slab 512), 4×H100 DDP (`h100_multi_ddp`), synthetic packed rows at the Qwen3 vocabulary, uncompiled, bf16-mixed, **model gradient checkpointing off** (config default), 20 steps with 15 timed; s/step is the mean over the 4 ranks (spread ≤ 0.004 s). Peak GB = `torch.cuda.max_memory_*() / 1e9`; an H100 holds 85.0 GB.

| teacher | rows | micro-batch × accum × GPUs | tokens / step | s / step | tokens / s | peak alloc GB | peak reserved GB | job |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | 4,096 | 8 × 2 × 4 | 262,144 | 4.144 | 63,250 | 80.60 | 83.29 | 2589379 |
| Qwen3-1.7B-Base | 8,192 | 4 × 4 × 4 | 524,288 | 9.186 | 57,070 | 70.74 | 80.26 | 2589379 |
| Qwen3-8B-Base | 4,096 | 8 × 2 × 4 | 262,144 | OOM | – | ≥ 82.7 at failure | – | 2589362 |
| Qwen3-8B-Base | 8,192 | 4 × 4 × 4 | 524,288 | OOM | – | ≥ 81.7 at failure | – | 2589362 |

The same probe with **model gradient checkpointing on** (`GRAD_CKPT=true`, non-reentrant; P6's recipe), every other setting unchanged:

| teacher | rows | micro-batch × accum × GPUs | tokens / step | s / step | tokens / s | peak alloc GB | peak reserved GB | job |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | 4,096 | 8 × 2 × 4 | 262,144 | 4.224 | 62,050 | 30.11 | 36.42 | 2589417 |
| Qwen3-1.7B-Base | 8,192 | 4 × 4 × 4 | 524,288 | 10.891 | 48,140 | 20.14 | 30.19 | 2589417 |
| Qwen3-8B-Base | 4,096 | 8 × 2 × 4 | 262,144 | 6.104 | 42,950 | 43.18 | 49.13 | 2589417 |
| Qwen3-8B-Base | 8,192 | 4 × 4 × 4 | 524,288 | 15.174 | 34,550 | 33.22 | 43.34 | 2589417 |

Prediction written before job 2589417 (note of 2026-09-30): 1.7B ≤ 50 GB and 8B ≤ 70 GB at 4,096 rows; step time +20–40% over the unchecked 1.7B. Outcome: PARTIAL. Both memory bounds hold (30.11 and 43.18 GB). Checkpointing cost +1.9% at 4,096 rows (4.144 → 4.224 s) and +18.6% at 8,192 (9.186 → 10.891 s), below the predicted range.

Teacher cost at P6's shapes, from these steps (an estimate: 15 timed steps on synthetic rows). With a 5B-token budget (90% at 4,096 rows = 17,166 steps; 10% at 8,192 = 954 steps), one P6 run takes ≈ 23.0 h with the 1.7B teacher and ≈ 33.1 h with the 8B (+44%). The budget is `min(5B, available commercial-safe)` (P4-M), so both shrink with the corpus.

## 5. What this licenses

- The port's compiled `ref_hybrid_m3` runs at the reference's speed and memory at 16,384 tokens.
- Chunk 128 for `hybrid_legal_base`: equivalent to chunk 64 within R1 and clearly faster at 4,096. The 204 ms gap is several SDs wide.
- Packed training costs the hybrid about 10–14% per step over unpacked at 2,048–8,192 tokens.
- The 1.7B teacher at P6's stage-A and stage-B shapes (micro-batch 8 × 2 at 4,096; 4 × 4 at 8,192) fits on 4×H100 without gradient checkpointing, at 4.14 and 9.19 s/step.
- The 8B teacher does not fit at those shapes without gradient checkpointing. With it (P6's recipe), both teachers fit at both shapes with ≥ 35 GB to spare per GPU; the 8B costs 1.44× the 1.7B step at 4,096 rows and 1.39× at 8,192.

## 6. Not licensed

- A claim that the uncompiled port is faster than the reference: the reference number is from another torch build and another run; one 10-iteration point is not a comparison.
- Any hybrid-vs-transformer speed claim beyond these uncompiled points: the hybrid is 2.4–3.4× slower uncompiled at inference; compiled training was not profiled here.
- Which teacher distils better: nothing here measures the student's quality under either teacher. The 8B was chosen by the pre-registered rule (memory and speed) and the user's preference, not by a quality measurement. A ~36× teacher/student size gap is large: distillation work reports that a much larger teacher does not always give a better small student.
- Step times on real shards: the probe's rows are synthetic (the KD cost does not depend on token values, but the packing does).

## 7. Open limitations

- 10 timed iterations per point; SDs reach 15% of the median on the long training points. Ratios near the ×1.2 bar (hybrid packed at 2,048: 1.14) are within that noise.
- The flex overhead on the all-attention model at 2,048–4,096 tokens is not explained here. The block mask is built outside compilation (`build_doc_block_mask`, `@torch.compiler.disable`), a fixed per-step cost that would weigh most at short rows. That is a hypothesis, not a measurement.
- Equivalence job 2589360 failed R1 on `ref_hybrid_m3` because of the checker, not the operator. `check_operator_equivalence.py` set every mixer's `chunk_size`, including the mLSTM's TFLA chunk (128 in that config), so the "chunk 64" row and the reset before the compile check ran the mLSTM at 64. Both read 4.518e-4. Fixed to sweep only the Mamba-3 chunk, test added; rerun job 2589378 passed (`ref_hybrid_m3`: chunk 64 0.000e+00, chunk 128 1.646e-6, compiled 1.685e-6; P4-S). Side observation from that failure: the legacy TFLA path's logits move by 4.5e-4 between mLSTM chunk 64 and 128 at L=512 in fp32. `ref_hybrid_m3` is never trained.
- KD probe vs P6 recipe: jobs 2589362 and 2589379 ran with `model.use_gradient_checkpointing=false` (the model default), while P6's recipe says "gradient checkpointing". The provisional teacher came from the rule on those runs. At the user's request, both teachers were then probed with checkpointing on (job 2589417, a new configuration, not a rerun), and the choice was left open to the user.
- Without checkpointing, the 1.7B run at 4,096 rows leaves under 2 GB of headroom (80.60 / 83.29 GB of 85.0). With it, 30.11 / 36.42 GB.

## 8. Reproduction

Key job ids: 2589361 (profile), 2589360 / 2589378 (equivalence; the first failed on the checker), 2589362 (KD probe; 8B OOMs), 2589379 (KD probe; 1.7B, `PROBES='t1p7b_L4096 t1p7b_L8192'`), 2589417 (KD probe, all four shapes, `GRAD_CKPT=true`). Wrappers: `scripts/slurm/profile.sh`, `scripts/slurm/equivalence.sh`, `scripts/slurm/kd_memory_probe.sh`. Configs: `configs/model/{ref_hybrid_m3,ref_transformer,hybrid_legal_base,transformer_legal_base}.yaml`. Every number above has a job id and a log path (R12).
