# Profile ladder, chunk size and distillation memory

Speed and memory of the hybrid and all-attention models, the Mamba-3/mLSTM chunk-size choice, and the memory cost of online logit distillation with two teacher sizes.

**Setup.** One H100 80 GB (profile, equivalence) and 4×H100 (distillation probe), bf16, batch 4, 10 timed iterations per point, random weights.

## Summary

1. The ported `ref_hybrid_m3` reproduces the upstream reference implementation's compiled chunk-128 point at L=16,384: 121.06 ms, 7.17 GB (reference ≈ 123 ms, 7.17 GB). Uncompiled at chunk 64 it runs at 571.43 ms, 16% *faster* than the reference's ≈ 680 ms, so outside the ±10% band.
2. Chunk size stays **128**. It passes the operator-equivalence check on `hybrid_legal_base` (chunk 64 vs 128: 1.68e-6 rel-max-err; compiled vs eager: 1.85e-6; tolerance 1e-4). At 4,096 packed training it is faster than chunk 64: 444.20 ms vs 648.59 ms median.
3. The packed (flex) path costs `hybrid_legal_base` ×1.14 / ×1.10 / ×1.10 over the unpacked SDPA step at 2,048 / 4,096 / 8,192 tokens, within the ×1.2 hypothesis. On the all-attention `transformer_legal_base` it costs ×1.83 / ×1.34 / ×1.08, outside the hypothesis at 2,048 and 4,096 tokens.
4. Distillation probe, Qwen3-8B-Base teacher, no gradient checkpointing: out of memory at 4,096-token rows (micro-batch 8) and at 8,192 (micro-batch 4). At the OOM each rank held 77.0 / 76.1 GiB (82.7 / 81.7 GB) allocated of 79.2 GiB (85.0 GB), failing in the slab KL.
5. Distillation probe, Qwen3-1.7B-Base teacher, no gradient checkpointing (rerun after a cache fix). At 4,096 rows (8 × 2 × 4 GPUs, 262,144 tokens/step): 4.144 s/step, peak 80.60 GB allocated / 83.29 GB reserved per rank. At 8,192 rows (4 × 4 × 4, 524,288 tokens/step): 9.186 s/step, 70.74 / 80.26 GB.
6. With gradient checkpointing on, both teachers fit at both shapes. 1.7B: 4.224 s/step at 30.11 GB (4,096 rows) and 10.891 s at 20.14 GB (8,192). 8B: 6.104 s at 43.18 GB and 15.174 s at 33.22 GB.
7. **Teacher selected: Qwen/Qwen3-8B-Base with gradient checkpointing.** The selection criterion (below) fails for the 8B without checkpointing and is met with it (43.18 GB ≤ 70 and 6.104 / 4.224 = 1.44× ≤ 1.6×). The first, provisional choice was the 1.7B, made from the probes without checkpointing.

## Hypotheses and criteria

Hypotheses and criteria were stated before the runs.

| Hypothesis (stated before the run) | Result | Outcome |
|---|---|---|
| `ref_hybrid_m3` reproduces the reference's L=16,384 numbers within 10%: compiled chunk 128 ≈ 123 ms, 7.17 GB | 121.06 ms (−1.6%), 7.169 GB | Confirmed |
| … uncompiled chunk 64 ≈ 680 ms | 571.43 ms (−16%) | Refuted (faster than the band) |
| `hybrid_legal_base` at 4,096 rows trains uncompiled within 35 GB per GPU with the 1.7B teacher | 80.60 GB allocated per rank (no gradient checkpointing) | Refuted |
| … within 60 GB with the 8B teacher | OOM at micro-batch 8 (> 79 GB per rank) | Refuted |
| The flex-attention packed path is within 1.2× of the unpacked SDPA time | hybrid 1.14 / 1.10 / 1.10; transformer 1.83 / 1.34 / 1.08 | Partial: holds for the hybrid, not for the all-attention model below 8,192 |

**Teacher criterion (stated before the numbers):** "8B if peak ≤ 70 GB at 4,096 rows and step time ≤ 1.6× the 1.7B step; else 1.7B". Without gradient checkpointing the 8B peak exceeds 82.7 GB (OOM), so the provisional choice was Qwen/Qwen3-1.7B-Base. With checkpointing on, the same criterion is met by the 8B: 43.18 GB ≤ 70 and 1.44× ≤ 1.6×. **Selected: `Qwen/Qwen3-8B-Base`, with gradient checkpointing on**, chosen as the teacher expected to give the best result. The training wrappers default to it: `DISTILL_CFG=qwen3_8b`, and `GRAD_CKPT=true` for the 8B.

**Chunk-size criterion (stated before the numbers):** "128 unless it fails the equivalence check or loses > 5% at 4,096". Chunk 128 passes the check and wins by 31.5% at 4,096. **Selected: `mamba3_chunk_size` = `mlstm_chunk_size` = 128** (unchanged).

## Results

### Sanity points (L=16,384, forward, bf16, batch 4)

| point | chunk (Mamba-3 / mLSTM, effective) | compiled | median ms | SD ms | peak GB |
|---|---|---|---|---|---|
| `ref_transformer` | – | no | 175.99 | 0.71 | 7.25 |
| `ref_hybrid_m3` | 64 / 128 | no | 571.43 | 10.90 | 7.19 |
| `ref_hybrid_m3` | 128 / 128 | yes | 121.06 | 2.61 | 7.17 |

### Chunk size (training, forward+backward, slab loss, packed, doc_len 1,000, L=4,096)

| chunk (effective) | median ms | SD ms | peak GB | equivalence check on `hybrid_legal_base` (L=2,048, fp32) |
|---|---|---|---|---|
| 128 / 128 | 444.20 | 48.97 | 28.21 | baseline |
| 64 / 64 | 648.59 | 73.82 | 26.86 | 1.68e-6, pass |

### Speed and memory

Uncompiled, bf16, batch 4, 10 iterations; median ms (SD) / peak GB. `effective_chunk_size` is 128/128 on every hybrid row (printed by the profiler from the built module).

Inference (forward, causal SDPA):

| L | `hybrid_legal_base` | `transformer_legal_base` | ratio (hybrid / transformer) |
|---|---|---|---|
| 2,048 | 47.13 (4.68) / 2.99 | 14.06 (1.16) / 3.03 | 3.35 |
| 8,192 | 176.85 (10.39) / 10.54 | 63.70 (7.81) / 10.57 | 2.78 |
| 16,384 | 364.94 (48.76) / 20.60 | 153.14 (3.38) / 20.66 | 2.38 |

Training (forward+backward, slab loss), unpacked SDPA vs packed flex (doc_len 1,000):

| L | hybrid unpacked | hybrid packed | packed / unpacked | transformer unpacked | transformer packed | packed / unpacked |
|---|---|---|---|---|---|---|
| 2,048 | 191.94 (22.05) / 15.19 | 219.37 (35.45) / 16.30 | 1.14 | 83.43 (22.91) / 9.09 | 153.02 (10.96) / 9.12 | 1.83 |
| 4,096 | 402.82 (44.47) / 25.98 | 444.20 (48.97) / 28.21 | 1.10 | 169.74 (31.36) / 13.65 | 226.66 (14.52) / 13.72 | 1.34 |
| 8,192 | 917.26 (134.53) / 47.56 | 1005.94 (153.21) / 52.01 | 1.10 | 353.96 (14.37) / 22.78 | 381.69 (20.48) / 22.91 | 1.08 |

One length per process, one Inductor cache per point.

### Distillation memory probe

`hybrid_legal_base`, online logit KD (α 0.5, T 2, slab 512), 4×H100 DDP (`h100_multi_ddp`), synthetic packed rows at the Qwen3 vocabulary, uncompiled, bf16-mixed, **model gradient checkpointing off** (config default), 20 steps with 15 timed; s/step is the mean over the 4 ranks (spread ≤ 0.004 s). Peak GB = `torch.cuda.max_memory_*() / 1e9`; an H100 holds 85.0 GB.

| teacher | rows | micro-batch × accum × GPUs | tokens / step | s / step | tokens / s | peak alloc GB | peak reserved GB |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | 4,096 | 8 × 2 × 4 | 262,144 | 4.144 | 63,250 | 80.60 | 83.29 |
| Qwen3-1.7B-Base | 8,192 | 4 × 4 × 4 | 524,288 | 9.186 | 57,070 | 70.74 | 80.26 |
| Qwen3-8B-Base | 4,096 | 8 × 2 × 4 | 262,144 | OOM | – | ≥ 82.7 at failure | – |
| Qwen3-8B-Base | 8,192 | 4 × 4 × 4 | 524,288 | OOM | – | ≥ 81.7 at failure | – |

The same probe with **model gradient checkpointing on** (`GRAD_CKPT=true`, non-reentrant), every other setting unchanged:

| teacher | rows | micro-batch × accum × GPUs | tokens / step | s / step | tokens / s | peak alloc GB | peak reserved GB |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | 4,096 | 8 × 2 × 4 | 262,144 | 4.224 | 62,050 | 30.11 | 36.42 |
| Qwen3-1.7B-Base | 8,192 | 4 × 4 × 4 | 524,288 | 10.891 | 48,140 | 20.14 | 30.19 |
| Qwen3-8B-Base | 4,096 | 8 × 2 × 4 | 262,144 | 6.104 | 42,950 | 43.18 | 49.13 |
| Qwen3-8B-Base | 8,192 | 4 × 4 × 4 | 524,288 | 15.174 | 34,550 | 33.22 | 43.34 |

Hypothesis (stated before this run): 1.7B ≤ 50 GB and 8B ≤ 70 GB at 4,096 rows; step time +20-40% over the unchecked 1.7B. Result: partially confirmed. Both memory bounds hold (30.11 and 43.18 GB). Checkpointing cost +1.9% at 4,096 rows (4.144 → 4.224 s) and +18.6% at 8,192 (9.186 → 10.891 s), below the hypothesised range.

Teacher cost at the pretraining shapes, from these steps (an estimate: 15 timed steps on synthetic rows). With a 5B-token budget (90% at 4,096 rows = 17,166 steps; 10% at 8,192 = 954 steps), one run takes ≈ 23.0 h with the 1.7B teacher and ≈ 33.1 h with the 8B (+44%). The budget is `min(5B, available commercial-safe)` (see `analysis/corpus_manifest.md`), so both shrink with the corpus.

## Discussion

- The port's compiled `ref_hybrid_m3` runs at the reference's speed and memory at 16,384 tokens.
- Chunk 128 for `hybrid_legal_base` is equivalent to chunk 64 within tolerance and clearly faster at 4,096. The 204 ms gap is several SDs wide.
- Packed training costs the hybrid about 10-14% per step over unpacked at 2,048-8,192 tokens.
- The 1.7B teacher at the two pretraining shapes (stage A, 4,096-token rows, and the longer-context stage B, 8,192-token rows; micro-batch 8 × 2 at 4,096; 4 × 4 at 8,192) fits on 4×H100 without gradient checkpointing, at 4.14 and 9.19 s/step.
- The 8B teacher does not fit at those shapes without gradient checkpointing. With it, both teachers fit at both shapes with ≥ 35 GB to spare per GPU; the 8B costs 1.44× the 1.7B step at 4,096 rows and 1.39× at 8,192.

## Limitations

- Not claimed: that the uncompiled port is faster than the reference. The reference number is from another torch build and another run; one 10-iteration point is not a comparison.
- Not claimed: any hybrid-vs-transformer speed result beyond these uncompiled points. The hybrid is 2.4-3.4× slower uncompiled at inference; compiled training was not profiled here.
- Not claimed: which teacher distils better. Nothing here measures the student's quality under either teacher. The 8B was chosen by the memory and speed criterion and a preference for the stronger teacher, not by a quality measurement. A ~36× teacher/student size gap is large: distillation work reports that a much larger teacher does not always give a better small student.
- Step times are not measured on real shards: the probe's rows are synthetic (the KD cost does not depend on token values, but the packing does).
- 10 timed iterations per point; SDs reach 15% of the median on the long training points. Ratios near the ×1.2 bar (hybrid packed at 2,048: 1.14) are within that noise.
- The flex overhead on the all-attention model at 2,048-4,096 tokens is not explained here. The block mask is built outside compilation (`build_doc_block_mask`, `@torch.compiler.disable`), a fixed per-step cost that would weigh most at short rows. That is a hypothesis, not a measurement.
- The first equivalence run failed on `ref_hybrid_m3` because of the checker, not the operator. `check_operator_equivalence.py` set every mixer's `chunk_size`, including the mLSTM's TFLA chunk (128 in that config), so the "chunk 64" row and the reset before the compile check ran the mLSTM at 64. Both read 4.518e-4. The checker was fixed to sweep only the Mamba-3 chunk, a test was added, and the rerun passed (`ref_hybrid_m3`: chunk 64 0.000e+00, chunk 128 1.646e-6, compiled 1.685e-6). Side observation from that failure: the legacy TFLA path's logits move by 4.5e-4 between mLSTM chunk 64 and 128 at L=512 in fp32. `ref_hybrid_m3` is never trained.
- The first two distillation probes ran with `model.use_gradient_checkpointing=false` (the model default), while the training recipe uses gradient checkpointing. The provisional teacher came from the criterion applied to those runs. Both teachers were then probed with checkpointing on (a new configuration, not a rerun).
- Without checkpointing, the 1.7B run at 4,096 rows leaves under 2 GB of headroom (80.60 / 83.29 GB of 85.0). With it, 30.11 / 36.42 GB.

## Reproduction

Wrappers: `scripts/slurm/profile.sh`, `scripts/slurm/equivalence.sh`, `scripts/slurm/kd_memory_probe.sh` (`PROBES='t1p7b_L4096 t1p7b_L8192'` for the 1.7B rerun; `GRAD_CKPT=true` for the checkpointed probe over all four shapes). Configs: `configs/model/{ref_hybrid_m3,ref_transformer,hybrid_legal_base,transformer_legal_base}.yaml`.
