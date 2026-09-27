# Hybrid Mamba–xLSTM Codebase Review and DACH Legal AI Blueprint

Sep 27, 2026 · @Krishan

## Reading status

Both branches were cloned in full from `krishankb-de/hybrid_model_mamba_xlstm` on 2026-09-27 and read end to end: `h100_efficiency` (HEAD `900acee`, "E6 complete: the optimised configuration produces byte-identical reports. PLAN COMPLETE.") and `h100_mamba3_v2` (HEAD `78ff48d`, "V5-E: doubling the budget is worse"). Every Python module under `hybrid_xmamba/` (≈8,700 lines), every model/dataset/trainer/distill yaml under `configs/`, the training and evaluation scripts, the tests, and every Markdown file in `analysis/` on both branches were read.

| What | `h100_efficiency` | `h100_mamba3_v2` |
| --- | --- | --- |
| `analysis/*.md` files | 10 (adds `EFFICIENCY_NOTE.md`) | 9 |
| Root plan documents | `CLAUDE.md`, `README.md`, `H100_SCALING_PLAN.md` (279 KB), `MAMBA3_PLAN_V2.md` (101 KB), `EFFICIENCY_PLAN.md` (43 KB), `HYBRID_ARCH_REFACTOR_PLAN.md`, `BIOMEDCLIP_KD_PLAN.md`, `JOINT_TRAINING_PLAN.md`, `GUIDE_70M_TRAINING.md` | same minus `EFFICIENCY_PLAN.md` |
| State files | `efficiency_state.json`, `mamba3_v2_state.json`, `h100_scaling_state.json` (456 KB), 3 older | same minus `efficiency_state.json` |
| Files that differ between branches | 23, all efficiency-plan additions (`EFFICIENCY_PLAN.md`, `analysis/EFFICIENCY_NOTE.md`, `efficiency_state.json`, `scripts/check_operator_equivalence.py`, four `profile_*_h100.sh` / `verify_*_h100.sh` wrappers, `performance_profile.py`, `mamba3_state.py`), plus the E5-C cleanup (`test_triton_fix.py`, `triton` requirement, and the "custom CUDA/Triton kernels" wording removed) |  |

Not readable and therefore not covered: the three PDFs in `Research_docs/` (binary, external analyses, not code), and anything that lives only on the HPI cluster (`outputs/`, `results/`, the MIMIC parquets, SLURM logs beyond the three in `hpi_results_logs/`). `analysis/mamba3_results.md` §6 differs between the branches: the efficiency branch replaces the "Transformer is faster at every length" paragraph with the compiled-chunk-128 result; where they disagree, the efficiency branch is the newer record.

## Repository map

`h100_efficiency` is `h100_mamba3_v2` plus one closed plan (`EFFICIENCY_PLAN.md`, phases E0–E6, opened and completed 2026-09-25); everything below applies to both unless marked. The repo is a PyTorch Lightning + Hydra research codebase; every published number was produced by SLURM jobs on the HPI aisc H100 cluster (`--partition=aisc-batch`, login node executes nothing scripted).

| Path | Role | Read this for |
| --- | --- | --- |
| `CLAUDE.md` | Session bootstrap; which plan is active, what is closed, corrections to older claims | The authoritative one-page state of the project |
| `MAMBA3_PLAN_V2.md` + `mamba3_v2_state.json` | Active plan-of-record (M0–M7 merged, V0–V4 complete, V5 3/5) | Defect, SSD migration, screen, full pipeline, V5 diagnostics |
| `EFFICIENCY_PLAN.md` + `efficiency_state.json` (efficiency branch only) | Second plan-of-record, inference speed only, COMPLETE | How 4.14× slower became 1.34× faster than FlashAttention |
| `H100_SCALING_PLAN.md` + `h100_scaling_state.json` | Closed baseline record, Phases 1–15 (retrieval chapter, report generation, supervisor reviews) | Every incumbent number the Mamba-3 arms are compared to |
| `HYBRID_ARCH_REFACTOR_PLAN.md`, `BIOMEDCLIP_KD_PLAN.md`, `JOINT_TRAINING_PLAN.md`, `GUIDE_70M_TRAINING.md` | Historical (70M A100 era) | HybridNorm, gate soft-cap, cu\_seqlens resets, WSD origin |
| `analysis/` | The record: `mamba3_results.md`, `EFFICIENCY_NOTE.md`, `PHASE14_SUPERVISOR_REVIEW.md`, `h100_scaling_results.md`, `PUBLISHED_BASELINES.md`, `PHASE15C_AUX_LOSS.md`, `phase14_results.md` (70M refactor), `baseline_probe_report.md`, `scan_error_bound.md`, `ARCHIVE_MANIFEST.md` | All quantitative results (sections below) |
| `hybrid_xmamba/models/configuration_hybrid.py` | `HybridConfig` dataclass, all architecture flags, `from_hydra` field filtering | Every knob the model has |
| `hybrid_xmamba/models/hybrid_lm.py` | `HybridLanguageModel`, `HybridTextEncoder`, `AttentionPooling`, O(1) cached decode, cached beam search | Model, decode paths, architecture fingerprint |
| `hybrid_xmamba/layers/hybrid_block.py` | `HybridBlock` factory: pre-norm → mixer → residual → MLP; HybridNorm post-norm FFN; capability-based `cu_seqlens` dispatch | Block topology |
| `hybrid_xmamba/layers/mamba3_block.py` + `kernels/ssd/` | `Mamba3Block` (Mamba-2 SSD with flags), `ssd_chunked_scan`, `ssd_step`, fp64 oracle | The live backbone |
| `hybrid_xmamba/layers/mamba_block.py` + `kernels/selective_scan/scan_interface.py` | Legacy Mamba-1 block; `legacy` (defective) and `exact` scans, fp64 reference | The incumbent and the defect |
| `hybrid_xmamba/layers/mlstm_block.py` + `kernels/tfla/tfla_interface.py` | mLSTM with exp gating, tanh soft-cap; chunk-parallel TFLA (`legacy`/`exact`) | The 3 mLSTM layers |
| `hybrid_xmamba/layers/attention_block.py` | RoPE causal attention mixer with doc-boundary mask (Phase 14A baseline) | The matched Transformer |
| `hybrid_xmamba/layers/slstm_block.py`, `rotary.py`, `normalization.py`, `activations.py` | sLSTM (unused in any shipped model), data-dependent RoPE for Mamba-3, RMSNorm | Support code |
| `hybrid_xmamba/models/prefix_mapper.py`, `vision_hybrid.py` | `ImagePrefixMapper` (k=32 prefix tokens from ViT patch grid); unused vision backbone | Medical-only conditioning, to be removed |
| `hybrid_xmamba/training/` | `lightning_module.py` (LM, contrastive, distill, joint, MQAR, report-gen modules), `optimizer.py`, `schedulers.py` (WSD), `metrics.py`, `moco_queue.py`, callbacks | Training loops |
| `configs/model/` | `hybrid_150m_m3`, `hybrid_150m_m3_rrg`, `hybrid_150m_v2`, `hybrid_150m_a1`, `transformer_150m_baseline(_rrg)`, 70M/350M/7B and pure baselines | Layer patterns, param-matched configs |
| `configs/dataset/`, `configs/distill/`, `configs/trainer/` | pubmed, wikitext, c4, fineweb, mqar, mimic/indiana; BioMedLM/PubMedBERT/BiomedCLIP KD; a100/h100/ddp/fsdp trainers | Data and hardware recipes |
| `scripts/` | `train_stage0_distill.py`, `train_report_generation.py`, `train_contrastive.py`, `evaluate_*.py`, `performance_profile.py`, `check_operator_equivalence.py`, `mamba3_arms.py`, `mamba3_state.py`, `bootstrap_compare.py`, `repair_generations.py`, \~40 SLURM wrappers | Reproduction |
| `tests/` | 326 tests: `test_mamba3_numerics.py` (69, fp64 oracles), `test_willi_parity.py` (217, config invariants), `test_scan_correctness.py` (8), layers/models/kernels | What the repo guarantees about itself |

Cluster-only artefacts (not in git): `outputs/` (\~83 GB; four Stage-0 backbones, the image tower, nine report-gen decoders), `results/` (hypothesis and reference dumps), `/sc/home/$USER/dataset/mimic_full/`. The four Stage-0 backbones (`h100_stage0_150m_m3` 11.674, `m8_stage0_A2_150m` 12.550, `h100_stage0_transformer_150m` 11.222, `h100_stage0_150m_v2` 13.18, 211 H100-hours in total) are PubMed-only, no patient data, and are the assets that carry over to legal work.

## Architecture as implemented

The live model, `hybrid_150m_m3` (184,192,200 parameters), is a 12-layer, 768-wide causal decoder with no attention layer: nine `Mamba3Block`s (with every Mamba-3 flag off, which is exactly Mamba-2 SSD) around three `mLSTMBlock`s at positions 5–7. It reads GPT-2 BPE, has no positional embedding, was trained at 1,024 positions, and runs at any length. Every mixer type is a drop-in `HybridBlock` mixer selected by `layer_pattern`, so the parameter-matched Transformer is the same model with `layer_pattern: ["attention"]`.

&#91;embedded content: hybrid\_150m\_m3 · 12-layer stack and one HybridBlock\]

The stack is one pass left to right; each block is pre-norm → mixer → residual, then a post-norm FFN (HybridNorm), and the same topology is mirrored exactly in the O(1) `step()` decode path.

| Setting | `hybrid_150m_m3` (live) | `hybrid_150m_v2` (incumbent) | `transformer_150m_baseline` |
| --- | --- | --- | --- |
| Parameters | 184,192,200 | 183,721,824 | 183,386,880 |
| Layers / dim / heads / MLP | 12 / 768 / 12 / ×4 GELU | 12 / 768 / 12 / ×4 | 15 / 768 / 12 / ×4 |
| `layer_pattern` | 4×mamba3, 3×mlstm, 5×mamba3 | 4×mamba, 3×mlstm, 5×mamba | 15×attention |
| Mixer params per SSM layer | 3,824,712 | 3,772,448 | 4·768² qkv+out |
| Recurrent state per layer | 24×64×128 = 196,608 (SSM) | 1,536×16 = 24,576 | KV cache grows with L |
| mLSTM state per layer | C 12×64×64 = 49,152 + n 768 | same | — |
| Norm topology | `hybrid` (RMSNorm on B and C; FFN post-norm) | `hybrid` (+ a Δ RMSNorm that erased the dt init) | `pre_rms`, RoPE θ=10,000 |
| Operators | SSD (division-free); `tfla_impl: exact` in `_rrg` and via `ARM=A2x` | `scan_impl: legacy`, `tfla_impl: legacy` (both defective) | `F.scaled_dot_product_attention` (FlashAttention) |
| Doc-boundary handling | boolean masks inside the scan; conv re-run on first k−1 positions | Python loop per (row, segment) | additive −inf mask |
| Decode | O(1) cache: `ssm_state`, conv window, `C`, `n` (\~7.1 MB fp32 at batch 1); cached beam search | none | no KV cache in this repo |
| Regularisation / init | dropout 0.1, normal 0.02, grad clip 1.0 (0.5 for report-gen) | same | same |

Mamba-3 flags on the live arm: `d_state 128`, `head_dim 64`, `ngroups 1`, `chunk_size 64` (128 at inference, verified equivalent), conv on (size 4), `dt ~ logU[1e-3, 1e-1]` with `dt_limit 1.0`, `a_mode static` (A \~ U\[1,16\] per head, log-space), trapezoid off, RoPE off, `bc_bias none`, `mimo_rank 1`, no out-projection norm. The `in_proj` is sized for every flag at once (2·1536 + 2·128 + 3·24 + 32 = 3,656 outputs), so the ablation arms A2–A6 are parameter-matched to within 0.014%.

What is and is not on the execution path:

- No hand-written Triton or CUDA kernel exists or ever ran (`TRITON_AVAILABLE = False`; `scan_triton.py`/`tfla_triton.py` deleted in `37f7964`). Every operator is pure PyTorch; `torch.compile` is an opt-in inference path under which Inductor emits its own Triton.
- The `legacy` selective scan and `legacy` TFLA divide by a clamped cumulative decay and annihilate a token's own contribution where the clamp fires (rel-max-err 0.92 at Δ=0.705, 0.88 for TFLA at the shipped forget-gate init). `exact` variants fix both; defaults stay `legacy` for byte-reproduction and every yaml pins the value (`test_every_recurrent_model_yaml_pins_the_operator_explicitly`).
- `mLSTMBlock._slow_forward` (`use_tfla=False`) computes a different function from TFLA (0.42 max abs at L=24): LSE-stabilised vs unstabilised with a signed clamped denominator. Only TFLA was ever trained; the `step()` cache reproduces TFLA, not the slow path.
- `sLSTMBlock` is implemented but in no shipped config; it has no `step()` and no `cu_seqlens` handling.
- `AttentionBlock` has no KV cache; the Transformer's uncached decode number in §6 of `mamba3_results.md` is flagged unverified.
- Cached decode prefill is token by token, so time-to-first-token is 4.0× slower than the chunked forward (1.79 s vs 0.45 s at prompt 256); exposing the carried state from `ssd_chunked_scan` is recorded follow-up.
- Retrieval-era code (`HybridTextEncoder`, `AttentionPooling`, `bidirectional_encode`, SimCSE/CLIP/MoCo/KD modules) and medical conditioning (`ImagePrefixMapper`, `ReportGenerationLightningModule`, CheXbert scoring) are live in the repo but irrelevant to a text-only legal decoder.

## Training and distillation pipeline

Every published backbone came from one recipe, `scripts/train_stage0_distill.py` on one H100: \~3B PubMed tokens, 120,000 steps, a frozen 2.7B BioMedLM teacher, and a hidden-state distillation loss that is weak by construction (mean-pooled cosine, not logits). The medical task then fine-tuned that backbone as an image-conditioned decoder; only the Stage-0 half transfers to legal work.

| Stage | Data | Objective | Recipe | Cost (H100) |
| --- | --- | --- | --- | --- |
| Stage-0 backbone (`train_stage0_distill.py`, `train_stage0_150m_h100.sh`) | HF `pubmed23n` article text, GPT-2 tokenizer, packed to 512-token rows with per-position doc ids (`cu_seqlens`), 2,000 held-out validation samples | L = 0.5·CE + 0.5·(1 − cos(proj(mean-pool student h), mean-pool teacher h)); teacher `stanford-crfm/BioMedLM` (2.7B, bf16, own 28,895-token BPE, so logit KD is impossible); projection 768→2560; NaN guard skips up to 50 batches | batch 16 × accum 3 = 48 rows (24,576 tokens/step), 120,000 steps ≈ 2.95B tokens; LR 4e-4 (√-width-scaled from the 70M's 6e-4), warmup 2,000, weight decay 0.1, grad clip 0.5, bf16-mixed, gradient checkpointing on (80 GB), `compile_model=false`, WSD schedule (warmup → 85% stable → 14% decay by 1 − √p, min ratio floor) | Mamba-3 57 h 23 m; incumbent v2 74 h; Transformer 23 h; screen arms 12,000 steps in 1 h 21 m (A2) to 5 h 45 m (A2x, exact TFLA ≈ 20% slower); A1 (exact Mamba-1 scan) 4 h 11 m, needs a 24 h wall |
| Screen (`screen_arms_h100.sh`, `mamba3_arms.py`) | same corpus, same data order | same loss | 12,000 steps, warmup 500, seeds 42 and 1234, `SAVE_TOP_K=0`; pre-registered bar 0.642 PPL = 2× the two-run seed SD | one arm per array task |
| Report-generation decoder (`train_report_generation.py`, `ReportGenerationLightningModule`) — medical only | MIMIC-CXR-JPG 191,462 / 1,433 / 2,663 frontal pairs at 320 px, target "Findings: … Impression: …" | causal CE over report tokens; `ImagePrefixMapper` prepends k=32 tokens from the BiomedCLIP ViT-B/16 patch grid; optional aux CheXpert head (λ=0, negative result) | decoder\_lr 1e-5, head\_lr 3e-4, wd 0.01, warmup 500, 12,000 steps, 4×H100 DDP, clip 0.5, seeds 42/43/44; pad = EOS and masked, so no stop token is ever supervised | \~1 h 19 m (Mamba-3) / 2 h 15 m–2 h 30 m (hybrid v2) / 26–40 min (Transformer) |
| Contrastive / retrieval stages (`train_contrastive.py`, `HybridContrastiveLightningModule`, `DistillContrastive…`, `JointMultiTask…`) — closed chapter | PubMed (SimCSE), MIMIC pairs (CLIP/SigLIP), BiomedCLIP KD, MoCo queue | InfoNCE / SigLIP / cosine-KD with attention pooling | canonical recipe frozen 2026-07-27 | historical |

Evaluation and reproduction machinery that is worth keeping:

- `scripts/evaluate_lm.py` (perplexity, bits-per-byte, throughput, peak memory), `evaluate_report_generation.py` (uncached `beam_search_decode`, beam 3, fixed 100 tokens, no EOS handling; `--scan-impl`, `--tfla-impl`, `--chunk-size` overrides announced in the log and refused for configs without the matching layer), `bootstrap_compare.py` (paired bootstrap, 1,000 resamples, per-label CIs), `analyze_generation_diversity.py` (duplicate clusters, distinct-n, self-BLEU), `repair_generations.py`.
- `scripts/performance_profile.py`: `--sweep`, `--backward`, `--per-layer`, `--attn-backend {auto,math,flash,efficient}`, `--chunk-size`, `--compile`; log-log exponent fits; prints the chunk size read back off the built module.
- `scripts/check_operator_equivalence.py`: fp64-oracle gate (operator, packed-document boundary inside a chunk, model logits across `chunk_size` and `torch.compile`; tolerance 1e-4).
- `scripts/mamba3_state.py tick|phase|note|readme|show|sync` with `--plan {mamba3,efficiency}`: plan checkboxes and JSON state kept in lockstep.
- `scripts/validate.sh`: Hydra invariants, `pytest -m "not cuda and not slow"`, CPU forward/backward over all five mixer types asserting every parameter receives a gradient. No CI exists; this is the only gate.
- Every model logs an `ARCH …` fingerprint at construction (layer counts, norm topology, `scan_impl`, `tfla_impl`, dt init, all Mamba-3 flags, parameter count); `run_metadata.json` beside each checkpoint records `prefix_k` and the config, and a >50%-missing-keys guard hard-fails mismatched loads.

What this pipeline does not have, and a legal build needs: a tokenizer for German, a logit-level teacher, an end-of-sequence token in the targets and a stop condition in decoding, sequence lengths beyond 512 in training (all Stage-0 rows are 512; `max_position_embeddings` 1,024), any instruction or preference tuning, and any retrieval-augmented input format.

## Quantitative results: efficiency (`h100_efficiency`)

With two inference-time settings that change no decoded token, the corrected Mamba-3 model is 1.15× faster than the parameter-matched FlashAttention Transformer at 8,192 tokens and 1.34× at 16,384, at memory parity (7.169 vs 7.152 GB); training remains 2.71× slower and 1.59× heavier at 2,048. All numbers: H100 80 GB, batch 4, bf16, random weights and token ids (throughput and memory do not depend on weights), one sequence length per process, one Inductor cache per point (jobs 2579631, 2579642, 2580198, 2582482, 2582775, 2583455).

**Inference, confirmed protocol (job 2582775), median ms / peak GB**

| L | Transformer (FlashAttention) | Mamba-3 baseline, chunk 64 | chunk 128 | compiled, chunk 64 | compiled, chunk 128 | Ours vs Transformer |
| --- | --- | --- | --- | --- | --- | --- |
| 8,192 | 71.26 / 3.776 | 269.99 | 195.27 | 70.48 | **62.09 / 3.824** | 1.15× faster |
| 16,384 | 164.85 / 7.152 | 562.22 / 7.090 | 402.96 | 140.35 | **122.90 / 7.169** | 1.34× faster |

Latency exponents 8,192→16,384: ours 0.985, attention 1.210 (baseline 1.058, chunk-128 1.045, compiled 0.994). Compile lowers the inference memory exponent to 0.544 from 0.643. The 122.90 ms headline was reproduced three times (122.90 / 123.03 / 122.94, 0.1% spread). The interim exponent of 0.47 was an artefact of a contaminated 8,192 point and is withdrawn.

**Inference, uncompiled default (job 2561023, Phase 14A-7 protocol)**

| L | Incumbent hybrid v2 | Mamba-3 (chunk 64) | Transformer |
| --- | --- | --- | --- |
| 256 | 17.6 ms / 1.22 GB | 20.6 / 0.51 | 4.9 / 0.51 |
| 2,048 | 120.6 / 5.71 | 86.2 / 1.24 | 16.1 / 1.24 |
| 16,384 | 915.6 / 42.00 | 678.2 / 7.09 | 163.9 / 7.15 |

Memory exponent: Mamba-3 0.643, Transformer 0.644, incumbent 0.863 (5.9× more memory at 16,384). Latency exponent uncompiled: 0.840 vs 0.844, i.e. 4.14× slower with no crossover before compile.

**Where the uncompiled time went at 16,384 (E0-A/B, 555 ms forward)**: 9 Mamba-3 layers 77.8%, of which `ssd_chunked_scan` 67.5% of the whole forward (374.9 ms); 3 mLSTM layers 13.4%; embedding, head, norms 8.8%. Amdahl bound if the SSD path were free: 4.50× (2.93× at 2,048). The incumbent spends 86.6% in its Mamba-1 layers. The cause was a Python `for` loop over 256 chunks × 9 layers = 2,304 sequential iterations per forward.

**Chunk-size sweep, uncompiled (E0-C): forward ms at L=4,096 / L=16,384 / scan-only at 16,384**: 64 → 135.6 / 557.8 / 377.0; **128 → 99.4 / 398.6 / 216.7**; 256 → 104.6 / 411.8 / 230.1; 512 → 130.7 / 512.8 / 331.6. A U-curve, not the predicted monotone 1.5–2.5×. Under `torch.compile`, 256 and 512 fail to build (Inductor `SplitScan` cumsum codegen `TypeError`), so 128 is a compiler ceiling, not a tuning choice.

**Fused vs unfused attention (E0-D, same Transformer, SDPA backend `auto` vs `math`)**: 2,048: 16.12 / 1.24 GB vs 81.93 / 2.43 (5.1×); 4,096: 33.32 / 2.09 vs 298.79 / 8.11 (9.0×); 8,192: 71.34 / 3.78 vs 1,195.95 / 30.45 (16.8×); 16,384: 163.86 / 7.15 vs out of memory. Exponents 0.871 vs 1.513 latency, 0.644 vs 1.201 memory. Against unfused attention the uncompiled Mamba-3 was already \~3× faster at 8,192 and ran at 16,384 where attention could not: the deficit was FlashAttention's kernel engineering, not the algorithm.

**Training step (forward + backward), median ms**

| L | Transformer | baseline | chunk 128 | compiled | compiled + chunk 128 |
| --- | --- | --- | --- | --- | --- |
| 1,024 | 29.65 / 3.99 GB | 259.05 (8.74×) | 121.84 (4.11×) | 56.62 (1.91×) | **36.35 (1.23×)** |
| 2,048 | 52.17 / 7.55 GB | 305.50 (5.86×) | 209.98 (4.02×) | 264.39 (5.07×) | **141.14 (2.71×)**; 12.02 GB (1.59×) |

Chunk 128 alone is a steady \~1.45× on training at +5–7% memory (partial table, job 2582482: L=512 101.43 → 77.02 ms; 1,024 168.56 → 119.47; 2,048 310.06 → 210.04; 4,096 543.40 → 371.46; 8,192 1,337.35 → 884.13). Uncompiled memory: 13.83 GB at 2,048, 54.4 GB at 8,192. Against the incumbent at 2,048 the corrected model trains 2.6× faster in 3.9× less memory (416 ms / 13.9 GB vs 1,096 / 54.0); the incumbent is out of memory at L ≥ 4,096, Mamba-3 trains to 8,192; training latency exponent 0.886 vs 1.285. The project's older "8× slower training at 2,048" figure does not reproduce (measured 5.86×) and must not be re-quoted.

**Decode, prompt 256, 64 new tokens, batch 1**: full recompute 0.03293 s/token (30.4 tok/s, growth 0.96×); cached O(1) 0.00635 s/token (157.6 tok/s, growth 1.00×) = 5.19× per token and flat in context. Time-to-first-token is 4.0× worse (1.79 s vs 0.45 s) because prefill steps token by token. The Transformer's uncached 0.337 s/token is excluded as not credible. The incumbent has no cache at all: its legacy TFLA computes no recurrence an O(1) step could reproduce.

**Equivalence gates (rule R1)**: operator vs fp64 oracle 2.1e-07; logits across chunk size 2.3e-05–3.4e-05; `torch.compile` logits 3.0e-05; tolerance 1e-4; packed-document boundary inside a chunk passes at every chunk size. Decoding 400 test studies at chunk 64 vs 128 (job 2583455): 0 of 400 reports differ; ROUGE-L 0.1874066, BLEU-1 0.2442838, BLEU-4 0.0518528 identical to every digit. This verifies `chunk_size`; `torch.compile`'s effect on decoded text is inferred by analogy (same-order logit perturbation), not measured.

**Earlier sweeps kept for the record**: Phase 14A-7 (hybrid v2 vs Transformer): L=256 24.72 ms / 1.22 GB vs 5.03 / 0.51; L=16,384 919 / 42.0 vs 164 / 7.15; training L=2,048 1,084 / 53.96 vs 52.6 / 7.56, L=4,096 OOM vs 107 / 14.7; the real decoder trained at 1.51 vs 7.96 it/s (5.3×). Retrieval-era 150M sweep (2026-07-28, L=16,384): pure Mamba-1 1,105 ms / 42.0 GB / 59,280 tok/s; hybrid 925 / 42.0 / 70,882; pure xLSTM 355 / 7.1 / 184,860; training L=2,048 Mamba 1,348 ms / 67.5 GB, hybrid 1,078 / 54.0, xLSTM 309 / 11.2; at L=256 hybrid 18.08 ms vs 19.78 (Mamba) / 21.12 (xLSTM). Note the sign: pure xLSTM was the fast, light one; the Mamba-1 layers carried the cost.

**Process finding**: a shared Inductor cache handed two arms one kernel and timed them identically to the microsecond; within one process, shapes compiled later measured worse (3.04× at 512 → 1.02× at 4,096). Every compiled number depends on what was compiled beside it unless measured one shape per process.

## Quantitative results: Mamba-3 backbone (`h100_mamba3_v2`)

Repairing the two recurrences is worth more than any architecture change measured: −16.0% validation perplexity at 12,000 steps on the unchanged Mamba-1 model, and 11.674 vs 13.18 at 120,000 steps for the corrected Mamba-3 backbone (77% of the gap to the matched Transformer's 11.222). That backbone gain does not reach the downstream task: across three seeds the Mamba-3 decoder is indistinguishable from the Transformer on all ten report-generation metrics and from the incumbent on eight of ten.

**The defect (fp64 sequential references; `tests/test_mamba3_numerics.py`, `tests/test_scan_correctness.py`)**

| Operator | Configuration | rel-max-err |
| --- | --- | --- |
| selective scan, chunk 64 | Δ = 0.705, this repo's init | 0.92 (1.031 in the 14C table; 92% of entries clamped) |
| selective scan, chunk 64 | Δ = 0.1, `dt_max` of the correct init | 0.39 (0.281 in 14C; 54% clamped) |
| selective scan | Δ ≤ 0.01, any chunk | 3.6e-08 to 6.7e-08, exact |
| mLSTM TFLA, chunk 64 | shipped `forget_gate_bias_init 0.0` | 0.88 (70.9% of entries hit the clamp) |
| Δ at init | `hybrid_150m_v2` vs reference logU\[1e-3, 1e-1\] | 0.807 (mean 0.8229 under `hybrid`) vs \~0.021 |

**Screen, 12,000 steps, PubMed + BioMedLM, seed 42 (and 1234), bar 0.642 PPL**

| Arm | Operator | Val PPL s42 / s1234 | Wall | Rate |
| --- | --- | --- | --- | --- |
| A0 | Mamba-1, both recurrences defective | 19.387 / 18.933 | 7:59:00 (2:36:50 bar column) | 1.36 it/s |
| A1 | Mamba-1, both corrected + Δ init + no Δ-norm | 16.294 | 4:11:13 | — |
| A2 | Mamba-3 SSD, d\_state 128, mLSTM still defective | 16.708 / 16.376 | 4:09:14 (1:20:55 bar column) | 2.18 it/s |
| A3 | A2 + exponential-trapezoidal | 16.719 | 1:28:00 | — |
| A4-lo / mid / hi | A2 + RoPE, theta\_max 0.002 / 0.02 / 0.2 | 16.431 / 16.534 / 16.199 (hi at s1234: 18.912) | \~1:30 | — |
| A4 / A5 / A6 | RoPE at theta\_max 1.0 (81 turns over 512 tokens) | 1166.7, collapsed | — | — |
| A2x | A2 + `tfla_impl exact` | 15.566 / 15.788 | 5:45:19 / 5:57 | 1.75 it/s |

Trapezoid: null (+0.011, 1.7% of the bar). RoPE: high-variance (cross-seed spread 2.713 vs A2's 0.332, 8.2×), rejected by the pre-registered rule; the parity-task capability claim reproduced only with Δ free to reach \~1 (rope off 62–64%, on 100%); under the reference dt init the reachable rotation is \~0.06 rad and the mechanism is dormant. A2x beats A2 by 0.865 on the two-seed mean and A1 by 0.728 at seed 42 (provisional, one A1 seed). Exact TFLA costs \~20% throughput vs A2.

**Stage-0, 120,000 steps, one seed per arm (job 2553736)**

| Backbone | Val PPL | vs incumbent |
| --- | --- | --- |
| `hybrid_150m_v2`, both defective | 13.180 | — |
| A2 (`m8_stage0_A2_150m`), SSD + 8× state, mLSTM defective | 12.550 | −0.630 (−4.8%) |
| `hybrid_150m_m3` (A2x), both corrected | 11.674 | −1.506 (−11.4%) |
| `transformer_150m_baseline` | 11.222 | −1.958 (−14.9%) |

Decomposition at 120K: scan repair + SSD + 8× state −0.630 (42%), mLSTM repair −0.876 (58%). At 12K the same split is −2.618 / −0.865 (75% / 25%): the two halves swap rank with training length, so quote both. The A2 row was trained from a dirty tree at `7fecc86`; no Stage-0 arm has a seed band, and the residual 0.452 is the size of the screen's seed spread (0.33–0.45), so "essentially matches" is supportable and "beats" is not.

**Report generation, official MIMIC-CXR test split n=2,663, beam 3, `prefix_k` 32, 13D image tower reused, decoder seeds 42/43/44 paired**

| Metric | Mamba-3 | Hybrid v2 | Transformer | vs hybrid | vs Transformer |
| --- | --- | --- | --- | --- | --- |
| ROUGE-L | .1953 ± .0029 | .1949 ± .0047 | .1952 ± .0021 | tie | tie |
| BLEU-1 | .2484 ± .0005 | .2508 ± .0034 | .2478 ± .0022 | tie | tie |
| BLEU-4 | .0579 ± .0009 | .0578 ± .0032 | .0575 ± .0016 | tie | tie |
| CheXbert-14-micro | .4480 ± .0188 | .4480 ± .0223 | .4443 ± .0153 | tie | tie |
| CheXbert-14-macro | .2715 ± .0121 | .2660 ± .0122 | .2692 ± .0106 | tie | tie |
| CheXbert-5-micro | .5044 ± .0257 | .5086 ± .0382 | .5032 ± .0226 | tie | tie |
| CheXbert-5-macro | .4170 ± .0164 | .4193 ± .0274 | .4170 ± .0165 | tie | tie |
| exact-match-14 | .0452 ± .0028 | .0380 ± .0058 | .0455 ± .0027 | +.0071 | tie |
| exact-match-5 | .2242 ± .0070 | .2163 ± .0019 | .2244 ± .0057 | +.0079 | tie |
| example-F1 | .3858 ± .0174 | .3790 ± .0214 | .3817 ± .0144 | tie | tie |

Claim rule (decision 10): paired mean > one baseline seed SD and sign at ≥ 2/3 seeds. Of 27 per-seed interval calls against the Transformer exactly one excluded zero (BLEU-1 at seed 44). Per-seed calls contradict each other: seed 42 Mamba-3 wins BLEU-4 and exact-match-14, hybrid wins CheXbert-5-micro; seed 43 Mamba-3 wins CheXbert-14-micro/14-macro/5-micro, hybrid wins BLEU-1; seed 44 Mamba-3 wins exact-match-14, hybrid wins BLEU-1. Against the retrieval-NN floor (ROUGE-L 0.1636, CheXbert-14-micro 0.4296, 14-macro 0.3014, example-F1 0.3691): text metrics and exact-match win 3/3, CheXbert-14-macro loses 3/3 (−.0180 / −.0294 / −.0422); Lung Lesion .000/.022/.000 and Pleural Other .033/.017/.000. Templating (job 2560259): 26.0% of generations in exact-duplicate clusters vs incumbent 29.2%, references 0.2%, floor 7.3%; distinct-2 .0350 vs .0299; self-BLEU-4 .6709 vs .6854.

**V5-A, operator sensitivity of the published 13D checkpoint (n=400, beam 3, exact vs legacy operators, same weights)**: ROUGE-L 0.1839 vs 0.1836 \[−0.0022, +0.0031\]; BLEU-1 0.2454 vs 0.2444; BLEU-4 0.0504 vs 0.0503; CheXbert-14-micro 0.4523 vs 0.4566 \[−0.0166, +0.0090\]; 14-macro 0.2613 vs 0.2589; 5-micro 0.5352 vs 0.5396; 5-macro 0.4286 vs 0.4322; exact-match-5 0.1950 vs 0.2025; exact-match-14 0.0375 vs 0.0250 \[+0.0025, +0.0250\] (15 vs 10 reports). Eight of nine tie; **227 of 400 reports (57%) decode to different text**, some to a different template.

**V5-D/E, the missing stop condition (13D, n=400)**: 301 of 400 reports end mid-sentence; only 61 duplicated sentences. Post-hoc repair (drop severed fragment, collapse repeats; 321 reports changed, tokens 23,325 → 21,298, mean 58.3 → 53.2): BLEU-1 0.2271 vs 0.2444 \[−0.0196, −0.0149\], BLEU-4 0.0473 vs 0.0503, ROUGE-L 0.1834 vs 0.1836 (tie), CheXbert all tie; brevity penalty 0.7124 vs 0.8003 while clipped unigram precision rose 0.3054 → 0.3188. Doubling the budget to 200 tokens: ROUGE-L 0.1644, BLEU-1 0.1983, BLEU-4 0.0394; mean length 114.1; 346 consecutive duplicate sentences; 17.9% of 5,451 sentences repeat; 71% still cut mid-sentence. Verdict: 100 tokens is at the metric optimum and the only real fix is a trained EOS.

**Historical numbers the record still cites**: 13D single seed ROUGE-L 0.1899 / CheXbert-14-micro 0.4736 / 14-macro 0.2800 / 5-micro 0.5522 / 5-macro 0.4487 / exact-match-14 0.0349 (the high CheXbert draw); Phase 14A seed-42 split decision (hybrid +0.0146 14-micro \[+0.0052, +0.0240\], Transformer ROUGE-L −0.0038 \[−0.0066, −0.0010\]) overturned by seeds 43/44; prefix sweep k=8/32/64 CheXbert-14-micro 0.4481 / 0.4736 / 0.4488 (Transformer 0.4590 / 0.4637, indifferent); 15C aux loss λ=0.1 / 0.5 validate macro 0.2853 / 0.2712 vs 0.2869 (negative); 13F oversampling negative at weights 5.0 and 2.0; 70M refactor MIMIC i2t R@10 8.23% → 10.45%; H100 retrieval 17.14% protocol-matched, 14.59% clean, stock BiomedCLIP 3.40%, Indiana flat at 3.90%; published field on example-F1 puts the project at 0.3790 ± 0.0214, between M2TR 0.308 and MKSG 0.371, level with CvT2DistilGPT2 0.384, below PromptMRG 0.476; on macro-F1 0.2660 vs PromptMRG 0.381 (\~0.2B) and the 7B–84B tier at 0.386–0.417.

## Qualitative findings, honesty constraints and known defects

The project's own record licenses a narrow set of claims and forbids a longer one; the legal work inherits both lists.

Licensed by the record:

- The shipped scan and TFLA did not compute their specified recurrences; the deviation is bounded, regression-tested, and changes what the model writes (57% of reports) without moving any published metric.
- Repairing them is worth −16.0% val PPL at 12K on the existing architecture; the corrected Mamba-3 backbone reaches 11.674 vs 13.18 at 120K.
- At matched parameters the corrected attention-free model is statistically indistinguishable from a matched Transformer on every downstream metric across three seeds.
- It matches FlashAttention's inference memory at every length and, compiled at chunk 128, is faster than FlashAttention at 8,192 and 16,384 tokens; it trains at lengths the incumbent cannot (to 8,192).
- The O(1) recurrent cache decodes 5.19× faster per token with flat cost in context.
- The conditioning interface (prefix length k=8→32, +0.0255 CheXbert-14-micro) mattered more than the sequence mixer (every hybrid–Transformer difference at matched k was smaller).

Not licensed:

- "Mamba-3 improves the downstream task"; "matches the Transformer on language modelling" (one seed each, 0.452 apart, no seed band); any claim from a single seed (seed SD is 3–15× the bootstrap half-width and per-seed calls contradict each other).
- Anything about the trapezoidal rule or complex state beyond "null at this scale and context length"; MIMO is plumbed, never run.
- Training-side efficiency: 2.71× slower and 1.59× the Transformer's memory at 2,048 even compiled; the compiled path is opt-in and sits against a compiler cliff.
- "Mamba for speed, xLSTM for storage": each Mamba-3 layer holds 196,608 state values, each mLSTM layer 49,152; in the retrieval-era sweep pure xLSTM was the fast, light architecture (355 ms / 7.1 GB vs pure Mamba's 1,105 ms / 42 GB at 16,384). The defensible description is two different fixed-memory update rules, neither of which can copy exact text.

Defects and traps recorded in the repo, each of which produced a confidently wrong number before it was caught:

| Defect or trap | Effect | Guard now in place |
| --- | --- | --- |
| `strict=False` checkpoint loading with wrong `DECODER_CKPT`, `--model-config` or `prefix_k` | loads a mismatched model without error; the `prefix_k` trap cost 0.0145 ROUGE-L, 4× the whole hybrid–Transformer gap | >50%-missing-keys hard fail; `prefix_k` resolved from `run_metadata.json`; "`Missing keys: 0` is not evidence" |
| `norm_topology` silently dropped when building `HybridConfig` (Phase 9); `scan_impl`/`tfla_impl`/`dt_init_strategy` reached the yaml but not the model (job 2513007) | HybridNorm weights loaded into a pre-norm model; A1 trained with the defects still in place | `HybridConfig.from_hydra` filters against `dataclasses.fields`; unknown `mamba3_*`/`mlstm_*`/`slstm_*` kwargs raise; `ARCH` fingerprint at step 0 |
| `theta_max` missing from the dataclass | every RoPE arm ran 1 rad/token and collapsed to 1166.7 PPL | Mamba-3 kwargs derived from `Mamba3Block.__init__`'s signature |
| sLSTM dispatched on a hard-coded layer-name tuple | recurrent state leaked across packed documents | capability flag `supports_cu_seqlens` cross-checked against the forward signature |
| Gate 6 of the old harness built with `use_fast_path=False` | only the buggy duplicate scan was ever tested; old kernel tests asserted shape/NaN and were CUDA-gated | one scan implementation shared by both paths; fp64 oracles run unconditionally on CPU |
| `.gitignore` blanket `*.md` with an allowlist | `ARCHIVE_MANIFEST.md` reported as committed while untracked | `test_every_analysis_deliverable_can_actually_enter_the_repo` |
| Login node refuses scripts; the refusal text word-splits | a 12K screen arm ran as a 120K A0 at ten times the length (job 2513581) | arm resolved on the compute node; `ARM=` env |
| `save_top_k=3` + `last.ckpt` = 9.6 GB per arm on a 200 GiB quota | 3 of 4 seed arms died on `Disk quota exceeded` | `SAVE_TOP_K` lever, 0 in every arm |
| Shared Inductor cache across profiling arms | two chunk sizes timed identically to 1 µs; a 0.47 exponent was published then withdrawn | one length per process, one cache per point; chunk size read back off the module |
| E6 wrapper defaulted to the incumbent checkpoint | `mamba3_chunk_size` set and never read; a five-decimal tie that measured nothing ("TRAINED with None") | eval raises on a chunk-size override for a config with no `mamba3` layer |
| No EOS in report targets; `beam_search_decode` runs a fixed 100 tokens; `length_penalty` inert | 75% of reports cut mid-sentence; longer budgets only add repetition | not fixed; "train the model to stop" is the recorded next lever |
| Pre-Phase-4 model: mLSTM `i_gate` pre-activation 14.25 (cap 15), block-7 hidden norm 59 vs 17 at block 0, 18.4% cross-document state leak | retrieval ceiling at 8–10% | soft-cap 15, `i` bias −10, HybridNorm, `cu_seqlens` resets |

Qualitative behaviour of the generator that will recur in a legal decoder unless designed out: output is formulaic (distinct-2 \~6× below the reference corpus, self-BLEU-4 \~2.1× above), length is constant regardless of the case (58.4 / 58.2 tokens on both splits), rare labels stay at F1 ≈ 0 through oversampling and an auxiliary loss, and a retrieval-copy baseline beats every generator arm on macro-F1 because copying a real document names rare items. The README on `main` still lists "Triton ≥ 2.1 (for GPU kernels)" in its requirements and omits `attention` from the valid `layer_pattern` entries; the efficiency branch removed the package docstring's "custom CUDA/Triton kernels" claim, `test_triton_fix.py` and the `triton` requirement, but the README requirements line survives.

## Pivot from medical to legal: what carries over, what changes, what goes

The backbone, its kernels, its decode cache, its packing, its test harness and its measurement discipline carry over unchanged; the tokenizer, teacher, corpus, targets, decode loop and sequence length must change; everything image- or retrieval-specific is dead weight. Nothing trained on MIMIC-CXR may leave the cluster, but all four Stage-0 backbones are PubMed-only and free to reuse.

| Keep unchanged | Change | Remove or leave dormant |
| --- | --- | --- |
| `HybridConfig`, `HybridBlock`, `Mamba3Block` + `kernels/ssd/` (scan, `ssd_step`, fp64 oracle), `mLSTMBlock` with `tfla_impl: exact`, `AttentionBlock` with doc-boundary mask, `cu_seqlens` packing, HybridNorm | Tokenizer: GPT-2 BPE is English-centric and fragments German legal compounds; adopt the teacher's multilingual tokenizer (Qwen3's 151,669-token BBPE if Qwen3 is the teacher) and measure fertility on a legal sample before and after | `ImagePrefixMapper`, `ReportGenerationLightningModule`, `train_report_generation.py`, CheXbert scoring and venv, `mimic_cxr_vendor/`, `build_mimic_cxr_local.py`, the `_rrg` yamls, `prefix_k`, `vit_*`, `aux_*` keys |
| O(1) cached decode, `reorder_cache`, `beam_search_cached` (token-identical to the uncached path by test) | Teacher: BioMedLM's mean-pooled cosine KD cannot transfer knowledge; use an open-weight teacher with a shared tokenizer and logit-level KL (KD infrastructure exists in `configs/distill/stage0_biomedlm.yaml` as (1−α)·CE + α·T²·KL, but the script implements hidden-state cosine) | `HybridTextEncoder`, `AttentionPooling`, `bidirectional_encode`, SimCSE/CLIP/SigLIP/MoCo/BiomedCLIP-KD modules, `evaluate_retrieval.py`, `evaluate_cxr_retrieval.py`, `evaluate_sts.py`, `train_contrastive.py`, `contrastive_eval_callback.py`, `moco_queue.py` |
| WSD scheduler, fused AdamW, bf16-mixed, gradient checkpointing, NaN guard, `SignalCheckpointCallback`, `run_metadata.json`, `ARCH` fingerprint | Corpus: PubMed → German/DACH statutes and case law with general-German replay; packing at 4–8K rows instead of 512, then a long-context stage at 8–16K (hybrids do not generalise beyond their trained length) | `VisionHybridModel`, `PatchEmbedding` |
| `performance_profile.py`, `check_operator_equivalence.py`, `bootstrap_compare.py`, `analyze_generation_diversity.py`, `mamba3_state.py`, `validate.sh`, the 326 tests | Targets and decoding: append an EOS to every supervised target, stop on it, add repetition control; add pointer/citation tokens to the vocabulary; constrained decoding over pointer ids | `sLSTMBlock` (no step, no resets, in no config); `slstm_*` keys can stay for schema parity |
| `layer_pattern` mechanism: attention is already a fifth mixer type, so `[m3, m3, m3, attention, mlstm, …]` builds today | Attention layers: 2–4 global attention layers need a KV cache in `AttentionBlock.step()` (none exists) and cache reordering in beam search; prefill should expose the carried state from `ssd_chunked_scan` instead of stepping token by token | The old A100/willi paths (`a100_single_gpu.yaml` Python 3.9 guards, `validate_for_willi.sh` shim) |
| Naming and provenance discipline: pinned `scan_impl`/`tfla_impl` in every yaml, pre-registered predictions, one-lever-per-arm, seed bands | Model naming and README: retire "custom CUDA/Triton kernels" and the `triton` requirement line on `main`; state that Inductor emits Triton under opt-in compile | Legacy operators (`legacy` scan, `legacy` TFLA, `MambaBlock`) can stay for reproduction but no new legal config should use them |

One structural mismatch to resolve before the legal model is trained: the doc-boundary reset. `Mamba3Block` masks boundaries inside the scan, but `mLSTMBlock._forward_segmented` and `MambaBlock._forward_segmented` run a Python loop per (row, segment). At 512-token PubMed rows this was tolerable; at 8–16K packed legal rows with many short statute paragraphs it will dominate the step, and it is the reason `compile_model` was pinned off for Stage-0. Either mask boundaries inside TFLA the way SSD does, or pack one document per row at long context.

## Target legal system design

The defensible unique claim is the package, not the mixer: a corrected Mamba-3 + mLSTM + sparse global-attention decoder that never generates statute text, cites by pointer into a numbered retrieved context, and is gated by a verifier, evaluated on DACH law. Two literature searches found no established language model with that combination; check properly before writing "first". Frame the thesis as "matches or beats much larger general models on citation-faithful DACH legal QA at a fraction of the cost, on-prem", never "beats frontier models".

&#91;embedded content: DACH legal QA system · retriever, hybrid decoder, renderer, verifier\]

One pass left to right and down: retrieval fixes what can be quoted, the decoder only chooses and connects, and the verifier decides what the user sees.

**Uniqueness, stated at the width the evidence supports.** Mamba-plus-attention hybrids are now standard (Jamba 1:7, Nemotron-H \~8% attention, Granite 4.0 9:1, Qwen3-Next 3:1, Kimi Linear 3:1). Mamba-plus-xLSTM exists in vision (VMAXL-UNet). This project's own numbers add what those do not: a measured, seed-banded parity with a matched Transformer at 184M; the mLSTM repair worth 58% of the backbone gain at full scale (so the mixture earns its place by measurement, not by story); memory parity and a 1.34× inference win over FlashAttention at 16K; a 5.19× O(1) decode. What it cannot claim: better reasoning than any Transformer, or that any sub-1B model beats GPT-4-class systems on German legal QA overall.

**Backbone.** Keep 12 layers at 768 for the thesis model and replace two Mamba-3 positions with global attention: `M3 M3 M3 A L L L M3 M3 A M3 M3` (2 of 12 = 17%, inside the 8–25% band the shipped hybrids validate; keep attention full, not sliding-window, because emitting pointers to sentence ids anywhere in the context is in-context recall, the task pure SSMs are measurably worst at). The KV cache for two attention layers at 16K is 2 × 2 × 16,384 × 768 × 2 bytes ≈ 101 MB per sequence in bf16, trivial next to the 7.09 GB activation peak. Rebalance MLP width or one Mamba-3 layer to stay parameter-matched. Re-test `d_state` 128 vs 64 once attention is interleaved: the exact log-space fix costs 19 MB at scalar-A, so the choice is quality, not memory. Add one multi-token-prediction head (cheap, doubles as a speculative drafter for the recurrent decode). Skip MoE and MIMO. Train at the deployment length (8–16K), packed with `cu_seqlens`, after fixing the mLSTM segment loop.

**Scale and initialisation.** Two roads, both compatible with this codebase. (a) From scratch at 150–350M on 5–10B German legal + general tokens using the existing Stage-0 entry point with a logit-level teacher: the honest thesis-scale option, \~57 h per 3B tokens on one H100 uncompiled, 2–4× less with compile + chunk 128. (b) Convert a permissively licensed Transformer (Qwen3-0.6B or 1.7B, Apache 2.0) into this hybrid by MambaInLlama/MOHAWK-style distillation: initialise Mamba-3 in\_proj slices from the teacher's Q/K/V/O, keep \~25% of layers as attention, distil with KL against Qwen3-8B logits on a German-heavy mix, then DPO; this inherits 36T tokens of pretraining and the multilingual tokenizer and is the only road to "beats a frontier model on the task". Either way, never train on GPT/Claude API outputs (both providers' terms forbid it); open-weight teachers only.

**Pointer citations (rule 1: never write statute text from memory).** Every retrieved sentence carries an id `[c17.s3]`; the decoder emits `<Q c17.s3>` for a quotation and `<CITE c17>` for a citation; the renderer inserts the exact sentence and the § label (e.g. `BGB §573 Abs. 2 Nr. 1`) from the passage metadata; decoding masks pointer ids not present in the retrieved set. Quotes are exact by construction and the model never generates a § number.

**Verifier (rule 2: nothing unverified reaches the user).** Three checks per answer: every § reference resolves to a retrieved passage; every quoted span string-matches its source after Unicode/whitespace normalisation (automatic with pointers); every sentence is entailed by the passage it cites, scored by a small German NLI cross-encoder. Failing sentences are removed; a failing main claim becomes "Die bereitgestellten Quellen beantworten diese Frage nicht." The verifier's log is also the citation-precision metric and the reward for GRPO.

**Retrieval (system-level, model-agnostic).** Statutes chunked at §/Artikel level with hierarchy and version dates; case law by Tenor/Tatbestand/Entscheidungsgründe then 300–500-token windows with Randnummer anchors; hybrid BM25 + dense (BGE-M3 or jina-v3, fine-tuned on synthetic (question, §) pairs) fused by reciprocal rank fusion, cross-encoder rerank of the top 50–100, top 3–8 chunks (2–4K tokens) into the context; jurisdiction (DE/AT/CH/EU) and valid-on date filters; nightly re-sync from GII, NeuRIS, RIS, Fedlex, OLDP. Keeping the context short turns the recurrent model's task from recall into reading, which is where the O(1) state is sufficient.

**Decode path.** Cached decode already exists and is token-identical to the uncached beam search; add EOS termination, repetition control, pointer masking, a KV cache for the two attention layers, best-of-n with verifier reranking (n = 4–16 is cheap at 157 tok/s and constant memory), and prefill from the chunked scan's carried state to remove the 4× time-to-first-token penalty.

**Training objectives after the backbone.** SFT on the pointer format with 20–30% unanswerable and false-premise items; DPO/KTO on verifier-passing vs verifier-failing pairs; GRPO with the verifier as reward (valid pointers, entailed claims, correct refusal, length penalty) plus a coverage term against reward hacking; verifier-filtered synthetic QA generated by open teachers (Qwen3-32B/235B, GLM-4.5-Air, gpt-oss-120b) over real retrieved passages.

## Build plan

Six gated phases over roughly twelve weeks, run with the same discipline the medical campaign used: one lever per arm, a pre-registered bar written before the job, a seed band before any claim, and a state file ticked after every job. The backbone screen is the critical path; corpus and retrieval work run beside it.

&#91;embedded content: build plan · 6 phases, 6 gates\]

Each band feeds the next; a gate that fails stops the chain and becomes a recorded negative, not a reason to iterate until it passes.

**Compute budget (measured where marked, otherwise estimated)**

| Item | Measured basis in this repo | Estimate for the legal build |
| --- | --- | --- |
| Screen arm, 150M, 12K steps, 512-token rows, 1 H100 | 1 h 21 m (A2) to 5 h 45 m (A2x, exact TFLA); A1 4 h 11 m | 4–6 arms × 2 seeds ≈ 30–60 H100-hours; compile + chunk 128 cut a training step 2.2–7× (1,024: 259 → 36 ms; 2,048: 306 → 141 ms), untested on a full run |
| Full backbone, 150M, \~3B tokens | 57 h 23 m on 1 H100, uncompiled, gradient checkpointing, batch 48 × 512 | 5–10B German tokens at 4–8K rows: 3–5 days on 4 H100 (training memory 13.9 GB at 2,048, 54.4 GB at 8,192 uncompiled, so per-GPU batch shrinks with length) |
| Conversion of Qwen3-0.6B / 1.7B into the hybrid | none here | 1–3 days on 4 H100 for 5–20B tokens, scaled from MambaInLlama's 8B / 4 days / 8 GPUs; unmeasured |
| Decoder SFT, 12K steps, 4 H100 | 1 h 19 m (Mamba-3 decoder) to 2 h 30 m (hybrid v2) | same order per SFT/DPO run; GRPO dominated by rollouts at 157 tok/s per sequence (cached decode, batch 1) |
| Evaluation | paired bootstrap minutes; 400-study decode \~1 h; efficiency sweep 1.5–3 h per job | NLI verifier replaces CheXbert; 3 seeds × 3 systems |
| Storage | 200 GiB home quota; 2.4 GB per 150M checkpoint; `SAVE_TOP_K=0` mandatory | budget \~40 GB for backbones + SFT arms; the corpus (5–10B tokens) needs its own scratch |

**Baselines, all on the identical retriever, prompts and passages**

1. The parameter-matched Transformer (`transformer_150m_baseline`, exists; retrain on the legal corpus).
2. Qwen3-0.6B or 1.7B fine-tuned through the identical SFT → DPO → GRPO pipeline: the baseline reviewers will ask for.
3. A large open model with retrieval (gpt-oss-120b or Qwen3-235B) as the quality ceiling; a frontier API with the same passages as a second ceiling, never as training data.
4. A retrieval-copy floor: return the top-ranked passage verbatim. The medical retrieval-NN floor beat every generator arm on macro-F1; a legal generator that does not beat verbatim copying has not contributed anything.
5. The incumbent `hybrid_150m_v2` recipe with `legacy` operators, as the ablation that shows the correction still matters on legal text.

**Screens to run before the full backbone (12K steps, 2 seeds each, bar = 2× seed SD)**: attention count 0 / 2 / 4 at matched parameters; `d_state` 128 vs 64 with attention present; multi-token-prediction head on/off; global vs sliding-window attention only if contexts exceed 32K. Measure MQAR (`configs/dataset/mqar.yaml` exists), a statute-recall probe (given a § in context, reproduce Absatz 2 verbatim), and late-position perplexity, not only mean PPL.

## Evaluation plan and defensible claims

The system can win on citation faithfulness, refusal, currency, cost, memory and on-prem deployment, and it will lose on open-ended reasoning; the thesis is stronger if it says both plainly. Every claim follows the rule already in the repo: paired mean above one baseline seed SD, sign at ≥ 2 of 3 seeds, bootstrap CIs reported beside it and never in its place.

| Dimension | Metric | Target | Can a 150M–2B hybrid + RAG + verifier win? |
| --- | --- | --- | --- |
| Citation accuracy | citation precision and recall: pointer resolves, and the cited passage entails the claim (NLI + human-annotated subset) | ≥ 95% precision vs a frontier model without the index | Yes, plausibly; structural, not model-size dependent |
| Quote fidelity | exact-match rate of rendered quotes vs source | 100% by construction | Yes |
| Refusal | precision and recall on unanswerable and false-premise items (20–30% of the test set) | strong F1; no unjustified refusals on answerable items | Yes, if trained for it |
| Currency | correctness on provisions amended in the last 12 months | nightly-synced index, version filter | Yes |
| Cost and latency | cost per 1,000 queries, p50/p95 latency, tokens/s, GPU memory at 8K / 32K / 128K context | O(1) decode memory; report memory per concurrent session | Yes, strongly (1.34× FlashAttention at 16K inference, 5.19× cached decode) |
| Long-document handling | throughput and recall at 100K+ tokens | tokens/s and memory; MQAR and statute-recall accuracy | Efficiency yes; recall depends on the attention layers |
| Multi-step legal reasoning, subsumption | BenGER-style subsumption; multi-hop statute cross-references (the MiniMax-M2 finding: hybrids hid multi-hop deficits behind standard benchmarks) | report the loss | No |
| General German quality | held-out general-German PPL | not a goal; check catastrophic forgetting with replay | No |

Evaluation sets: GerLayQA (21,538 lay questions → BGB paragraphs, evaluation only; forum provenance), GerLeRB (367 queries over 90,798 statute paragraphs), GerDaLIR and LegalQuAD for case-law retrieval, plus a project-written held-out set of 200–300 questions graded by 2–3 DACH jurists with double annotation and reported agreement. Retrieval is scored with nDCG@10, Recall@{5,10,20} and MRR per jurisdiction; the Stanford RegLab typology (correct / incorrect / misgrounded, plus responsiveness) is the field's bar, with commercial legal RAG tools hallucinating 17–33% of the time.

Statistics: ≥ 3 seeds per arm, paired by seed and by study; bootstrap CI = sampling uncertainty over questions, seed SD = training uncertainty, kept visibly separate; no t-test on 2 degrees of freedom. Report every pre-registered prediction next to its outcome, including the wrong ones, as `MAMBA3_PLAN_V2.md` and `EFFICIENCY_PLAN.md` do.

Claims the design can support once measured: "comparable or better citation-faithful accuracy on DACH statutes than a same-size Transformer and a fine-tuned Qwen3 through the same pipeline, at lower memory and latency, on-prem"; "never emits an unresolvable citation and never misquotes a statute, by construction"; "linear latency in context length where attention is superlinear". Claims it cannot support: beating frontier models in general; better legal reasoning than GPT-4/5-class systems; anything about the hybrid mixer improving quality over attention, which the medical record measured as a null across three seeds.

## Open issues, risks and next decisions

The two decisions that shape everything else are the initialisation road (from scratch at 150–350M vs conversion of Qwen3-0.6B/1.7B) and whether the DACH corpus is collected under the university research exception or under licences; both should be decided before Phase 1 starts.

- [ ] **Initialisation road.** From scratch keeps the thesis story clean and reuses `train_stage0_distill.py` unchanged apart from tokenizer and teacher; conversion is the only route to frontier-competitive knowledge and inherits Qwen3's tokenizer, but needs new code (in\_proj initialisation from Q/K/V/O, layer-wise alignment, logit KL) that this repo does not have. Decide by compute: 4 H100 for a week supports either; only conversion supports 1.7B.
- [ ] **Corpus legality.** §60d UrhG text-and-data mining covers non-commercial research on lawfully accessible works, but excludes research organisations cooperating with a private company that has decisive influence and preferential access to results. If the model is built for or with XAINAG/Noxtua, use only the open sources (GII, RII, NeuRIS, OLDP under ODbL, DIP, EUR-Lex CC BY 4.0, RIS CC BY 4.0, Fedlex, BGer CC BY 4.0, opencaselaw CC0 self-declared) or licensed data used within its terms and with written authorisation. Keep a per-document licence tag and a commercial-safe subset from day one; exclude MultiLegalPile's NC subsets and GerLayQA from anything shippable.
- [ ] **Personal data in case law.** Published decisions are pseudonymised, not anonymised; run a German-LER-based scrub (PER, RR, AN, STR) before pretraining, never answer "who was the party", and rely on Art. 6(1)(f) GDPR plus Art. 89 research safeguards.
- [ ] **Attention KV cache and cached beam search with attention layers.** `AttentionBlock` has none; `supports_cached_decode()` is all-or-nothing, so the cached path is unavailable until `AttentionBlock.step()` exists.
- [ ] **mLSTM document-boundary loop** (`_forward_segmented`) at 8–16K packed rows; either mask inside TFLA or pack one document per row.
- [ ] **EOS supervision and stop condition**: the single largest untested lever in the medical record (75% of reports truncated); must be in place before any legal SFT.
- [ ] **Training-side cost**: 2.71× slower and 1.59× the memory of the Transformer at 2,048 even compiled; `torch.compile` for training was never run on a full job; `chunk_size=256/512` fail to compile. Budget for the uncompiled path.
- [ ] **Stage-0 seed band**: no backbone in the project's history has one; the legal screen should run 2 seeds per arm from the start.
- [ ] **Tokenizer fertility on German legal text**: unmeasured; expected well above 1.8 tokens/word with GPT-2 BPE.
- [ ] **Multi-hop deficit risk**: MiniMax reverted M2 to full attention after hybrids matched on standard benchmarks but lost on multi-hop reasoning at scale; the evaluation must include multi-hop cross-reference questions from the first screen.
- [ ] **Reward hacking in GRPO**: the verifier rewards trivially entailed citations and over-refusal unless a coverage/recall term and a human-checked held-out set are included.
- [ ] **README and naming on `main`**: remove the Triton requirement line and the kernel claim; add `attention` to the valid `layer_pattern` list; rename the project for the legal domain.
- [ ] **DUA obligations**: everything under `outputs/h100_report_gen_*`, `h100_kd_*`, `h100_6*` and `results/` stays on the cluster or approved storage; only the four PubMed Stage-0 backbones and `analysis/` may leave.
- [ ] **EU AI Act and RDG framing**: a general legal-information assistant is not Annex III high-risk unless used by or for courts; Article 50 transparency duties apply from 2 August 2026; frame outputs as legal information with sources and a "keine Rechtsberatung" notice; obtain local advice for AT and CH before any commercial launch. This document is not legal advice.

Sources for the numbers above: `analysis/mamba3_results.md`, `analysis/EFFICIENCY_NOTE.md`, `EFFICIENCY_PLAN.md`, `MAMBA3_PLAN_V2.md`, `analysis/PHASE14_SUPERVISOR_REVIEW.md`, `analysis/h100_scaling_results.md`, `analysis/PUBLISHED_BASELINES.md`, `analysis/PHASE15C_AUX_LOSS.md`, `analysis/scan_error_bound.md`, `analysis/ARCHIVE_MANIFEST.md`, `CLAUDE.md` and the state JSON files, all read on 2026-09-27 at the branch heads named in the reading status; the legal-domain data, licensing and architecture context comes from the three research reports supplied with the request.
