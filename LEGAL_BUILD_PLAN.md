# LEGAL_BUILD_PLAN — a DACH legal, citation-faithful, retrieval-gated hybrid Mamba-3 / xLSTM decoder

> **Status:** plan of record, opened 2026-09-27. **State:** `legal_build_state.json`. **Helper:** `python3 scripts/plan_state.py ...`
> **Resume rule:** every session starts with `python3 scripts/plan_state.py resume` and reads only the phase it prints. The checkboxes in this file are ground truth; the JSON is the resumable record. Tick through the helper, never by hand.
> **Repository rule:** git belongs to the user. The user initialised the repository on 2026-09-27 (remote `origin` = `github.com/krishankb-de/legal_ai_hybrid_mamba_xlstm_model`, one commit containing `README.md`) and decides every commit and push. The implementing model runs no git command unless the user asks for that specific command, and never pushes on its own.
> **Reference rule:** `Reference/hybrid_model_mamba_xlstm` is the medical codebase being ported. It is read-only, is deleted at **P1-Z** by the user, and is never read again afterwards; `Docs/REFERENCE_PORT_MAP.md` carries what the port needs.
> **Blueprint:** `Docs/Hybrid Mamba–xLSTM Codebase Review and DACH Legal AI Blueprint.md` (read on 2026-09-27; cited by section from the port map, not re-read wholesale).
> **Testing and CI:** §8. Every code box ships its tests; `bash scripts/validate.sh` is the local gate; `.github/workflows/ci.yml` runs the same gates plus the Linux-only and pushed-tree checks on every push. The workflow is inert until the user pushes; nothing in this plan pushes.

---

## 1. Why this plan exists

The reference project built and measured a corrected Mamba-3 + mLSTM decoder on the HPI aisc H100 cluster for chest-X-ray report generation. Its own record licenses a narrow set of claims (memory parity with FlashAttention at 16K; 1.34× inference speed compiled at chunk 128; a 5.19× O(1) cached decode; a −16% validation-perplexity repair of two defective recurrences; statistical parity with a matched Transformer across three seeds) and forbids a longer one. The blueprint concludes that the defensible unique contribution for a legal system is **the package, not the mixer**: a corrected hybrid decoder that never generates statute text, cites by pointer into a numbered retrieved context, and is gated by a verifier, evaluated on DACH law.

This plan builds that system from a clean, production-grade repo (`lexhybrid`), reusing the reference's backbone, kernels, decode cache, packing, tests and measurement discipline, and replacing the tokenizer, teacher, corpus, targets, decode loop and sequence length. Everything is executed by a smaller model than the one that wrote this plan, one checkbox at a time, locally on a Mac first and then on the aisc cluster through `sbatch`.

## 2. Locked decisions

| # | Decision | Value | Source |
|---|---|---|---|
| 1 | Initialisation road | From scratch first (Qwen3 tokenizer, logit-KL from an open `Qwen3-*-Base` teacher). Qwen3 conversion is P11, optional, on user instruction only | user, 2026-09-27 |
| 2 | Corpus | Open-licensed sources **plus** a research-exception arm; every document carries `licence` and `commercial_safe`; `research_only` documents never enter a shippable arm | user, 2026-09-27 |
| 3 | GPUs | 1 H100 per screen arm; 4 H100 DDP for full backbone and SFT runs | user, 2026-09-27 |
| 4 | Reference deletion | P1-Z, after the port gate, user-confirmed | user, 2026-09-27 |
| 5 | Package name | `lexhybrid`; Python ≥ 3.11; PyTorch ≥ 2.5; `pytorch_lightning` ≥ 2.4; Hydra; editable install, no `sys.path` shims | plan |
| 6 | Tokenizer / EOS | Qwen3 tokenizer (`Qwen/Qwen3-1.7B-Base`); EOS is Qwen3's own `<\|endoftext\|>` (id 151,643) so the teacher's target at document ends is meaningful; student `vocab_size = 151936` | plan (blueprint §Pivot; review) |
| 7 | Added special tokens (SFT only, never in KD) | `<\|q\|>`, `<\|cite\|>`, `<\|c1\|>`..`<\|c16\|>`, `<\|s1\|>`..`<\|s64\|>`, `<\|unanswerable\|>`, `<\|passage\|>`, `<\|/passage\|>`, `<\|question\|>`, `<\|answer\|>` (87 tokens ≤ 267 free ids) | plan |
| 8 | Teacher | `Qwen/Qwen3-1.7B-Base` by default; `Qwen/Qwen3-8B-Base` only if P4-U measures it fits at 4,096 rows on 4×H100 | plan |
| 9 | KD loss | online; `(1−α)·CE + α·T²·KL(teacher_T ‖ student_T)`, α=0.5, T=2, computed per 512-token slab under activation checkpointing; teacher receives per-document `position_ids` and a block-diagonal mask; the doc-boundary label mask applies to KL too | plan (review) |
| 10 | Model | 12 layers, dim 768, 12 heads, pattern `M3 M3 M3 A L L L M3 M3 A M3 M3`, `norm_topology hybrid`, `tfla_impl exact`, `d_state 128`, `mamba3_chunk_size 128`, `mlstm_chunk_size 128`, `mlstm_forget_gate_bias_init 3.0`, `rope_theta 500000`, `dropout 0.0`, tied embeddings, `max_position_embeddings 8192` | plan |
| 11 | Parameter counting | `non_embedding = total − embedding − (lm_head if untied)`; both counts reported always; the base model is **104.1M non-embedding / 220.8M total** and is never called "150M" | review |
| 12 | Sequence lengths | screens at 2,048-token rows; main pretraining at 4,096; long-context stage at 8,192; 32K/128K numbers are efficiency-only | blueprint; review |
| 13 | Corpus tooling | LER scrub `flair/ner-german-legal` (PER/RR/AN/STR); dedup `datasketch` MinHash 5-gram/128 perms/Jaccard ≥ 0.8 **within a source**; copies of one text in different sources are all kept, packed and linked in the dedup log, and a citation names every source that holds the passage; replay FineWeb-2 `deu_Latn` (ODC-By) | plan; user, 2026-09-29 (cross-source copies) |
| 14 | Retrieval / verifier tooling | `bm25s` + `BAAI/bge-m3` + RRF + `BAAI/bge-reranker-v2-m3`; NLI `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, threshold pre-registered before calibration | plan |
| 15 | Ceilings | large open model `gpt-oss-120b` with the same passages (Qwen3-235B does not fit 4×80 GB) or skipped and said so; a frontier API only as an optional USER ACTION and never as training data | blueprint |
| 16 | Statistics | 2 seeds per screen arm, 3 seeds for any claim; claim rule = paired mean > one baseline seed SD **and** sign at ≥ 2/3 seeds; bootstrap CIs (1,000 resamples by question) beside, never instead | reference decision 10 |
| 17 | Legacy operators | only `configs/model/hybrid_legal_legacy.yaml` may set `scan_impl: legacy` / `tfla_impl: legacy`; it is PPL/probe-only (no cached decode exists for it) | plan |
| 18 | Environment | uv project with one `uv.lock` for macOS arm64 and Linux x86_64; `.python-version` 3.11 (the cluster interpreter); `torch==2.11.0` everywhere: PyPI wheel on macOS, `2.11.0+cu128` from the PyTorch cu128 index on Linux (the newest build for the cluster's CUDA 12.8 driver — torch 2.12+ has no cu128 build); every environment is built with `uv sync --locked` and verified by `scripts/check_env.py` | user, 2026-09-27 |
| 19 | CI | GitHub Actions `.github/workflows/ci.yml` on every push and pull request: lint, tests on Linux (Python 3.11 and 3.12), pushed-tree hygiene, packaging; weekly network smoke and macOS clean install; GPU checks stay on the cluster; inert until the user pushes | user, 2026-09-27 |
| 20 | S5 parameter band | the legacy recipe's +2.36% non-embedding is a stated difference inside ±2.5%, not reshaped (the pre-registered ≤ 1% did not hold for the 9 × mamba + 3 × mlstm shape); S5 stays PPL/probe-only; a hybrid win is conservative, a legacy win inside the margin inconclusive | user, 2026-09-27 |
| 21 | S5 forget bias | `mlstm_forget_gate_bias_init 3.0`, the base's value: S5's one lever is the legacy operator; the reference's effective 0.0 came from defect 3, and at 0.0 the kernel falls onto its sequential path at chunk 128 | user, 2026-09-27 |
| 22 | LER scrub environment | `flair/ner-german-legal` runs in its own uv project `envs/scrub/` (own lock, like `.venv-vllm`); the main environment keeps transformers 5 and never imports flair; if flair does not install cleanly there, a transformers-5-compatible model trained on the same German LER data, evaluated on its test split before use | user, 2026-09-27 |

## 3. Rules (R1–R14)

- **R1 Equivalence.** Any change to an operator (`ssd_chunked_scan`, TFLA, selective scan, decode step) passes the fp64-oracle gate at rel-max-err ≤ 1e-6 (operator) and the model-logit gate at ≤ 1e-4 (`scripts/check_operator_equivalence.py`) before it is ticked.
- **R2 Memory.** No change regresses the inference memory result (parity with FlashAttention at 16K, reference 7.09 vs 7.15 GB); P4 measures it and P9 re-measures it.
- **R3 One lever per arm.** Every arm differs from its control in exactly one named lever; arms that are not parameter-matched carry their delta in every table.
- **R4 Predict first.** Every measurement phase opens with an italic pre-registered prediction and a bar written before the job is submitted; the RESULTS block reports prediction → outcome, including wrong ones.
- **R5 Seed band before claim.** Two seeds per screen arm, three for any claim; decision 16's claim rule verbatim; no t-test on two degrees of freedom.
- **R6 Checkpoint budget.** `SAVE_TOP_K=0` on every arm, `1` + `last.ckpt` on pipeline inputs; `outputs/` lives on `$SCRATCH_ROOT`, never in `$HOME`; `du` before every submission.
- **R7 One compiler cache per measured point.** Every `torch.compile` number is measured one shape per process with its own `TORCHINDUCTOR_CACHE_DIR`; the profiler prints the chunk size read back off the module.
- **R8 No legacy operators** outside `hybrid_legal_legacy.yaml` and the `ref_*` reference replicas (which keep the reference's operators so their parameter counts and P4's sanity point match it); every recurrent yaml pins `tfla_impl` (and `scan_impl` when it has Mamba-1 layers) explicitly.
- **R9 Licence discipline.** Every document carries `licence`, `jurisdiction`, `commercial_safe`; `build_corpus` refuses `licence=unknown`; `research_only` never enters a `commercial_safe` shard (test in `validate.sh`); no checkpoint trained with `research_only` data leaves the cluster.
- **R10 No API-model training data.** Never train on GPT/Claude/Gemini API outputs; synthetic data comes from open-weight teachers only.
- **R11 Personal data.** LER scrub (PER, RR, AN, STR) runs before pretraining and before any index build; the verifier rejects answers naming a PER entity; "who was the party" is a refusal.
- **R12 Provenance.** Every number in `analysis/` carries a job id and a log path; every model logs its `ARCH` fingerprint at construction; every run writes `run_metadata.json` with the `.sync_stamp` tree hash and the commit id read from `.git/HEAD` when present (read as a file, never by running git); "tooling is not a tick — the gate is the measurement".
- **R13 Tests with the code.** A box that adds or changes code adds or changes its tests in the same box; every test belongs to a layer and a marker of §8; a skip always carries its reason; the coverage floor is never lowered without a note; an intermittent test is a defect, never retried until green.
- **R14 One gate, three venues.** `validate.sh --ci` and the CI `test` job run the same pytest selection; a check that cannot run on the Mac is listed in §8.3 and runs in CI or on the cluster; a red CI run on a pushed branch blocks the current box until it is fixed.

## 4. Baselines every arm is measured against

Reference numbers (H100 80 GB, bf16, batch 4, random weights unless stated; `analysis/mamba3_results.md`, `analysis/EFFICIENCY_NOTE.md` of the reference; job ids in the port map §14). They are sanity checks for P4 and the yardstick for P9's efficiency table; the legal model's quality numbers come from this plan's own arms.

| Quantity | Reference value |
|---|---|
| Inference, L=16,384: Mamba-3 compiled chunk 128 / FlashAttention Transformer | 122.90 ms, 7.169 GB / 164.85 ms, 7.152 GB (1.34× faster, memory parity) |
| Inference, uncompiled chunk 64, L=2,048 / 16,384 | 86.2 ms, 1.24 GB / 678.2 ms, 7.09 GB (Transformer 16.1 / 163.9 ms) |
| Training step, L=2,048: Transformer / hybrid compiled chunk 128 | 52.17 ms, 7.55 GB / 141.14 ms, 12.02 GB (2.71× slower, 1.59× memory) |
| Training memory, hybrid uncompiled, L=2,048 / 8,192 | 13.9 GB / 54.4 GB |
| Cached decode, prompt 256, batch 1 | 157.6 tok/s, 5.19× per token vs full recompute; time-to-first-token 4.0× worse (token-by-token prefill) |
| Screen, 12K steps, PubMed, seed 42/1234 | A0 19.387/18.933 (defective), A2 16.708/16.376, A2x (exact TFLA) 15.566/15.788; bar 0.642 PPL = 2× seed SD |
| Stage-0, 120K steps, PubMed val PPL | hybrid_150m_m3 11.674; hybrid_150m_v2 (legacy) 13.18; transformer_150m_baseline 11.222 |
| Wall, one H100 | 12K-step screen arm 1 h 21 m (A2) to 5 h 45 m (A2x, exact TFLA); 120K-step backbone 57 h 23 m |

## 5. Design

**Backbone.** `HybridLanguageModel` = embedding → 12 × `HybridBlock` (pre-norm → mixer → residual; HybridNorm post-norm FFN from block 1) → RMSNorm → tied LM head. Mixers by `layer_pattern`: `mamba3` (Mamba-2 SSD, scalar-A per head, `d_state 128`, boundary masks inside the scan), `mlstm` (exact TFLA, boundary masks inside the kernel after P2-D), `attention` (RoPE, per-document positions, flex-attention block mask in packed training, KV cache in decode), `mamba` (legacy Mamba-1, ablation only). Packed rows carry `doc_ids: (B, L)` integer document ids; every mixer resets state where the id changes; the loss masks the prediction across a boundary.

**Decode.** One-pass prefill from each mixer's carried state; O(1) step per token for the recurrent mixers, KV append for the two attention layers; greedy / sampling / beam over the cache; EOS stop; repetition penalty; pointer-token FSM that masks `<|cK|>`/`<|sJ|>` ids outside the retrieved set; best-of-n hook for verifier reranking.

**System (P7).** Retriever (statute chunks at §/Artikel with hierarchy and version dates; case law by Tenor/Tatbestand/Gründe then 300–500-token windows with Randnummer anchors; BM25 + dense + RRF + rerank; jurisdiction and valid-on filters) → decoder (pointer citations only) → renderer (inserts the exact sentence and the § label from metadata; appends the "keine Rechtsberatung" notice) → verifier (pointer resolves; quote string-matches; NLI entailment; party-identity rule). Nothing unverified reaches the user.

**Training.** Pretraining = CE + logit-KL on packed 4,096-token rows, WSD schedule, then a short 8,192-row stage. Post-training = SFT on the pointer format with 20–30% unanswerable items → DPO/KTO on verifier pass/fail pairs → GRPO only if a pre-registered condition holds.

**Ablation ladder (OFAT, not factorial).** S1 is the base and the control; every other screen arm flips one lever against it (§P5). Arms whose lever changes the parameter count are reported with the delta.

## 6. Compute and storage budget (estimates; P4 replaces them with measurements)

| Item | Estimate | Basis |
|---|---|---|
| Screen arm, 12K steps, 2,048 rows, ~32K tokens/step, 1.7B teacher, 1 H100 | 6–10 h each; 7 arms × 2 seeds ≈ 100–140 H100-h | reference 1 h 21 m – 5 h 45 m at 512 rows and 24.6K tokens/step |
| Full backbone, 5B tokens, 4,096 rows, 4×H100 DDP, ~262K tokens/step, ~19K steps | 1–2 days uncompiled; stage B (0.5B tokens at 8,192 rows) ~6 h | reference 14–18K tok/s per H100 with a 2.6B teacher |
| Transformer baseline | same order | |
| SFT, 12K steps, 4×H100 | 1.5–3 h per arm; 7 arms | reference 1 h 19 m – 2 h 30 m |
| Synthetic QA, Qwen3-32B in vLLM, 1 H100 | ~65 GB weights; hours per 50K items | |
| Checkpoint | ≈ 4.3 GB (fp32 weights + two Adam moments for 220.8M params); P4-N measures it | |
| Storage | `outputs/` on scratch; ≤ 40 GB of checkpoints kept at any time (one `hybrid_legal_base` checkpoint = 2.65 GB, P4-O job 2589377: ≤ 15 kept); 200 GB home quota untouched | reference quota incident |

## 7. Phases

Every phase: purpose → where it runs → boxes → `**Gate:**`. Box text names the file, the test and the verify command. `USER ACTION —` boxes stop the session. Boxes that say `sbatch` come in pairs (submit, verdict) and need `--evidence job=<id>`. `Pn-Z` is always the gate box.

### P0 — Plan of record

Purpose: create the plan, state, helper, bootstrap, port map, template and README (this session, Stage 0). Runs: Mac, no GPU, no downloads.

- [x] **P0-A** Write `LEGAL_BUILD_PLAN.md` (this file) in the helper's grammar: `### Pn — title`, `- [ ] **Pn-X**` boxes, one `**Gate:**` per phase, `Pn-Z` gate boxes, italic predictions in measurement phases.
- [x] **P0-B** Write `scripts/plan_state.py` (stdlib only) with `init|sync|resume|section|tick|note|next-action|phase|next|block|unblock|job|artifact|evidence|verdict|show|session-end|check|readme` and the tick guards (phase, USER ACTION, sbatch evidence). Verify: the smoke test in the scratchpad passes every guard.
- [x] **P0-C** Write `CLAUDE.md` (≤ 120 lines): bootstrap command, awk fallback, rules (a)–(j), command table, hostname rule.
- [x] **P0-D** Write `Docs/REFERENCE_PORT_MAP.md`: source → destination for every module, verbatim signatures, the 18 defects with line references, SLURM headers, env block, WSD formula, ARCH format, oracle-test structure, validate gates, gitignore guards, handoff recipe, measured numbers, incidents, stack.
- [x] **P0-E** Write `Docs/analysis_TEMPLATE.md` and `README.md`.
- [x] **P0-F** `python3 scripts/plan_state.py init` creates `legal_build_state.json`; fill `decisions`, `pre_registered.rules`, `scope`, `status`, `next_action`; `current_phase = P1`.
- [x] **P0-Z** Gate: `python3 scripts/plan_state.py check` exits 0; `show` lists P0 complete and P1..P11 pending; nothing under `Reference/` modified; no `.git`; no GPU, ssh or cluster action; no downloads.

**Gate:** `check` exit 0 and the six deliverables exist.

### P1 — Bootstrap and faithful port

Purpose: a runnable `lexhybrid` package whose blocks compute exactly what the reference blocks compute, with renames only (`cu_seqlens → doc_ids`, package name, no `sys.path`, no git). Defects are **not** fixed here (P2), so the parity test can hold. Runs: Mac, CPU only. Port map: `Docs/REFERENCE_PORT_MAP.md` §1–§10. Precondition: `Reference/hybrid_model_mamba_xlstm` present.

- [x] **P1-A** `pyproject.toml` as a uv project (decision 18): package `lexhybrid`, `requires-python >= 3.11`; dependencies `torch==2.11.0`, `pytorch-lightning>=2.4`, `hydra-core`, `omegaconf`, `transformers`, `datasets`, `einops`, `pyyaml`, `numpy`, `packaging`, `tqdm`, `tensorboard`; dependency group `dev` (`pytest`, `pytest-cov`, `ruff`, `pathspec`); `[tool.uv.sources]` torch from the explicit PyTorch cu128 index on Linux; `[tool.uv] environments` macOS arm64 + Linux x86_64; `[tool.pytest.ini_options]` with `--strict-markers` and markers `slow cuda multigpu linux_only network reference` (the reference's `[tool:pytest]` in `pytest.ini` was ignored, defect 14); `[tool.ruff]`; `[tool.coverage]`; hatch wheel and sdist targets with root-anchored includes (`tests/test_packaging.py` pins the archive contents); `.python-version` = `3.11`; `.gitignore` (port map §12) and `.rsync-exclude` (rsync rules: `+ tests/fixtures/**` before the `*.pt` exclude, because rsync is first-match-wins and has no `!` negation; `tests/test_rsync_rules.py` runs the local rsync on a temporary tree, including `--delete`); package skeleton `lexhybrid/{__init__,config,layers,kernels,models,decoding,data,training,retrieval,verifier,renderer,eval,utils}/__init__.py`; `configs/`, `scripts/slurm/`, `tests/`, `analysis/`, `data/manifests/.gitkeep`. Verify: `uv lock` writes `uv.lock`; `uv lock --check` exits 0.
- [x] **P1-B** `.venv` via `uv sync --locked` (uv fetches CPython 3.11 when the Mac has none). Verify: `.venv/bin/python -c "import sys, torch; print(sys.version, torch.__version__)"` prints 3.11.x and 2.11.0; `lexhybrid` imports from a directory outside the repo (editable install, no `sys.path` shims).
- [x] **P1-C** `lexhybrid/config/hybrid_config.py` ← `hybrid_xmamba/models/configuration_hybrid.py` (port map §1): drop `slstm_*`, `pooling_strategy`, `bidirectional_encode`, `proj_head_dropout`, `use_cache`, `use_tfla`; keep everything else including `scan_impl`/`tfla_impl` defaults `legacy` for parity; `from_hydra` unchanged. Test: `tests/test_config.py::test_from_hydra_filters_unknown_keys_and_keeps_every_field`, `::test_invalid_values_are_rejected` (an `slstm` layer pattern among the cases). Verify: `.venv/bin/python -m pytest tests/test_config.py -q`.
- [x] **P1-D** `lexhybrid/layers/normalization.py` (`RMSNorm` exactly as the reference, no upcast yet) and `lexhybrid/layers/activations.py` (`exponential_activation`, `silu_activation`) ← port map §2. Test: `tests/test_layers.py::test_rmsnorm_shape_and_unit_rms`.
- [x] **P1-E** `lexhybrid/layers/rotary.py` ← `hybrid_xmamba/layers/rotary.py` (`cumulative_angles`, `_segment_baseline`, `apply_rotary`; `cu_seqlens → doc_ids`). Test: `tests/test_layers.py::test_cumulative_angles_restart_per_document`.
- [x] **P1-F** `lexhybrid/kernels/ssd/{ssd_interface,ssd_reference}.py` ← `hybrid_xmamba/kernels/ssd/` (port map §3), verbatim apart from `doc_ids`; export `segsum, ssd_chunked_scan, ssd_step, ssd_sequential_reference`. Test: `tests/test_numerics_fp64.py::test_ssd_chunked_matches_sequential_reference[shape,chunk]` (rel-max-err ≤ 1e-6; padding) and `::test_ssd_document_boundaries_match_reference[mid-chunk,on-chunk-edge,multi-doc]`.
- [x] **P1-G** `lexhybrid/kernels/tfla/{tfla_interface,tfla_reference}.py` ← `hybrid_xmamba/kernels/tfla/tfla_interface.py` plus the fp64 `sequential_mlstm_fp64` oracle moved out of the reference tests into `tfla_reference.py` (port map §4). Both `legacy` and `exact` kept for parity. Test: `tests/test_numerics_fp64.py::test_tfla_matches_sequential_reference[impl,forget_bias,chunk]`, the `legacy` cases as `xfail(strict=True)`.
- [x] **P1-H** `lexhybrid/kernels/selective_scan/scan_interface.py` ← `hybrid_xmamba/kernels/selective_scan/scan_interface.py` (`selective_scan`, `selective_scan_parallel`, `selective_scan_exact`, fp64 reference). Test: `tests/test_numerics_fp64.py::test_selective_scan_chunked_matches_sequential_reference` and `::test_selective_scan_public_api_matches_sequential_reference` with the reference's `DELTAS`/`CHUNKS` and `xfail(strict=True)` for legacy.
- [x] **P1-I** `lexhybrid/layers/mamba3_block.py` ← `hybrid_xmamba/layers/mamba3_block.py` (port map §5) unchanged apart from `doc_ids`. Test: `tests/test_layers.py::test_mamba3_step_reproduces_the_chunked_forward` (from the reference's `test_mamba3_numerics.py` step tests).
- [x] **P1-J** `lexhybrid/layers/mlstm_block.py` ← `hybrid_xmamba/layers/mlstm_block.py` (port map §6) with `_slow_forward` and `use_tfla` **removed** (gate 3 of validate must run the shipping operator), `_forward_segmented` kept for now. Test: `tests/test_layers.py::test_mlstm_step_reproduces_the_shipping_exact_tfla_operator`.
- [x] **P1-K** `lexhybrid/layers/attention_block.py` ← `hybrid_xmamba/layers/attention_block.py` (port map §7) unchanged. Test: `tests/test_layers.py::test_attention_is_causal_and_blocks_cross_document`.
- [x] **P1-L** `lexhybrid/layers/mamba_block.py` ← `hybrid_xmamba/layers/mamba_block.py` (legacy Mamba-1, `scan_impl` legacy/exact, `post_model_init`, `_forward_segmented`). Test: `tests/test_layers.py::test_mamba1_forward_backward`.
- [x] **P1-M** `lexhybrid/layers/hybrid_block.py` ← `hybrid_xmamba/layers/hybrid_block.py` (port map §8) with the `slstm` branch removed and `_mamba3_params()` kept; the `mlstm` allow-list bug is preserved for parity (fixed in P2-A). Test: `tests/test_layers.py::test_hybrid_block_all_layer_types` (pre-norm), `::test_hybrid_norm_ffn_is_post_residual_from_block_one` (HybridNorm), `::test_typo_in_a_prefixed_mixer_option_raises`.
- [x] **P1-N** `lexhybrid/models/hybrid_lm.py` ← `hybrid_xmamba/models/hybrid_lm.py` lines 1–600 (port map §9): `CausalLMOutput`, `HybridEmbedding`, `HybridLanguageModel` with `forward`, `architecture_fingerprint`, `_init_weights`, `generate` (uncached), `supports_cached_decode`, `allocate_inference_cache`, `step_logits`, `prefill` (token-by-token, for now), `_filter_logits`, `generate_cached`, `reorder_cache`, `beam_search_cached`, `get_num_params`, `get_layer_types`; drop `prefix_embeds`, `AttentionPooling`, `HybridTextEncoder`; fingerprint moved to `lexhybrid/utils/arch_fingerprint.py`; the uncached beam search the cached one was tested against (`scripts/evaluate_report_generation.py::beam_search_decode`, minus the image prefix) becomes `lexhybrid/decoding/generate.py::beam_search_uncached`. Test: `tests/test_models.py::test_forward_backward_all_mixers_every_parameter_gets_a_gradient`, `::test_arch_fingerprint_tokens`, `tests/test_decoding.py::test_cached_beam_equals_uncached`.
- [x] **P1-O** `lexhybrid/training/{optimizer,schedulers,metrics}.py` ← `hybrid_xmamba/training/` (`get_parameter_groups`, `configure_optimizer`, `wsd_factor`, `WSDScheduler`, `beta2_for_step`, `apply_beta2_schedule`, `compute_perplexity`, `compute_mqar_accuracy`; rename `compute_bits_per_byte` to `compute_bits_per_token`). Test: `tests/test_training.py::test_wsd_shape_matches_the_reference_record` (at 120,000 steps and warmup 2,000: D=16,800, S=101,200, decay start 103,200) and `::test_beta2_anneal`.
- [x] **P1-P** `lexhybrid/training/callbacks.py` (`SignalCheckpointCallback` ← `signal_callbacks.py`) and `lexhybrid/training/lightning_module.py` (`HybridLightningModule` ← lines 19–283 of the reference module: `training_step`, `validation_step`, `configure_optimizers` with wsd/cosine/linear/constant, manual `clip_grad_norm_`, β2 anneal). Test: `tests/test_training.py::test_lightning_module_trains_two_steps_on_cpu[wsd,cosine,linear,constant]`.
- [x] **P1-Q** `lexhybrid/utils/run_metadata.py` (fields of the reference minus `git_*`, plus `sync_stamp` read from `.sync_stamp` if present and `tree_hash` computed over `lexhybrid/ configs/ scripts/`) and `lexhybrid/utils/checkpoint.py` (`strip_prefixes` from `evaluate_lm.py:223-262`, `load_state_dict_guarded` with the >50%-missing hard-fail / >5% warning from `train_report_generation.py:271-287`, and `checkpoint_arch.py`'s `FINGERPRINTS`/`infer_layer_types`/`infer_architecture` without `slstm`). Test: `tests/test_utils.py::test_guarded_load_refuses_mismatch_and_accepts_match`, `::test_infer_layer_types_and_architecture`.
- [x] **P1-R** `scripts/evaluate_lm.py` ← reference (perplexity, bits-per-token, throughput ladder, peak memory; uses `load_state_dict_guarded`; `--dataset` takes a packed parquet shard path instead of `wikitext|pubmed`) and `scripts/verify_handoff.py` ← `verify_stage0_handoff.py` (builds `HybridLanguageModel` from a yaml, loads `model.`-prefixed keys from `last.ckpt`, PASS = 0 missing / 0 unexpected). Both build their model through `lexhybrid/config/loading.py` (`load_model_config`: the one yaml → `HybridConfig.from_hydra` path every script uses, FM5). Test: `tests/test_scripts.py::test_verify_handoff_pass_and_fail` on a tiny model.
- [x] **P1-S** `scripts/performance_profile.py` ← reference (`--sweep --backward --per-layer --decode --attn-backend --chunk-size --compile --dtype`, median timing, OOM rows, log-log exponent fit, `effective_chunk_size` read back) and `scripts/check_operator_equivalence.py` ← reference (`R1_TOLERANCE = 1e-4`; operator, boundary-inside-chunk, model-logit and compile gates). Test: `tests/test_scripts.py::test_check_operator_equivalence_passes_on_cpu`.
- [x] **P1-T** `lexhybrid/eval/text_metrics.py` ← `rouge_l_score`, `corpus_bleu` from the dropped `scripts/evaluate_report_generation.py` (verbatim, stdlib only); `scripts/bootstrap_compare.py` ← reference, importing them from there, CheXbert parts dropped, generalised to per-item metric files (`--metrics-a/-b` JSONL); `scripts/analyze_generation_diversity.py` ← reference with the medical verdict text generalised. Test: `tests/test_scripts.py::test_bootstrap_paired_ci_sign`, `tests/test_eval.py::test_rouge_l_known_values`, `::test_corpus_bleu_known_values`.
- [x] **P1-U** `tests/test_numerics_fp64.py` complete (port map §10): `DELTAS`, `CHUNKS`, `TOL = 1e-6`, `_xfail_if(strict=True)`, CPU-collected, never CUDA-gated; covers SSD (contiguous, boundary mid-chunk, boundary on edge, multiple docs, `step` = forward), TFLA exact/legacy, selective scan exact/legacy, feature-off bit-identity (zero-angle RoPE, λ=1 trapezoid, zero-init bias). Verify: `.venv/bin/python -m pytest tests/test_numerics_fp64.py -q` → all pass or strict-xfail.
- [x] **P1-V** `configs/config.yaml` (defaults model/dataset/trainer/callbacks/distill: null; `seed: 42`; `output_dir: ${oc.env:SCRATCH_ROOT,.}/outputs/${experiment_name}`), `configs/callbacks/default.yaml`, `configs/trainer/{cpu_debug,h100_single_gpu,h100_multi_ddp}.yaml` (port map §11; DDP strategy built in code), `configs/dataset/synthetic.yaml` + `lexhybrid/data/synthetic.py` (seeded random-token rows with `doc_ids`, for smoke tests and profiling), and the two reference models as `configs/model/ref_hybrid_m3.yaml` and `configs/model/ref_transformer.yaml` (GPT-2 vocab, kept for P4's sanity point; their parameter counts must equal the reference's 184,192,200 and 183,386,880 exactly) plus `tests/test_config_invariants.py` (Hydra compose of every yaml; recurrent yamls pin operators; exact reference parameter counts). Verify: `.venv/bin/python -m pytest tests/test_config_invariants.py -q`.
- [x] **P1-W** `scripts/validate.sh` (port map §13; modes: default, `--full`, `--ci`): gate 0 `python3 scripts/plan_state.py check`; gate 1 environment (`uv lock --check`, `scripts/check_env.py`); gate 2 static (`ruff check`, `ruff format --check`, `bash -n` and `shellcheck` on every shell script); gate 3 Hydra invariants over every `configs/model/*.yaml` (`scripts/check_configs.py`); gate 4 pytest with the §8.3 selection (`--ci` adds coverage and JUnit output); gate 5 CPU forward/backward over `[mamba3, mlstm, attention, mamba]` with the shipping operators asserting every parameter receives a gradient (`scripts/smoke_model.py`); gate 6 repository hygiene (`scripts/check_repo_hygiene.py`, `.gitignore`-aware in every mode; the CI `hygiene` job runs `--ci` on the pristine checkout); plus `tests/test_plan_state.py` (the helper against a temp copy of the plan: init, tick, guards, check). Verify: `bash scripts/validate.sh` exits 0.
- [x] **P1-W1** `scripts/check_env.py`: interpreter equals `.python-version` (or `--expect-python`), every installed distribution equals its `uv.lock` pin for this platform (via `uv export --frozen`), key imports and versions, `lexhybrid` installed editable from this repo, `torch.version.cuda == "12.8"` on Linux and `None` on macOS; exits 1 on any mismatch. Test: `tests/test_check_env.py` (pure functions on synthetic lock/installed tables). Verify: `.venv/bin/python scripts/check_env.py` exits 0.
- [x] **P1-W2** `scripts/check_repo_hygiene.py`: local mode walks the tree minus `.gitignore` (via `pathspec`), `--ci` walks the checkout (= the pushed tree); fails on files > 5 MB, checkpoints/data outside `tests/fixtures/`, a `Reference/` directory in CI, CRLF line endings, case-insensitive path collisions, secret-shaped strings, shell scripts without a shebang, and critical files (`README.md`, `CLAUDE.md`, `LEGAL_BUILD_PLAN.md`, `legal_build_state.json`, `uv.lock`, `pyproject.toml`, `.python-version`, `.github/workflows/ci.yml`, `Docs/*.md`, `analysis/*.md`) that `.gitignore` would exclude. Test: `tests/test_repo_hygiene.py` on temp trees. Verify: `.venv/bin/python scripts/check_repo_hygiene.py` exits 0.
- [x] **P1-W3** `.github/workflows/ci.yml` per §8.4: jobs `lint`, `test` (Python 3.11 and 3.12), `hygiene`, `package` on every push and pull request; `macos` and `network` on the weekly schedule and manual dispatch; `astral-sh/setup-uv` + `uv sync --locked` everywhere; read-only permissions; concurrency cancels superseded runs. Verify: `actionlint .github/workflows/ci.yml` exits 0.
- [x] **P1-W4** `tests/test_ci_workflow.py`: the workflow triggers on push and pull request, every job installs with `uv sync --locked` and never `pip install`, the `test` matrix contains the `.python-version` interpreter, the `test` job runs `bash scripts/validate.sh --ci`, the `hygiene` job runs `check_repo_hygiene.py --ci`, `lint` runs `actionlint` and `shellcheck`, network tests run only on schedule/dispatch, permissions are read-only; and `validate.sh`'s marker selections equal §8.3. Verify: the test file passes.
- [x] **P1-W5** Coverage floor: run `bash scripts/validate.sh --full --coverage`, record the measured `lexhybrid/` line coverage in the note, set `[tool.coverage.report] fail_under` to it rounded down to the nearest 5. Verify: `bash scripts/validate.sh --ci` exits 0 locally (the Linux-only tests skip with their reason on the Mac).
- [x] **P1-X** `scripts/dump_reference_parity.py`: imports `Reference/hybrid_model_mamba_xlstm/hybrid_xmamba` by path; for each of `mamba3`, `mlstm`, `attention`, `mamba` builds the reference block at dim 64 (`d_state 16`, `head_dim 32`, `tfla_impl exact`, reference chunk sizes, `mlstm_*` unset), seeds 0, input (B=2, L=96) with `doc_ids` `[0]*40+[1]*56`, and writes `tests/fixtures/reference_parity/<type>.pt` = a list of cases `{kwargs, state_dict, input, doc_ids, output_doc, output_nodoc, step_output}` in fp32 (several flag combinations per block, e.g. Mamba-3 with trapezoid + RoPE, mLSTM exact and legacy, Mamba-1 legacy and exact), plus `model.pt` for a 4-layer `HybridLanguageModel` (vocab 256, which keeps `model.pt` under the 5 MB hygiene limit: logits with and without `doc_ids`, loss, and cached beam-search tokens for a decodable pattern). Test: `tests/test_reference_fixtures_fresh.py` (marked `reference`: regenerating the fixtures reproduces the stored ones). Verify: `ls tests/fixtures/reference_parity | wc -l` → 5.
- [x] **P1-Y** `tests/test_reference_parity.py`: loads each fixture's `state_dict` into the NEW block or model (strict), runs fp32 forward with and without `doc_ids` (and `step` where the block has one), asserts `max|Δ| ≤ 1e-6` for blocks, `≤ 5e-6` for whole-model logits and loss (cross-platform fp32 BLAS: fixtures from macOS arm64 differ by up to 1.4e-6 on Linux x86_64, CI run 36322991795), and identical beam-search tokens. Never skipped, no `reference` marker. And `tests/test_port_map.py`: every destination of a P1 row of `Docs/REFERENCE_PORT_MAP.md` §0 exists (a later row's once its Box is ticked) and is not excluded by `.gitignore`; every Box is a real checkbox whose text names its destination; while `Reference/` exists every source exists; no `hybrid_xmamba` or `cu_seqlens` string under `lexhybrid/ scripts/ tests/` (fixtures excluded); no `sys.path.insert` in `scripts/*.py`; no `git` call in `lexhybrid/` or `scripts/`. And `tests/test_plan_references.py`: every test file and test function a P1 box names (a later box's once it is ticked) exists. Verify: the three files green; `bash scripts/validate.sh` exit 0.
- [x] **P1-Z** USER ACTION — delete `Reference/`. The model prints the three preconditions (P1-Y green, `validate.sh` exit 0, `test_port_map.py` green) and stops with: "P1 gate passed; reply 'delete reference' to proceed." The user deletes the folder (or tells the model to). Tick only with `--user-confirmed "<the user's words>"`; then `python3 scripts/plan_state.py next`.

**Gate:** `bash scripts/validate.sh --full` exit 0; `tests/test_reference_parity.py` green on all five fixtures; `tests/test_port_map.py` green; `actionlint` exit 0; `Reference/` gone.

### P2 — Correctness fixes, decoder deltas, legal configs

Purpose: fix the 18 recorded defects, add what a legal decoder needs (attention KV cache, one-pass prefill, EOS/pointer decoding, boundary-masked loss, slab-wise KL), and write the legal model configs. Runs: Mac, CPU. Port map §15 lists each defect with the reference line. Every box ships its own test; the parity fixtures from P1 stay green wherever a fix does not touch the fixture's configuration (state-dict load, `mlstm_*` unset, reference chunk sizes) and are updated with a note where it must change (P2-E RMSNorm upcast changes fp32 numerics below 1e-6 only; if a fixture moves more than 1e-6, record why in the note).

- [x] **P2-A** Config plumbing: `HybridBlock` strips the `mlstm_` prefix like `mamba3_` (defect 2) so `mlstm_gate_soft_cap`, `mlstm_input_gate_bias_init`, `mlstm_forget_gate_bias_init`, `mlstm_chunk_size` reach `mLSTMBlock`; add `mamba3_expand_factor` (defect 4) and `mlstm_chunk_size` (default 128) to `HybridConfig`. Tests: `tests/test_layers.py::test_mlstm_kwargs_reach_block`, `::test_mamba3_expand_factor_independent`.
- [x] **P2-B** `mLSTMBlock.post_model_init()` re-applies the gate-bias inits after `_init_weights` (defect 3); `HybridLanguageModel` calls it. Test: `tests/test_models.py::test_mlstm_gate_bias_survives_model_init` (i bias −10, f bias = configured value inside the built model).
- [x] **P2-C** TFLA: `apply_tfla(..., chunk_size)` takes the size from the block (defect 9); delete the dead `m_state` block; make the `_EXP_SAFE` fallback a config-time choice (`tfla_fallback: "error"|"sequential"`, default `error` — the model must be configured so the factorised path holds at init). Tests: `tests/test_numerics_fp64.py::test_tfla_fast_path_at_init[chunk=128, forget_bias=3.0]` (asserts no fallback), `::test_tfla_chunk_size_reaches_kernel`.
- [x] **P2-D** TFLA boundary masks inside the kernel (defect 15): apply `same` to the intra-chunk scores, gate the carried numerator/denominator with `carry_ok`, gate the chunk-end state with `end_ok` (mirroring `ssd_interface.py`), pad `f` with 1.0 and `seg` with the last real id; delete `mLSTMBlock._forward_segmented`; the sequential fallback (if enabled) also resets at boundaries; `tfla_reference.sequential_mlstm_fp64(..., doc_ids)` gains the reset. Test: `tests/test_numerics_fp64.py::test_tfla_packed_equals_per_document[chunk,L]` at 1e-6, boundaries mid-chunk and on chunk edges.
- [x] **P2-E** dtype: `RMSNorm` computes in fp32 and casts back (defect 10); `ssd_chunked_scan` allocates `state` in fp32 explicitly (defect 16); `mLSTMBlock.step` keeps `C`, `n` in fp32 like `Mamba3Block.step`. Tests: `tests/test_numerics_fp64.py::test_ssd_state_is_fp32_under_autocast`, `tests/test_layers.py::test_mlstm_step_state_stays_fp32`.
- [x] **P2-F** `ssd_chunked_scan` returns `(y, final_state)` with `seg` padded by the last real id and the `.item()` sync removed (defect 7); `Mamba3Block.forward(x, cache, doc_ids)` writes `ssm_state`, `conv_state` (last `conv_size−1` pre-conv columns), `seen`, `B_prev`/`x_prev`, `angle_state` (fp64, exposed from `cumulative_angles`) when `cache is not None`. Test: `tests/test_numerics_fp64.py::test_ssd_final_state_matches_oracle[L=1,63,64,65,200; docs on/off]` and `tests/test_layers.py::test_mamba3_forward_then_step_equals_step_only`.
- [x] **P2-G** TFLA returns `(h, C_final, n_final)`; `mLSTMBlock.forward` writes `C`, `n`, `seen` when `cache is not None`. Test: `tests/test_layers.py::test_mlstm_forward_then_step_equals_step_only`.
- [x] **P2-H** `AttentionBlock`: per-document RoPE positions from `doc_ids` (restart at each boundary), `_rope_tables` rebuilt with `self.rope_theta` (defect 8), config default `rope_theta 500000`. Test: `tests/test_layers.py::test_attention_positions_restart_per_document`.
- [x] **P2-I** `AttentionBlock.step(x_t, cache)` + `allocate_inference_cache(batch, device, dtype, max_seq_len)` with `{"k": (B,H,Lmax,Dh), "v": (B,H,Lmax,Dh), "seen": int}`; RoPE at position `seen`; attend over `[:seen+1]`; `supports_step = True`; `HybridLanguageModel.allocate_inference_cache(..., max_seq_len)`; `forward` writes K/V into the cache when given (defect 5). Tests: `tests/test_decoding.py::test_attention_step_matches_forward`, `::test_reorder_cache_with_attention`.
- [x] **P2-J** Packed training attention via `torch.nn.attention.flex_attention` with a block mask from `doc_ids` (defect 18); dense-mask path kept as the CPU/reference fallback. Test: `tests/test_layers.py::test_flex_block_mask_equals_dense` (marked `linux_only`: runs in CI on the Inductor CPU backend; the CUDA variant is in `tests/test_gpu.py`, run by P4-J1).
- [x] **P2-K** One-pass `HybridLanguageModel.prefill(hidden, caches)` = each layer's forward with its cache, then `lm_head(final_norm(hidden[:, -1]))`. Tests: `tests/test_decoding.py::test_prefill_equals_uncached[fp32 logits 1e-5; bf16 argmax]` for the hybrid, the Transformer and the legacy-free base pattern.
- [x] **P2-L** Loss mask at document boundaries (defect 6): `shift_labels[doc_ids[:,1:] != doc_ids[:,:-1]] = -100`, EOS supervised, `n_supervised_tokens` logged. Test: `tests/test_models.py::test_loss_masks_boundary_keeps_eos` (row starting mid-document included).
- [x] **P2-M** Init then tie (defect 17): `_init_weights` runs before tying; `tie_word_embeddings` default `True`; `get_num_params(non_embedding=True)` = total − embedding − (lm_head if untied); fingerprint prints `params=` and `params_nonembed=`. Test: `tests/test_models.py::test_param_counts_tied_untied`.
- [x] **P2-N** `lexhybrid/decoding/generate.py`: uncached reference `greedy`, `sample`, `beam_search` with EOS termination and repetition penalty (the reference tested its cached beam against a dropped script; this is the new reference). Test: `tests/test_decoding.py::test_uncached_beam_stops_on_eos`.
- [x] **P2-O** Cached `greedy_cached`/`sample_cached` with EOS stop, repetition penalty, `max_new_tokens`, batch > 1 with per-row finished flags. Test: `tests/test_decoding.py::test_cached_greedy_equals_uncached`.
- [x] **P2-P** `beam_search_cached` with EOS handling (finished hypotheses kept and length-normalised), `length_penalty` applied, `reorder_cache` over attention caches (defect 11). Test: `tests/test_decoding.py::test_cached_beam_equals_uncached_with_eos`.
- [x] **P2-Q** `lexhybrid/decoding/pointer_constraints.py`: `SpecialTokens` registry (decision 7 names) and a `PointerFSM` that, given the retrieved set `{(c, [s...])}`, masks `<|cK|>` ids not retrieved and `<|sJ|>` ids not in the current chunk, and requires `<|q|>`/`<|cite|>` before a `<|cK|>`. Test: `tests/test_decoding.py::test_pointer_fsm_never_emits_unretrieved_pointer` (random logits, 1,000 steps).
- [x] **P2-R** `best_of_n(model, prompt, n, scorer)` hook in `decoding/generate.py` (n cached samples, returns the scorer's argmax; the P7 verifier is the scorer). Test: `tests/test_decoding.py::test_best_of_n_picks_scorer_max`.
- [x] **P2-S** `lexhybrid/models/mtp_head.py`: multi-token-prediction head (`mtp_n=2`, `mtp_loss_weight`, off by default), sharing the tied embedding. Test: `tests/test_models.py::test_mtp_head_off_is_bit_identical` and `::test_mtp_loss_backward`.
- [x] **P2-T** `lexhybrid/training/distill.py`: `slab_ce_kl(hidden, labels, teacher_hidden, lm_head, teacher_lm_head, alpha, T, slab=512)` under `torch.utils.checkpoint` (never materialises `(B,L,V)`); `teacher_forward_packed(teacher, input_ids, doc_ids)` with per-document `position_ids` and a block-diagonal 4-D mask; label mask shared with CE. Tests: `tests/test_training.py::test_slab_kl_equals_full_kl_small`, `::test_teacher_position_ids_isolation` (tiny random `Qwen2`-architecture model from `transformers` config; doc i+1 logits equal the standalone run at 1e-3), `::test_peak_memory_scales_with_slab_not_row` (CPU allocator counts).
- [x] **P2-U** Lightning module for pretraining (`PretrainLightningModule`): CE + optional KL, WSD, β2 anneal, manual clip, `compile_model` key honoured, NaN guard that skips the optimizer step instead of returning a leaf tensor, DDP strategy built as `DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)` in `scripts/train_pretrain.py`, `val_check_interval = val_every_opt_steps × accumulate_grad_batches` computed in the script and asserted (defect 12); resume from `ckpt_path` when the wrapper passes one. Tests: `tests/test_training.py::test_val_check_interval_product`, `::test_nan_guard_skips_step`.
- [x] **P2-V** `configs/model/hybrid_legal_base.yaml` (decision 10; declares every flag so overrides can name them), `configs/model/transformer_legal_base.yaml` (15 attention layers, `pre_rms`, same LR/warmup/clip; delta stated in a comment), `configs/model/hybrid_legal_legacy.yaml` (9×mamba + 3×mlstm, `scan_impl legacy`, `tfla_impl legacy`, decision 17). Test: `tests/test_config_invariants.py::test_legal_yamls_compose_and_pin_operators`.
- [x] **P2-W** Screen variants `hybrid_legal_attn0.yaml` (9×M3+3×L), `hybrid_legal_attn4.yaml` (5×M3+4×A+3×L, positions 3,6,9,11), `hybrid_legal_ds64.yaml`, `hybrid_legal_mtp.yaml`; `tests/test_config_invariants.py::test_param_bands` records each config's non-embedding and total counts into `analysis/param_counts.md` and asserts: base vs Transformer within ±2.5% non-embedding; every screen variant's delta vs base printed and ≤ 3%; legacy vs base within ±2.5%, delta reported (pre-registered ≤ 1%; measured +2.36%; decision 20). (Expected from the module definitions: base 104.1M; Transformer +2.0%; attn0 +2.8%; attn4 −2.8%; ds64 −0.75%.)
- [x] **P2-X** `scripts/screen_arms.py`: `Arm(config, overrides, seed, isolates, expect, walltime)` table for S0–S6 (§P5), `list|env|verify [--full]`; `verify` builds each arm through `HybridConfig.from_hydra` and asserts its `expect` tokens appear in the ARCH fingerprint; `--full` asserts the parameter band. Test: `tests/test_scripts.py::test_screen_arms_verify_reduced`.
- [x] **P2-Y** `torch.compile` smoke: `tests/test_models.py::test_compile_forward_chunk128` compiles `hybrid_legal_base` at reduced size on the Inductor CPU backend and asserts logits equal eager at 1e-4 and the TFLA fast path is taken (marked `linux_only`: runs in CI; the CUDA variant runs in P4-J1).
- [x] **P2-Y1** `tests/test_gpu.py` (marked `cuda`: SDPA flash vs math in bf16, flex vs dense on GPU, SSD/TFLA bf16-autocast vs fp32, `torch.compile` vs eager at 1e-4, cached decode on GPU, peak-memory sanity) and `tests/test_multigpu.py` (marked `multigpu`: 2-process DDP step with `DDPStrategy(find_unused_parameters=False)`, gradients identical across ranks). Both collect and skip with their reason on the Mac and in CI.
- [x] **P2-Z** Gate: `bash scripts/validate.sh` exit 0; `tests/test_decoding.py` green (cached = uncached in fp32, argmax-identical in bf16, all three families incl. attention); `tests/test_numerics_fp64.py::test_tfla_packed_equals_per_document` green; `analysis/param_counts.md` written; `python3 scripts/plan_state.py next`.

**Gate:** validate exit 0; decode identity; packed-TFLA equality; parameter table written.

### P3 — Data pipeline at smoke scale

Purpose: every data component exists, is tested on fixtures, and runs end to end on ≤ 1 GB of downloads. Nothing here is at full scale; nothing needs the cluster. Runs: Mac. Precondition: P2-Z.

- [x] **P3-A** `lexhybrid/data/schema.py`: `Document(id, source, jurisdiction, doc_type, licence, commercial_safe, valid_from, valid_to, citation_id, url, sha256, retrieved_at, text, sections: list[Section(label, absatz, satz_idx, randnummer, text)])` as a dataclass with `to_json`/`from_json`, and `Docs/DOCUMENT_SCHEMA.md` describing every field and the `citation_id` grammar (`BGB §573 Abs. 2 Nr. 1`, `ABGB §1295`, `OR Art. 97`, `BGH VIII ZR 12/20 Rn. 14`). Test: `tests/test_data.py::test_document_roundtrip`.
- [x] **P3-B** `Docs/CORPUS_LICENCE_REGISTER.md` (one row per source: name, jurisdiction, doc types, licence, commercial-safe, gated?, URL, access method, notes) and `lexhybrid/data/corpus/licences.py` (`LICENCES` table, `is_commercial_safe`, `require_known`). `build_corpus` refuses `licence=unknown`. Test: `tests/test_data.py::test_unknown_licence_refused`.
- [x] **P3-C** `lexhybrid/data/tokenizer.py`: loads `Qwen/Qwen3-1.7B-Base`'s tokenizer, adds decision 7's specials, exposes `eos_id = 151643`, `pointer_ids()`; asserts `len(tokenizer) ≤ 151936`. Test: `tests/test_data.py::test_tokenizer_specials_fit_vocab` (downloads ~10 MB once; `HF_HOME=data/hf`).
- [x] **P3-D** Fertility measurement (R4): write the prediction first into `analysis/tokenizer_fertility.md` ("Qwen3 < 1.8 tokens/word on a 1M-word DACH legal sample; GPT-2 > 2.2"), then `scripts/measure_fertility.py` over the P3-F..M fixtures plus a public sample, tokens/word per tokenizer and per source; outcome recorded beside the prediction.
- [x] **P3-E** `lexhybrid/data/corpus/collectors/base.py`: `class Collector(Protocol): name, licence, jurisdiction; iter_documents(limit: int | None) -> Iterator[Document]`; shared `http_get` with retry/backoff and rate limit; `sha256`; `lexhybrid/data/corpus/manifest.py` writes `data/manifests/<source>.jsonl` (one line per document: every schema field but `text`). CLI `python3 -m lexhybrid.data.corpus.collectors.<src> --limit N --out data/raw/<src>`. Every collector box below ships (a) an offline parser test on its fixtures and (b) a `network`-marked live smoke (`--limit 2`) that the weekly CI `network` job runs. Test: `tests/test_data.py::test_manifest_roundtrip`.
- [x] **P3-E1** Dependencies for P3 in a box of their own (CLAUDE.md rule m): `requests` declared as a direct dependency (it was transitive), `datasketch` for P3-S (pulls `scipy`); `uv lock`, `uv sync --locked`, `scripts/check_env.py` green; the lock diff in the note. `flair` (P3-R) is decided in P3-R: current flair requires transformers < 5. Verify: `uv lock --check` exits 0; `bash scripts/validate.sh` exits 0.
- [x] **P3-F** Collector GII (gesetze-im-internet.de: federal statutes, XML, amtliches Werk §5 UrhG; captures `Stand` date, §, Absatz, Satz). 3 fixture records under `tests/fixtures/collectors/gii/`; offline parser test; `--limit 20` run exits 0; `licence`, `jurisdiction=DE`, `valid_from` non-null.
- [x] **P3-G** Collector RII (rechtsprechung-im-internet.de: federal court decisions; Tenor/Tatbestand/Gründe; Randnummern). Same acceptance as P3-F.
- [x] **P3-H** Collector OLDP (openlegaldata.io API, ODbL; court, date, ECLI). Same acceptance; `licence=ODbL-1.0`.
- [x] **P3-I** Collector EUR-Lex (German language, CC BY 4.0 reuse notice; CELEX id, Artikel). Same acceptance; `jurisdiction=EU`.
- [x] **P3-J** Collector RIS (Austrian Rechtsinformationssystem, CC BY 4.0; Bundesrecht + Judikatur; Inkrafttreten dates). Same acceptance; `jurisdiction=AT`.
- [x] **P3-K** Collector Fedlex (Swiss federal law, open; SR number, Artikel; versions). Same acceptance; `jurisdiction=CH`.
- [x] **P3-L** Collector BGer (Swiss Federal Supreme Court decisions; CC BY 4.0 as published; Erwägungen). Same acceptance; `jurisdiction=CH`.
- [x] **P3-M** Collector DIP (Bundestag documentation, open; Drucksachen). Same acceptance; `doc_type=parliament`.
- [x] **P3-N** Collector FineWeb-2 `deu_Latn` (HF `datasets` streaming, ODC-By 1.0; general-German replay). `--limit 20`; `doc_type=general`, `commercial_safe=true`.
- [x] **P3-O** Collector MultiLegalPile NC subsets (HF, gated; `research_only=true`, `commercial_safe=false`). If `HF_TOKEN` is unset locally, fixtures only and a note; P4-B supplies the cluster token.
- [x] **P3-P** Eval-set fetchers `lexhybrid/data/evalsets/{gerlerb,gerdalir}.py` (retrieval queries/qrels; eval only; licence recorded). `--limit 20` run; tests on fixtures.
- [x] **P3-Q** Eval-set fetchers `lexhybrid/data/evalsets/{legalquad,gerlayqa}.py` (QA; eval only; GerLayQA explicitly excluded from training). Same acceptance.
- [x] **P3-R1** Scrub environment in a box of its own (rule m; decision 22): `envs/scrub/` is a separate uv project (its own `pyproject.toml`, `.python-version` 3.11 and `uv.lock`) with `flair` for `flair/ner-german-legal`; `scripts/scrub_ner_worker.py` runs in it and imports nothing from `lexhybrid` (JSONL documents in, entity spans out); `scripts/check_env.py --scrub` checks `envs/scrub/.venv` against its lock; the main `.venv` never imports flair. If flair does not install cleanly there, record why and use a transformers-5-compatible model trained on the same German LER data, evaluated on its test split first. Verify: `uv lock --check --project envs/scrub` exits 0; `bash scripts/validate.sh` exits 0.
- [x] **P3-R** `lexhybrid/data/corpus/scrub_ler.py`: German-LER scrub with `flair/ner-german-legal` run by P3-R1's worker (PER, RR, AN, STR → typed placeholders), licence of the model verified and written into the register, scrub log (`data/manifests/scrub_<source>.jsonl` with hashed entities), throughput measured (docs/s on the Mac) and recorded. Test: `tests/test_data.py::test_scrub_replaces_person_names` on fixtures.
- [x] **P3-S** `lexhybrid/data/corpus/dedup.py`: `datasketch` MinHash, 5-gram shingles, 128 permutations, Jaccard ≥ 0.8, exact-hash pre-pass. Test: `tests/test_data.py::test_dedup_catches_planted_near_duplicate`.
- [x] **P3-T** `lexhybrid/data/corpus/changelog.py`: provisions amended since a date (GII `Stand`, RIS Inkrafttreten, Fedlex versions) → `data/manifests/changelog.jsonl`; this is P9's currency subset. Test: `tests/test_data.py::test_changelog_from_fixture_versions`.
- [x] **P3-U** `lexhybrid/data/packing.py`: EOS-delimited packing (`<|endoftext|>` after every document, `doc_ids` incremented after the EOS) into parquet shards with `input_ids`, `doc_ids`, `source`, `licence`, `commercial_safe` at row lengths 4,096 and 8,192 (`--row-len`); leftovers carried to the next row, not dropped. Test: `tests/test_data.py::test_packing_roundtrip_doc_ids` (decode each document back; boundaries where expected).
- [x] **P3-V** `lexhybrid/data/datasets.py`: by-document validation split (random 0.1%, minimum 2,000 documents, asserted), mixture yaml `configs/dataset/mixture_pretrain.yaml` (legal 70 / general 30 by tokens; per-source weights; arm `commercial_safe` vs `research`), `configs/dataset/legal_smoke.yaml`. Tests: `tests/test_data.py::test_val_split_by_document`, `::test_no_research_only_in_commercial_safe_shard` (also run by `validate.sh`).
- [x] **P3-W** Probes `lexhybrid/data/probes/{mqar,statute_recall}.py`: MQAR generator (reference `configs/dataset/mqar.yaml` semantics) and statute-recall (given a § in context, reproduce Absatz 2 verbatim; scored exact-match and char-F1). Tests on fixtures; `scripts/evaluate_probes.py` runs both on any checkpoint.
- [x] **P3-X** Probe `lexhybrid/data/probes/multihop.py`: two-hop cross-reference questions built from statutes that cite another § (the answer needs both passages); 200 items from fixtures; scored by exact-match. Test: `tests/test_data.py::test_multihop_items_need_two_passages`.
- [x] **P3-Y** `lexhybrid/data/sft_format.py`: the RAG prompt as ONE document (no `<|endoftext|>` inside): `<|question|> … <|passage|><|c1|> sentence-numbered text … <|/passage|> … <|answer|>`; target sentences with `<|q|><|cK|><|sJ|>` and `<|cite|><|cK|>` pointers, ending in `<|endoftext|>`; `<|unanswerable|>` items; render/parse round-trip. Test: `tests/test_data.py::test_sft_format_roundtrip`.
- [x] **P3-Z** Gate: `bash scripts/validate.sh` exit 0; a 100-row 4,096 shard built from the fixtures round-trips through `hybrid_legal_base` on CPU (`scripts/train_pretrain.py trainer=cpu_debug dataset=legal_smoke trainer.max_steps=2`); `analysis/tokenizer_fertility.md` has prediction and outcome; `Docs/CORPUS_LICENCE_REGISTER.md` has a row for every collector; `python3 scripts/plan_state.py next`.

**Gate:** validate exit 0; smoke shard trains 2 steps on CPU; fertility written; register complete.

### P4 — Cluster bring-up, corpus at scale, profiling

*Pre-registered prediction (R4): the ported `ref_hybrid_m3` reproduces the reference's L=16,384 numbers within 10% (compiled chunk 128 ≈ 123 ms, 7.17 GB; uncompiled chunk 64 ≈ 680 ms); `hybrid_legal_base` at 4,096 rows trains uncompiled within 35 GB per GPU with the 1.7B teacher and within 60 GB with the 8B teacher; the flex-attention packed path is within 1.2× of the unpacked SDPA time.*

Purpose: reach the aisc cluster from this repo without git, build the environment, fetch weights, build the corpus at scale, and measure before training. Runs: Mac (ssh one-liners) + aisc `sbatch`. Port map §11–§14 (SLURM headers, env block, incidents).

- [x] **P4-A** USER ACTION — copy `scripts/slurm/cluster.env.example` to `scripts/slurm/cluster.env` (gitignored) and fill `CLUSTER_HOST` (ssh alias), `CLUSTER_REPO=/sc/home/<user>/lexhybrid`, `SCRATCH_ROOT=/sc/scratch/<user>/lexhybrid`; confirm `ssh $CLUSTER_HOST true` exits 0 from the Mac. The model stops here until the user confirms.
- [x] **P4-B** USER ACTION — accept the MultiLegalPile terms on huggingface.co and place the token at `~/.hf_token` (mode 600) on the cluster; jobs export `HF_TOKEN=$(cat ~/.hf_token)` when the file exists.
- [x] **P4-C** USER ACTION — confirm scratch quota: paste `df -h /sc/scratch/$USER` and `du -sh $HOME` outputs; the plan needs ≥ 300 GB scratch free.
- [x] **P4-D** `scripts/sync_to_cluster.sh` (`rsync -az --delete --exclude-from=.rsync-exclude ./ $CLUSTER_HOST:$CLUSTER_REPO/`; writes `.sync_stamp` = ISO date + tree hash before syncing) and `scripts/sync_from_cluster.sh <subdir>` (pulls only `logs/` or `analysis/` into `cluster/<subdir>/`, never `--delete`, never `outputs/`); first sync done. Verify: `ssh $CLUSTER_HOST "ls $CLUSTER_REPO/scripts/slurm"` lists the wrappers.
- [x] **P4-E** `scripts/slurm/setup_env.sh` (port map §11: CPU job, `--exclude=ga03`, installs `uv` if missing, `uv sync --locked` — torch `2.11.0+cu128` comes from the lock — then `scripts/check_env.py`) submitted with `sbatch`; record the job id.
- [x] **P4-F** Verdict on the sbatch env job: pull the log; `check_env.py` must report Python 3.11, `torch 2.11.0+cu128`, CUDA 12.8 and every package equal to `uv.lock`; record its summary in evidence (the lock is the pin, so no separate requirements file).
- [x] **P4-G** `scripts/slurm/fetch_hf.sh` (CPU job, `HF_HUB_OFFLINE=0`, downloads `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-8B-Base`, `Qwen/Qwen3-0.6B-Base`, `flair/ner-german-legal`, `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, the NLI model into `$HF_HOME`) submitted with `sbatch`.
- [x] **P4-H** Verdict on the sbatch fetch job: `du -sh $HF_HOME/hub` recorded; every model directory present.
- [x] **P4-I** `scripts/slurm/preflight.sh` (CPU job: `screen_arms.py verify --full`; `pytest -m "not cuda and not slow and not reference"`; ARCH line printed for every arm) submitted with `sbatch`.
- [x] **P4-J** Verdict on the sbatch preflight job: "PRE-FLIGHT PASSED" in the log; test count recorded.
- [x] **P4-J1** `scripts/slurm/gpu_tests.sh` (1 H100: `pytest -m cuda tests/`, plus `check_env.py`) submitted with `sbatch`; record the job id.
- [x] **P4-J2** Verdict on the sbatch GPU-test job: every `cuda` test passed (count and log path in evidence); a failure is fixed before any screen job is submitted.
- [x] **P4-J3** `scripts/slurm/multigpu_tests.sh` (2 H100: `pytest -m multigpu tests/`) submitted with `sbatch --gpus=2`; record the job id.
- [x] **P4-J4** Verdict on the sbatch multi-GPU job: DDP smoke passed with identical gradients across ranks; evidence records the job id and log path.
- [x] **P4-K** `scripts/slurm/build_corpus_array.sh` (CPU array, one task per collector, `--limit` unset, writes `data/raw/<src>` and manifests on scratch) submitted with `sbatch --array`.
- [ ] **P4-L** Verdict on the sbatch corpus array: per-source document and token counts; then `scripts/slurm/pack_corpus.sh` (scrub → dedup → pack at 4,096 and 8,192, commercial-safe and research arms as separate shard sets) submitted with `sbatch` (evidence carries both job ids).
- [ ] **P4-M** Verdict on the sbatch pack job: `analysis/corpus_manifest.md` from the template with tokens per source / jurisdiction / licence / arm, dedup removal rate, scrub entity counts; token budget for P6 set to `min(5B, available commercial-safe)`.
- [x] **P4-N** `scripts/slurm/probe_ckpt_size.sh` (10-step `hybrid_legal_base` run on 1 GPU writing `last.ckpt` to scratch) submitted with `sbatch`.
- [x] **P4-O** Verdict on the sbatch probe job: checkpoint size in GB → storage budget row in §6 and `evidence.ckpt_gb`; probe outputs deleted.
- [x] **P4-P** `scripts/slurm/profile.sh` (port map §11 point() protocol: one length per process, fresh `TORCHINDUCTOR_CACHE_DIR`; arms: `ref_hybrid_m3` sanity at 16,384 chunk 64 / compiled chunk 128; `ref_transformer` at 16,384; `hybrid_legal_base` and `transformer_legal_base` inference at 2,048/8,192/16,384 and training at 2,048/4,096/8,192 with packed `doc_ids` (flex) and without) submitted with `sbatch`.
- [x] **P4-Q** Verdict on the sbatch profile job: `analysis/profile_results.md` (prediction vs outcome; chunk-size decision by the rule "128 unless it fails R1 or loses > 5% at 4,096"); `effective_chunk_size` checked in every row.
- [x] **P4-R** `scripts/slurm/equivalence.sh` (`check_operator_equivalence.py --chunk-sizes 64 128 --compile` on GPU) submitted with `sbatch`.
- [x] **P4-S** Verdict on the sbatch equivalence job: exit 0 at 1e-4; compile smoke on the cluster (P2-Y's test under CUDA) green.
- [x] **P4-T** `scripts/slurm/kd_memory_probe.sh` (4×H100 DDP, 20 steps each: 1.7B teacher at 4,096 and 8,192 rows; 8B teacher at 4,096 and 8,192; records s/step and peak GB per rank) submitted with `sbatch --gpus=4`.
- [x] **P4-U** Verdict on the sbatch KD probe: `decisions.teacher` set by the rule "8B if peak ≤ 70 GB at 4,096 rows and step time ≤ 1.6× the 1.7B step; else 1.7B"; numbers into `analysis/profile_results.md`.
- [x] **P4-V** `scripts/slurm/watch.sh` (port map §11: squeue/sacct/preemption/per-run checkpoint listing/log error grep; `source`-able; no python) and `tests/test_slurm_wrappers.py` (every wrapper: `--partition=aisc-batch`, `--account=aisc`, `--gpus` not `--gres`, `--exclude=ga03,gx17v1,gx13v1`, `--requeue` + `--open-mode=append` on training wrappers, `%A_%a` logs on arrays, `ARM` resolved inside, no `git` call, `cd "${SLURM_SUBMIT_DIR}"`).
- [ ] **P4-Z** Gate: P4-F, P4-J, P4-J2, P4-J4, P4-S verdicts green; `analysis/corpus_manifest.md` and `analysis/profile_results.md` written with job ids; `decisions.teacher` recorded; `bash scripts/validate.sh` exit 0; `python3 scripts/plan_state.py next`.

**RESULTS — measured 2026-09-30, jobs 2589361, 2589360.** gx07, 1×H100, bf16, batch 4, 10 timed iterations per point (22-point profile ladder; R1 at fp32). P4-Q; full tables in `analysis/profile_results.md`.
| arm | seed | val PPL | late-25% PPL | MQAR | statute-recall | multi-hop | s/step | peak GB |
|---|---|---|---|---|---|---|---|---|
| hybrid_legal_base train 4,096 packed, chunk 128 | – | – | – | – | – | – | 0.444 | 28.21 |
| hybrid_legal_base train 4,096 packed, chunk 64 | – | – | – | – | – | – | 0.649 | 26.86 |
| ref_hybrid_m3 fwd 16,384, chunk 128, compiled | – | – | – | – | – | – | 0.121 | 7.17 |
| ref_hybrid_m3 fwd 16,384, chunk 64, uncompiled | – | – | – | – | – | – | 0.571 | 7.19 |
*Prediction:* ported `ref_hybrid_m3` within 10% of the reference at L=16,384 (compiled chunk 128 ≈ 123 ms, 7.17 GB; uncompiled chunk 64 ≈ 680 ms); `hybrid_legal_base` at 4,096 rows within 35 GB (1.7B teacher) / 60 GB (8B teacher); flex packed path within 1.2× of unpacked SDPA → *Outcome:* PARTIAL — compiled point −1.6% / 7.169 GB; uncompiled 571 ms (−16%, outside the band); packed/unpacked 1.14/1.10/1.10 for the hybrid but 1.83/1.34/1.08 for the transformer; 8B teacher OOM at micro-batch 8 (job 2589362); 1.7B pending (P4-U).
*Gate:* PASS — "128 unless it fails R1 or loses > 5% at 4,096", chunk 128: R1 1.68e-6 ≤ 1e-4 and 444 vs 649 ms (31.5% faster), 128 kept.

**RESULTS — measured 2026-09-30, jobs 2589379, 2589362.** gx07, 4×H100, bf16-mixed, `hybrid_legal_base` + online KD, synthetic packed rows, 20 steps (15 timed), model gradient checkpointing off. P4-U; table in `analysis/profile_results.md` §4.
| arm | seed | val PPL | late-25% PPL | MQAR | statute-recall | multi-hop | s/step | peak GB |
|---|---|---|---|---|---|---|---|---|
| KD 1.7B, 4,096 rows, 8 × 2 × 4 | 42 | – | – | – | – | – | 4.144 | 80.60 |
| KD 1.7B, 8,192 rows, 4 × 4 × 4 | 42 | – | – | – | – | – | 9.186 | 70.74 |
| KD 8B, 4,096 rows, 8 × 2 × 4 | 42 | – | – | – | – | – | OOM | ≥ 82.7 |
| KD 8B, 8,192 rows, 4 × 4 × 4 | 42 | – | – | – | – | – | OOM | ≥ 81.7 |
*Prediction:* `hybrid_legal_base` at 4,096 rows trains uncompiled within 35 GB per GPU with the 1.7B teacher and within 60 GB with the 8B teacher → *Outcome:* REFUTED — 1.7B peaks at 80.60 GB; 8B does not fit (both without gradient checkpointing, which P6's recipe lists).
*Gate:* PASS (decision taken) — "8B if peak ≤ 70 GB at 4,096 rows and step time ≤ 1.6× the 1.7B step; else 1.7B", 8B ≥ 82.7 GB (OOM), above 70 → `decisions.teacher` = Qwen/Qwen3-1.7B-Base (provisional).
*Follow-up (job 2589417, gradient checkpointing on as in P6's recipe, at the user's request):* 1.7B 4.224 s / 30.11 GB, 8B 6.104 s / 43.18 GB at 4,096 rows; the same rule passes the 8B (43.18 ≤ 70, 1.44× ≤ 1.6×). **`decisions.teacher` = Qwen/Qwen3-8B-Base with gradient checkpointing** (2026-10-01; user: "choose whichever would give the best result, I am thinking of 8B").

**Gate:** env, preflight, GPU tests, multi-GPU tests and equivalence green; manifest and profile written; teacher decided by measurement.

### P5 — Pre-registered screen

*Pre-registered prediction (R4): S1 (attn-2) beats S0 (attn-0) on statute-recall and multi-hop by more than the bar and ties on val PPL; S2 (attn-4) ties S1 on everything and is not adopted (simplest-arm rule); S3 (d_state 64) is within the bar on PPL and worse on late-position PPL; S4 (MTP) is within the bar; S5 (legacy recipe) is worse on val PPL by more than 1.0; S6 (α=0, no KD) is worse on val PPL by more than the bar. Bar = 2× the two-seed SD of S1, floor 0.10 PPL.*

Purpose: choose the backbone configuration by measurement, one lever per arm. Runs: aisc, 1 H100 per arm, 12,000 steps, 2,048-token rows, micro-batch 8 × accum 2 (32,768 tokens/step), warmup 500, LR 4e-4, clip 1.0, WSD, `SAVE_TOP_K=0`, `VAL_EVERY` 4,000, seeds 42 and 43, teacher per `decisions.teacher`, commercial-safe shards.

| Arm | Config | Lever vs S1 | Parameter-matched? |
|---|---|---|---|
| S1 | `hybrid_legal_base` | control | — |
| S0 | `hybrid_legal_attn0` | no attention | no (+2.8%, delta reported) |
| S2 | `hybrid_legal_attn4` | four attention layers | no (−2.8%) |
| S3 | `hybrid_legal_ds64` | `mamba3_d_state 64` | no (−0.75%) |
| S4 | `hybrid_legal_mtp` | MTP head on | no (head params, reported) |
| S5 | `hybrid_legal_legacy` | incumbent recipe, legacy operators; PPL/probe-only; own walltime | no (+2.36%, delta reported; decision 20) |
| S6 | `hybrid_legal_base` + `distill.alpha=0` | no KD | yes |

Decision rule, written before the numbers: the **simplest** arm (S1 unless a lever is needed) advances unless another arm beats it on the two-seed mean val PPL by more than the bar **and** does not lose statute-recall or multi-hop by more than the bar; a lever that wins PPL but loses recall is recorded and not adopted. Every arm is scored on val PPL, late-25% PPL, MQAR, statute-recall, multi-hop, s/step, peak GB.

- [x] **P5-A** Write the prediction above and the bar rule into `pre_registered.P5` (`plan_state.py evidence P5 prediction=... bar_rule=...`) before any submission; `analysis/screen_results.md` created from the template with the prediction and an empty table.
- [x] **P5-B** `scripts/slurm/screen_array.sh` (port map §11 array header; `ARMS` env list; `ARM` resolved on the compute node by `screen_arms.py env`; `unset EXPERIMENT`; `SAVE_TOP_K=0`; greps the ARCH line at step 0 against the arm's `expect` tokens and exits non-zero on mismatch) and `scripts/slurm/train_pretrain_1gpu.sh` it delegates to. `tests/test_slurm_wrappers.py` extended.
- [ ] **P5-C** Submit S1 seeds 42 and 43 with `sbatch --array=0-1` (`ARMS="S1-s42 S1-s43"`); record both job ids; end the session.
- [ ] **P5-D** Verdict on the sbatch S1 jobs: val PPL per seed, two-seed SD, `bar_value = max(2×SD, 0.10)` written into this section and `pre_registered.P5.bar_value`; if the floor binds, say so.
- [ ] **P5-E** Submit S0, S2, S3, S4, S6 × seeds 42/43 with `sbatch --array=0-9` and S5 × 2 seeds separately with its own walltime (`sbatch --array=0-1 --time=24:00:00`); record job ids; end the session.
- [ ] **P5-F** Verdict on the sbatch arm jobs: per-arm, per-seed val PPL and late-25% PPL, s/step, peak GB into `analysis/screen_results.md`; requeued/failed arms noted with the `sacct` state.
- [ ] **P5-G** `scripts/slurm/eval_probes.sh` (1 GPU; `evaluate_probes.py` on every arm's `last.ckpt`: MQAR, statute-recall, multi-hop at 2,048) submitted with `sbatch`.
- [ ] **P5-H** Verdict on the sbatch probe job: probe columns filled; seed SD per column.
- [ ] **P5-I** Apply the decision rule verbatim; write RESULTS (prediction → outcome per arm), the chosen config into `decisions.backbone`, and the verdict; unexpected results are recorded, not re-run.
- [ ] **P5-Z** Gate: winner named by the rule with two seeds per arm; `analysis/screen_results.md` complete with every job id; `python3 scripts/plan_state.py next`.

**Gate:** decision by the pre-registered rule; results block with job ids.

### P6 — Full backbone pretraining

*Pre-registered prediction (R4): the winner reaches a lower commercial-safe val PPL than `transformer_legal_base` trained identically by less than one seed SD (parity, not a win); late-25% PPL at 8,192 rows improves after stage B by ≥ 3% relative; statute-recall at 8K ≥ its 2K value; both hybrid seeds finish within 10% of each other on val PPL.*

Purpose: the backbone the whole system stands on. Runs: aisc, 4×H100 DDP. Recipe: `decisions.backbone` config; teacher per `decisions.teacher`; α=0.5, T=2; WSD, warmup 2,000, LR 4e-4, clip 1.0, bf16-mixed, gradient checkpointing, micro-batch 8 × accum 2 × 4 GPUs at 4,096 rows (≈ 262K tokens/step); `SAVE_TOP_K=1` + `last.ckpt`, `every_n_train_steps 1000`; `val_every_opt_steps 1000` (interval = 1000 × accum); requeue resumes from `last.ckpt`; `outputs/` on scratch. Stage A: 90% of the token budget at 4,096 rows. Stage B: the last 10% at 8,192 rows (micro-batch 4 × accum 4), resumed from stage A's `last.ckpt` with the WSD decay re-anchored to the combined horizon. **After P6-E, set `current_phase` to P7 and work P7 locally while the jobs run; job-bound boxes P6-G..M may be ticked from P7.**

- [x] **P6-A** `scripts/slurm/train_pretrain_4gpu.sh` (port map §11 header with `--gpus=4`, `--mem=256G`, `--cpus-per-task=32`, `--time=3-00:00:00`, `--requeue`, `--open-mode=append`; `CKPT=$OUT/checkpoints/last.ckpt` passed as `+resume_from_checkpoint=` when present; `NUM_GPUS` check against `torch.cuda.device_count()`; `du -sh $SCRATCH_ROOT/outputs` printed first) and `scripts/train_pretrain.py` (Hydra entry: `PretrainLightningModule`, `DDPStrategy(...)`, the `val_check_interval` product, `write_run_metadata`, ARCH print). `tests/test_slurm_wrappers.py::test_pretrain_wrapper_resumes_from_last` (static check of the wrapper text).
- [ ] **P6-B** USER ACTION — decide whether a `research_only` arm (commercial-safe + NC shards) also trains (doubles P6 compute). Record the answer in `decisions.nc_arm`.
- [ ] **P6-C** Submit hybrid seed 42, stage A, with `sbatch --gpus=4 scripts/slurm/train_pretrain_4gpu.sh`; record the job id; artifact `P6.hybrid.s42.stageA_dir`.
- [ ] **P6-D** Submit hybrid seed 43, stage A, with `sbatch --gpus=4`; record; artifact.
- [ ] **P6-E** Submit `transformer_legal_base` seed 42, stage A, with `sbatch --gpus=4`; record; artifact; then `python3 scripts/plan_state.py phase P7` and end the session.
- [ ] **P6-F** If `decisions.nc_arm` is yes: submit the research arm with `sbatch --gpus=4`; else tick with `--evidence job=skipped reason=decisions.nc_arm=no`.
- [ ] **P6-G** Verdict on the sbatch hybrid s42 stage-A job: final val PPL, late-25% PPL, tokens seen, wall, preemptions (`sacct`), ARCH line; RESULTS row.
- [ ] **P6-H** Verdict on the sbatch hybrid s43 stage-A job: same.
- [ ] **P6-I** Verdict on the sbatch Transformer stage-A job: same.
- [ ] **P6-J** Submit stage B for the three runs with `sbatch --gpus=4 --dependency=afterok:<A-job>` where still needed (or directly when A finished); record three job ids.
- [ ] **P6-K** Verdict on the sbatch stage-B jobs: val PPL at 8,192 rows, late-25% PPL before/after stage B; artifacts `P6.<arm>.<seed>.last_ckpt`.
- [ ] **P6-L** `scripts/slurm/verify_handoff.sh` (CPU job; `verify_handoff.py` on each `last.ckpt` against its yaml; PASS = 0 missing / 0 unexpected) submitted with `sbatch` ×3 and verdict recorded in the same box (evidence carries the three job ids and PASS lines).
- [ ] **P6-M** `scripts/slurm/eval_probes.sh` at 8,192 (MQAR, statute-recall, multi-hop) on the three stage-B checkpoints submitted with `sbatch`; verdict in the same box.
- [ ] **P6-N** `analysis/backbone_results.md` from the template: seed table (2 hybrid seeds with SD, 1 Transformer — **descriptive only, no claim**), probes, prediction → outcome, job ids, handoff PASS lines.
- [ ] **P6-Z** Gate: three stage-B checkpoints with handoff PASS; `analysis/backbone_results.md` complete; `python3 scripts/plan_state.py next` (if `current_phase` is already P7, tick this box with `--force` and record it).

**Gate:** three verified backbones; results block; no claim from one seed.

### P7 — Retrieval, renderer, verifier

*Pre-registered prediction (R4): hybrid BM25 + dense + rerank reaches nDCG@10 ≥ 0.45 on GerLeRB and ≥ 0.35 on GerDaLIR; the NLI verifier at the pre-registered threshold reaches ≥ 0.90 precision on the 100-pair calibration set.*

Purpose: the system around the decoder, built from the P3 schema, while P6 trains. Runs: Mac for code and tests; aisc 1 GPU for the dense index and the retrieval eval. Precondition: P3-Z (and P6-E submitted).

- [x] **P7-A** `lexhybrid/retrieval/chunking.py::chunk_statute`: § / Artikel units with hierarchy (Gesetz → Abschnitt → § → Absatz → Satz), `valid_from/to`, sentence ids `sJ`; from `Document.sections`. Test: `tests/test_retrieval.py::test_statute_chunks_carry_hierarchy_and_dates`.
- [x] **P7-B** `chunk_case_law`: Tenor / Tatbestand / Entscheidungsgründe, then 300–500-token windows with Randnummer anchors. Test: `tests/test_retrieval.py::test_case_windows_keep_randnummer`.
- [x] **P7-C1** Dependencies for P7 in a box of their own (CLAUDE.md rule m): `bm25s[stem]` (decision 14; the `stem` extra pulls `PyStemmer`, the Snowball German stemmer) in `pyproject.toml`; `uv lock`, `uv sync --locked`, `scripts/check_env.py` green (its key imports gain `bm25s` and `Stemmer`); the lock diff in the note. The cluster `.venv` differs from the lock until `setup_env.sh` runs again, so that job goes first when the cluster is back (every GPU job runs `check_env.py`). Verify: `uv lock --check` exits 0; `bash scripts/validate.sh` exits 0.
- [x] **P7-C** `lexhybrid/retrieval/bm25.py` (`bm25s`, German stemming/stopwords, index build + query) and `scripts/build_index.py --kind bm25`. Test: `tests/test_retrieval.py::test_bm25_finds_planted_passage`.
- [ ] **P7-D** `lexhybrid/retrieval/dense.py` (`BAAI/bge-m3` embeddings, FAISS or numpy index) and `scripts/slurm/build_dense_index.sh` (1 GPU) submitted with `sbatch`; verdict in the same box (index size, docs/s).
- [ ] **P7-E** `lexhybrid/retrieval/fusion.py` (reciprocal rank fusion) and `rerank.py` (`BAAI/bge-reranker-v2-m3`, top 50 → 3–8 chunks ≤ 4K tokens). Test: `tests/test_retrieval.py::test_rrf_and_rerank_orderings`.
- [ ] **P7-F** `lexhybrid/retrieval/filters.py` (jurisdiction DE/AT/CH/EU; valid-on date; doc_type) and `index.py` (manifest of index versions; nightly re-sync stub `scripts/resync_index.py` that only records what would change). Test: `tests/test_retrieval.py::test_valid_on_filter`.
- [ ] **P7-G** `lexhybrid/renderer/render.py`: `<|q|><|cK|><|sJ|>` → the exact sentence text in quotes with its `citation_id` label; `<|cite|><|cK|>` → `(BGB §573 Abs. 2 Nr. 1)`; appends "Hinweis: Dies ist eine rechtliche Information, keine Rechtsberatung."; refuses pointers outside the retrieved set. Tests: `tests/test_renderer.py::test_quote_is_verbatim`, `::test_notice_always_appended`.
- [ ] **P7-H** `lexhybrid/verifier/pointers.py` (every pointer resolves) and `quotes.py` (rendered quote equals source after NFC + whitespace normalisation). Test: `tests/test_verifier.py::test_pointer_and_quote_checks_adversarial` (synthetic suite incl. off-by-one sentence, altered Unicode, missing chunk).
- [ ] **P7-I** `lexhybrid/verifier/nli.py`: pre-register the entailment threshold (0.5 on the entailment probability) in `pre_registered.P7` **before** running `scripts/calibrate_nli.py` on a 100-pair hand-built set (`data/manifests/nli_calibration.jsonl`); report precision/recall at the pre-registered and at the best threshold; the pre-registered one ships unless precision < 0.85.
- [ ] **P7-J** `lexhybrid/verifier/verifier.py`: sentence-level pipeline (pointer → quote → NLI → party-identity rule: reject any answer containing a PER placeholder or a name matched by the LER model; "who was the party" → refusal); failing sentences removed; a failing main claim → "Die bereitgestellten Quellen beantworten diese Frage nicht."; verifier log = citation-precision metric. Tests: `tests/test_verifier.py::test_party_identity_refusal`, `::test_no_scrub_log_entity_in_index_chunks`.
- [ ] **P7-K** `scripts/slurm/eval_retrieval.sh` (nDCG@10, R@5/10/20, MRR per jurisdiction on GerLeRB and GerDaLIR; BM25 / dense / fused / reranked arms) submitted with `sbatch`; verdict in the same box → `analysis/retrieval_results.md` (prediction → outcome).
- [ ] **P7-L** `scripts/answer.py`: end-to-end CLI (question → retrieve → build the P3-Y prompt → decode with pointer FSM and best-of-n → render → verify → print) runnable on CPU with a tiny checkpoint. Test: `tests/test_pipeline.py::test_answer_cli_end_to_end_tiny`.
- [ ] **P7-Z** Gate: verifier synthetic suite 100%; renderer notice test green; `analysis/retrieval_results.md` written with job ids; `bash scripts/validate.sh` exit 0; `python3 scripts/plan_state.py phase P8` once P6-Z is also ticked.

**Gate:** verifier and renderer tests green; retrieval numbers written.

### P8 — Post-training

*Pre-registered prediction (R4): after SFT alone, citation precision (pointer + NLI) ≥ 0.85 and truncation 0%; after DPO ≥ 0.95, so GRPO is not triggered; the hybrid and the Transformer are within one seed SD of each other on citation precision and refusal F1; Qwen3-0.6B through the same pipeline is ahead on multi-hop by more than the SD.*

Purpose: teach the pointer format, refusal and stopping. Runs: aisc 4×H100 (SFT/DPO), 1 H100 (vLLM). Precondition: P6-Z and P7-Z. Seeds 42/43/44 for hybrid and Transformer; 1 seed for Qwen3-0.6B.

- [ ] **P8-A** USER ACTION — approve ~65 GB scratch for `Qwen/Qwen3-32B` weights and the vLLM environment.
- [ ] **P8-B** `scripts/slurm/setup_vllm.sh` (CPU job; `.venv-vllm` separate from `.venv` because vLLM pins torch) and `scripts/slurm/vllm_smoke.sh` on `Qwen3-0.6B-Base` (1 GPU) submitted with `sbatch`; verdict in the same box; then `fetch_hf.sh MODELS=Qwen/Qwen3-32B`.
- [ ] **P8-C** `scripts/generate_synthetic_qa.py` (over real retrieved passages from P7; questions incl. 20–30% unanswerable / false-premise / party-identity; answers in the pointer format; verifier-filtered; open teacher only, R10) and `scripts/slurm/synth_qa.sh` submitted with `sbatch`; verdict in the same box (items kept / rejected by the verifier).
- [ ] **P8-D** `scripts/build_sft_data.py`: SFT shards from synthetic QA + LegalQuAD-style items (eval sets excluded), every target ends in `<|endoftext|>`, unanswerable share asserted in [0.2, 0.3], held-out dev split by question. Test: `tests/test_data.py::test_sft_shards_end_with_eos_and_unanswerable_share`.
- [ ] **P8-E** `lexhybrid/training/sft_module.py` (loss on answer tokens only; pointer tokens supervised) and `scripts/slurm/train_sft_4gpu.sh`; `scripts/train_sft_hf.py` for the Qwen3-0.6B baseline (same data, same steps). Tests: `tests/test_training.py::test_sft_masks_prompt_tokens`.
- [ ] **P8-F** Submit SFT hybrid seeds 42/43/44 with `sbatch --gpus=4` (three jobs); record ids.
- [ ] **P8-G** Submit SFT Transformer seeds 42/43/44 with `sbatch --gpus=4`; record ids.
- [ ] **P8-H** Submit SFT Qwen3-0.6B seed 42 with `sbatch --gpus=4`; record id; end the session.
- [ ] **P8-I** Verdict on the seven sbatch SFT jobs: dev loss, citation precision on dev (via `scripts/evaluate_citation.py`, P9-A may be pulled forward), truncation rate; table in `analysis/posttrain_results.md`.
- [ ] **P8-J** `scripts/build_dpo_pairs.py` (best-of-n samples from each SFT model; verifier pass vs fail pairs; coverage kept balanced) and `lexhybrid/training/dpo_module.py` (DPO/KTO). Test: `tests/test_training.py::test_dpo_loss_prefers_chosen`.
- [ ] **P8-K** Submit DPO for the seven arms with `sbatch --gpus=4`; verdict in the same box (citation precision before/after; refusal F1).
- [ ] **P8-L** GRPO decision by the pre-registered condition: if any hybrid seed's citation precision after DPO < 0.95, implement `lexhybrid/training/grpo_module.py` (verifier reward + coverage term + length penalty; stop rule: stop when precision ≥ 0.95 or after 2,000 steps, whichever first) and submit with `sbatch --gpus=4`; otherwise tick with `--evidence job=skipped reason=precision>=0.95`.
- [ ] **P8-M** USER ACTION — hand over the jurist annotation kit (`analysis/annotation_kit/`: 200–300 questions with retrieved passages and the three systems' answers, blinded; double-annotation instructions; agreement sheet). The user arranges grading; results return in P9-H.
- [ ] **P8-N** Truncation audit: `scripts/audit_truncation.py` over every post-trained model's dev decodes; must be 0% at the budget (all end on EOS); recorded per arm.
- [ ] **P8-Z** Gate: 3-seed SFT (+DPO) checkpoints for hybrid and Transformer, 1 for Qwen3-0.6B; truncation 0%; `analysis/posttrain_results.md` with prediction → outcome and job ids; `python3 scripts/plan_state.py next`.

**Gate:** seven post-trained models, 0% truncation, results block.

### P9 — Evaluation, benchmarks, writeup

*Pre-registered prediction (R4): the hybrid matches the Transformer within one seed SD on citation precision, quote fidelity (100% by construction for both) and refusal F1; it beats the retrieval-copy floor on citation-faithful answer accuracy at all 3 seeds; it decodes at ≥ 4× the Transformer's tokens/s at 32K context with flat memory; it loses to Qwen3-0.6B on multi-hop by more than the SD and to gpt-oss-120b on everything but cost, latency and memory; general-German PPL after post-training is within 5% of the backbone's.*

Purpose: the thesis table, with every pre-registered prediction next to its outcome. Runs: aisc + Mac.

- [ ] **P9-A** `scripts/evaluate_citation.py`: citation precision/recall (pointer resolves + NLI entails; recall against the annotated gold passages where available), quote fidelity (exact match after normalisation), refusal precision/recall/F1 on unanswerable and false-premise items, answer accuracy where gold exists; outputs per-question JSONL for the bootstrap. Test: `tests/test_eval.py::test_citation_metrics_on_synthetic`.
- [ ] **P9-B** Currency subset (questions whose answer changed after the changelog date, P3-T) with the valid-on filter on/off; general-German PPL (forgetting) on a held-out FineWeb-2 shard; probes at 8K. `scripts/slurm/eval_quality.sh` over all seven systems submitted with `sbatch`; verdict in the same box.
- [ ] **P9-C** Efficiency at 8K / 32K / 128K (pre-registered as efficiency-only for a model trained at 8K): cached decode tokens/s, time-to-first-token, memory per session, best-of-n cost, plus the P4 profiler ladder re-run on the final checkpoint; `scripts/slurm/eval_efficiency.sh` submitted with `sbatch` (one length per process); verdict in the same box.
- [ ] **P9-D** Baselines: retrieval-copy floor (top-ranked passage verbatim) and the legacy-operator ablation (`hybrid_legal_legacy` after P5 at S5's checkpoint, PPL/probe-only, no latency); both scored by P9-A where applicable.
- [ ] **P9-E** Large open ceiling: `gpt-oss-120b` in vLLM with the same passages and the same pointer prompt, `scripts/slurm/eval_ceiling.sh` submitted with `sbatch --gpus=4`; if it does not fit or load, tick with `--evidence job=skipped reason=...` and say so in the writeup.
- [ ] **P9-F** USER ACTION — optional: provide a frontier API key for a second ceiling (evaluation only, never training data, R10); if declined, record it.
- [ ] **P9-G** `scripts/bootstrap_compare.py` runs for every pair (hybrid vs Transformer, vs Qwen3-0.6B, vs floor) per seed, 1,000 resamples by question; seed SD computed across seeds; both reported in one table; the claim rule (decision 16) applied verbatim per metric.
- [ ] **P9-H** Human-annotated subset ingested when it returns (`scripts/ingest_annotations.py`; agreement reported; per-system correct / incorrect / misgrounded / responsive typology); until then the table row says "pending".
- [ ] **P9-I** `analysis/legal_results.md` from the template: summary findings incl. nulls; seed tables; predictions → outcomes; claims table "licensed / not licensed"; reproduction with every job id.
- [ ] **P9-Z** Gate: every pre-registered prediction of P4–P9 reported beside its outcome; no claim from one seed; every number carries a job id and a log path; `python3 scripts/plan_state.py next`.

**Gate:** results document complete under the claim rule.

### P10 — Release hygiene

Purpose: make the repo and its claims honest and reproducible; hand the git decision to the user. Runs: Mac.

- [ ] **P10-A** `README.md`: what the system is, the licensed claims list from `analysis/legal_results.md`, what is not claimed, the "Hinweis … keine Rechtsberatung" and EU AI Act Art. 50 transparency framing, DACH scope, no medical data anywhere.
- [ ] **P10-B** `Docs/RUNBOOK.md`: reproduction from an empty scratch (sync → env → fetch → corpus → screen → backbone → system → post-training → evaluation) with the exact wrapper names and env levers.
- [ ] **P10-C** `Docs/CORPUS_LICENCE_REGISTER.md` final (every source, every version date) and `analysis/ARCHIVE_MANIFEST.md`: what may leave the cluster (commercial-safe checkpoints, `analysis/`, manifests without text); what may not (any checkpoint or shard touched by `research_only` data; raw NC text).
- [ ] **P10-D** `analysis/LITERATURE_CHECK.md`: the prior-work check for "Mamba + xLSTM + sparse attention + pointer citations + verifier on DACH law" before the word "first" appears anywhere; if it exists, the README says "to our knowledge" and cites it.
- [ ] **P10-E** `python3 scripts/plan_state.py readme` refreshes the progress table; `bash scripts/validate.sh --ci` exit 0; `tests/test_port_map.py` still green.
- [ ] **P10-F** USER ACTION — release commit: the user commits and pushes the release state, or tells the model to run specific git commands (then it runs exactly those, after `check_repo_hygiene.py` passes, and nothing else). Tick with `--user-confirmed`.
- [ ] **P10-G** USER ACTION — CI on the pushed branch: after the user pushes, the model reads the run with `gh run list` / `gh run view` (only if the user has authorised `gh`; otherwise the user pastes the result) and every job must be green. Tick with `--user-confirmed`.
- [ ] **P10-Z** Gate: every P10 box ticked; `bash scripts/validate.sh --full` exit 0.

**Gate:** README, runbook, register, archive manifest, literature check written; the release state is committed and pushed by the user; CI is green on the pushed branch.

### P11 — Optional: Qwen3 conversion road

Purpose: the only road to frontier-competitive knowledge (blueprint §Scale). **Not scheduled.** Opened only when the user says so after P9.

- [ ] **P11-A** Design note `Docs/CONVERSION_DESIGN.md`: MOHAWK/MambaInLlama-style initialisation of Mamba-3 `in_proj` slices from Qwen3-0.6B/1.7B Q/K/V/O, 25% of layers kept as attention, layer-wise alignment, KL against `Qwen3-8B-Base`, then P8's pipeline; compute estimate.
- [ ] **P11-B** Implementation and screen per the design note, under R1–R12 and a new pre-registered prediction.
- [ ] **P11-Z** Gate: only on user instruction; results into `analysis/conversion_results.md`.

**Gate:** user instruction to open; otherwise this phase stays pending forever and that is correct.

## 8. Testing and continuous integration

Testing is part of every box, not a phase of its own: a box that adds code adds its tests in the same box, and a box is ticked only when `bash scripts/validate.sh` exits 0 after the change (CLAUDE.md rule c). Three venues run tests — the Mac, GitHub Actions and the aisc cluster — and each check runs in the venue that has what it needs.

### 8.1 Test layers

| Layer | What it proves | Examples | Marker | Runs on |
|---|---|---|---|---|
| Static | code, shell, workflow and plan are well-formed | `ruff check`, `ruff format --check`, `bash -n`, `shellcheck`, `actionlint`, `plan_state.py check`, `uv lock --check` | — (not pytest) | Mac, CI |
| Environment | the venv is the lock, on the pinned interpreter | `scripts/check_env.py` | — | Mac, CI, cluster |
| Unit | one function or module behaves | config, layers, schedulers, parsers, verifier rules | none | Mac, CI |
| Numerical oracle | an operator equals its fp64 reference | SSD, TFLA, selective scan, `step` vs forward | none (`slow` for wide sweeps) | Mac, CI |
| Parity | the port computes what the reference computed | `tests/test_reference_parity.py` fixtures | none | Mac, CI |
| Equivalence | two paths that must agree do | cached vs uncached decode, packed vs per-document, chunk-size invariance, compile vs eager | none / `linux_only` | Mac, CI |
| Config invariants | every yaml composes and pins what it must | Hydra compose, operator pins, parameter bands, exact reference counts | none | Mac, CI |
| Integration | components work together on CPU | 2-step Lightning run, eval scripts on a tiny checkpoint, answer CLI | none / `slow` | Mac, CI |
| Wrapper static | SLURM wrappers follow the cluster rules | directives, `ARM` resolved inside the job, no `--gres`, resume path | none | Mac, CI |
| Repository hygiene | the tree that would be pushed is clean | size limit, no data or checkpoints, no secrets, no case collisions, `.gitignore` allowlist | — | Mac (ignore-aware), CI (pushed tree) |
| Linux-only | what macOS cannot run | Inductor CPU compile with gcc, FlexAttention on CPU, case-sensitive imports | `linux_only` | CI |
| GPU | CUDA numerics and memory | bf16 SDPA/flex, compile on GPU, cached decode on GPU | `cuda` | cluster |
| Multi-GPU | DDP wiring | 2-GPU DDP step, gradients identical across ranks | `multigpu` | cluster |
| Network | live external sources still parse | collectors with `--limit 2` against GII, RIS, Fedlex, … | `network` | CI (weekly, manual) |

### 8.2 Markers and skips

Markers are declared in `pyproject.toml` under `--strict-markers`. `tests/conftest.py` skips with a stated reason, never silently: `linux_only` unless `sys.platform == "linux"`; `cuda` unless `torch.cuda.is_available()`; `multigpu` unless at least two CUDA devices; `network` unless `LEXHYBRID_NETWORK_TESTS=1`; `reference` when `Reference/hybrid_model_mamba_xlstm` is absent. `slow` runs unless deselected. Every test seeds itself. No test touches the network or `$HF_HOME` unless it is marked `network`; what tests need lives under `tests/fixtures/`.

### 8.3 Where each check runs

| Venue | Command | Pytest selection | Blocks |
|---|---|---|---|
| Mac, every code box | `bash scripts/validate.sh` | `not slow and not cuda and not multigpu and not network` | ticking the box |
| Mac, end of phase | `bash scripts/validate.sh --full` | `not cuda and not multigpu and not network` | ticking `Pn-Z` |
| GitHub Actions, every push and pull request | `.github/workflows/ci.yml` → `bash scripts/validate.sh --ci` on Linux x86_64, Python 3.11 and 3.12, plus the `lint`, `hygiene` and `package` jobs | `not cuda and not multigpu and not network` (Linux-only tests run here) | a red run blocks the current box until fixed |
| GitHub Actions, weekly and manual | same workflow, `schedule` / `workflow_dispatch` | `network`; macOS clean install | a failure becomes a note and a fix box |
| aisc cluster | `scripts/slurm/preflight.sh`, `gpu_tests.sh`, `multigpu_tests.sh` | CPU suite in the cluster venv; `cuda`; `multigpu` | the P4 gate, before any screen |

**Checks the Mac cannot run, and CI does:** installing the exact lock on Linux x86_64, including the `torch 2.11.0+cu128` build the cluster uses; the cluster interpreter (Python 3.11) on the cluster's OS family; imports on a case-sensitive filesystem; Inductor CPU compilation with gcc and FlexAttention on CPU; a clean-machine install with no leftovers from a working venv; building the wheel and importing it without the source tree; the tree exactly as pushed (no `Reference/`, data, checkpoints, secrets, oversized files or case collisions); the weekly live-source smoke.

**Checks neither the Mac nor CI can run, and the cluster does:** CUDA and bf16 kernels, FlashAttention and flex attention on GPU, `torch.compile` on GPU, DDP across GPUs, real throughput and memory, and SLURM behaviour (requeue, resume from `last.ckpt`).

### 8.4 CI workflow (`.github/workflows/ci.yml`)

Triggers: `push` (all branches), `pull_request`, `workflow_dispatch`, weekly `schedule`. Permissions: `contents: read`. Concurrency: one run per ref; a newer push cancels the older run. Every job that installs dependencies uses `astral-sh/setup-uv` and `uv sync --locked`; no job resolves dependencies with pip (the one `uv pip install --no-deps` puts the built wheel into the `package` job's venv).

| Job | Runner | Steps | When |
|---|---|---|---|
| `lint` | ubuntu-latest | `uv lock --check`; `uv sync --locked --only-group dev`; `ruff check`; `ruff format --check`; `bash -n` and `shellcheck -x` on every `*.sh`; `actionlint` | every push and PR |
| `test` | ubuntu-latest × Python 3.11, 3.12 | `uv sync --locked --python <v>`; `check_env.py`; `bash scripts/validate.sh --ci` (coverage floor enforced); JUnit and coverage uploaded as artifacts | every push and PR |
| `hygiene` | ubuntu-latest | `check_repo_hygiene.py --ci` on the checkout, which is the pushed tree; `python3 scripts/plan_state.py check`; stdlib only, no install | every push and PR |
| `package` | ubuntu-latest | `uv build`; a venv from the lock without the project (`uv sync --locked --no-install-project --no-dev`); the wheel added with `uv pip install --no-deps`; every module imported and the smoke gate run from outside the source tree | every push and PR |
| `macos` | macos-14 (arm64) | clean `uv sync --locked`; `check_env.py`; `bash scripts/validate.sh` | weekly and manual |
| `network` | ubuntu-latest | `uv sync --locked`; `LEXHYBRID_NETWORK_TESTS=1 pytest -m network` | weekly and manual |

The workflow does nothing until the user pushes; nothing in this plan pushes. `tests/test_ci_workflow.py` keeps the workflow and `validate.sh` in step.

### 8.5 Environment reproducibility

Decision 18: one `uv.lock` for macOS arm64 and Linux x86_64, `.python-version` 3.11, `torch==2.11.0` (PyPI on macOS, `+cu128` on Linux). The Mac `.venv`, every CI runner and the cluster `.venv` are built with `uv sync --locked`; `scripts/check_env.py` verifies the interpreter, every installed version against the lock, the key imports, the editable install and the CUDA build. A dependency change is one box: edit `pyproject.toml`, run `uv lock`, run `validate.sh`, and put the lock diff in the note.

### 8.6 Coverage and flaky tests

CI measures line coverage of `lexhybrid/` with `pytest-cov`. P1-W5 sets the floor to the measured value rounded down to the nearest 5 in `pyproject.toml`; a phase that adds coverage raises it, and it is never lowered without a note. An intermittent test is a defect: fix it, or mark it `xfail(strict=True)` with the reason.

### 8.7 Testing obligations per phase

- **P1:** an oracle test for every ported operator; parity fixtures for every block and the model; config, wrapper and plan-state tests; the CI workflow and its static test.
- **P2:** a regression test per defect; equivalence properties (cached = uncached, packed = per-document, prefill = uncached, flex = dense) in fp32 and bf16; compile smoke under `linux_only`; the `cuda` and `multigpu` test files.
- **P3:** an offline fixture test and a `network` live smoke per collector; packing, split, licence and scrub tests; tokenizer tests that run offline once the tokenizer is cached.
- **P4:** the GPU and multi-GPU jobs pass before any screen; preflight runs the CPU suite in the cluster venv.
- **P5–P9:** every script that produces a number has a CPU test on a tiny model; every SLURM wrapper is covered by `tests/test_slurm_wrappers.py`.
- **P10:** CI is green on the pushed branch (P10-G), or recorded as inert.

## 9. Risks (FL1–FL16)

| # | Risk | Early warning | Mitigation |
|---|---|---|---|
| FL1 | Silent config drop — a lever reaches the yaml but not the mixer (the reference's costliest failure) | ARCH line at step 0 lacks the arm's `expect` token | `screen_arms.py verify --full` in preflight; wrapper greps the ARCH line and exits non-zero; `from_hydra` everywhere |
| FL2 | Port diverges from the reference after a "fix" | `tests/test_reference_parity.py` red | fixtures dumped before deletion, never skipped; every defect fix ships its own test |
| FL3 | Licence contamination | a `research_only` row in a `commercial_safe` shard; any `licence=unknown` | `build_corpus` refuses unknown; shard test in `validate.sh`; register is the single source |
| FL4 | Checkpoint bloat / quota (≈ 4.3 GB per checkpoint) | `du -sh $HOME` > 150 GB; probe checkpoint larger than expected | `outputs/` on scratch; `SAVE_TOP_K=0` on arms, 1 + last on pipeline runs; `du` before every submission |
| FL5 | Preemption restarts from step 0 | two ARCH lines in one log; `sacct` shows REQUEUED | `--open-mode=append`; `SignalCheckpointCallback`; wrapper passes `ckpt_path=last.ckpt`; `tests/test_slurm_wrappers.py` |
| FL6 | Login-node execution | "This command is not allowed on the login node!" | CLAUDE.md allowlist; `ARM` resolved inside the job; nothing scripted outside `sbatch` |
| FL7 | Executor drift on the plan itself | `check` fails; ticks without notes; wrong phase | `check` as gate 0; tick guards; `resume` prints only the current phase |
| FL8 | KD memory: full-vocab KL at 4K rows | OOM at P4-T | slab-wise KL under checkpointing; teacher decided by measurement |
| FL9 | Exact TFLA on the sequential fallback at init | `test_tfla_fast_path_at_init` red; step time 5× the profile | `mlstm_forget_gate_bias_init 3.0`; fallback is an error unless configured |
| FL10 | Truncation / no stop condition (75% of the medical reports were cut) | any dev decode without EOS | EOS on every SFT target; P8-N audit gate at 0% |
| FL11 | Underpowered screen | S1 seed SD > 0.5 PPL | S1 seeds first; bar floor 0.10; a null is recorded as a null |
| FL12 | GRPO reward hacking (trivially entailed citations, over-refusal) | citation precision up while answer accuracy or coverage down | coverage term; stop rule; GRPO only when the pre-registered condition holds |
| FL13 | Multi-hop deficit hidden by mean PPL | S0/S1 tie on PPL, differ on multi-hop | multi-hop probe from the first screen; reported in every table |
| FL14 | Training-side cost at long rows (54 GB at 8K uncompiled) | stage-B step time > 3× stage A | budget the uncompiled path; compile only after P4-S; micro-batch 4 at 8K |
| FL15 | Environment drift between the Mac, CI and the cluster | `uv lock --check` fails; `check_env.py` mismatch; CI red on one platform only | one `uv.lock`; `uv sync --locked` everywhere; `check_env.py` in validate, CI and every cluster job |
| FL16 | Green on macOS, broken on Linux (case-insensitive paths, gcc/Inductor, flex on CPU, the cu128 build) | CI `test` red while local validate is green | Linux-only tests in CI on every push; the cluster build of torch is the one CI installs |

## 10. State-tracking contract

After **every** meaningful change — file written and its test green, job submitted, job finished, number recorded, gate reached — tick the checkbox and update the state through the helper, never by hand:

```
python3 scripts/plan_state.py tick P2-D --evidence validate='exit 0, 212 passed, 2026-10-04'
python3 scripts/plan_state.py job add 2601234 --phase P5 --box P5-C --arm S1-s42 --log logs/screen_2601234_0.log
python3 scripts/plan_state.py tick P5-D --evidence job=2601234,2601235 log=cluster/logs/screen_2601234_0.log s42=9.871 s43=9.902 bar_value=0.10
python3 scripts/plan_state.py verdict P5 "S1 advances"
python3 scripts/plan_state.py next
```

`legal_build_state.json` is the resumable record; the checkboxes above are ground truth. Local commands are written `python3 scripts/...` or `.venv/bin/python ...` so they are never pasted into the aisc login node, which executes nothing scripted.

## 11. Cluster runbook (aisc)

Claude runs on the Mac. Every cluster interaction is one of these, from the Mac, with `scripts/slurm/cluster.env` sourced:

```bash
bash scripts/sync_to_cluster.sh                                        # rsync + .sync_stamp
ssh $CLUSTER_HOST "cd $CLUSTER_REPO && sbatch scripts/slurm/setup_env.sh"
ssh $CLUSTER_HOST "cd $CLUSTER_REPO && squeue --me"
ssh $CLUSTER_HOST "sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode -P"
ssh $CLUSTER_HOST "cd $CLUSTER_REPO && tail -50 logs/<name>_<id>.log"
bash scripts/sync_from_cluster.sh logs                                 # into cluster/logs/
ssh $CLUSTER_HOST "cd $CLUSTER_REPO && ARMS='S1-s42 S1-s43' sbatch --array=0-1 scripts/slurm/screen_array.sh"
ssh $CLUSTER_HOST "cd $CLUSTER_REPO && NUM_GPUS=4 sbatch --gpus=4 scripts/slurm/train_pretrain_4gpu.sh"
ssh $CLUSTER_HOST "cd $CLUSTER_REPO && du -sh $SCRATCH_ROOT/outputs $HOME"
```

Never on the login node: `python`, `bash script.sh`, `source .venv/bin/activate`, heredocs, `du -sh $HOME` as a script. `#SBATCH --gpus=N` in a file is not overridden by an env var: a 4-GPU run needs `--gpus=4` on the `sbatch` line and its own wrapper. `--gres` is never used. `ga03` is an ARM node; `gx13v1` has a faulty GPU; both stay excluded.

If Claude is ever run on the cluster itself (hostname `lx*`), the same rules hold without the `ssh` prefix, and nothing scripted runs outside `sbatch`.

## 12. Blueprint phase map

| Blueprint band | Plan phases |
|---|---|
| Decisions before Phase 1 (initialisation road, corpus legality) | §2 decisions 1–2 |
| Backbone screen (attention count, `d_state`, MTP, probes) | P4 (profiling) + P5 |
| Corpus and tokenizer | P3 + P4-K..M |
| Full backbone at deployment length | P6 |
| Retrieval, renderer, verifier | P7 |
| SFT → DPO → GRPO, synthetic QA | P8 |
| Evaluation and defensible claims | P9 |
| Release, AI Act framing, archive | P10 |
| Conversion road | P11 (optional) |

## 13. Unresolved questions

- How many commercial-safe legal tokens the open sources actually yield (P4-M answers; the 5B budget shrinks to what exists).
- Whether the 8B teacher fits at 4,096 rows on 4×H100 with the student's slab-wise KL (P4-U answers).
- Whether GerLeRB / GerDaLIR licences allow redistribution of derived indices (register row; eval-only until confirmed).
- Whether jurist annotation can be arranged (P8-M); without it, the human-graded row stays "pending" and the claims table says so.
- Whether `gpt-oss-120b` loads under the cluster's vLLM/torch pins (P9-E; skipped and said so otherwise).

## 14. Amendment log

| Date | Change | Requested by |
|---|---|---|
| 2026-09-27 | Plan opened (Stage 0, Fable 5.1): phases P0–P11, rules R1–R12, risks FL1–FL14 | user |
| 2026-09-27 | Repository rule updated: the user initialised git and pushed `README.md` to `github.com/krishankb-de/legal_ai_hybrid_mamba_xlstm_model` at 13:50; git stays the user's, the model runs no git command unless asked; P10-F becomes the user's release commit | observed, user action |
| 2026-09-27 | Testing and CI (Opus 5.5 session): §8 rewritten as the testing strategy with markers, venues and the CI workflow; decisions 18 (uv lock, Python 3.11, torch 2.11.0/cu128) and 19 (CI); rules R13, R14; risks FL15, FL16; boxes P1-W1..W5, P2-Y1, P4-J1..J4, P10-F, P10-G added (the git decision moved from P10-Z to P10-F); P1-A, P1-B, P1-N, P1-T, P1-V, P1-W, P1-X, P1-Y, P2-J, P2-Y, P3-E, P4-A, P4-E, P4-F, P4-Z text amended | user |
| 2026-09-27 | P1 implementation (Opus 5.5): §3 heading R1–R14; R8 also admits the `ref_*` replicas; P1-R names `lexhybrid/config/loading.py`; P1-V `output_dir` puts `outputs/` under `$SCRATCH_ROOT`; P1-X model fixture vocab 256 (5 MB limit); P1-Y port-map test scope; P1-W gate 6 wording; P1 test names match the tests as written, and P1-Y adds `tests/test_plan_references.py` so they stay that way; `.rsync-exclude` include rule fixed (the `!` line was gitignore syntax and dropped the parity fixtures from a sync) with `tests/test_rsync_rules.py`; sdist includes anchored to the root (an unanchored `README.md` pulled in `Reference/**/README.md`) with `tests/test_packaging.py`; §8.4 job steps match `ci.yml` | implementation |
| 2026-09-27 | CI on the pushed branch (runs 36322586655, 36322991795): `check_env.py` parses uv output under `FORCE_COLOR=1`; P1-Y whole-model parity tolerance 5e-6 (Linux BLAS differs from the macOS fixtures by up to 1.4e-6; blocks stay at 1e-6) | implementation |
| 2026-09-27 | P3 (Opus 5.5): box P3-E1 added so P3's dependency changes sit in a box of their own (rule m) | implementation |
| 2026-09-27 | Decisions 20–22 from the user's message: S5 "report the +2.36% as a stated difference; don't reshape" and "Forget bias: 3.0"; the LER scrub "keep flair/ner-german-legal, but run it in its own isolated scrub environment ... Don't downgrade the main environment", "a small separate uv project with its own lock, added in a box of its own (rule m), with check_env extended to cover it". P2-W box text and the P5 arm table amended (S5 band ±2.5%, delta reported); the strict xfail `test_legacy_is_within_one_percent_of_the_base` replaced by `test_legacy_delta_is_the_stated_difference`; `screen_arms.py` S5 `band_pct` 2.5 (`verify --full` exits 0); box P3-R1 added and P3-R runs in its environment | user |
| 2026-09-29 | GPU cluster in maintenance; on the user's instruction ("proceed with backlogs or pending tasks which are not dependent on the GPU or the server") the Mac-only work ran ahead: P4-V ticked; P5-A, P5-B and P6-A ticked with `--force` while `current_phase` stays P4. How the open P4 job boxes are carried out, without changing their text: **P4-K** `--limit all`, except FineWeb-2, capped at `FINEWEB_LIMIT` pages (default 2,000,000, about the 30% general share of a ≤ 5B-token budget) and shuffled across dumps; a requeued or timed-out collection resumes through an on-disk HTTP cache (`--http-cache`). **P4-L** the LER scrub measured ~1,000 characters/s on a CPU (P3-R), so it runs first as `scripts/slurm/scrub_array.sh` (one H100 per source part), then `pack_corpus.sh` does dedup → pack, streaming (both row lengths from one tokenisation; the scrub parts read back in collection order); the verdict carries the corpus, scrub and pack job ids. The P5 screen's 2,048-token rows are cut from the 4,096 shards (`datasets.shard_directory`), so nothing is packed at 2,048. **P4-P** `performance_profile.py` gains `--doc-len` (packed rows), `--loss slab` (the training loss) and `--mlstm-chunk-size`; CSV/JSON go to `logs/profile_<job>/` because the rsync push `--delete`s job output under `analysis/`. DDP runs as one SLURM task with `LightningEnvironment` (no srun). | user (instruction); implementation |
| 2026-09-29 | Decision 13 amended from the user's answer ("keep both and when cited cite both so that not only 1 citation works but many"; chose "Every copy" for pretraining): deduplication drops a duplicate only inside one source; a copy in another source (RII and OLDP of one decision, a web page reproducing a statute, the Multi Legal Pile copy of an OLDP decision in the research arm) is kept, packed with its own source and linked in `dedup.jsonl` (`dropped: false`); the retrieval index groups the copies and the renderer (P7-G) cites every source of a quoted or cited passage (`decisions.cite_every_copy`). P7-A/P7-B also started on the user's instruction, with two backward-compatible schema fields: `Document.hierarchy` (the act's structural headings above a provision) and `Section.part` (the part of a decision a paragraph belongs to). | user |
| 2026-09-30 | Box P7-C1 added before P7-C so P7's dependency change (`bm25s[stem]`, pulling `PyStemmer`) sits in a box of its own (rule m), on the user's go-ahead ("Yes proceed with P7-C"); the cluster `.venv` must be rebuilt (`setup_env.sh`) before the next cluster job that runs `check_env.py`. | user |
