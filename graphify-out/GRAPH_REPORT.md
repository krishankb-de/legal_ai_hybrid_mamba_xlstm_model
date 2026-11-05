# Graph Report - legal_ai_hybrid_mamba_xlstm_model  (2026-11-05)

## Corpus Check
- 261 files · ~260,093 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 61 file(s) not represented in the graph (top: .jsonl 30, .xml 12, (none) 6)

## Summary
- 4140 nodes · 9667 edges · 195 communities (168 shown, 27 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 507 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `198e28c1`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- packing.py
- HybridLanguageModel
- test_data.py
- Document
- test_retrieval.py
- test_layers.py
- evalsets/base.py
- sequential_mlstm_reference
- performance_profile.py
- rel_max_err
- test_eval.py
- bootstrap_compare.py
- run_metadata.py
- test_dpo.py
- test_models.py
- register
- test_numerics_fp64.py
- generate.py
- test_collect_offline.py
- train_pretrain.py
- scrub_ner_worker.py
- evaluate_val_ppl.py
- family_lm
- collectors/base.py
- PretrainLightningModule
- check_operator_equivalence.py
- tfla_forward_parallel
- AttentionBlock
- RMSNorm
- test_scripts.py
- HybridBlock
- test_verifier.py
- pathlib
- HybridConfig
- SpikeDiagnosticsCallback
- schema.py
- fedlex.py
- datasets.py
- dedup.py
- screen_arms.py
- test_training.py
- Mamba3Block
- DenseIndex
- statute_recall.py
- mLSTMBlock
- test_decoding.py
- Chunk
- test_slurm_wrappers.py
- test_synth_qa.py
- segment_ids
- check_env.py
- sft_format.py
- verify
- torch
- LayerSplit
- os
- test_every_planned_wrapper_exists
- eurlex.py
- test_check_env.py
- test_config_invariants.py
- scrub_ler.py
- .__init__
- MDebertaNLI
- code
- HybridLightningModule
- scrub_file
- load_model_config
- greedy
- build_dpo_pairs.py
- test_code_hygiene.py
- _ddp_fit
- CausalLMAdapter
- test_ci_workflow.py
- test_reference_parity.py
- train_dpo.py
- cumulative_angles
- build_dense_index.sh
- lexhybrid
- directives
- eval_retrieval.sh
- test_rsync_rules.py
- _BlockParser
- train_sft.py
- MambaBlock
- DPOLightningModule
- resync_index.py
- test_repo_hygiene.py
- German-law pretraining corpus: tokens, deduplication, scrub
- filters.py
- build_doc_block_mask
- build_sft_data.py
- probe_ckpt_size.sh
- SFTLightningModule
- check_repo_hygiene.py
- screen_table.py
- Profile ladder, chunk size and distillation memory
- README.md
- Architecture screen: ablations of the hybrid backbone
- BM25Index
- gpu_tests.sh
- multigpu_tests.sh
- train_pretrain_1gpu.sh
- Contributing
- check_configs.py
- ScheduleSampler
- StepStatsCallback
- .allowed
- conftest.py
- equivalence.sh
- kd_memory_probe.sh
- Architecture
- judge.py
- _doc
- evaluate_lm.py
- argparse
- preflight.sh
- profile.sh
- validate.sh
- test_judge.py
- Tokenizer fertility on DACH legal text
- _Paras
- .from_json
- train_sft_hf.py
- Backbone results: the hybrid and a Transformer baseline at 8,192 tokens
- pytest
- analyze_generation_diversity.py
- build_corpus_array.sh
- pack_corpus.sh
- OpenAICompatible
- measure_fertility.py
- corpus_counts.sh
- diag_compile.sh
- eval_probes.sh
- fetch_hf.sh
- scrub_array.sh
- setup_env.sh
- train_pretrain_4gpu.sh
- answer.py
- test_diag_spike_reruns_the_arm_observed_in_its_own_directory
- test_imports.py
- ByteTokenizer
- judge_answers.py
- filter_logits
- _executable
- cleanup.sh
- inspect_ckpt.sh
- screen_array.sh
- screen_metrics.sh
- verify_handoff.sh
- test_diagnose_compile.py
- collectors/__init__.py
- corpus/__init__.py
- evalsets/__init__.py
- data/__init__.py
- kernels/__init__.py
- renderer/__init__.py
- retrieval/__init__.py
- training/__init__.py
- diag_spike.sh
- watch.sh
- sync_from_cluster.sh
- sync_to_cluster.sh
- test_judge_parses_scores_and_caches
- lexhybrid
- lexhybrid-scrub
- checkpoint.py
- test_compile_forward_chunk128
- test_pipeline.py
- dump_reference_parity.py
- collections
- teacher_forward_packed
- test_tfla_wide_chunk_factorises_per_block_on_gpu
- test_tfla_wide_chunks_factorise_per_block_exactly
- Judge
- sample
- schedulers.py
- check_run_settings
- synth_qa.sh
- train_sft_4gpu.sh
- hf_run
- test_sync_scripts.py
- Corpus licence register
- train_dpo_4gpu.sh
- _units
- generated_tokens
- paragraphs
- setup_vllm.sh
- build_dpo_pairs.sh
- build_sft_data.sh
- eval_gold_passages.sh
- eval_system_ablation.sh
- vllm_server.sh
- lexhybrid-vllm

## God Nodes (most connected - your core abstractions)
1. `Document` - 132 edges
2. `HybridLanguageModel` - 109 edges
3. `Chunk` - 99 edges
4. `HybridConfig` - 72 edges
5. `AnswerRecord` - 47 edges
6. `verify()` - 47 edges
7. `load_model_config()` - 46 edges
8. `Section` - 41 edges
9. `sandbox()` - 38 edges
10. `http_get()` - 32 edges

## Surprising Connections (you probably didn't know these)
- `Data pipeline` --references--> `Document`  [INFERRED]
  docs/ARCHITECTURE.md → lexhybrid/data/schema.py
- `Citation ids` --references--> `parse_citation()`  [INFERRED]
  docs/DOCUMENT_SCHEMA.md → lexhybrid/data/schema.py
- `Hypotheses and method` --references--> `encode_document()`  [INFERRED]
  analysis/tokenizer_fertility.md → lexhybrid/data/tokenizer.py
- `Tokenizer and pointer vocabulary` --references--> `encode_document()`  [INFERRED]
  docs/ARCHITECTURE.md → lexhybrid/data/tokenizer.py
- `How it answers` --references--> `MDebertaNLI`  [INFERRED]
  README.md → lexhybrid/verifier/nli.py

## Import Cycles
- None detected.

## Communities (195 total, 27 thin omitted)

### Community 0 - "packing.py"
Cohesion: 0.09
Nodes (25): Citation ids, Document schema, Fields, main(), pack(), Packer, Path, EOS-delimited packing into fixed-length rows and parquet shards. Every document… (+17 more)

### Community 1 - "HybridLanguageModel"
Cohesion: 0.05
Nodes (35): Model, Probes scored on any checkpoint: MQAR, statute recall, multi-hop., __getattr__(), lexhybrid -- a DACH legal, citation-faithful, retrieval-gated hybrid Mamba-3 /…, HybridLanguageModel, no_grad, Tensor, Final norm and LM head: residual stream -> logits. (+27 more)

### Community 2 - "test_data.py"
Cohesion: 0.03
Nodes (73): BGerCollector, EURLexCollector, FineWeb2DECollector, GIICollector, parse_toc(), ``(slug, xml.zip URL)`` for every law in ``gii-toc.xml``., read_zip_xml(), MultiLegalPileCollector (+65 more)

### Community 3 - "Document"
Cohesion: 0.05
Nodes (72): html_parser, licence_flags(), main(), ``licence``, ``commercial_safe`` and ``research_only`` as the register defines…, A collector module's entry point: ``cli``, then end the process at once with…, utc_now(), follows(), parse_bger() (+64 more)

### Community 4 - "test_retrieval.py"
Cohesion: 0.06
Nodes (54): _anchor(), chunk_case_law(), chunk_document(), chunk_statute(), _common(), _label(), provision_heading(), ``(grain, text)`` for one Absatz: its sentences (``Satz 2``) and its items… (+46 more)

### Community 5 - "test_layers.py"
Cohesion: 0.08
Nodes (48): _bc_bias_pair(), _continue_with_steps(), _drive_lambda_to_one(), m3_block(), no_grad_params(), parametrize, Block-level behaviour of the four mixers and HybridBlock. Ported from the…, Isolation plus correctness: document B equals running document B on its own. (+40 more)

### Community 6 - "evalsets/base.py"
Cohesion: 0.08
Nodes (51): gzip, cli(), EvalSetInfo, Passage, Path, QAItem, QASet, Query (+43 more)

### Community 7 - "sequential_mlstm_reference"
Cohesion: 0.10
Nodes (22): dtype, Tensor, float64 sequential oracle for the recurrence ``tfla_forward_parallel``…, Plain-loop mLSTM recurrence; inputs (B, H, L, D), output (B, H, L, D) in…, sequential_mlstm_reference(), _packed_tfla_inputs(), The legacy error is governed entirely by whether f_cum underflows the 1e-6…, The overflow guard: re-centring the whole chunk produced NaN here; re-centring… (+14 more)

### Community 8 - "performance_profile.py"
Cohesion: 0.08
Nodes (41): contextlib, csv, Embedding and every block: the residual stream (B, L, D) BEFORE the final norm.…, attention_backend(), available_configs(), build_model(), compile_for(), effective_chunk_sizes() (+33 more)

### Community 9 - "rel_max_err"
Cohesion: 0.14
Nodes (25): Chunk-parallel SSD scan. h_t = exp(dt_t A) h_{t-1} + sum_terms coeff_t (B_t…, ssd_chunked_scan(), Term, parametrize, Tensor, Resets are boolean masks inside the scan; the on-chunk-edge case is where an…, Perturbing document A must leave document B bit-identical, not merely close., The state after the last token, including a partial last chunk -- where the… (+17 more)

### Community 10 - "test_eval.py"
Cohesion: 0.06
Nodes (53): answer_words(), cited_passages(), doc_of(), gold_relation(), kept_segments(), _mean(), Segment, _ratio() (+45 more)

### Community 11 - "bootstrap_compare.py"
Cohesion: 0.10
Nodes (27): corpus_bleu(), _lcs_length(), _ngram_counts(), Counter, Sentence ROUGE-L and corpus BLEU, stdlib only. Ported verbatim from the…, Sentence-level ROUGE-L F-measure (Lin, 2004)., Corpus-level BLEU-N with brevity penalty, one reference per hypothesis., rouge_l_score() (+19 more)

### Community 12 - "run_metadata.py"
Cohesion: 0.16
Nodes (16): Any, DictConfig, Path, ``run_metadata.json``: what produced a checkpoint, written at the start of…, sha256 over the sorted (relative path, file sha256) pairs of the source tree., Contents of ``.sync_stamp`` (date + tree hash of the last sync to the cluster),…, The checked-out commit id, read from ``.git/HEAD`` and its ref file; "" if…, Persist ``<output_dir>/run_metadata.json`` and return its path. (+8 more)

### Community 13 - "test_dpo.py"
Cohesion: 0.06
Nodes (56): _argv(), _dpo_overrides(), _fake_trainer(), _pair_rows(), _pairs(), Preference optimisation: pair selection and balance, the answer-token log-…, Only the exact ``<|unanswerable|>`` is a good refusal: a kept answer is bad,…, Target 1/4 of refusal pairs: 1 refusal + 6 answer pairs keep 3 answer pairs… (+48 more)

### Community 14 - "test_models.py"
Cohesion: 0.10
Nodes (36): _delta_at_init(), _delta_model(), parametrize, Model-level behaviour of HybridLanguageModel., The model's weight pass zeroed both gate biases; `post_model_init` restores the…, A `.to(bfloat16)` model without autocast once died in the SSD einsums., A document's last token (its EOS) does not predict the next document's first;…, Init then tie; `non_embedding = total - embedding - (lm_head if untied)`, so… (+28 more)

### Community 15 - "register"
Cohesion: 0.15
Nodes (19): file_sha256(), IndexVersion, latest(), Path, Every registered version, oldest first; ``[]`` when the registry does not exist…, The highest version, or ``None`` when nothing is registered., Record the index in ``index_dir`` as the next version, or return the…, One registered index build. See the module docstring. (+11 more)

### Community 16 - "test_numerics_fp64.py"
Cohesion: 0.09
Nodes (34): _audit_inputs(), _clamp_fire_fraction(), _delta_cases(), _measure(), Operator-level numerical correctness against float64 sequential oracles. Ported…, ``use_fast_path=False`` shares the one implementation (the reference once had a…, ``scan_impl="legacy"`` must stay the original operator bit for bit (the…, The shipped configuration runs exact TFLA on its factorised path at init. With… (+26 more)

### Community 17 - "generate.py"
Cohesion: 0.13
Nodes (34): Advance, Decoding, apply_repetition_penalty(), _beam_core(), beam_search(), beam_search_cached(), beam_search_uncached(), best_of_n() (+26 more)

### Community 18 - "test_collect_offline.py"
Cohesion: 0.05
Nodes (45): FetchError, RuntimeError, ``fetch(url)``'s response, or None when it raised ``FetchError``., A request failed after every retry., Fetch one document, or skip it (logged) when its fetch fails after every retry,…, SkipFailedDocuments, api_key(), DIPCollector (+37 more)

### Community 19 - "train_pretrain.py"
Cohesion: 0.07
Nodes (39): FinalCheckpointCallback, Callback, Training callbacks: checkpoint-on-signal (ported from the reference) and step…, End training after optimizer step ``stop_at_step`` while the LR schedule keeps…, Write ``<checkpoint_dir>/last.ckpt`` when training ends normally.…, Write ``interrupt.ckpt`` when the process is signalled; a no-op after the first…, SignalCheckpointCallback, StopAtStepCallback (+31 more)

### Community 20 - "scrub_ner_worker.py"
Cohesion: 0.22
Nodes (12): cap(), chunks(), default_cache_dir(), load_tagger(), main(), flush(), Entities of every document, the sentences of all ``docs`` tagged in ONE…, German legal-entity recognition for the LER scrub. Runs in the scrub… (+4 more)

### Community 21 - "evaluate_val_ppl.py"
Cohesion: 0.10
Nodes (26): Normalise ``model._orig_mod.`` / ``_orig_mod.model.`` / ``_orig_mod.`` /…, strip_prefixes(), document_offsets(), late_context_sums(), load_model(), main(), no_grad, Path (+18 more)

### Community 22 - "family_lm"
Cohesion: 0.11
Nodes (20): family_lm(), _prefill_then_greedy(), parametrize, The one-pass prefill leaves every cache exactly where token-by-token decoding…, In bf16 the two paths round differently, so the claim is argmax identity…, A packed prompt: decoding continues each row's last document as if it were…, Cached greedy equals the uncached reference token for token, batch 3 with rows…, With EOS, length normalisation and a repetition penalty, cached beam search… (+12 more)

### Community 23 - "collectors/base.py"
Cohesion: 0.06
Nodes (42): _cached(), cli(), Collector, http_get(), http_post(), HttpCache, parse_limit(), Path (+34 more)

### Community 24 - "PretrainLightningModule"
Cohesion: 0.07
Nodes (29): Dataset, Tensor, Seeded random-token rows with packed-document ids, for smoke tests and…, ``num_rows`` rows of ``row_len`` random tokens, each packed from several…, SyntheticPackedDataset, PretrainLightningModule, Tensor, Clip and log the gradient norm; a non-finite norm clears the gradients (step… (+21 more)

### Community 25 - "check_operator_equivalence.py"
Cohesion: 0.13
Nodes (20): Mamba-2/3 state space dual (SSD) kernels., _heads_per_group(), dtype, Tensor, Sequential reference for the SSD recurrence, and the single-token decode step.…, Advance the SSD recurrence by one token. h_t = exp(dt_t * A) h_{t-1} +…, Ground-truth SSD scan: a plain loop, in float64 by default. Args: doc_ids:…, ssd_sequential_reference() (+12 more)

### Community 26 - "tfla_forward_parallel"
Cohesion: 0.13
Nodes (21): TFLA (tiled linear attention) kernels for the mLSTM., apply_tfla(), _blockwise_scores(), _intra_chunk_sequential(), RuntimeError, Tensor, Chunk-parallel TFLA (tiled linear attention) for the mLSTM. Forward is O(L/C)…, Chunk-parallel TFLA. Args: q, k, v: (B, H, L, D) i_gate: input gates (B, H, L,… (+13 more)

### Community 27 - "AttentionBlock"
Cohesion: 0.09
Nodes (28): apply_rope(), AttentionBlock, build_doc_boundary_attn_mask(), build_rope_cache(), compiled_flex_attention(), positions_within_documents(), device, dtype (+20 more)

### Community 28 - "RMSNorm"
Cohesion: 0.08
Nodes (33): LayerType, exponential_activation(), Tensor, SiLU (swish) activation., Exponential gate activation of the mLSTM input gate., silu_activation(), Causal self-attention mixer with RoPE. A *mixer*, not a model class: registered…, create_hybrid_blocks() (+25 more)

### Community 29 - "test_scripts.py"
Cohesion: 0.05
Nodes (32): fixture, The ported command-line scripts run end to end on CPU with tiny models., A tiny model yaml and a Lightning-style checkpoint for it., The profiler's slab step computes what PretrainLightningModule trains on: the…, Every arm builds through HybridConfig.from_hydra at reduced width, runs a…, The learning-rate arms are S0 and S1 with only `model.learning_rate` changed,…, Full size on `meta` with the bands: every arm passes; S5 carries +2.36% as a…, Guard: if an arm's lever never reaches the model, its ARCH line lacks the token. (+24 more)

### Community 30 - "HybridBlock"
Cohesion: 0.10
Nodes (17): HybridBlock, Tensor, (batch, seq_len, dim) -> (batch, seq_len, dim). Argument order matches…, Per-layer decode state, or None if this mixer has no step (Mamba-1)., One token through mixer + MLP with residuals, mirroring ``forward``'s topology…, One decoder block: ``x + mixer(norm1(x))``, then the MLP (pre-norm, or…, Under `hybrid`, block >= 1 computes norm2(x + mlp(x)); block 0 stays pre-norm., The reference filtered on unprefixed names, so every `mlstm_*` option fell back… (+9 more)

### Community 31 - "test_verifier.py"
Cohesion: 0.04
Nodes (121): _check_sentences(), _labels(), pointer_text(), Segment, ValueError, The renderer: the model's pointer answer as the text a reader sees. The answer…, The rendered answer of parsed segments against the retrieved passages (prompt…, The rendered answer of the model's raw answer text (``<|endoftext|>`` allowed… (+113 more)

### Community 32 - "pathlib"
Cohesion: 0.04
Nodes (90): collections_abc, dataclasses, hashlib, json, Pointer-constrained decoding. The legal model never generates statute text: it…, build_sets(), gold_chunk_ids(), The gold-passage sanity check: with retrieval assumed solved, does the model… (+82 more)

### Community 33 - "HybridConfig"
Cohesion: 0.06
Nodes (33): HybridConfig, Validate and auto-compute derived parameters., The per-mixer arguments for layer ``layer_idx`` (a readable summary; not used…, Every architectural knob of the hybrid decoder. One dataclass serves every…, Build a config from a Hydra ``cfg.model`` node, keeping every field the…, Convert the config to a plain dictionary., Create a config from a dictionary produced by ``to_dict``., Write ``config.json`` into ``save_directory`` and return its path. (+25 more)

### Community 34 - "SpikeDiagnosticsCallback"
Cohesion: 0.10
Nodes (15): flush(), Callback, Tensor, Trainer, SpikeDiagnosticsCallback, hook(), hook(), hook() (+7 more)

### Community 35 - "schema.py"
Cohesion: 0.07
Nodes (37): datetime, io, Element, Gesetze im Internet: federal statutes and ordinances of Germany. Source:…, Text of a gii-norm content element: list items on their own lines, whitespace…, The date of the last amendment in the header's ``Stand`` comment, else the…, render(), walk() (+29 more)

### Community 36 - "fedlex.py"
Cohesion: 0.09
Nodes (33): act_code(), act_meta(), acts_query(), article_sections(), FedlexCollector, _inline(), _is_text(), latest_versions() (+25 more)

### Community 37 - "datasets.py"
Cohesion: 0.05
Nodes (60): itertools, build_source_shards(), build_source_shards_stream(), effective_weights(), mixture_from_config(), PackedMixture, Dataset, Path (+52 more)

### Community 38 - "dedup.py"
Cohesion: 0.15
Nodes (20): deduplicate(), deduplicate_signatures(), deduplicate_stream(), DedupLog, keep_order(), minhash(), Deduplication of the training corpus: exact, then near-duplicate. 1. **Exact.**…, ``deduplicate_signatures`` over documents streamed once, their MinHashes… (+12 more)

### Community 39 - "screen_arms.py"
Cohesion: 0.19
Nodes (15): NamedTuple, Arm, build_config(), check_tokens(), env_lines(), _fmt(), hydra_overrides(), _inference_non_embedding() (+7 more)

### Community 40 - "test_training.py"
Cohesion: 0.04
Nodes (62): Summed CE and summed masked KL for one slab (recomputed in the backward pass)., Slab-wise ``(1 - alpha) * CE + alpha * T^2 * KL`` without (B, L, V) logits.…, slab_ce_kl(), _slab_terms(), configure_optimizer(), get_parameter_groups(), Any, Module (+54 more)

### Community 41 - "Mamba3Block"
Cohesion: 0.11
Nodes (15): heavy_tail_activation(), Mamba3Block, Tensor, (Re-)apply the SSM initializations after the parent model's global weight pass.…, Per-head state transition, strictly negative. Shape (nheads,) or (batch, len,…, (batch, seqlen, dim) -> (batch, seqlen, dim). Argument order is load-bearing:…, The last ``conv_size - 1`` pre-convolution columns, (batch, channels, conv_size…, Exponential-trapezoidal discretization (paper Prop. 1). h_t = alpha_t h_{t-1} +… (+7 more)

### Community 42 - "DenseIndex"
Cohesion: 0.10
Nodes (27): fused_retriever(), retrieve(), Encoder, Path, The index's ranking: BM25, fused with ``dense/`` when it exists and ``encode``…, _sets_of(), easy(), related() (+19 more)

### Community 43 - "statute_recall.py"
Cohesion: 0.08
Nodes (32): gold_nll(), no_grad, Mean per-token NLL of ``target_ids`` following ``prompt_ids`` (natural log).…, generate(), MQARConfig, MQARItem, no_grad, Multi-query associative recall, MQAR, with the upstream reference's… (+24 more)

### Community 44 - "mLSTMBlock"
Cohesion: 0.10
Nodes (19): mLSTMBlock, Tensor, (Re-)apply the gate-bias inits after the parent model's global weight pass.…, Soft-capped gates: exp input gate (computed in fp32), sigmoid forget and output…, (batch, seq_len, dim) -> (batch, seq_len, dim). Args: x: input. cache: an EMPTY…, mLSTM decode state: the matrix cell ``C`` and the normalizer ``n``, both zeros,…, Advance one token. ``(batch, dim)`` or ``(batch, 1, dim)`` in, ``(batch, dim)``…, Soft cap via tanh: maps x to ``tanh(x / cap) * cap``, range (-cap, cap). (+11 more)

### Community 45 - "test_decoding.py"
Cohesion: 0.10
Nodes (31): Problems with the pointers in a decoded sequence; empty when every pointer is…, validate_pointers(), cached_lm(), Decoding. Cached decode must be an equivalence with full recomputation. The…, All-or-nothing: Mamba-1 has no step(), so the legacy stack cannot decode with a…, The KV-cache step reproduces the full forward, from an empty cache and after a…, After a packed prompt the cache masks earlier documents and RoPE continues the…, Beams reorder the KV cache row-wise, and cached beam search over a stack with… (+23 more)

### Community 46 - "Chunk"
Cohesion: 0.05
Nodes (63): concurrent_futures, Exception, Chunk, estimate_tokens(), One retrievable passage., Ranked, Cross-encoder reranking of fused candidates into the prompt's passages. The…, The first ``limit`` candidates re-scored and sorted best first (ties keep the… (+55 more)

### Community 47 - "test_slurm_wrappers.py"
Cohesion: 0.07
Nodes (45): fixture, SLURM wrappers under `scripts/slurm/`. Static checks on every `*.sh` there: the…, Each run's last.ckpt against configs/model/<cfg>.yaml, a PASS/FAIL line per…, The default sources' scrubbed parts go to the bm25 build with the pack's dedup…, The eval guard reads manifests of the full sets, fetched online into a…, The ARCH line of the real base model (S1), built on meta from the screen's own…, A repository copy with the wrappers, screen_arms.py and a fake interpreter;…, A requeued run finds its last.ckpt and passes it on. (+37 more)

### Community 48 - "test_synth_qa.py"
Cohesion: 0.07
Nodes (36): http_server, socket, _executable(), _free_port(), jobs(), nli(), parametrize, Synthetic QA (scripts/generate_synthetic_qa.py) and the vLLM jobs, offline. The… (+28 more)

### Community 49 - "segment_ids"
Cohesion: 0.10
Nodes (22): forward_fill_from_starts(), pad_segments(), device, Tensor, Document segments inside packed rows, shared by the SSD and TFLA kernels. A…, (batch, seqlen) bool, True where a new document begins; position 0 is False., Monotone segment index per position, (batch, seqlen) long; all zeros without…, Extend ``seg`` by ``pad`` positions that join the LAST real segment. Padding… (+14 more)

### Community 50 - "check_env.py"
Cohesion: 0.13
Nodes (22): importlib_metadata, packaging_markers, packaging_requirements, packaging_utils, packaging_version, Requirement, check_cuda_build(), check_editable() (+14 more)

### Community 51 - "sft_format.py"
Cohesion: 0.09
Nodes (47): Cite, encode(), Example, parse(), Passage, Quote, The retrieval-grounded SFT format: one prompt, pointer answers. The prompt is…, ``(prompt, target)`` as strings with the special-token markup. (+39 more)

### Community 52 - "verify"
Cohesion: 0.10
Nodes (26): asks_for_party(), ClaimCheck, _precision(), NLI, The question asks who a party of a case is (German or English patterns)., The verified, rendered answer of the model's raw ``answer`` (see the module…, _refusal(), verify() (+18 more)

### Community 53 - "torch"
Cohesion: 0.07
Nodes (38): HeadFn, Gold-answer likelihood: a continuous probe score that an exact-match floor…, Activation functions used by the mixers., boundary_masked_labels(), CausalLMOutput, ``HybridLanguageModel``: embedding -> N x HybridBlock -> RMSNorm -> LM head.…, Language-model forward pass. Args: input_ids: (B, L) token ids; mutually…, Output of a causal language model forward pass. (+30 more)

### Community 54 - "LayerSplit"
Cohesion: 0.11
Nodes (11): _delta_ms(), LayerSplit, hook(), hook(), _mark(), Start/stop marker: a CUDA event on GPU, a wall clock on CPU., Per-`HybridBlock` timing, aggregated by mixer type. Markers are queued on the…, {mixer type: mean ms per forward}, plus a per-layer-index breakdown. (+3 more)

### Community 55 - "os"
Cohesion: 0.13
Nodes (20): einops, Mamba-1 selective scan (legacy ablation only)., Tensor, Mamba-1 selective scan (legacy incumbent; used only by the…, Exact selective scan -- the recurrence the Mamba-1 block is specified to…, Exact chunked selective scan -- no division, no clamp. Flips which axis is…, Public selective-scan entry point, computed in fp32 and cast back.…, Legacy chunk-parallel selective scan (divide-and-clamp; see module docstring).… (+12 more)

### Community 57 - "eurlex.py"
Cohesion: 0.17
Nodes (17): act_code(), _find(), _inline(), _local(), oj_date(), parse_eurlex_xhtml(), Element, EUR-Lex: EU regulations and directives in German. Source: the Publications… (+9 more)

### Community 58 - "test_check_env.py"
Cohesion: 0.10
Nodes (15): env(), skipif, slow, scripts/check_env.py: the environment-equals-lock check., The CI condition, reproduced: the real uv, FORCE_COLOR=1, the real lock., Run from the main environment, `--scrub` must fail: no flair, transformers 5,…, The live venv equals the lock (skipped in fast mode; CI and --full run it)., FORCE_COLOR=1 made uv colour its "# via" lines, and they failed to parse. (+7 more)

### Community 59 - "test_config_invariants.py"
Cohesion: 0.11
Nodes (16): counts(), fixture, parametrize, slow, Configuration invariants: the living spec of what every yaml guarantees., One factor at a time: each screen yaml equals the base in every field but its…, S5 (legacy recipe): the <= 1% target did not hold (+2.36%); it is reported as a…, Structural parity with the reference: same modules, same shapes, to the… (+8 more)

### Community 60 - "scrub_ler.py"
Cohesion: 0.09
Nodes (36): hmac, Entity, entity_hash(), _pattern(), placeholders(), The German legal-entity scrub: names out before training. Entities come from…, Surface form -> placeholder, numbered per tag in order of first appearance., Replace every occurrence of the propagating forms (longest first, whole words… (+28 more)

### Community 61 - ".__init__"
Cohesion: 0.38
Nodes (4): MTPModule, Module, Tensor, One prediction depth: two norms, a 2D -> D projection, and one ``HybridBlock``.

### Community 62 - "MDebertaNLI"
Cohesion: 0.22
Nodes (8): entailment_index(), MDebertaNLI, The class index whose label is ``entailment`` (any case); ``ValueError`` unless…, mDeBERTa XNLI as an ``NLI`` callable: softmax over the classes, the entailment…, ENTAILMENT_THRESHOLD is the fixed 0.5 threshold; MDebertaNLI loads lazily and…, Regression: ``truncation=True`` is ``longest_first``, which cut the…, test_nli_threshold_is_pre_registered_and_labels_are_read_by_name(), test_nli_truncates_the_premise_never_the_claim()

### Community 63 - "code"
Cohesion: 0.12
Nodes (16): code(), The two shell lookups in fetch_hf.sh must find the constants the loaders use., The command as documented, the same check on the legal base, the CUDA smoke., ARM is resolved inside the job by `screen_arms.py env`, EXPERIMENT from the…, fetch_hf.sh fetched Qwen3-1.7B-Base at the tokenizer's pinned commit, which…, The scrub array and the counts read the same German-market sources; the pack…, A source in configs/dataset/mixture_pretrain.yaml without shards stops every…, runs_training() (+8 more)

### Community 64 - "HybridLightningModule"
Cohesion: 0.14
Nodes (9): Limitations, HybridLightningModule, Optimizer, Tensor, Token-weighted sums for the epoch's ``val/perplexity``. Logged per batch with…, Clip gradients and log the pre-clip norm (manual, so foreach AdamW and AMP…, A WSD scheduler with an absolute warmup; records the decay window for the beta2…, Training loop around a ``HybridLanguageModel``. Args: model: the model to… (+1 more)

### Community 65 - "scrub_file"
Cohesion: 0.17
Nodes (17): iter_part(), main(), part_name(), Path, Each document with its entities, in order, from ONE worker process: the…, Entities per document id from the NER worker in the scrub environment, and its…, The documents on lines ``part, part + parts, part + 2 * parts, ...`` of…, ``<source>.jsonl``, or ``<source>.part-<k>-of-<n>.jsonl`` for one part of a… (+9 more)

### Community 66 - "load_model_config"
Cohesion: 0.11
Nodes (25): Configuration dataclass for the hybrid decoder. Ported from the reference…, Model configuration and the yaml loader every script uses., available_model_configs(), load_model_config(), load_model_yaml(), Any, Path, Load a model yaml into a ``HybridConfig`` -- the one path every script uses. A… (+17 more)

### Community 67 - "greedy"
Cohesion: 0.12
Nodes (16): greedy(), Uncached greedy decoding: the full forward is recomputed for every new token…, Bigram, log_table(), ``logits[b, t] = table[ids[b, t]]``: the next-token scores given the last token., Rows of probabilities -> log-probabilities (so a beam's score is a sum of known…, Finished hypotheses leave the beam with their EOS and are ranked by…, The reference's `length_penalty` divided every live beam by the same length, so… (+8 more)

### Community 68 - "build_dpo_pairs.py"
Cohesion: 0.05
Nodes (51): decode_answer(), prompt_ids(), The prompt's token ids, up to and including ``<|answer|>`` (the positions…, Answer tokens as the pointer markup: special tokens by name, so no tokenizer…, cache_dir(), load_tokenizer(), main(), The student's tokenizer: Qwen3's own, plus the SFT pointer specials.… (+43 more)

### Community 69 - "test_code_hygiene.py"
Cohesion: 0.18
Nodes (17): _code_files(), git_calls_in_python(), git_calls_in_shell(), _is_sys_path_edit(), _names_git(), AST, parametrize, Path (+9 more)

### Community 70 - "_ddp_fit"
Cohesion: 0.11
Nodes (13): _ddp_fit(), _IndexRows, _RankRows, Under DDP (two CPU ranks, gloo) each rank reads its own rows, r, r + 2, ...;…, Row i carries its index, so a run's data order can be read back., Lightning restarts a plain loader at row 0 on resume (screen run S5's…, Stage B: resumed from stage A's last step onto a shorter schedule of its own…, Writes the row indices each DDP rank trains on to <dir>/rank<r>.txt. (+5 more)

### Community 71 - "CausalLMAdapter"
Cohesion: 0.22
Nodes (4): CausalLMAdapter, Module, Tensor, A Hugging Face causal LM behind the hybrid's interfaces: the…

### Community 72 - "test_ci_workflow.py"
Cohesion: 0.16
Nodes (12): fixture, The CI workflow and the local validation script stay in step. PyYAML parses the…, Fast excludes slow/cuda/multigpu/network; --full and --ci exclude…, steps_text(), test_every_installing_job_uses_the_lock_and_never_pip(), test_hygiene_runs_on_a_pristine_checkout(), test_lint_runs_every_static_tool(), test_package_job_imports_the_wheel_outside_the_checkout() (+4 more)

### Community 73 - "test_reference_parity.py"
Cohesion: 0.20
Nodes (13): inspect, _block_defaults(), block_params(), _config_defaults(), load(), model_params(), _per_document(), parametrize (+5 more)

### Community 74 - "train_dpo.py"
Cohesion: 0.09
Nodes (30): copy, collate_pairs(), _pad(), PairSchedule, Dataset, Preference optimisation of an SFT model: DPO and KTO against a frozen reference…, ``num_samples`` pairs on a seeded schedule (one permutation per pass), or every…, (prompt, answer) rows as right-padded ``input_ids`` and ``labels`` (-100 off… (+22 more)

### Community 75 - "cumulative_angles"
Cohesion: 0.17
Nodes (14): SSD scan, plus the trapezoidal term and the complex-state rotation when…, apply_rotary(), cumulative_angles(), cumulative_angles_fp64(), Tensor, Data-dependent rotary embeddings for the complex-valued SSM. Mamba-3 Sec 3.2…, Theta_t = sum_{i<=t} dt_i * theta_i, accumulated in float64 and wrapped to [0,…, ``cumulative_angles`` before the final cast: float64, wrapped to [0, 2pi).… (+6 more)

### Community 76 - "build_dense_index.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_dense_index.sh script

### Community 77 - "lexhybrid"
Cohesion: 0.12
Nodes (17): Acknowledgements and data attribution, Architecture at a glance, Cluster usage, Contributing, Data pipeline and retrieval index, How it answers, Installation, Key properties (+9 more)

### Community 78 - "directives"
Cohesion: 0.10
Nodes (24): directives(), parametrize, An array wrapper picks its work by SLURM_ARRAY_TASK_ID and logs each task on…, A `--gpus` directive is not overridden by an environment variable: a multi-GPU…, check_env.py reads uv.lock through `uv`; a missing uv failed with 'uv not found…, The 4-GPU header, and the body it delegates to passes last.ckpt as…, `#SBATCH --key=value` lines before the first command, trailing comments dropped., The Qwen3-8B-Base teacher fits only with gradient checkpointing. (+16 more)

### Community 79 - "eval_retrieval.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, eval_retrieval.sh script

### Community 80 - "test_rsync_rules.py"
Cohesion: 0.11
Nodes (19): shutil, subprocess, tarfile, dists(), fixture, The built distributions contain the package and nothing else (CI `package`…, parametrize, skipif (+11 more)

### Community 82 - "train_sft.py"
Cohesion: 0.11
Nodes (23): best_val_checkpoint(), collate(), Dataset, The ``save_top_k`` lowest-``val/loss`` checkpoints, ``best-step<N>.ckpt``,…, One split (``train.parquet`` / ``dev.parquet``) as ``{input_ids, labels}``…, Right-pad ``input_ids`` (with 0: no real position sees a pad) and ``labels``…, SFTSplit, ModelCheckpoint (+15 more)

### Community 83 - "MambaBlock"
Cohesion: 0.19
Nodes (9): MambaBlock, Tensor, Re-apply the Mamba dt init after the model's global weight pass (no-op unless…, (batch, seq_len, dim) -> (batch, seq_len, dim); ``doc_ids`` resets state per…, Non-fast-path scan: the same implementation as ``selective_scan`` with chunk…, Per-(row, segment) wrapper that resets the SSM state at document boundaries., Mamba-1 mixer: input projection, depthwise causal conv, selective SSM, gated…, The declared capability and the real signature must not drift apart. (+1 more)

### Community 84 - "DPOLightningModule"
Cohesion: 0.13
Nodes (18): answer_logps(), dpo_loss(), DPOLightningModule, kto_loss(), Tensor, Args: those of ``HybridLightningModule`` plus reference: the frozen reference…, The loss and the logged rewards (``beta * (pi - ref)``) of a ``collate_pairs``…, Mean DPO loss over pairs; every argument is (B,) summed answer log-… (+10 more)

### Community 85 - "resync_index.py"
Cohesion: 0.20
Nodes (15): document_digest(), DocumentDigest, indexed_documents(), ``{document id: digest}`` of the documents an index was built from…, What an index records of one input document (a ``documents.jsonl`` row)., ``index.json`` of an index directory; ``FileNotFoundError`` when it is not an…, read_manifest(), compare() (+7 more)

### Community 86 - "test_repo_hygiene.py"
Cohesion: 0.23
Nodes (12): make_tree(), parametrize, Path, scripts/check_repo_hygiene.py on synthetic trees., test_a_missing_critical_file_is_reported(), test_an_ignored_analysis_document_is_reported(), test_case_collisions_are_caught(), test_clean_tree_passes() (+4 more)

### Community 87 - "German-law pretraining corpus: tokens, deduplication, scrub"
Cohesion: 0.15
Nodes (12): Budget rule, Deduplication (drop within a source, keep and link across), Discussion, Efficiency, Estimate vs exact, German-law pretraining corpus: tokens, deduplication, scrub, Limitations, Reproduction (+4 more)

### Community 88 - "filters.py"
Cohesion: 0.21
Nodes (14): date, _allowed(), _bound(), _day(), filter_chunks(), _parse(), Retrieval filters: jurisdiction, validity date and document type.…, A ``date`` from a date, a datetime (its day) or an ISO date string. (+6 more)

### Community 89 - "build_doc_block_mask"
Cohesion: 0.14
Nodes (13): Limitations, disable, build_doc_block_mask(), FlexAttention ``BlockMask`` for causal attention inside each document of a…, _packed_doc_ids(), linux_only, Tensor, Semantics of the block mask, checked with eager FlexAttention (runs… (+5 more)

### Community 90 - "build_sft_data.py"
Cohesion: 0.10
Nodes (30): _unit(), question_key(), A hash of the question with its form normalised (NFC, case, whitespace, final…, German sentences of a passage: split after ``.``, ``!`` or ``?`` before an…, split_sentences(), build(), evalset_guard(), is_eval() (+22 more)

### Community 91 - "probe_ckpt_size.sh"
Cohesion: 0.15
Nodes (12): BATCH_SIZE, CKPT_EVERY, DATASET_CONFIG, DISTILL_CFG, EXPERIMENT, EXTRA_OVERRIDES, MAX_STEPS, MODEL_CONFIG (+4 more)

### Community 92 - "SFTLightningModule"
Cohesion: 0.12
Nodes (8): Args: those of ``HybridLightningModule`` (``learning_rate`` is the plain value…, ``(mean CE over the supervised targets, their count)``; position t predicts…, SFTLightningModule, _CaptureLogger, The answer loss never uses the MTP head; DDP would wait for gradients it never…, Keeps every ``log_metrics`` call in memory: ``rows = [(step, {tag: value})]``., test_sft_freezes_an_mtp_head_outside_its_loss(), __init__()

### Community 93 - "check_repo_hygiene.py"
Cohesion: 0.33
Nodes (11): check_critical(), check_files(), list_files(), load_ignore(), main(), Path, Critical files must exist; locally they must also survive ``.gitignore``. In…, Repository hygiene: the tree that would be (or was) pushed is clean. Two modes,… (+3 more)

### Community 94 - "screen_table.py"
Cohesion: 0.33
Nodes (11): apply_rule(), arm_means(), build_rows(), _f(), main(), parse_run(), The architecture screen's seed table, per-arm means and the decision rule.…, read_eval() (+3 more)

### Community 95 - "Profile ladder, chunk size and distillation memory"
Cohesion: 0.18
Nodes (10): Chunk size (training, forward+backward, slab loss, packed, doc_len 1,000, L=4,096), Discussion, Distillation memory probe, Hypotheses and criteria, Profile ladder, chunk size and distillation memory, Reproduction, Results, Sanity points (L=16,384, forward, bf16, batch 4) (+2 more)

### Community 96 - "README.md"
Cohesion: 0.32
Nodes (3): Parameter counts of the legal model configs, The predicted counts and bands, and analysis/param_counts.md is current., test_param_bands()

### Community 97 - "Architecture screen: ablations of the hybrid backbone"
Cohesion: 0.17
Nodes (11): Architecture screen: ablations of the hybrid backbone, Discussion, Efficiency, Hypotheses and selection rule, Paired comparisons, Per-seed calls (where the seeds disagree), Reproduction, Results (+3 more)

### Community 98 - "BM25Index"
Cohesion: 0.11
Nodes (28): BM25Index, Path, The index's tokens of each text (lower-cased, German stopwords out, stemmed)., A ``bm25s`` index over chunks, queried by text., Index ``chunks`` (anything with ``id`` and ``text``:…, The best ``k`` chunks for ``text``, best first; chunks sharing no term score 0…, _stem(), tokenize() (+20 more)

### Community 99 - "gpu_tests.sh"
Cohesion: 0.20
Nodes (10): CUDA_LAUNCH_BLOCKING, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PATH, PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, gpu_tests.sh script (+2 more)

### Community 100 - "multigpu_tests.sh"
Cohesion: 0.20
Nodes (10): CUDA_LAUNCH_BLOCKING, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PATH, PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, multigpu_tests.sh script (+2 more)

### Community 101 - "train_pretrain_1gpu.sh"
Cohesion: 0.20
Nodes (10): CUDA_LAUNCH_BLOCKING, HF_DATASETS_CACHE, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, opt(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF (+2 more)

### Community 102 - "Contributing"
Cohesion: 0.15
Nodes (10): Cluster jobs, Continuous integration, Contributing, Dependencies, Lint and style, Repository hygiene, Setup, Tests (+2 more)

### Community 103 - "check_configs.py"
Cohesion: 0.27
Nodes (9): hydra, hydra_core_global_hydra, check_model(), compose_model(), main(), model_names(), validate.sh gate 3: every model yaml composes through Hydra and pins what it…, Problems with ``configs/model/<name>.yaml``; empty when it passes. (+1 more)

### Community 104 - "ScheduleSampler"
Cohesion: 0.19
Nodes (6): _rank_world(), A training mixture's rows for one rank, from where the run stands. Sample ``i``…, Every row of an evaluation set across the ranks, each rank the same count…, ScheduleSampler, ShardSampler, test_schedule_sampler_splits_ranks_evenly_and_skips_what_was_trained()

### Community 106 - ".allowed"
Cohesion: 0.23
Nodes (7): process(), Tensor, The state after ``token``; raises ``ValueError`` if the grammar forbids it., Boolean (vocab,) mask of the tokens the grammar allows next., Set every forbidden logit to ``-inf``, row by row; ``generated`` (B, T) are the…, A ``logits_processor(logits, seqs)`` for the decoders in…, State

### Community 107 - "conftest.py"
Cohesion: 0.12
Nodes (19): importlib_util, reference, load_script(), fixture, pytest_collection_modifyitems(), Shared test configuration. Markers are declared in pyproject.toml under…, Every test starts from the same RNG state, so failures reproduce., Import ``scripts/<name>.py`` as a module without touching sys.path. (+11 more)

### Community 108 - "equivalence.sh"
Cohesion: 0.22
Nodes (9): CUDA_LAUNCH_BLOCKING, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, equivalence.sh script, step() (+1 more)

### Community 109 - "kd_memory_probe.sh"
Cohesion: 0.22
Nodes (9): DATASET_CONFIG, GRAD_CKPT, MAX_STEPS, MODEL_CONFIG, NUM_GPUS, probe(), SAVE_TOP_K, kd_memory_probe.sh script (+1 more)

### Community 110 - "Architecture"
Cohesion: 0.25
Nodes (8): Answer pipeline, Architecture, Data pipeline, Renderer, Retrieval, Tokenizer and pointer vocabulary, Training, Verifier

### Community 111 - "judge.py"
Cohesion: 0.20
Nodes (24): answer_text(), correctness_prompt(), faithfulness_prompt(), judge_prompts(), _listed(), _numbered(), precision_prompt(), The independent answer judge: a strong LLM, not the verifier, scores the… (+16 more)

### Community 112 - "_doc"
Cohesion: 0.07
Nodes (40): admits(), build_corpus(), Corpus assembly gate: which documents may enter which training arm. Two arms:…, Whether ``doc`` (already licence-checked) belongs in ``arm``., Yield the documents of ``arm``; raise ``LicenceError`` on the first licence…, check_document(), is_commercial_safe(), Licence (+32 more)

### Community 113 - "evaluate_lm.py"
Cohesion: 0.14
Nodes (13): evaluate_perplexity(), load_model(), main(), measure_throughput(), ParquetRows, Dataset, no_grad, Tensor (+5 more)

### Community 114 - "argparse"
Cohesion: 0.33
Nodes (9): argparse, _fmt(), latest_version(), main(), Path, Validation and training curves of the screen runs, read from their TensorBoard…, ``<run>/logs/<exp>/version_<n>`` with the highest n that holds an events file., read_scalars() (+1 more)

### Community 115 - "preflight.sh"
Cohesion: 0.25
Nodes (8): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PATH, PYTHONUNBUFFERED, preflight.sh script, step(), TORCHINDUCTOR_CACHE_DIR

### Community 116 - "profile.sh"
Cohesion: 0.25
Nodes (8): CUDA_LAUNCH_BLOCKING, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, point(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, profile.sh script

### Community 117 - "validate.sh"
Cohesion: 0.39
Nodes (8): fail(), FAILED, pass(), PASSED, run_gate(), validate.sh script, warn(), WARNED

### Community 118 - "test_judge.py"
Cohesion: 0.09
Nodes (26): requests, fake_post(), post(), FakeResponse, _qa_row(), The judge's HTTP client, its CLI, and its calibration items and scoring (all…, Patch ``requests.post``; ``reply(prompt)`` gives the completion. Returns the…, Every production prompt gets its items; the split goes by context or provision… (+18 more)

### Community 119 - "Tokenizer fertility on DACH legal text"
Cohesion: 0.25
Nodes (7): Discussion, Hypotheses and method, Limitations, Reproduction, Results, Summary, Tokenizer fertility on DACH legal text

### Community 120 - "_Paras"
Cohesion: 0.25
Nodes (3): _Paras, HTMLParser, The ``<div class="para">`` blocks as ``(bold, text)``: ``bold`` is the text of…

### Community 121 - ".from_json"
Cohesion: 0.07
Nodes (31): Change, changes_since(), main(), Path, The currency changelog: which provisions changed since a date. A provision's…, write_changelog(), Hex SHA-256 of the UTF-8 text: the identity dedup and the manifests key on., sha256_text() (+23 more)

### Community 122 - "train_sft_hf.py"
Cohesion: 0.24
Nodes (12): ``lm`` frozen but for rank-``rank`` LoRA adapters on ``LORA_TARGETS`` and the…, wrap_lora(), adapt(), load_lm(), main(), DictConfig, main, The Hugging Face baseline of supervised fine-tuning: ``sft.hf_model``… (+4 more)

### Community 123 - "Backbone results: the hybrid and a Transformer baseline at 8,192 tokens"
Cohesion: 0.18
Nodes (10): Backbone results: the hybrid and a Transformer baseline at 8,192 tokens, Discussion, Handoff, Limitations, Predictions, Probes on the stage-B checkpoints, by context length, Reproduction, Results (+2 more)

### Community 124 - "pytest"
Cohesion: 0.15
Nodes (13): pytest, CUDA checks. Marked ``cuda``: they collect everywhere and skip, with the…, CUDA variant of the compiled legal base check: the compiled legal base equals…, Under an outer torch.compile, a block mask traced inside the graph was wrong…, _reduced_base(), test_attn_impl_auto_takes_flex_on_gpu(), test_compile_equals_eager_on_gpu(), test_compiled_flex_attention_matches_eager_per_row() (+5 more)

### Community 125 - "analyze_generation_diversity.py"
Cohesion: 0.16
Nodes (18): analyse(), distinct_n(), duplicate_clusters(), main(), _ngram_counts(), Mean BLEU-4 of each text against a sample of the OTHERS. Higher = more…, Render the comparison table plus the verdict., Boilerplate / duplicate-template analysis for generated answers, with controls.… (+10 more)

### Community 126 - "build_corpus_array.sh"
Cohesion: 0.33
Nodes (5): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_corpus_array.sh script

### Community 127 - "pack_corpus.sh"
Cohesion: 0.33
Nodes (5): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, pack_corpus.sh script, TOKENIZERS_PARALLELISM

### Community 128 - "OpenAICompatible"
Cohesion: 0.33
Nodes (4): OpenAICompatible, ``POST {base_url}/chat/completions``: one user message, temperature 0; returns…, A draft verdict inside a reasoning block is not the reply; a 200 without…, test_a_reasoning_draft_and_a_malformed_success_are_not_verdicts()

### Community 129 - "measure_fertility.py"
Cohesion: 0.12
Nodes (25): functools, encode_document(), Token ids of raw document text; special-token strings inside it are split into…, qwen3_encoder(), count(), main(), manifest_counts(), Path (+17 more)

### Community 130 - "corpus_counts.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, corpus_counts.sh script

### Community 131 - "diag_compile.sh"
Cohesion: 0.40
Nodes (4): HF_HUB_OFFLINE, PYTHONUNBUFFERED, diag_compile.sh script, TORCHINDUCTOR_CACHE_DIR

### Community 132 - "eval_probes.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, eval_probes.sh script

### Community 133 - "fetch_hf.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, fetch_hf.sh script

### Community 134 - "scrub_array.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, scrub_array.sh script

### Community 135 - "setup_env.sh"
Cohesion: 0.40
Nodes (4): PYTHONUNBUFFERED, setup_env.sh script, UV_CACHE_DIR, UV_LINK_MODE

### Community 136 - "train_pretrain_4gpu.sh"
Cohesion: 0.40
Nodes (4): NUM_GPUS, SAVE_TOP_K, train_pretrain_4gpu.sh script, TRAINER_CFG

### Community 137 - "answer.py"
Cohesion: 0.09
Nodes (30): CrossEncoderScorer, ``bge-reranker-v2-m3`` as a ``Scorer``: one logit per (question, passage) pair,…, load_sft_model(), The SFT model of ``checkpoint``: the hybrid of ``config``, or a…, The verifier log line: no answer text, so it can be kept beside the manifests., VerifierResult, add_arguments(), answer() (+22 more)

### Community 138 - "test_diag_spike_reruns_the_arm_observed_in_its_own_directory"
Cohesion: 0.40
Nodes (5): _compose(), The fake interpreter recorded `distill=null` happily; real Hydra refused it…, Loss-spike rerun: S1-s42 with the screen settings, seed and schedule unchanged…, test_diag_spike_reruns_the_arm_observed_in_its_own_directory(), test_every_train_argument_list_composes_under_real_hydra()

### Community 139 - "test_imports.py"
Cohesion: 0.24
Nodes (10): importlib, pkgutil, _exists_with_exact_case(), _internal_targets(), AST, parametrize, Path, Every module imports, and every internal import names the file with its exact… (+2 more)

### Community 140 - "ByteTokenizer"
Cohesion: 0.14
Nodes (19): _jsonl(), _qa_file(), _qa_row(), One evaluation-only set's manifest: the raw questions' hashes, as the…, QA rows with the writer's ``info.json`` beside them, free for training unless…, build_sft_data writes train and dev splits in which every target ends in…, QA items are all answerable: when they hold the share below 0.2, a seeded…, A QA file without info.json, or one that does not free its items for training;… (+11 more)

### Community 141 - "judge_answers.py"
Cohesion: 0.21
Nodes (13): add_judge_args(), judge_from_args(), judge_record(), ArgumentParser, ``(scores, details)``: a score per metric (``None`` when not applicable or not…, Namespace, check_validated(), main() (+5 more)

### Community 142 - "filter_logits"
Cohesion: 0.25
Nodes (7): filter_logits(), Temperature, then top-k, then nucleus (top-p) filtering; filtered entries…, choose(), Shared sampling filter (``lexhybrid.decoding.filter_logits``)., test_filter_logits(), generator(), run()

### Community 143 - "_executable"
Cohesion: 0.40
Nodes (5): _executable(), Path, The log is the job's own stdout, and GNU grep refuses to read its output file…, test_fetch_hf_models_override_fetches_only_those_at_main_unless_pinned(), test_kd_probe_summary_reads_its_own_log()

### Community 162 - "test_judge_parses_scores_and_caches"
Cohesion: 0.18
Nodes (10): average_precision(), first_json_object(), _parse(), The first ``{...}`` in ``text`` that parses as a JSON object (code fences and…, ``(read(reply object), reply object)``, retried once; ``(None, {"raw":…, RAGAS context precision: precision@k averaged over the positions k of the…, read(), Read (+2 more)

### Community 167 - "checkpoint.py"
Cohesion: 0.12
Nodes (27): detect_prefix(), _first_segments(), infer_architecture(), infer_layer_types(), InferredArchitecture, load_every_key(), load_state_dict_guarded(), _mixer_params() (+19 more)

### Community 168 - "test_compile_forward_chunk128"
Cohesion: 0.40
Nodes (4): linux_only, slow, Hybrid_legal_base at reduced width -- chunk 128 for both Mamba-3 and mLSTM,…, test_compile_forward_chunk128()

### Community 169 - "test_pipeline.py"
Cohesion: 0.10
Nodes (20): fixture, End-to-end answer pipeline on CPU: index -> retrieve -> prompt -> constrained…, With best-of-n's samples scripted (every token one the FSM allows), the…, Half of the samples or more being the model's own refusal decides, though a…, Questions in either form: ``answers`` are gold answers; ``gold`` citations…, No questions and no gold prompts: a clean exit and an empty summary, with no…, A sample cut off by the budget scores below any finished one, even a refused…, The CLI retrieves the planted passage, decodes under the pointer FSM and prints… (+12 more)

### Community 170 - "dump_reference_parity.py"
Cohesion: 0.31
Nodes (9): block_cases(), build_all(), import_reference(), inputs(), main(), model_cases(), Register the reference package without running its __init__ and without…, Dump the reference-parity fixtures -- run once, while Reference/ still exists.… (+1 more)

### Community 171 - "collections"
Cohesion: 0.36
Nodes (7): collections, calibrate(), main(), Path, Calibrate the verifier's entailment threshold on hand-labelled premise/claim…, read_pairs(), scores()

### Community 172 - "teacher_forward_packed"
Cohesion: 0.32
Nodes (8): no_grad, Tensor, ``(position_ids, attention_mask)`` for a Hugging Face decoder on packed rows.…, The teacher's final hidden states (after its final norm), (B, L, D_teacher),…, teacher_forward_packed(), teacher_inputs_packed(), A packed row reaches the teacher with per-document position ids and a block-…, test_teacher_position_ids_isolation()

### Community 173 - "test_tfla_wide_chunk_factorises_per_block_on_gpu"
Cohesion: 0.33
Nodes (7): _packed_doc_ids(), parametrize, Tensor, On the GPU, a 128-chunk whose log-decay overflows one factorisation (forget…, CUDA variant of tests/test_layers.py::test_flex_block_mask_equals_dense: the…, test_flex_block_mask_equals_dense_on_gpu(), test_tfla_wide_chunk_factorises_per_block_on_gpu()

### Community 174 - "test_tfla_wide_chunks_factorise_per_block_exactly"
Cohesion: 0.33
Nodes (7): 256 positions (two chunks of 128), two rows, boundaries inside and on chunk…, Trained forget gates widened a 128-chunk's log-decay past fp32 (half-range 43.2…, In fp32 the per-block path forms no inf, forward or backward (a masked inf…, test_tfla_per_block_path_is_finite_and_close_in_fp32(), test_tfla_wide_chunks_factorise_per_block_exactly(), _tfla_grads(), _wide_tfla_inputs()

### Community 175 - "Judge"
Cohesion: 0.18
Nodes (12): Complete, Judge, Path, A ``Complete`` behind the reply cache; ``calls`` counts the completions…, Another endpoint, model or prompt version never sees a cached verdict (a…, Items of one type alone never meet the bar: the metrics without items are named., 429, 5xx, lost connections and timeouts wait 2, 4, 8, 16 s between five tries…, test_cache_key_and_bad_cache_entries() (+4 more)

### Community 176 - "sample"
Cohesion: 0.33
Nodes (6): Uncached sampling with temperature / top-k / top-p; ``generator`` makes it…, sample(), test_sample_is_reproducible_and_top_k_one_is_greedy(), Same logits, same generator: the KV-cached sampler and the uncached ``sample``…, test_hf_sampling_through_its_cache_matches_the_uncached_reference(), sample()

### Community 177 - "schedulers.py"
Cohesion: 0.12
Nodes (19): LambdaLR, Apply the beta2 schedule when WSD and ``beta2_schedule`` are both active., apply_beta2_schedule(), beta2_for_step(), Optimizer, Warmup-Stable-Decay (WSD) learning-rate schedule and the beta2 anneal (ported…, Set every param group's beta2 to the scheduled value in place; returns it for…, WSD learning-rate multiplier at ``step``. warmup: linear from… (+11 more)

### Community 178 - "check_run_settings"
Cohesion: 0.40
Nodes (5): check_run_settings(), Path, One output directory, one run. A job that finds ``last.ckpt`` there resumes it,…, The saved settings guard a directory: same settings resume, other settings are…, test_a_run_directory_holds_one_run()

### Community 179 - "synth_qa.sh"
Cohesion: 0.20
Nodes (8): scripts_slurm_lib_vllm_server_sh, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, synth_qa.sh script, HF_HOME, HF_HUB_OFFLINE, vllm_smoke.sh script

### Community 180 - "train_sft_4gpu.sh"
Cohesion: 0.22
Nodes (9): HF_DATASETS_CACHE, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, opt(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, train_sft_4gpu.sh script (+1 more)

### Community 181 - "hf_run"
Cohesion: 0.40
Nodes (5): hf_run(), fixture, A real model's next-token distribution made uniform at every position over EOS,…, A tiny Qwen3 SFT checkpoint on the same splits: its config has the tokenizer's…, _rig_unigram()

### Community 182 - "test_sync_scripts.py"
Cohesion: 0.07
Nodes (36): CompletedProcess, SimpleNamespace, stat, `--docs-per-batch` puts the sentences of several documents into ONE predict…, test_scrub_worker_tags_a_batch_of_documents_in_one_predict_call(), __init__(), Two runs of one system and seed (SFT and DPO) get their own outputs, so the…, counting() (+28 more)

### Community 183 - "Corpus licence register"
Cohesion: 0.50
Nodes (4): Corpus licence register, Evaluation sets (never trained on), Models the pipeline runs, Training sources (collectors)

### Community 184 - "train_dpo_4gpu.sh"
Cohesion: 0.29
Nodes (7): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, opt(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, train_dpo_4gpu.sh script

### Community 185 - "_units"
Cohesion: 0.50
Nodes (4): A paragraph's sentences; one longer than a window (a table, a list of motions)…, ``(section, sentence)`` windows of one part: a window closes at a paragraph…, _units(), _windows()

### Community 186 - "generated_tokens"
Cohesion: 0.67
Nodes (3): generated_tokens(), The answer part of a decoded row: after the prompt, up to and including the…, test_generated_tokens_cuts_after_eos()

### Community 187 - "paragraphs"
Cohesion: 0.33
Nodes (6): is_heading(), paragraphs(), A page's footnotes interrupt the paragraph that runs on to the next page ("...…, The paper's paragraphs as ``(is_heading, text)``, cleaned of the PDF's layout…, _rejoin_across_footnotes(), test_dip_paragraph_rules()

### Community 191 - "setup_vllm.sh"
Cohesion: 0.33
Nodes (5): PYTHONUNBUFFERED, setup_vllm.sh script, UV_CACHE_DIR, UV_LINK_MODE, UV_PROJECT_ENVIRONMENT

### Community 195 - "build_dpo_pairs.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_dpo_pairs.sh script

### Community 196 - "build_sft_data.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_sft_data.sh script

### Community 197 - "eval_gold_passages.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, eval_gold_passages.sh script

### Community 198 - "eval_system_ablation.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, eval_system_ablation.sh script

## Knowledge Gaps
- **237 isolated node(s):** `lexhybrid-scrub`, `lexhybrid-vllm`, `lexhybrid`, `build_corpus_array.sh script`, `HF_HOME` (+232 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1774 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **27 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HybridLanguageModel` connect `HybridLanguageModel` to `test_data.py`, `performance_profile.py`, `answer.py`, `test_dpo.py`, `filter_logits`, `test_models.py`, `test_numerics_fp64.py`, `train_pretrain.py`, `evaluate_val_ppl.py`, `PretrainLightningModule`, `check_operator_equivalence.py`, `RMSNorm`, `test_scripts.py`, `HybridBlock`, `HybridConfig`, `datasets.py`, `screen_arms.py`, `test_compile_forward_chunk128`, `test_pipeline.py`, `test_training.py`, `checkpoint.py`, `test_decoding.py`, `test_slurm_wrappers.py`, `sft_format.py`, `torch`, `os`, `test_config_invariants.py`, `HybridLightningModule`, `load_model_config`, `test_reference_parity.py`, `train_dpo.py`, `train_sft.py`, `DPOLightningModule`, `evaluate_lm.py`, `.from_json`, `pytest`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Why does `Document` connect `Document` to `packing.py`, `measure_fertility.py`, `test_data.py`, `HybridLanguageModel`, `test_retrieval.py`, `register`, `test_collect_offline.py`, `collectors/base.py`, `test_verifier.py`, `pathlib`, `schema.py`, `fedlex.py`, `datasets.py`, `dedup.py`, `test_pipeline.py`, `statute_recall.py`, `sft_format.py`, `eurlex.py`, `scrub_ler.py`, `scrub_file`, `resync_index.py`, `BM25Index`, `Architecture`, `_doc`, `.from_json`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Why does `HybridConfig` connect `HybridConfig` to `HybridLanguageModel`, `test_data.py`, `test_layers.py`, `answer.py`, `test_dpo.py`, `test_models.py`, `test_numerics_fp64.py`, `train_pretrain.py`, `test_scripts.py`, `HybridBlock`, `datasets.py`, `screen_arms.py`, `test_training.py`, `test_pipeline.py`, `checkpoint.py`, `test_decoding.py`, `torch`, `test_config_invariants.py`, `load_model_config`, `test_reference_parity.py`, `train_dpo.py`, `train_sft.py`, `check_configs.py`, `.from_json`, `pytest`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 57 inferred relationships involving `Document` (e.g. with `Data pipeline` and `Document schema`) actually correct?**
  _`Document` has 57 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `HybridLanguageModel` (e.g. with `Model` and `__getattr__()`) actually correct?**
  _`HybridLanguageModel` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 60 inferred relationships involving `Chunk` (e.g. with `answer_words()` and `cited_passages()`) actually correct?**
  _`Chunk` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `HybridConfig` (e.g. with `Model` and `load_model_config()`) actually correct?**
  _`HybridConfig` has 6 INFERRED edges - model-reasoned connections that need verification._