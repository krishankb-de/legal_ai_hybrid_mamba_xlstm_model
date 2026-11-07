# Graph Report - legal_ai_hybrid_mamba_xlstm_model  (2026-11-07)

## Corpus Check
- 266 files · ~266,527 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 62 file(s) not represented in the graph (top: .jsonl 30, .xml 12, (none) 6)

## Summary
- 4208 nodes · 9765 edges · 193 communities (162 shown, 31 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 512 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `21a213ba`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Tensor
- answer.py
- test_data.py
- collectors/base.py
- HybridConfig
- test_layers.py
- evalsets/base.py
- dedup.py
- performance_profile.py
- test_pipeline.py
- .forward
- bootstrap_compare.py
- train_dpo.py
- http_get
- test_models.py
- resync_index.py
- rel_max_err
- generate.py
- Resp
- train_pretrain.py
- test_numerics_fp64.py
- evaluate_val_ppl.py
- test_decoding.py
- eurlex.py
- PretrainLightningModule
- analyze_generation_diversity.py
- cumulative_angles
- AttentionBlock
- torch
- test_scripts.py
- Chunk
- test_synth_qa.py
- CausalLMAdapter
- HybridLanguageModel
- SpikeDiagnosticsCallback
- FetchError
- fedlex.py
- datasets.py
- chunk_dict
- DIPCollector
- test_training.py
- Mamba3Block
- eval_retrieval.py
- ssd_interface.py
- ._gates
- SpecialTokens
- generate_synthetic_qa.py
- sandbox
- teacher
- check_operator_equivalence.py
- check_env.py
- Example
- test_verifier.py
- filter_logits
- sequential_mlstm_reference
- filters.py
- test_slurm_wrappers.py
- _tfla_model
- test_check_env.py
- test_config_invariants.py
- .from_json
- slab_cross_entropy
- test_dpo.py
- code
- HybridLightningModule
- collections_abc
- load_model_config
- greedy
- prompt_ids
- test_ci_workflow.py
- test_eval.py
- NLI verifier threshold on 100 hand-built statute pairs
- Corpus licence register
- screen_metrics.py
- build_judge_calibration.py
- eval_gold_passages.py
- build_dense_index.sh
- lexhybrid
- parametrize
- eval_retrieval.sh
- test_rsync_rules.py
- _BlockParser
- train_sft.py
- MambaBlock
- dpo_module.py
- measure_fertility.py
- test_repo_hygiene.py
- German-law pretraining corpus: tokens, deduplication, scrub
- tfla_forward_parallel
- screen_arms.py
- build
- probe_ckpt_size.sh
- _CaptureLogger
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
- test_tfla_chunk_size_reaches_kernel
- strip_prefixes
- callbacks.py
- PointerFSM
- conftest.py
- equivalence.sh
- kd_memory_probe.sh
- Architecture
- judge.py
- _doc
- Judge calibration and validation on gold labels
- AnswerRecord
- preflight.sh
- profile.sh
- validate.sh
- test_judge.py
- Tokenizer fertility on DACH legal text
- _Paras
- parse_citation
- Backbone results: the hybrid and a Transformer baseline at 8,192 tokens
- test_gpu.py
- ScheduleSampler
- build_corpus_array.sh
- pack_corpus.sh
- _s1_fingerprint
- run_metadata.py
- corpus_counts.sh
- diag_compile.sh
- eval_probes.sh
- fetch_hf.sh
- scrub_array.sh
- setup_env.sh
- train_pretrain_4gpu.sh
- ParquetRows
- test_diag_spike_reruns_the_arm_observed_in_its_own_directory
- test_verify_handoff_checks_every_run_and_fails_if_one_does
- test_eval_probes_scores_named_runs_at_another_length
- test_reference_parity.py
- HybridBlock
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
- Judge
- lexhybrid
- lexhybrid-scrub
- test_utils.py
- Retrieval on GerLeRB and GerDaLIR
- MDebertaNLI
- record_metrics
- test_multigpu.py
- StopAtStepCallback
- sys
- test_hf_sampling_through_its_cache_matches_the_uncached_reference
- StepStatsCallback
- Document
- synth_qa.sh
- train_sft_4gpu.sh
- test_sync_scripts.py
- train_dpo_4gpu.sh
- FakeQwen
- schedulers.py
- setup_arm_env.sh
- setup_vllm.sh
- test_tfla_wide_chunks_factorise_per_block_exactly
- pathlib
- build_dpo_pairs.sh
- build_sft_data.sh
- eval_gold_passages.sh
- eval_system_ablation.sh
- vllm_server.sh
- lexhybrid-arm
- test_compile_forward_chunk128
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
9. `sandbox()` - 41 edges
10. `http_get()` - 32 edges

## Surprising Connections (you probably didn't know these)
- `Data pipeline` --references--> `Document`  [INFERRED]
  docs/ARCHITECTURE.md → lexhybrid/data/schema.py
- `Hypotheses and method` --references--> `encode_document()`  [INFERRED]
  analysis/tokenizer_fertility.md → lexhybrid/data/tokenizer.py
- `Tokenizer and pointer vocabulary` --references--> `encode_document()`  [INFERRED]
  docs/ARCHITECTURE.md → lexhybrid/data/tokenizer.py
- `How it answers` --references--> `MDebertaNLI`  [INFERRED]
  README.md → lexhybrid/verifier/nli.py
- `Model` --references--> `HybridConfig`  [INFERRED]
  docs/ARCHITECTURE.md → lexhybrid/config/hybrid_config.py

## Import Cycles
- None detected.

## Communities (193 total, 31 thin omitted)

### Community 0 - "Tensor"
Cohesion: 0.13
Nodes (9): no_grad, Tensor, Final norm and LM head: residual stream -> logits., Uncached sampling (``lexhybrid.decoding.sample``): the full forward per new…, One token of hidden state -> next-token logits, advancing every layer's cache., Consume a prompt in ONE chunked forward per layer, filling every cache. Each…, ``generate`` over the caches (``lexhybrid.decoding.sample_cached``): one…, Reindex every cached tensor along the batch axis (beam search). Beams reorder… (+1 more)

### Community 1 - "answer.py"
Cohesion: 0.05
Nodes (54): ClaimCheck, pooled_citation_precision(), _precision(), NLI, The verifier log line: no answer text, so it can be kept beside the manifests., Passed cited claims over cited claims, pooled over many answers (``None`` if…, The verified, rendered answer of the model's raw ``answer`` (see the module…, _refusal() (+46 more)

### Community 2 - "test_data.py"
Cohesion: 0.04
Nodes (68): BGerCollector, EURLexCollector, FineWeb2DECollector, GIICollector, parse_toc(), ``(slug, xml.zip URL)`` for every law in ``gii-toc.xml``., read_zip_xml(), MultiLegalPileCollector (+60 more)

### Community 3 - "collectors/base.py"
Cohesion: 0.04
Nodes (85): datetime, html_parser, io, cli(), Collector, licence_flags(), main(), parse_limit() (+77 more)

### Community 4 - "HybridConfig"
Cohesion: 0.07
Nodes (30): HybridConfig, Validate and auto-compute derived parameters., The per-mixer arguments for layer ``layer_idx`` (a readable summary; not used…, Every architectural knob of the hybrid decoder. One dataclass serves every…, Convert the config to a plain dictionary., Create a config from a dictionary produced by ``to_dict``., Write ``config.json`` into ``save_directory`` and return its path., HybridEmbedding (+22 more)

### Community 5 - "test_layers.py"
Cohesion: 0.06
Nodes (59): mLSTMBlock, (Re-)apply the gate-bias inits after the parent model's global weight pass.…, mLSTM decode state: the matrix cell ``C`` and the normalizer ``n``, both zeros,…, mLSTM mixer: per-dimension exponential input gate, sigmoid forget gate, matrix…, _bc_bias_pair(), _continue_with_steps(), _drive_lambda_to_one(), m3_block() (+51 more)

### Community 6 - "evalsets/base.py"
Cohesion: 0.07
Nodes (55): gzip, html, cli(), EvalSetInfo, Passage, Path, QAItem, QASet (+47 more)

### Community 7 - "dedup.py"
Cohesion: 0.11
Nodes (27): deduplicate(), deduplicate_signatures(), deduplicate_stream(), DedupLog, keep_order(), minhash(), Deduplication of the training corpus: exact, then near-duplicate. 1. **Exact.**…, ``deduplicate_signatures`` over documents streamed once, their MinHashes… (+19 more)

### Community 8 - "performance_profile.py"
Cohesion: 0.05
Nodes (55): contextlib, csv, Summed CE and summed masked KL for one slab (recomputed in the backward pass)., Slab-wise ``(1 - alpha) * CE + alpha * T^2 * KL`` without (B, L, V) logits.…, slab_ce_kl(), _slab_terms(), attention_backend(), available_configs() (+47 more)

### Community 9 - "test_pipeline.py"
Cohesion: 0.08
Nodes (27): pytest, Two runs of one system and seed (SFT and DPO) get their own outputs, so the…, test_eval_gold_passages_wrapper_keeps_runs_apart_and_reuses_sets(), run(), fixture, End-to-end answer pipeline on CPU: index -> retrieve -> prompt -> constrained…, With best-of-n's samples scripted (every token one the FSM allows), the…, Half of the samples or more being the model's own refusal decides, though a… (+19 more)

### Community 11 - "bootstrap_compare.py"
Cohesion: 0.18
Nodes (15): align(), evaluate_subset(), main(), paired_bootstrap(), paired_items(), _paired_mean(), Resample item indices ONCE per draw and score both systems on them., Paired bootstrap comparison of two systems scored on the same items. Ported… (+7 more)

### Community 12 - "train_dpo.py"
Cohesion: 0.10
Nodes (26): copy, load_sft_model(), PairSchedule, Dataset, The SFT model of ``checkpoint``: the hybrid of ``config``, or a…, ``num_samples`` pairs on a seeded schedule (one permutation per pass), or every…, build_callbacks(), build_dataloaders() (+18 more)

### Community 13 - "http_get"
Cohesion: 0.07
Nodes (29): _cached(), http_get(), http_post(), HttpCache, Path, RateLimiter, Successful responses on disk, keyed by the request (method, URL, query, JSON…, GET with a per-host rate limit and exponential backoff on 429/5xx and… (+21 more)

### Community 14 - "test_models.py"
Cohesion: 0.09
Nodes (38): _delta_at_init(), _delta_model(), parametrize, Model-level behaviour of HybridLanguageModel., The model's weight pass zeroed both gate biases; `post_model_init` restores the…, A `.to(bfloat16)` model without autocast once died in the SSD einsums., A document's last token (its EOS) does not predict the next document's first;…, Init then tie; `non_embedding = total - embedding - (lm_head if untied)`, so… (+30 more)

### Community 15 - "resync_index.py"
Cohesion: 0.08
Nodes (36): document_digest(), DocumentDigest, file_sha256(), indexed_documents(), IndexVersion, latest(), Path, ``{document id: digest}`` of the documents an index was built from… (+28 more)

### Community 16 - "rel_max_err"
Cohesion: 0.13
Nodes (26): Chunk-parallel SSD scan. h_t = exp(dt_t A) h_{t-1} + sum_terms coeff_t (B_t…, ssd_chunked_scan(), Term, parametrize, Tensor, Resets are boolean masks inside the scan; the on-chunk-edge case is where an…, Perturbing document A must leave document B bit-identical, not merely close., The state after the last token, including a partial last chunk -- where the… (+18 more)

### Community 17 - "generate.py"
Cohesion: 0.13
Nodes (35): Advance, apply_repetition_penalty(), _beam_core(), beam_search(), beam_search_cached(), best_of_n(), _check_one_sample(), generated_tokens() (+27 more)

### Community 18 - "Resp"
Cohesion: 0.07
Nodes (27): SimpleNamespace, FakeWeb, ``get``/``post`` answering from ``routes`` (regex on the URL -> function(url,…, One document's connection dropped five times and the whole RII crawl (3.8 h,…, Resp, _ris_page(), test_bger_collector_pages_by_search_after_and_skips_blocked_decisions(), html() (+19 more)

### Community 19 - "train_pretrain.py"
Cohesion: 0.10
Nodes (32): Build a config from a Hydra ``cfg.model`` node, keeping every field the…, best_val_checkpoint(), The ``save_top_k`` lowest-``val/loss`` checkpoints, ``best-step<N>.ckpt``,…, ModelCheckpoint, pytorch_lightning_loggers, pytorch_lightning_strategies, build_callbacks(), build_dataloaders() (+24 more)

### Community 20 - "test_numerics_fp64.py"
Cohesion: 0.11
Nodes (36): Mamba-1 selective scan (legacy ablation only)., Tensor, Mamba-1 selective scan (legacy incumbent; used only by the…, Exact selective scan -- the recurrence the Mamba-1 block is specified to…, Exact chunked selective scan -- no division, no clamp. Flips which axis is…, Public selective-scan entry point, computed in fp32 and cast back.…, Legacy chunk-parallel selective scan (divide-and-clamp; see module docstring).…, selective_scan() (+28 more)

### Community 21 - "evaluate_val_ppl.py"
Cohesion: 0.15
Nodes (17): mixture_from_config(), The dataset a ``name: packed`` dataset yaml describes, for ``split`` train or…, boundary_masked_labels(), Next-token targets ``labels[:, 1:]`` with every cross-document prediction set…, document_offsets(), late_context_sums(), load_model(), main() (+9 more)

### Community 22 - "test_decoding.py"
Cohesion: 0.08
Nodes (40): beam_search_uncached(), The original name of ``beam_search`` (ported from the reference's…, cached_lm(), family_lm(), _prefill_then_greedy(), parametrize, Decoding. Cached decode must be an equivalence with full recomputation. The…, The one-pass prefill leaves every cache exactly where token-by-token decoding… (+32 more)

### Community 23 - "eurlex.py"
Cohesion: 0.18
Nodes (17): act_code(), _find(), _inline(), _local(), oj_date(), parse_eurlex_xhtml(), Element, EUR-Lex: EU regulations and directives in German. Source: the Publications… (+9 more)

### Community 24 - "PretrainLightningModule"
Cohesion: 0.06
Nodes (36): Dataset, Tensor, Seeded random-token rows with packed-document ids, for smoke tests and…, ``num_rows`` rows of ``row_len`` random tokens, each packed from several…, SyntheticPackedDataset, PretrainLightningModule, Tensor, Clip and log the gradient norm; a non-finite norm clears the gradients (step… (+28 more)

### Community 25 - "analyze_generation_diversity.py"
Cohesion: 0.16
Nodes (18): analyse(), distinct_n(), duplicate_clusters(), main(), _ngram_counts(), Mean BLEU-4 of each text against a sample of the OTHERS. Higher = more…, Render the comparison table plus the verdict., Boilerplate / duplicate-template analysis for generated answers, with controls.… (+10 more)

### Community 26 - "cumulative_angles"
Cohesion: 0.17
Nodes (14): apply_rotary(), cumulative_angles(), cumulative_angles_fp64(), Tensor, Data-dependent rotary embeddings for the complex-valued SSM. Mamba-3 Sec 3.2…, Theta_t = sum_{i<=t} dt_i * theta_i, accumulated in float64 and wrapped to [0,…, ``cumulative_angles`` before the final cast: float64, wrapped to [0, 2pi).…, Forward-fill each segment's start value across the segment. (+6 more)

### Community 27 - "AttentionBlock"
Cohesion: 0.06
Nodes (41): Limitations, disable, apply_rope(), AttentionBlock, build_doc_block_mask(), build_doc_boundary_attn_mask(), build_rope_cache(), compiled_flex_attention() (+33 more)

### Community 28 - "torch"
Cohesion: 0.06
Nodes (49): einops, Gold-answer likelihood: a continuous probe score that an exact-match floor…, forward_fill_from_starts(), device, Tensor, Document segments inside packed rows, shared by the SSD and TFLA kernels. A…, (batch, seqlen) bool, True where a new document begins; position 0 is False., Monotone segment index per position, (batch, seqlen) long; all zeros without… (+41 more)

### Community 29 - "test_scripts.py"
Cohesion: 0.05
Nodes (32): fixture, The ported command-line scripts run end to end on CPU with tiny models., A tiny model yaml and a Lightning-style checkpoint for it., The profiler's slab step computes what PretrainLightningModule trains on: the…, Every arm builds through HybridConfig.from_hydra at reduced width, runs a…, The learning-rate arms are S0 and S1 with only `model.learning_rate` changed,…, Full size on `meta` with the bands: every arm passes; S5 carries +2.36% as a…, Guard: if an arm's lever never reaches the model, its ARCH line lacks the token. (+24 more)

### Community 30 - "Chunk"
Cohesion: 0.06
Nodes (62): German sentences of a passage: split after ``.``, ``!`` or ``?`` before an…, split_sentences(), _anchor(), Chunk, chunk_case_law(), chunk_document(), chunk_statute(), _common() (+54 more)

### Community 31 - "test_synth_qa.py"
Cohesion: 0.13
Nodes (21): http_server, socket, stat, generator(), _executable(), _free_port(), index(), jobs() (+13 more)

### Community 32 - "CausalLMAdapter"
Cohesion: 0.16
Nodes (7): CausalLMAdapter, Module, Tensor, A Hugging Face causal LM behind the hybrid's interfaces: the…, Args: those of ``HybridLightningModule`` (``learning_rate`` is the plain value…, ``(mean CE over the supervised targets, their count)``; position t predicts…, SFTLightningModule

### Community 33 - "HybridLanguageModel"
Cohesion: 0.08
Nodes (22): __getattr__(), lexhybrid -- a DACH legal, citation-faithful, retrieval-gated hybrid Mamba-3 /…, HybridLanguageModel, True when every mixer has an O(1) ``step``; all-or-nothing by design, because…, One cache per layer. Recurrent mixers hold a fixed-size state; attention layers…, Parameter count: ``total`` counts every distinct tensor once, so a tied head is…, The mixer type of every block, in order., Hybrid Mamba-3 / mLSTM / attention causal language model. (+14 more)

### Community 34 - "SpikeDiagnosticsCallback"
Cohesion: 0.11
Nodes (14): Callback, Tensor, Trainer, SpikeDiagnosticsCallback, hook(), hook(), hook(), hook() (+6 more)

### Community 35 - "FetchError"
Cohesion: 0.17
Nodes (12): FetchError, RuntimeError, ``fetch(url)``'s response, or None when it raised ``FetchError``., A request failed after every retry., test_skip_failed_documents_stops_on_an_outage(), fail(), An act CELLAR does not serve as German XHTML (HTTP 404, e.g. 31993L0013) is…, RIS answers a page past the end with HTTP 500 ("Die Seitennummer ist höher als… (+4 more)

### Community 36 - "fedlex.py"
Cohesion: 0.09
Nodes (33): act_code(), act_meta(), acts_query(), article_sections(), FedlexCollector, _inline(), _is_text(), latest_versions() (+25 more)

### Community 37 - "datasets.py"
Cohesion: 0.04
Nodes (68): build_source_shards(), build_source_shards_stream(), effective_weights(), PackedMixture, Dataset, Path, Tensor, Pretraining data over packed shards: the split, the mixture and the arms.… (+60 more)

### Community 38 - "chunk_dict"
Cohesion: 0.19
Nodes (12): chunk_dict(), add(), Per-question metrics and the run summary on verified synthetic records: a…, A first pointer outside the prompt selects nothing: the next pointer does not…, The judge reads the first JSON object of a reply and retries an unparseable one…, test_citation_metrics_on_synthetic(), nli(), record() (+4 more)

### Community 39 - "DIPCollector"
Cohesion: 0.32
Nodes (5): api_key(), DIPCollector, ``types``: the Drucksache types, collected in turn, newest first;…, test_dip_collector_follows_the_cursor(), page()

### Community 40 - "test_training.py"
Cohesion: 0.05
Nodes (60): no_grad, Tensor, ``(position_ids, attention_mask)`` for a Hugging Face decoder on packed rows.…, The teacher's final hidden states (after its final norm), (B, L, D_teacher),…, teacher_forward_packed(), teacher_inputs_packed(), collate(), Right-pad ``input_ids`` (with 0: no real position sees a pad) and ``labels``… (+52 more)

### Community 41 - "Mamba3Block"
Cohesion: 0.08
Nodes (24): _mamba3_params(), _mlstm_params(), _prefixed_kwargs(), The ``mamba3_*`` options this dispatcher forwards, read off the block's…, Every ``mLSTMBlock`` parameter, read off its signature like ``_mamba3_params``., Kwargs for a prefixed mixer: ``shared`` unprefixed names first, then…, _signature_params(), heavy_tail_activation() (+16 more)

### Community 42 - "eval_retrieval.py"
Cohesion: 0.13
Nodes (23): CrossEncoderScorer, ``bge-reranker-v2-m3`` as a ``Scorer``: one logit per (question, passage) pair,…, Path, Reading a JSONL file that a killed job may have been appending to., The file's non-empty whole lines; a last line left without its newline is cut…, read_complete_lines(), dense_index(), load_set() (+15 more)

### Community 43 - "ssd_interface.py"
Cohesion: 0.29
Nodes (7): device, Tensor, Chunked SSD scan -- the Mamba-2/3 state space dual. Why this exists rather than…, Lower-triangular segment sums: ``out[..., t, s] = sum_{i=s+1}^{t} v_i`` for ``s…, Monotone segment index per position (see ``lexhybrid.kernels.segments``)., _segment_ids(), segsum()

### Community 44 - "._gates"
Cohesion: 0.27
Nodes (6): Tensor, Soft-capped gates: exp input gate (computed in fp32), sigmoid forget and output…, (batch, seq_len, dim) -> (batch, seq_len, dim). Args: x: input. cache: an EMPTY…, Advance one token. ``(batch, dim)`` or ``(batch, 1, dim)`` in, ``(batch, dim)``…, Soft cap via tanh: maps x to ``tanh(x / cap) * cap``, range (-cap, cap)., _tanh_soft_cap()

### Community 45 - "SpecialTokens"
Cohesion: 0.14
Nodes (10): K for ``<|cK|>``, else None., J for ``<|sJ|>``, else None., Ids of the 87 SFT-only special tokens (never used in distillation)., The token strings, in registry order (the order ``from_offset`` assigns ids)., Consecutive ids from ``first_id`` -- tests, and any vocabulary that appends…, Ids from a tokenizer that has the tokens added (they are added to the Qwen3…, SpecialTokens, The base vocabulary's other added tokens and the unused rows above the specials… (+2 more)

### Community 46 - "generate_synthetic_qa.py"
Cohesion: 0.06
Nodes (59): concurrent_futures, Exception, Hit, _ids(), Reciprocal rank fusion of retrieval rankings (Cormack, Clarke and Büttcher…, Fuse rankings (each a list of ``Hit`` or chunk ids, best first) into one…, reciprocal_rank_fusion(), Ranked (+51 more)

### Community 47 - "sandbox"
Cohesion: 0.07
Nodes (30): _executable(), fixture, Path, The default sources' scrubbed parts go to the bm25 build with the pack's dedup…, A repository copy with the wrappers, screen_arms.py and a fake interpreter;…, The log is the job's own stdout, and GNU grep refuses to read its output file…, One sequence length per process, a fresh Inductor cache per point, the…, Task i scrubs part i % PARTS of source i / PARTS; every task uses the same HMAC… (+22 more)

### Community 48 - "teacher"
Cohesion: 0.13
Nodes (19): nli(), parametrize, Quotas met: no teacher call, same rows. A log cut mid-line by a kill loses only…, The teacher answers over the prompt's passages in the normal format. A refusal…, Rounds smaller than the limit add up: 32 errors in a row stop the run, whatever…, Attempts run out with too few answerable rows: a seeded sample of refusals…, Rows are capped kind by kind at the new count's quotas, with no teacher call. A…, Pure, so threads may call it in any order. A question names its source's… (+11 more)

### Community 49 - "check_operator_equivalence.py"
Cohesion: 0.13
Nodes (20): Mamba-2/3 state space dual (SSD) kernels., _heads_per_group(), dtype, Tensor, Sequential reference for the SSD recurrence, and the single-token decode step.…, Advance the SSD recurrence by one token. h_t = exp(dt_t * A) h_{t-1} +…, Ground-truth SSD scan: a plain loop, in float64 by default. Args: doc_ids:…, ssd_sequential_reference() (+12 more)

### Community 50 - "check_env.py"
Cohesion: 0.13
Nodes (22): importlib_metadata, packaging_markers, packaging_requirements, packaging_utils, packaging_version, Requirement, check_cuda_build(), check_editable() (+14 more)

### Community 51 - "Example"
Cohesion: 0.06
Nodes (65): Cite, encode(), Example, parse(), Passage, Quote, ``(prompt, target)`` as strings with the special-token markup., The ``Example`` back from ``prompt + target`` (as ``render`` writes them). (+57 more)

### Community 52 - "test_verifier.py"
Cohesion: 0.07
Nodes (48): entity_hash(), Every source holding the passage: the chunk's own first, then its copies'…, sources(), Answer verification: pointers, quotes, entailment and scrub leaks., candidate_forms(), leaked_entities(), Scrub leaks into the index: no scrubbed name in a retrievable chunk. The scrub…, Each capitalised word of ``text`` and each run of 2..``MAX_RUN`` capitalised… (+40 more)

### Community 53 - "filter_logits"
Cohesion: 0.33
Nodes (5): filter_logits(), Temperature, then top-k, then nucleus (top-p) filtering; filtered entries…, choose(), Shared sampling filter (``lexhybrid.decoding.filter_logits``)., test_filter_logits()

### Community 54 - "sequential_mlstm_reference"
Cohesion: 0.09
Nodes (24): dtype, Tensor, float64 sequential oracle for the recurrence ``tfla_forward_parallel``…, Plain-loop mLSTM recurrence; inputs (B, H, L, D), output (B, H, L, D) in…, sequential_mlstm_reference(), On the GPU, a 128-chunk whose log-decay overflows one factorisation (forget…, test_tfla_wide_chunk_factorises_per_block_on_gpu(), _packed_tfla_inputs() (+16 more)

### Community 55 - "filters.py"
Cohesion: 0.21
Nodes (14): date, _allowed(), _bound(), _day(), filter_chunks(), _parse(), Retrieval filters: jurisdiction, validity date and document type.…, A ``date`` from a date, a datetime (its day) or an ISO date string. (+6 more)

### Community 56 - "test_slurm_wrappers.py"
Cohesion: 0.15
Nodes (17): SLURM wrappers under `scripts/slurm/`. Static checks on every `*.sh` there: the…, A requeued run finds its last.ckpt and passes it on., The array index picks the arm; screen_arms.py resolves it ON THE NODE; the…, The wrappers the pipeline names are all here., Only the named shapes are rerun: the 1.7B shapes are rerun alone; the 8B shapes…, Stage B: INIT_FROM (stage A's last.ckpt) while the stage has none of its own;…, test_every_planned_wrapper_exists(), test_four_gpu_wrapper_runs_ddp_through_the_same_body() (+9 more)

### Community 57 - "_tfla_model"
Cohesion: 0.40
Nodes (5): The shipped configuration runs exact TFLA on its factorised path at init. With…, The record behind the fp32 range limit: forget bias 0 gives f = 0.5 and a half-…, test_tfla_fast_path_at_init(), test_tfla_reference_forget_bias_factorises_per_block(), _tfla_model()

### Community 58 - "test_check_env.py"
Cohesion: 0.07
Nodes (21): shutil, subprocess, tarfile, env(), skipif, slow, scripts/check_env.py: the environment-equals-lock check., The CI condition, reproduced: the real uv, FORCE_COLOR=1, the real lock. (+13 more)

### Community 59 - "test_config_invariants.py"
Cohesion: 0.11
Nodes (16): counts(), fixture, parametrize, slow, Configuration invariants: the living spec of what every yaml guarantees., One factor at a time: each screen yaml equals the base in every field but its…, S5 (legacy recipe): the <= 1% target did not hold (+2.36%); it is reported as a…, Structural parity with the reference: same modules, same shapes, to the… (+8 more)

### Community 60 - ".from_json"
Cohesion: 0.06
Nodes (45): Entity, iter_part(), main(), part_name(), _pattern(), placeholders(), Path, Surface form -> placeholder, numbered per tag in order of first appearance. (+37 more)

### Community 61 - "slab_cross_entropy"
Cohesion: 0.15
Nodes (12): HeadFn, MTPHead, MTPModule, Module, Tensor, One prediction depth: two norms, a 2D -> D projection, and one ``HybridBlock``., ``depths`` sequential ``MTPModule``s; ``blocks`` are built by the model's block…, Mean over depths of each depth's boundary-masked cross-entropy. Args: hidden:… (+4 more)

### Community 62 - "test_dpo.py"
Cohesion: 0.05
Nodes (63): _unit(), _argv(), _dpo_overrides(), _fake_trainer(), hf_run(), _pair_rows(), _pairs(), fixture (+55 more)

### Community 63 - "code"
Cohesion: 0.09
Nodes (25): code(), directives(), A `--gpus` directive is not overridden by an environment variable: a multi-GPU…, The two shell lookups in fetch_hf.sh must find the constants the loaders use., The command as documented, the same check on the legal base, the CUDA smoke., The 4-GPU header, and the body it delegates to passes last.ckpt as…, ARM is resolved inside the job by `screen_arms.py env`, EXPERIMENT from the…, `#SBATCH --key=value` lines before the first command, trailing comments dropped. (+17 more)

### Community 64 - "HybridLightningModule"
Cohesion: 0.10
Nodes (19): LambdaLR, HybridLightningModule, Optimizer, Tensor, Token-weighted sums for the epoch's ``val/perplexity``. Logged per batch with…, Clip gradients and log the pre-clip norm (manual, so foreach AdamW and AMP…, A WSD scheduler with an absolute warmup; records the decay window for the beta2…, Training loop around a ``HybridLanguageModel``. Args: model: the model to… (+11 more)

### Community 65 - "collections_abc"
Cohesion: 0.05
Nodes (98): collections_abc, decode_answer(), The retrieval-grounded SFT format: one prompt, pointer answers. The prompt is…, Answer tokens as the pointer markup: special tokens by name, so no tokenizer…, Pointer-constrained decoding. The legal model never generates statute text: it…, answer_words(), cited_passages(), kept_segments() (+90 more)

### Community 66 - "load_model_config"
Cohesion: 0.11
Nodes (26): Configuration dataclass for the hybrid decoder. Ported from the reference…, Model configuration and the yaml loader every script uses., available_model_configs(), load_model_config(), load_model_yaml(), Any, Path, Load a model yaml into a ``HybridConfig`` -- the one path every script uses. A… (+18 more)

### Community 67 - "greedy"
Cohesion: 0.12
Nodes (17): greedy(), Uncached greedy decoding: the full forward is recomputed for every new token…, Bigram, log_table(), ``logits[b, t] = table[ids[b, t]]``: the next-token scores given the last token., Rows of probabilities -> log-probabilities (so a beam's score is a sum of known…, Finished hypotheses leave the beam with their EOS and are ranked by…, The reference's `length_penalty` divided every live beam by the same length, so… (+9 more)

### Community 68 - "prompt_ids"
Cohesion: 0.09
Nodes (26): prompt_ids(), The prompt's token ids, up to and including ``<|answer|>`` (the positions…, Pointers, balance(), build_pairs(), draw_samples(), is_good(), main() (+18 more)

### Community 69 - "test_ci_workflow.py"
Cohesion: 0.08
Nodes (29): fixture, The CI workflow and the local validation script stay in step. PyYAML parses the…, Fast excludes slow/cuda/multigpu/network; --full and --ci exclude…, steps_text(), test_every_installing_job_uses_the_lock_and_never_pip(), test_hygiene_runs_on_a_pristine_checkout(), test_lint_runs_every_static_tool(), test_package_job_imports_the_wheel_outside_the_checkout() (+21 more)

### Community 70 - "test_eval.py"
Cohesion: 0.11
Nodes (27): evaluate(), ndcg_at_k(), Ranking metrics over qrels, as trec_eval computes them: nDCG@k (gain =…, Mean nDCG@10, R@5/10/20 and MRR over the judged queries (a query with no…, recall_at_k(), reciprocal_rank(), corpus_bleu(), _lcs_length() (+19 more)

### Community 71 - "NLI verifier threshold on 100 hand-built statute pairs"
Cohesion: 0.22
Nodes (8): NLI verifier threshold on 100 hand-built statute pairs, Not licensed, Open limitations, Prediction and outcome, Reproduction, Results, Summary, What this licenses

### Community 72 - "Corpus licence register"
Cohesion: 0.50
Nodes (4): Corpus licence register, Evaluation sets (never trained on), Models the pipeline runs, Training sources (collectors)

### Community 73 - "screen_metrics.py"
Cohesion: 0.39
Nodes (8): _fmt(), latest_version(), main(), Path, Validation and training curves of the screen runs, read from their TensorBoard…, ``<run>/logs/<exp>/version_<n>`` with the highest n that holds an events file., read_scalars(), report()

### Community 74 - "build_judge_calibration.py"
Cohesion: 0.09
Nodes (33): importlib, Index ``chunks`` (anything with ``id`` and ``text``:…, pkgutil, build(), faithfulness_items(), _item(), main(), _name() (+25 more)

### Community 75 - "eval_gold_passages.py"
Cohesion: 0.10
Nodes (27): build_sets(), decode(), fused_retriever(), retrieve(), Encoder, NLI, Path, ``(sets, counts)``: the prompt sets of every item with a gold passage (see the… (+19 more)

### Community 76 - "build_dense_index.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_dense_index.sh script

### Community 77 - "lexhybrid"
Cohesion: 0.12
Nodes (17): Acknowledgements and data attribution, Architecture at a glance, Cluster usage, Contributing, Data pipeline and retrieval index, How it answers, Installation, Key properties (+9 more)

### Community 78 - "parametrize"
Cohesion: 0.13
Nodes (15): parametrize, An array wrapper picks its work by SLURM_ARRAY_TASK_ID and logs each task on…, check_env.py reads uv.lock through `uv`; a missing uv failed with 'uv not found…, The Qwen3-8B-Base teacher fits only with gradient checkpointing., An ARCH line without an expected token, or with a forbidden one, stops the job…, The batch partition is preemptible. A training job is requeued, and its log is…, test_array_wrappers_log_per_task(), test_body() (+7 more)

### Community 79 - "eval_retrieval.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, eval_retrieval.sh script

### Community 80 - "test_rsync_rules.py"
Cohesion: 0.19
Nodes (13): parametrize, skipif, `.rsync-exclude` does what `scripts/sync_to_cluster.sh` needs. rsync is first-…, An unanchored `data/probes/` also dropped `lexhybrid/data/probes/`. Mirror…, `sync_to_cluster.sh` runs `rsync --delete`; excluded paths are protected on the…, rules(), test_data_excludes_are_anchored_to_the_root(), test_delete_keeps_what_the_cluster_owns() (+5 more)

### Community 82 - "train_sft.py"
Cohesion: 0.09
Nodes (25): Dataset, Supervised fine-tuning: cross-entropy on the answer tokens only, for the hybrid…, One split (``train.parquet`` / ``dev.parquet``) as ``{input_ids, labels}``…, SFTSplit, check_run_settings(), Path, One output directory, one run. A job that finds ``last.ckpt`` there resumes it,…, numpy (+17 more)

### Community 83 - "MambaBlock"
Cohesion: 0.19
Nodes (9): MambaBlock, Tensor, Re-apply the Mamba dt init after the model's global weight pass (no-op unless…, (batch, seq_len, dim) -> (batch, seq_len, dim); ``doc_ids`` resets state per…, Non-fast-path scan: the same implementation as ``selective_scan`` with chunk…, Per-(row, segment) wrapper that resets the SSM state at document boundaries., Mamba-1 mixer: input projection, depthwise causal conv, selective SSM, gated…, The declared capability and the real signature must not drift apart. (+1 more)

### Community 84 - "dpo_module.py"
Cohesion: 0.10
Nodes (25): Tests, answer_logps(), collate_pairs(), dpo_loss(), DPOLightningModule, kto_loss(), _pad(), Module (+17 more)

### Community 85 - "measure_fertility.py"
Cohesion: 0.08
Nodes (40): functools, cache_dir(), encode_document(), load_tokenizer(), main(), ``data/hf/hub`` unless the environment names a Hugging Face cache., Qwen3's tokenizer, with the pointer specials unless ``with_specials=False``.…, Token ids of raw document text; special-token strings inside it are split into… (+32 more)

### Community 86 - "test_repo_hygiene.py"
Cohesion: 0.23
Nodes (12): make_tree(), parametrize, Path, scripts/check_repo_hygiene.py on synthetic trees., test_a_missing_critical_file_is_reported(), test_an_ignored_analysis_document_is_reported(), test_case_collisions_are_caught(), test_clean_tree_passes() (+4 more)

### Community 87 - "German-law pretraining corpus: tokens, deduplication, scrub"
Cohesion: 0.15
Nodes (12): Budget rule, Deduplication (drop within a source, keep and link across), Discussion, Efficiency, Estimate vs exact, German-law pretraining corpus: tokens, deduplication, scrub, Limitations, Reproduction (+4 more)

### Community 88 - "tfla_forward_parallel"
Cohesion: 0.12
Nodes (23): pad_segments(), Extend ``seg`` by ``pad`` positions that join the LAST real segment. Padding…, TFLA (tiled linear attention) kernels for the mLSTM., apply_tfla(), _blockwise_scores(), _intra_chunk_sequential(), RuntimeError, Tensor (+15 more)

### Community 89 - "screen_arms.py"
Cohesion: 0.19
Nodes (15): NamedTuple, Arm, build_config(), check_tokens(), env_lines(), _fmt(), hydra_overrides(), _inference_non_embedding() (+7 more)

### Community 90 - "build"
Cohesion: 0.13
Nodes (18): question_key(), A hash of the question with its form normalised (NFC, case, whitespace, final…, build(), evalset_guard(), is_eval(), main(), Path, qa_info() (+10 more)

### Community 91 - "probe_ckpt_size.sh"
Cohesion: 0.15
Nodes (12): BATCH_SIZE, CKPT_EVERY, DATASET_CONFIG, DISTILL_CFG, EXPERIMENT, EXTRA_OVERRIDES, MAX_STEPS, MODEL_CONFIG (+4 more)

### Community 92 - "_CaptureLogger"
Cohesion: 0.11
Nodes (9): _CaptureLogger, _LargestTensor, _RankRows, Writes the row indices each DDP rank trains on to <dir>/rank<r>.txt., The answer loss never uses the MTP head; DDP would wait for gradients it never…, Records the numel of the largest tensor any op produces (forward and backward)., Keeps every ``log_metrics`` call in memory: ``rows = [(step, {tag: value})]``., test_sft_freezes_an_mtp_head_outside_its_loss() (+1 more)

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
Cohesion: 0.08
Nodes (29): BM25Index, Path, The index's tokens of each text (lower-cased, German stopwords out, stemmed)., A ``bm25s`` index over chunks, queried by text., The best ``k`` chunks for ``text``, best first; chunks sharing no term score 0…, _stem(), tokenize(), DenseIndex (+21 more)

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
Cohesion: 0.22
Nodes (8): Cluster jobs, Continuous integration, Contributing, Dependencies, Lint and style, Repository hygiene, Setup, Validation

### Community 104 - "strip_prefixes"
Cohesion: 0.20
Nodes (13): Normalise ``model._orig_mod.`` / ``_orig_mod.model.`` / ``_orig_mod.`` /…, strip_prefixes(), _f(), inspect(), main(), optimizer_names(), part_of(), Module (+5 more)

### Community 105 - "callbacks.py"
Cohesion: 0.15
Nodes (9): Training callbacks: checkpoint-on-signal (ported from the reference) and step…, Write ``interrupt.ckpt`` when the process is signalled; a no-op after the first…, SignalCheckpointCallback, Loss-spike diagnostics for one training run. Two screening runs collapsed to…, pytorch_lightning, pytorch_lightning_callbacks, signal, statistics (+1 more)

### Community 106 - "PointerFSM"
Cohesion: 0.11
Nodes (24): Decoding, PointerFSM, process(), Tensor, The pointer grammar for one retrieved context. Args: specials: the special-…, The state after ``token``; raises ``ValueError`` if the grammar forbids it., Boolean (vocab,) mask of the tokens the grammar allows next., Set every forbidden logit to ``-inf``, row by row; ``generated`` (B, T) are the… (+16 more)

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
Cohesion: 0.22
Nodes (9): Answer pipeline, Architecture, Data pipeline, Model, Renderer, Retrieval, Tokenizer and pointer vocabulary, Training (+1 more)

### Community 111 - "judge.py"
Cohesion: 0.11
Nodes (41): add_judge_args(), answer_text(), correctness_prompt(), faithfulness_prompt(), judge_from_args(), judge_prompts(), judge_record(), _listed() (+33 more)

### Community 112 - "_doc"
Cohesion: 0.07
Nodes (40): admits(), build_corpus(), Corpus assembly gate: which documents may enter which training arm. Two arms:…, Whether ``doc`` (already licence-checked) belongs in ``arm``., Yield the documents of ``arm``; raise ``LicenceError`` on the first licence…, check_document(), is_commercial_safe(), Licence (+32 more)

### Community 113 - "Judge calibration and validation on gold labels"
Cohesion: 0.13
Nodes (14): 1. Validation half, prompt version 2 (accuracy / κ / precision / recall; unparsed in brackets), 2. Accuracy by variant, validation half, 3. Calibration half, prompt version 2, with the check judges (accuracy / κ), Bar and outcome, Judge calibration and validation on gold labels, Not licensed, Open limitations, Reproduction (+6 more)

### Community 114 - "AnswerRecord"
Cohesion: 0.08
Nodes (37): gold_chunk_ids(), Each item's gold-relevant chunk ids: its ``gold_chunk_ids`` the index holds,…, AnswerRecord, Path, read_records(), write_records(), main(), The records an earlier, preempted decode wrote whole; a line it cut short is… (+29 more)

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
Cohesion: 0.07
Nodes (36): first_json_object(), OpenAICompatible, The first ``{...}`` in ``text`` that parses as a JSON object (code fences and…, ``POST {base_url}/chat/completions``: one user message, temperature 0; returns…, requests, fake_post(), post(), FakeResponse (+28 more)

### Community 119 - "Tokenizer fertility on DACH legal text"
Cohesion: 0.25
Nodes (7): Discussion, Hypotheses and method, Limitations, Reproduction, Results, Summary, Tokenizer fertility on DACH legal text

### Community 120 - "_Paras"
Cohesion: 0.25
Nodes (3): _Paras, HTMLParser, The ``<div class="para">`` blocks as ``(bold, text)``: ``bold`` is the text of…

### Community 121 - "parse_citation"
Cohesion: 0.04
Nodes (66): Limitations, Citation ids, Document schema, Fields, Probes scored on any checkpoint: MQAR, statute recall, multi-hop., gold_nll(), no_grad, Mean per-token NLL of ``target_ids`` following ``prompt_ids`` (natural log).… (+58 more)

### Community 123 - "Backbone results: the hybrid and a Transformer baseline at 8,192 tokens"
Cohesion: 0.18
Nodes (10): Backbone results: the hybrid and a Transformer baseline at 8,192 tokens, Discussion, Handoff, Limitations, Predictions, Probes on the stage-B checkpoints, by context length, Reproduction, Results (+2 more)

### Community 124 - "test_gpu.py"
Cohesion: 0.12
Nodes (19): _packed_doc_ids(), parametrize, Tensor, CUDA checks. Marked ``cuda``: they collect everywhere and skip, with the…, CUDA variant of the compiled legal base check: the compiled legal base equals…, The slab loss's peak memory follows the slab, not the row: at the Qwen3…, Under an outer torch.compile, a block mask traced inside the graph was wrong…, CUDA variant of tests/test_layers.py::test_flex_block_mask_equals_dense: the… (+11 more)

### Community 125 - "ScheduleSampler"
Cohesion: 0.19
Nodes (6): _rank_world(), A training mixture's rows for one rank, from where the run stands. Sample ``i``…, Every row of an evaluation set across the ranks, each rank the same count…, ScheduleSampler, ShardSampler, test_schedule_sampler_splits_ranks_evenly_and_skips_what_was_trained()

### Community 126 - "build_corpus_array.sh"
Cohesion: 0.33
Nodes (5): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_corpus_array.sh script

### Community 127 - "pack_corpus.sh"
Cohesion: 0.33
Nodes (5): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, pack_corpus.sh script, TOKENIZERS_PARALLELISM

### Community 128 - "_s1_fingerprint"
Cohesion: 0.33
Nodes (4): The eval guard reads manifests of the full sets, fetched online into a…, The ARCH line of the real base model (S1), built on meta from the screen's own…, _s1_fingerprint(), test_build_sft_data_guards_with_complete_manifests_fetched_once()

### Community 129 - "run_metadata.py"
Cohesion: 0.16
Nodes (16): Any, DictConfig, Path, ``run_metadata.json``: what produced a checkpoint, written at the start of…, sha256 over the sorted (relative path, file sha256) pairs of the source tree., Contents of ``.sync_stamp`` (date + tree hash of the last sync to the cluster),…, The checked-out commit id, read from ``.git/HEAD`` and its ref file; "" if…, Persist ``<output_dir>/run_metadata.json`` and return its path. (+8 more)

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

### Community 137 - "ParquetRows"
Cohesion: 0.15
Nodes (10): evaluate_perplexity(), main(), measure_throughput(), ParquetRows, Dataset, no_grad, Tensor, Rows of a packed parquet shard: ``input_ids`` (and ``doc_ids`` when present),… (+2 more)

### Community 138 - "test_diag_spike_reruns_the_arm_observed_in_its_own_directory"
Cohesion: 0.40
Nodes (5): _compose(), The fake interpreter recorded `distill=null` happily; real Hydra refused it…, Loss-spike rerun: S1-s42 with the screen settings, seed and schedule unchanged…, test_diag_spike_reruns_the_arm_observed_in_its_own_directory(), test_every_train_argument_list_composes_under_real_hydra()

### Community 142 - "test_reference_parity.py"
Cohesion: 0.20
Nodes (14): inspect, _block_defaults(), block_params(), _config_defaults(), load(), model_params(), _per_document(), parametrize (+6 more)

### Community 143 - "HybridBlock"
Cohesion: 0.09
Nodes (21): LayerType, create_hybrid_blocks(), HybridBlock, Tensor, (batch, seq_len, dim) -> (batch, seq_len, dim). Argument order matches…, Per-layer decode state, or None if this mixer has no step (Mamba-1)., One token through mixer + MLP with residuals, mirroring ``forward``'s topology…, ``num_layers`` blocks cycling through ``layer_pattern``; block 0 is marked… (+13 more)

### Community 162 - "Judge"
Cohesion: 0.13
Nodes (16): Complete, average_precision(), Judge, _parse(), Path, A ``Complete`` behind the reply cache; ``calls`` counts the completions…, ``(read(reply object), reply object)``, retried once; ``(None, {"raw":…, RAGAS context precision: precision@k averaged over the positions k of the… (+8 more)

### Community 167 - "test_utils.py"
Cohesion: 0.12
Nodes (25): detect_prefix(), _first_segments(), infer_architecture(), infer_layer_types(), InferredArchitecture, load_every_key(), load_state_dict_guarded(), _mixer_params() (+17 more)

### Community 169 - "Retrieval on GerLeRB and GerDaLIR"
Cohesion: 0.18
Nodes (10): GerDaLIR (12,234 queries), GerLeRB (367 queries), Not licensed, Open limitations, Prediction and outcome, Reproduction, Results, Retrieval on GerLeRB and GerDaLIR (+2 more)

### Community 170 - "MDebertaNLI"
Cohesion: 0.15
Nodes (14): entailment_index(), MDebertaNLI, The class index whose label is ``entailment`` (any case); ``ValueError`` unless…, mDeBERTa XNLI as an ``NLI`` callable: softmax over the classes, the entailment…, calibrate(), main(), Path, Calibrate the verifier's entailment threshold on hand-labelled premise/claim… (+6 more)

### Community 171 - "record_metrics"
Cohesion: 0.12
Nodes (21): doc_of(), gold_relation(), _mean(), _ratio(), One summary per (system, seed, arm), in order of first appearance (see the…, The document of a chunk id (``<document id>#<n>``)., ``(doc_id, copies) -> bool``: a chunk is a gold passage or a copy of one when…, One record's metrics (see the module docstring). (+13 more)

### Community 172 - "test_multigpu.py"
Cohesion: 0.40
Nodes (5): _ddp_worker(), Multi-GPU checks. Marked ``multigpu``: they collect everywhere and skip, with…, Two processes, different data, one all-reduced gradient: identical on every…, test_ddp_step_gradients_identical_across_ranks(), torch_multiprocessing

### Community 173 - "StopAtStepCallback"
Cohesion: 0.10
Nodes (13): FinalCheckpointCallback, Callback, End training after optimizer step ``stop_at_step`` while the LR schedule keeps…, Write ``<checkpoint_dir>/last.ckpt`` when training ends normally.…, StopAtStepCallback, _IndexRows, Row i carries its index, so a run's data order can be read back., Lightning restarts a plain loader at row 0 on resume (screen run S5's… (+5 more)

### Community 175 - "sys"
Cohesion: 0.12
Nodes (22): hydra, hydra_core_global_hydra, ``lm`` frozen but for rank-``rank`` LoRA adapters on ``LORA_TARGETS`` and the…, wrap_lora(), check_model(), compose_model(), main(), model_names() (+14 more)

### Community 176 - "test_hf_sampling_through_its_cache_matches_the_uncached_reference"
Cohesion: 0.67
Nodes (3): Same logits, same generator: the KV-cached sampler and the uncached ``sample``…, test_hf_sampling_through_its_cache_matches_the_uncached_reference(), sample()

### Community 178 - "Document"
Cohesion: 0.06
Nodes (45): Change, changes_since(), main(), Path, The currency changelog: which provisions changed since a date. A provision's…, write_changelog(), Fetch one document, or skip it (logged) when its fetch fails after every retry,…, SkipFailedDocuments (+37 more)

### Community 179 - "synth_qa.sh"
Cohesion: 0.20
Nodes (8): scripts_slurm_lib_vllm_server_sh, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, synth_qa.sh script, HF_HOME, HF_HUB_OFFLINE, vllm_smoke.sh script

### Community 180 - "train_sft_4gpu.sh"
Cohesion: 0.22
Nodes (9): HF_DATASETS_CACHE, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, opt(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, train_sft_4gpu.sh script (+1 more)

### Community 182 - "test_sync_scripts.py"
Cohesion: 0.19
Nodes (21): CompletedProcess, counting(), decode(), Records reach the arm files as they are produced, so a run cut mid-question…, test_ablation_appends_each_record_and_resumes_after_preemption(), flaky(), calls(), _load_tree_hash() (+13 more)

### Community 184 - "train_dpo_4gpu.sh"
Cohesion: 0.29
Nodes (7): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, opt(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, train_dpo_4gpu.sh script

### Community 185 - "FakeQwen"
Cohesion: 0.22
Nodes (4): FakeQwen, Just enough of a Hugging Face tokenizer for ``load_tokenizer``'s checks., test_tokenizer_cache_and_cli(), test_tokenizer_loader_checks_eos_and_vocabulary()

### Community 187 - "schedulers.py"
Cohesion: 0.09
Nodes (23): Apply the beta2 schedule when WSD and ``beta2_schedule`` are both active., configure_optimizer(), get_parameter_groups(), Any, Module, Optimizer, Optimizer construction with weight-decay parameter groups (ported from the…, Split trainable parameters into a decayed and a non-decayed group by name… (+15 more)

### Community 190 - "setup_arm_env.sh"
Cohesion: 0.33
Nodes (5): PYTHONUNBUFFERED, setup_arm_env.sh script, UV_CACHE_DIR, UV_LINK_MODE, UV_PROJECT_ENVIRONMENT

### Community 191 - "setup_vllm.sh"
Cohesion: 0.33
Nodes (5): PYTHONUNBUFFERED, setup_vllm.sh script, UV_CACHE_DIR, UV_LINK_MODE, UV_PROJECT_ENVIRONMENT

### Community 193 - "test_tfla_wide_chunks_factorise_per_block_exactly"
Cohesion: 0.33
Nodes (7): 256 positions (two chunks of 128), two rows, boundaries inside and on chunk…, Trained forget gates widened a 128-chunk's log-decay past fp32 (half-range 43.2…, In fp32 the per-block path forms no inf, forward or backward (a masked inf…, test_tfla_per_block_path_is_finite_and_close_in_fp32(), test_tfla_wide_chunks_factorise_per_block_exactly(), _tfla_grads(), _wide_tfla_inputs()

### Community 194 - "pathlib"
Cohesion: 0.06
Nodes (54): argparse, collections, dataclasses, hashlib, hmac, itertools, json, The German legal-entity scrub: names out before training. Entities come from… (+46 more)

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

### Community 204 - "test_compile_forward_chunk128"
Cohesion: 0.40
Nodes (4): linux_only, slow, Hybrid_legal_base at reduced width -- chunk 128 for both Mamba-3 and mLSTM,…, test_compile_forward_chunk128()

## Knowledge Gaps
- **267 isolated node(s):** `lexhybrid-arm`, `lexhybrid-scrub`, `lexhybrid-vllm`, `lexhybrid`, `build_corpus_array.sh script` (+262 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1821 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **31 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HybridLanguageModel` connect `HybridLanguageModel` to `Tensor`, `_s1_fingerprint`, `test_data.py`, `HybridConfig`, `performance_profile.py`, `test_pipeline.py`, `.forward`, `train_dpo.py`, `test_models.py`, `HybridBlock`, `test_reference_parity.py`, `train_pretrain.py`, `test_numerics_fp64.py`, `evaluate_val_ppl.py`, `test_decoding.py`, `PretrainLightningModule`, `torch`, `test_scripts.py`, `datasets.py`, `test_utils.py`, `test_training.py`, `test_multigpu.py`, `check_operator_equivalence.py`, `filter_logits`, `test_slurm_wrappers.py`, `_tfla_model`, `test_config_invariants.py`, `schedulers.py`, `slab_cross_entropy`, `test_dpo.py`, `HybridLightningModule`, `pathlib`, `load_model_config`, `test_compile_forward_chunk128`, `train_sft.py`, `dpo_module.py`, `screen_arms.py`, `strip_prefixes`, `Architecture`, `parse_citation`, `test_gpu.py`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Why does `Document` connect `Document` to `test_data.py`, `collectors/base.py`, `fedlex.py`, `pathlib`, `datasets.py`, `DIPCollector`, `dedup.py`, `test_pipeline.py`, `http_get`, `Architecture`, `resync_index.py`, `_doc`, `test_verifier.py`, `measure_fertility.py`, `eurlex.py`, `parse_citation`, `.from_json`, `Chunk`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `Chunk` connect `Chunk` to `collections_abc`, `pathlib`, `answer.py`, `prompt_ids`, `BM25Index`, `chunk_dict`, `test_eval.py`, `eval_gold_passages.py`, `record_metrics`, `generate_synthetic_qa.py`, `AnswerRecord`, `Example`, `test_verifier.py`, `test_judge.py`, `filters.py`, `test_dpo.py`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 57 inferred relationships involving `Document` (e.g. with `Data pipeline` and `Document schema`) actually correct?**
  _`Document` has 57 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `HybridLanguageModel` (e.g. with `Model` and `__getattr__()`) actually correct?**
  _`HybridLanguageModel` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 60 inferred relationships involving `Chunk` (e.g. with `answer_words()` and `cited_passages()`) actually correct?**
  _`Chunk` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `HybridConfig` (e.g. with `Model` and `load_model_config()`) actually correct?**
  _`HybridConfig` has 6 INFERRED edges - model-reasoned connections that need verification._