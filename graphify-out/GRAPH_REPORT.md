# Graph Report - legal_ai_hybrid_mamba_xlstm_model  (2026-11-05)

## Corpus Check
- 261 files · ~260,283 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 61 file(s) not represented in the graph (top: .jsonl 30, .xml 12, (none) 6)

## Summary
- 4143 nodes · 9671 edges · 204 communities (174 shown, 30 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 507 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8661de3f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- packing.py
- HybridLanguageModel
- test_data.py
- pathlib
- test_retrieval.py
- test_layers.py
- evalsets/base.py
- sequential_mlstm_reference
- performance_profile.py
- rel_max_err
- test_eval.py
- bootstrap_compare.py
- write_run_metadata
- test_dpo.py
- test_models.py
- index.py
- test_numerics_fp64.py
- generate.py
- test_collect_offline.py
- train_pretrain.py
- scrub_ner_worker.py
- evaluate_val_ppl.py
- test_decoding.py
- http_get
- SyntheticPackedDataset
- check_operator_equivalence.py
- tfla_forward_parallel
- AttentionBlock
- torch
- test_scripts.py
- HybridBlock
- verifier.py
- AnswerRecord
- HybridConfig
- SpikeDiagnosticsCallback
- parse_citation
- fedlex.py
- datasets.py
- dedup.py
- screen_arms.py
- test_training.py
- Mamba3Block
- DenseIndex
- evaluate_probes.py
- ._gates
- SpecialTokens
- generate
- sandbox
- test_synth_qa.py
- ssd_interface.py
- check_env.py
- sft_format.py
- test_verifier.py
- hybrid_lm.py
- LayerSplit
- test_multigpu.py
- test_slurm_wrappers.py
- eurlex.py
- test_check_env.py
- test_config_invariants.py
- scrub_ler.py
- .__init__
- _stub_transformers
- code
- HybridLightningModule
- test_renderer.py
- load_model_config
- Bigram
- build_dpo_pairs.py
- test_code_hygiene.py
- _schedule_fit
- CausalLMAdapter
- test_ci_workflow.py
- test_reference_parity.py
- train_dpo.py
- cumulative_angles
- build_dense_index.sh
- lexhybrid
- parametrize
- eval_retrieval.sh
- test_rsync_rules.py
- _BlockParser
- run
- MambaBlock
- dpo_module.py
- build_shards.py
- test_repo_hygiene.py
- German-law pretraining corpus: tokens, deduplication, scrub
- eval_gold_passages.py
- build_index.py
- build_sft_data.py
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
- check_configs.py
- ScheduleSampler
- StepStatsCallback
- PointerFSM
- conftest.py
- equivalence.sh
- kd_memory_probe.sh
- Architecture
- argparse
- _doc
- evaluate_lm.py
- eval_system_ablation.py
- preflight.sh
- profile.sh
- validate.sh
- test_judge.py
- Tokenizer fertility on DACH legal text
- _Paras
- Document
- train_sft_hf.py
- Backbone results: the hybrid and a Transformer baseline at 8,192 tokens
- test_gpu.py
- test_http_cache_replays_responses_without_the_network
- build_corpus_array.sh
- pack_corpus.sh
- sft_module.py
- measure_fertility.py
- corpus_counts.sh
- diag_compile.sh
- eval_probes.sh
- fetch_hf.sh
- scrub_array.sh
- setup_env.sh
- train_pretrain_4gpu.sh
- Chunk
- test_diag_spike_reruns_the_arm_observed_in_its_own_directory
- test_imports.py
- ByteTokenizer
- record_metrics
- parse_rii_xml
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
- Judge
- lexhybrid
- lexhybrid-scrub
- test_utils.py
- SourceShards
- corpus_counts.py
- dump_reference_parity.py
- MDebertaNLI
- PretrainLightningModule
- build_source_shards_stream
- test_tfla_wide_chunks_factorise_per_block_exactly
- SFTLightningModule
- test_hf_sampling_through_its_cache_matches_the_uncached_reference
- lightning_module.py
- FakeQwen
- synth_qa.sh
- train_sft_4gpu.sh
- hf_run
- test_sync_scripts.py
- test_packaging.py
- train_dpo_4gpu.sh
- PairSchedule
- test_eval_gold_passages_wrapper_keeps_runs_apart_and_reuses_sets
- Results
- Results
- html_blocks
- test_http_teacher_retries_and_the_reply_parsers_are_strict
- setup_vllm.sh
- _tfla_model
- test_param_bands
- test_packing_with_the_qwen3_tokenizer
- build_dpo_pairs.sh
- build_sft_data.sh
- eval_gold_passages.sh
- eval_system_ablation.sh
- vllm_server.sh
- .__init__
- test_tfla_chunk_size_reaches_kernel
- counting
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
- `Hypotheses and method` --references--> `encode_document()`  [INFERRED]
  analysis/tokenizer_fertility.md → lexhybrid/data/tokenizer.py
- `Tokenizer and pointer vocabulary` --references--> `encode_document()`  [INFERRED]
  docs/ARCHITECTURE.md → lexhybrid/data/tokenizer.py
- `Model` --references--> `HybridConfig`  [INFERRED]
  docs/ARCHITECTURE.md → lexhybrid/config/hybrid_config.py
- `Corpus licence register` --references--> `build_corpus()`  [INFERRED]
  docs/CORPUS_LICENCE_REGISTER.md → lexhybrid/data/corpus/build.py

## Import Cycles
- None detected.

## Communities (204 total, 30 thin omitted)

### Community 0 - "packing.py"
Cohesion: 0.13
Nodes (18): main(), pack(), Packer, Path, EOS-delimited packing into fixed-length rows and parquet shards. Every document…, Parquet shards ``shard-00000.parquet``... of at most ``rows_per_shard`` rows,…, Parquet shards ``shard-00000.parquet``... of at most ``rows_per_shard`` rows., Rows of ``row_len`` tokens from one source's documents, fed one tokenised… (+10 more)

### Community 1 - "HybridLanguageModel"
Cohesion: 0.05
Nodes (38): __getattr__(), lexhybrid -- a DACH legal, citation-faithful, retrieval-gated hybrid Mamba-3 /…, boundary_masked_labels(), HybridLanguageModel, no_grad, Tensor, Language-model forward pass. Args: input_ids: (B, L) token ids; mutually…, Embedding and every block: the residual stream (B, L, D) BEFORE the final norm.… (+30 more)

### Community 2 - "test_data.py"
Cohesion: 0.03
Nodes (77): BGerCollector, EURLexCollector, FineWeb2DECollector, GIICollector, read_zip_xml(), MultiLegalPileCollector, court_abbreviation(), OLDPCollector (+69 more)

### Community 3 - "pathlib"
Cohesion: 0.05
Nodes (71): collections_abc, datetime, hashlib, io, itertools, json, cli(), Collector (+63 more)

### Community 4 - "test_retrieval.py"
Cohesion: 0.05
Nodes (65): _anchor(), chunk_case_law(), chunk_document(), chunk_statute(), _common(), _label(), provision_heading(), ``(grain, text)`` for one Absatz: its sentences (``Satz 2``) and its items… (+57 more)

### Community 5 - "test_layers.py"
Cohesion: 0.07
Nodes (56): mLSTMBlock, mLSTM decode state: the matrix cell ``C`` and the normalizer ``n``, both zeros,…, mLSTM mixer: per-dimension exponential input gate, sigmoid forget gate, matrix…, _bc_bias_pair(), _continue_with_steps(), _drive_lambda_to_one(), m3_block(), no_grad_params() (+48 more)

### Community 6 - "evalsets/base.py"
Cohesion: 0.09
Nodes (50): gzip, cli(), EvalSetInfo, Passage, Path, QAItem, QASet, Query (+42 more)

### Community 7 - "sequential_mlstm_reference"
Cohesion: 0.10
Nodes (22): dtype, Tensor, float64 sequential oracle for the recurrence ``tfla_forward_parallel``…, Plain-loop mLSTM recurrence; inputs (B, H, L, D), output (B, H, L, D) in…, sequential_mlstm_reference(), _packed_tfla_inputs(), The legacy error is governed entirely by whether f_cum underflows the 1e-6…, The overflow guard: re-centring the whole chunk produced NaN here; re-centring… (+14 more)

### Community 8 - "performance_profile.py"
Cohesion: 0.07
Nodes (42): contextlib, csv, attention_backend(), available_configs(), build_model(), compile_for(), _delta_ms(), effective_chunk_sizes() (+34 more)

### Community 9 - "rel_max_err"
Cohesion: 0.13
Nodes (28): Chunk-parallel SSD scan. h_t = exp(dt_t A) h_{t-1} + sum_terms coeff_t (B_t…, ssd_chunked_scan(), dtype, Ground-truth SSD scan: a plain loop, in float64 by default. Args: doc_ids:…, ssd_sequential_reference(), Term, parametrize, Tensor (+20 more)

### Community 10 - "test_eval.py"
Cohesion: 0.09
Nodes (33): One summary per (system, seed, arm), in order of first appearance (see the…, score(), _share(), evaluate(), ndcg_at_k(), Ranking metrics over qrels, as trec_eval computes them: nDCG@k (gain =…, Mean nDCG@10, R@5/10/20 and MRR over the judged queries (a query with no…, recall_at_k() (+25 more)

### Community 11 - "bootstrap_compare.py"
Cohesion: 0.06
Nodes (46): corpus_bleu(), _lcs_length(), _ngram_counts(), Counter, Sentence ROUGE-L and corpus BLEU, stdlib only. Ported verbatim from the…, Sentence-level ROUGE-L F-measure (Lin, 2004)., Corpus-level BLEU-N with brevity penalty, one reference per hypothesis., rouge_l_score() (+38 more)

### Community 12 - "write_run_metadata"
Cohesion: 0.18
Nodes (14): Any, DictConfig, Path, sha256 over the sorted (relative path, file sha256) pairs of the source tree., Contents of ``.sync_stamp`` (date + tree hash of the last sync to the cluster),…, The checked-out commit id, read from ``.git/HEAD`` and its ref file; "" if…, Persist ``<output_dir>/run_metadata.json`` and return its path., read_git_head() (+6 more)

### Community 13 - "test_dpo.py"
Cohesion: 0.06
Nodes (53): _argv(), _dpo_overrides(), _fake_trainer(), _pair_rows(), _pairs(), Preference optimisation: pair selection and balance, the answer-token log-…, Only the exact ``<|unanswerable|>`` is a good refusal: a kept answer is bad,…, Target 1/4 of refusal pairs: 1 refusal + 6 answer pairs keep 3 answer pairs… (+45 more)

### Community 14 - "test_models.py"
Cohesion: 0.10
Nodes (36): _delta_at_init(), _delta_model(), parametrize, Model-level behaviour of HybridLanguageModel., The model's weight pass zeroed both gate biases; `post_model_init` restores the…, A `.to(bfloat16)` model without autocast once died in the SSD einsums., A document's last token (its EOS) does not predict the next document's first;…, Init then tie; `non_embedding = total - embedding - (lm_head if untied)`, so… (+28 more)

### Community 15 - "index.py"
Cohesion: 0.09
Nodes (35): document_digest(), DocumentDigest, file_sha256(), indexed_documents(), IndexVersion, latest(), Path, Index versions: a registry of the retrieval indexes that have been built.… (+27 more)

### Community 16 - "test_numerics_fp64.py"
Cohesion: 0.11
Nodes (36): Mamba-1 selective scan (legacy ablation only)., Tensor, Mamba-1 selective scan (legacy incumbent; used only by the…, Exact selective scan -- the recurrence the Mamba-1 block is specified to…, Exact chunked selective scan -- no division, no clamp. Flips which axis is…, Public selective-scan entry point, computed in fp32 and cast back.…, Legacy chunk-parallel selective scan (divide-and-clamp; see module docstring).…, selective_scan() (+28 more)

### Community 17 - "generate.py"
Cohesion: 0.12
Nodes (41): Advance, apply_repetition_penalty(), _beam_core(), beam_search(), beam_search_cached(), beam_search_uncached(), best_of_n(), _check_one_sample() (+33 more)

### Community 18 - "test_collect_offline.py"
Cohesion: 0.09
Nodes (28): api_key(), DIPCollector, ``types``: the Drucksache types, collected in turn, newest first;…, FakeWeb, The collectors' and fetchers' network loops, offline (CI has no network). Each…, The OLDP API stopped paging past page 10, so the collector reads the bulk dump…, ``get``/``post`` answering from ``routes`` (regex on the URL -> function(url,…, One document's connection dropped five times and the whole RII crawl (3.8 h,… (+20 more)

### Community 19 - "train_pretrain.py"
Cohesion: 0.07
Nodes (38): FinalCheckpointCallback, Callback, Training callbacks: checkpoint-on-signal (ported from the reference) and step…, End training after optimizer step ``stop_at_step`` while the LR schedule keeps…, Write ``<checkpoint_dir>/last.ckpt`` when training ends normally.…, Write ``interrupt.ckpt`` when the process is signalled; a no-op after the first…, SignalCheckpointCallback, StopAtStepCallback (+30 more)

### Community 20 - "scrub_ner_worker.py"
Cohesion: 0.22
Nodes (12): cap(), chunks(), default_cache_dir(), load_tagger(), main(), flush(), Entities of every document, the sentences of all ``docs`` tagged in ONE…, German legal-entity recognition for the LER scrub. Runs in the scrub… (+4 more)

### Community 21 - "evaluate_val_ppl.py"
Cohesion: 0.11
Nodes (23): Normalise ``model._orig_mod.`` / ``_orig_mod.model.`` / ``_orig_mod.`` /…, strip_prefixes(), document_offsets(), late_context_sums(), load_model(), main(), no_grad, Path (+15 more)

### Community 22 - "test_decoding.py"
Cohesion: 0.09
Nodes (34): cached_lm(), family_lm(), _prefill_then_greedy(), parametrize, Decoding. Cached decode must be an equivalence with full recomputation. The…, The one-pass prefill leaves every cache exactly where token-by-token decoding…, In bf16 the two paths round differently, so the claim is argmax identity…, A packed prompt: decoding continues each row's last document as if it were… (+26 more)

### Community 23 - "http_get"
Cohesion: 0.05
Nodes (52): _cached(), FetchError, http_get(), http_post(), RuntimeError, RateLimiter, ``fetch(url)``'s response, or None when it raised ``FetchError``., GET with a per-host rate limit and exponential backoff on 429/5xx and… (+44 more)

### Community 24 - "SyntheticPackedDataset"
Cohesion: 0.10
Nodes (24): Dataset, Tensor, Seeded random-token rows with packed-document ids, for smoke tests and…, ``num_rows`` rows of ``row_len`` random tokens, each packed from several…, SyntheticPackedDataset, _cpu_trainer(), _diag_fit(), Stage B ends at step 18,037 with checkpoints every 1,000: without… (+16 more)

### Community 25 - "check_operator_equivalence.py"
Cohesion: 0.23
Nodes (11): check_model(), check_operator(), main(), _operands(), Model level: do the logits move? This is what a decoded token actually sees., The equivalence gate every operator or speed change has to pass. Ported from…, max |a-b| / max(|b|, eps), computed in fp64., Operator level: does changing chunk_size change the function the scan computes? (+3 more)

### Community 26 - "tfla_forward_parallel"
Cohesion: 0.11
Nodes (26): forward_fill_from_starts(), pad_segments(), Tensor, Extend ``seg`` by ``pad`` positions that join the LAST real segment. Padding…, For each position, ``values`` at the most recent start at or before it along…, TFLA (tiled linear attention) kernels for the mLSTM., apply_tfla(), _blockwise_scores() (+18 more)

### Community 27 - "AttentionBlock"
Cohesion: 0.05
Nodes (45): Limitations, disable, apply_rope(), AttentionBlock, build_doc_block_mask(), build_doc_boundary_attn_mask(), build_rope_cache(), compiled_flex_attention() (+37 more)

### Community 28 - "torch"
Cohesion: 0.07
Nodes (38): einops, Gold-answer likelihood: a continuous probe score that an exact-match floor…, device, Document segments inside packed rows, shared by the SSD and TFLA kernels. A…, (batch, seqlen) bool, True where a new document begins; position 0 is False., Monotone segment index per position, (batch, seqlen) long; all zeros without…, segment_ids(), segment_starts() (+30 more)

### Community 29 - "test_scripts.py"
Cohesion: 0.05
Nodes (30): fixture, The ported command-line scripts run end to end on CPU with tiny models., A tiny model yaml and a Lightning-style checkpoint for it., Every arm builds through HybridConfig.from_hydra at reduced width, runs a…, The learning-rate arms are S0 and S1 with only `model.learning_rate` changed,…, Full size on `meta` with the bands: every arm passes; S5 carries +2.36% as a…, Guard: if an arm's lever never reaches the model, its ARCH line lacks the token., The method: whole documents until the word budget, documents over the cap… (+22 more)

### Community 30 - "HybridBlock"
Cohesion: 0.09
Nodes (21): LayerType, create_hybrid_blocks(), HybridBlock, Tensor, (batch, seq_len, dim) -> (batch, seq_len, dim). Argument order matches…, Per-layer decode state, or None if this mixer has no step (Mamba-1)., One token through mixer + MLP with residuals, mirroring ``forward``'s topology…, ``num_layers`` blocks cycling through ``layer_pattern``; block 0 is marked… (+13 more)

### Community 31 - "verifier.py"
Cohesion: 0.08
Nodes (60): answer_words(), cited_passages(), kept_segments(), Segment, Citation metrics of answer records (``lexhybrid.eval.records``), per question…, The segments of the claims the verifier kept, i.e. what the rendered answer…, Each pointer's passage as a 0-based index into ``chunks``, in order; ``None``…, The answer's own words: its text with each quote as the source sentences,… (+52 more)

### Community 32 - "AnswerRecord"
Cohesion: 0.11
Nodes (21): AnswerRecord, Path, One system's answer to one evaluation question: the JSONL record every…, read_records(), write_records(), main(), Truncation audit: the share of decodes that did not end on the EOS within their…, An AnswerRecord carries its prompt's passages in full and survives the JSONL… (+13 more)

### Community 33 - "HybridConfig"
Cohesion: 0.06
Nodes (34): HybridConfig, Validate and auto-compute derived parameters., The per-mixer arguments for layer ``layer_idx`` (a readable summary; not used…, Every architectural knob of the hybrid decoder. One dataclass serves every…, Build a config from a Hydra ``cfg.model`` node, keeping every field the…, Convert the config to a plain dictionary., Create a config from a dictionary produced by ``to_dict``., Write ``config.json`` into ``save_directory`` and return its path. (+26 more)

### Community 34 - "SpikeDiagnosticsCallback"
Cohesion: 0.11
Nodes (14): Callback, Tensor, Trainer, SpikeDiagnosticsCallback, hook(), hook(), hook(), hook() (+6 more)

### Community 35 - "parse_citation"
Cohesion: 0.09
Nodes (27): Limitations, Citation ids, Document schema, Fields, build_items(), numbered_once(), MultihopItem, no_grad (+19 more)

### Community 36 - "fedlex.py"
Cohesion: 0.09
Nodes (33): act_code(), act_meta(), acts_query(), article_sections(), FedlexCollector, _inline(), _is_text(), latest_versions() (+25 more)

### Community 37 - "datasets.py"
Cohesion: 0.11
Nodes (26): effective_weights(), mixture_from_config(), PackedMixture, Dataset, Tensor, Pretraining data over packed shards: the split, the mixture and the arms.…, Each admitted source's share of the rows: its group's share, split by weight…, Rows of the admitted sources as training items ``{input_ids, labels,… (+18 more)

### Community 38 - "dedup.py"
Cohesion: 0.17
Nodes (18): deduplicate(), deduplicate_signatures(), deduplicate_stream(), DedupLog, keep_order(), minhash(), Deduplication of the training corpus: exact, then near-duplicate. 1. **Exact.**…, ``deduplicate_signatures`` over documents streamed once, their MinHashes… (+10 more)

### Community 39 - "screen_arms.py"
Cohesion: 0.19
Nodes (15): NamedTuple, Arm, build_config(), check_tokens(), env_lines(), _fmt(), hydra_overrides(), _inference_non_embedding() (+7 more)

### Community 40 - "test_training.py"
Cohesion: 0.05
Nodes (48): WSD learning-rate multiplier at ``step``. warmup: linear from…, wsd_factor(), _compose_sft(), _full_kd(), _kd_inputs(), _LargestTensor, _pretrain_state(), parametrize (+40 more)

### Community 41 - "Mamba3Block"
Cohesion: 0.06
Nodes (32): _mamba3_params(), _mlstm_params(), _prefixed_kwargs(), The ``mamba3_*`` options this dispatcher forwards, read off the block's…, Every ``mLSTMBlock`` parameter, read off its signature like ``_mamba3_params``., Kwargs for a prefixed mixer: ``shared`` unprefixed names first, then…, _signature_params(), heavy_tail_activation() (+24 more)

### Community 42 - "DenseIndex"
Cohesion: 0.06
Nodes (48): fused_retriever(), retrieve(), Encoder, Path, The index's ranking: BM25, fused with ``dense/`` when it exists and ``encode``…, Hit, The index's tokens of each text (lower-cased, German stopwords out, stemmed)., The best ``k`` chunks for ``text``, best first; chunks sharing no term score 0… (+40 more)

### Community 43 - "evaluate_probes.py"
Cohesion: 0.07
Nodes (35): Probes scored on any checkpoint: MQAR, statute recall, multi-hop., gold_nll(), no_grad, Mean per-token NLL of ``target_ids`` following ``prompt_ids`` (natural log).…, generate(), MQARConfig, MQARItem, no_grad (+27 more)

### Community 44 - "._gates"
Cohesion: 0.27
Nodes (6): Tensor, Soft-capped gates: exp input gate (computed in fp32), sigmoid forget and output…, (batch, seq_len, dim) -> (batch, seq_len, dim). Args: x: input. cache: an EMPTY…, Advance one token. ``(batch, dim)`` or ``(batch, 1, dim)`` in, ``(batch, dim)``…, Soft cap via tanh: maps x to ``tanh(x / cap) * cap``, range (-cap, cap)., _tanh_soft_cap()

### Community 45 - "SpecialTokens"
Cohesion: 0.10
Nodes (18): K for ``<|cK|>``, else None., J for ``<|sJ|>``, else None., Ids of the 87 SFT-only special tokens (never used in distillation)., The token strings, in registry order (the order ``from_offset`` assigns ids)., Consecutive ids from ``first_id`` -- tests, and any vocabulary that appends…, Ids from a tokenizer that has the tokens added (they are added to the Qwen3…, SpecialTokens, decode() (+10 more)

### Community 46 - "generate"
Cohesion: 0.07
Nodes (34): Exception, Attempt, cap_rows(), check_answer(), finish(), fit_share(), fused_rankings(), generate() (+26 more)

### Community 47 - "sandbox"
Cohesion: 0.06
Nodes (30): fixture, Each run's last.ckpt against configs/model/<cfg>.yaml, a PASS/FAIL line per…, The default sources' scrubbed parts go to the bm25 build with the pack's dedup…, The eval guard reads manifests of the full sets, fetched online into a…, The ARCH line of the real base model (S1), built on meta from the screen's own…, A repository copy with the wrappers, screen_arms.py and a fake interpreter;…, One sequence length per process, a fresh Inductor cache per point, the…, Task i scrubs part i % PARTS of source i / PARTS; every task uses the same HMAC… (+22 more)

### Community 48 - "test_synth_qa.py"
Cohesion: 0.08
Nodes (37): http_server, shutil, socket, generator(), _executable(), _free_port(), jobs(), run() (+29 more)

### Community 49 - "ssd_interface.py"
Cohesion: 0.16
Nodes (13): Mamba-2/3 state space dual (SSD) kernels., device, Tensor, Chunked SSD scan -- the Mamba-2/3 state space dual. Why this exists rather than…, Lower-triangular segment sums: ``out[..., t, s] = sum_{i=s+1}^{t} v_i`` for ``s…, Monotone segment index per position (see ``lexhybrid.kernels.segments``)., _segment_ids(), segsum() (+5 more)

### Community 50 - "check_env.py"
Cohesion: 0.13
Nodes (22): importlib_metadata, packaging_markers, packaging_requirements, packaging_utils, packaging_version, Requirement, check_cuda_build(), check_editable() (+14 more)

### Community 51 - "sft_format.py"
Cohesion: 0.11
Nodes (39): Cite, encode(), Example, Passage, Quote, The retrieval-grounded SFT format: one prompt, pointer answers. The prompt is…, ``(prompt, target)`` as strings with the special-token markup., Token ids of one example: ``input_ids``, ``labels`` (-100 over the prompt) and… (+31 more)

### Community 52 - "test_verifier.py"
Cohesion: 0.04
Nodes (71): Answer verification: pointers, quotes, entailment and scrub leaks., entailment_index(), The class index whose label is ``entailment`` (any case); ``ValueError`` unless…, _balanced_body(), check_quote(), check_quotes(), extract_quotes(), _find_quote() (+63 more)

### Community 53 - "hybrid_lm.py"
Cohesion: 0.14
Nodes (12): filter_logits(), Temperature, then top-k, then nucleus (top-p) filtering; filtered entries…, choose(), CausalLMOutput, ``HybridLanguageModel``: embedding -> N x HybridBlock -> RMSNorm -> LM head.…, Shared sampling filter (``lexhybrid.decoding.filter_logits``)., Output of a causal language model forward pass., architecture_fingerprint() (+4 more)

### Community 54 - "LayerSplit"
Cohesion: 0.17
Nodes (8): LayerSplit, hook(), hook(), _mark(), Start/stop marker: a CUDA event on GPU, a wall clock on CPU., Per-`HybridBlock` timing, aggregated by mixer type. Markers are queued on the…, {mixer type: mean ms per forward}, plus a per-layer-index breakdown., timed()

### Community 55 - "test_multigpu.py"
Cohesion: 0.33
Nodes (6): _ddp_worker(), Multi-GPU checks. Marked ``multigpu``: they collect everywhere and skip, with…, Two processes, different data, one all-reduced gradient: identical on every…, test_ddp_step_gradients_identical_across_ranks(), torch_distributed, torch_multiprocessing

### Community 56 - "test_slurm_wrappers.py"
Cohesion: 0.15
Nodes (17): SLURM wrappers under `scripts/slurm/`. Static checks on every `*.sh` there: the…, A requeued run finds its last.ckpt and passes it on., The array index picks the arm; screen_arms.py resolves it ON THE NODE; the…, The wrappers the pipeline names are all here., Only the named shapes are rerun: the 1.7B shapes are rerun alone; the 8B shapes…, Stage B: INIT_FROM (stage A's last.ckpt) while the stage has none of its own;…, test_every_planned_wrapper_exists(), test_four_gpu_wrapper_runs_ddp_through_the_same_body() (+9 more)

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
Cohesion: 0.06
Nodes (51): hmac, Entity, entity_hash(), main(), part_name(), _pattern(), placeholders(), Path (+43 more)

### Community 61 - ".__init__"
Cohesion: 0.38
Nodes (4): MTPModule, Module, Tensor, One prediction depth: two norms, a 2D -> D projection, and one ``HybridBlock``.

### Community 62 - "_stub_transformers"
Cohesion: 0.12
Nodes (9): SimpleNamespace, `--docs-per-batch` puts the sentences of several documents into ONE predict…, test_scrub_worker_tags_a_batch_of_documents_in_one_predict_call(), __init__(), Stand-in tokenizer and model for ``MDebertaNLI``: one token per word, three…, Regression: ``truncation=True`` is ``longest_first``, which cut the…, _stub_transformers(), __call__() (+1 more)

### Community 63 - "code"
Cohesion: 0.09
Nodes (25): code(), directives(), A `--gpus` directive is not overridden by an environment variable: a multi-GPU…, The two shell lookups in fetch_hf.sh must find the constants the loaders use., The command as documented, the same check on the legal base, the CUDA smoke., The 4-GPU header, and the body it delegates to passes last.ckpt as…, ARM is resolved inside the job by `screen_arms.py env`, EXPERIMENT from the…, `#SBATCH --key=value` lines before the first command, trailing comments dropped. (+17 more)

### Community 64 - "HybridLightningModule"
Cohesion: 0.10
Nodes (18): HybridLightningModule, Optimizer, Tensor, Token-weighted sums for the epoch's ``val/perplexity``. Logged per batch with…, Clip gradients and log the pre-clip norm (manual, so foreach AdamW and AMP…, A WSD scheduler with an absolute warmup; records the decay window for the beta2…, Training loop around a ``HybridLanguageModel``. Args: model: the model to…, compute_accuracy() (+10 more)

### Community 65 - "test_renderer.py"
Cohesion: 0.10
Nodes (39): _check_sentences(), _labels(), Segment, ValueError, The renderer: the model's pointer answer as the text a reader sees. The answer…, The rendered answer of parsed segments against the retrieved passages (prompt…, The rendered answer of the model's raw answer text (``<|endoftext|>`` allowed…, Every source holding the passage: the chunk's own first, then its copies'… (+31 more)

### Community 66 - "load_model_config"
Cohesion: 0.09
Nodes (29): Configuration dataclass for the hybrid decoder. Ported from the reference…, Model configuration and the yaml loader every script uses., available_model_configs(), load_model_config(), load_model_yaml(), Any, Path, Load a model yaml into a ``HybridConfig`` -- the one path every script uses. A… (+21 more)

### Community 67 - "Bigram"
Cohesion: 0.13
Nodes (13): Bigram, log_table(), ``logits[b, t] = table[ids[b, t]]``: the next-token scores given the last token., Rows of probabilities -> log-probabilities (so a beam's score is a sum of known…, Finished hypotheses leave the beam with their EOS and are ranked by…, The reference's `length_penalty` divided every live beam by the same length, so…, CTRL penalty: a token already in the history has its positive logit divided…, test_length_penalty_is_not_inert() (+5 more)

### Community 68 - "build_dpo_pairs.py"
Cohesion: 0.10
Nodes (35): decode_answer(), parse(), prompt_ids(), The ``Example`` back from ``prompt + target`` (as ``render`` writes them)., The prompt's token ids, up to and including ``<|answer|>`` (the positions…, Answer tokens as the pointer markup: special tokens by name, so no tokenizer…, Pointers, answer_pointers() (+27 more)

### Community 69 - "test_code_hygiene.py"
Cohesion: 0.18
Nodes (17): _code_files(), git_calls_in_python(), git_calls_in_shell(), _is_sys_path_edit(), _names_git(), AST, parametrize, Path (+9 more)

### Community 70 - "_schedule_fit"
Cohesion: 0.17
Nodes (8): _IndexRows, Row i carries its index, so a run's data order can be read back., Lightning restarts a plain loader at row 0 on resume (screen run S5's…, Stage B: resumed from stage A's last step onto a shorter schedule of its own…, _schedule_fit(), _SeenRows, test_a_resumed_run_reads_on_through_the_schedule_and_matches_an_unbroken_one(), test_a_stage_on_a_new_schedule_starts_it_at_row_0_and_a_requeue_reads_on()

### Community 71 - "CausalLMAdapter"
Cohesion: 0.22
Nodes (4): CausalLMAdapter, Module, Tensor, A Hugging Face causal LM behind the hybrid's interfaces: the…

### Community 72 - "test_ci_workflow.py"
Cohesion: 0.16
Nodes (12): fixture, The CI workflow and the local validation script stay in step. PyYAML parses the…, Fast excludes slow/cuda/multigpu/network; --full and --ci exclude…, steps_text(), test_every_installing_job_uses_the_lock_and_never_pip(), test_hygiene_runs_on_a_pristine_checkout(), test_lint_runs_every_static_tool(), test_package_job_imports_the_wheel_outside_the_checkout() (+4 more)

### Community 73 - "test_reference_parity.py"
Cohesion: 0.20
Nodes (14): inspect, _block_defaults(), block_params(), _config_defaults(), load(), model_params(), _per_document(), parametrize (+6 more)

### Community 74 - "train_dpo.py"
Cohesion: 0.09
Nodes (34): copy, best_val_checkpoint(), The ``save_top_k`` lowest-``val/loss`` checkpoints, ``best-step<N>.ckpt``,…, check_run_settings(), Path, One output directory, one run. A job that finds ``last.ckpt`` there resumes it,…, ModelCheckpoint, omegaconf (+26 more)

### Community 75 - "cumulative_angles"
Cohesion: 0.25
Nodes (7): cumulative_angles(), Theta_t = sum_{i<=t} dt_i * theta_i, accumulated in float64 and wrapped to [0,…, test_cumulative_angles_restart_per_document(), Prop. 4's ordering constraint: B_{t-1} must carry its own Theta_{t-1}., test_angle_accumulation_stays_accurate_at_length(), test_rope_angles_reset_per_document(), test_rotate_then_shift_is_not_the_same_as_shift_then_rotate()

### Community 76 - "build_dense_index.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_dense_index.sh script

### Community 77 - "lexhybrid"
Cohesion: 0.12
Nodes (16): Acknowledgements and data attribution, Architecture at a glance, Cluster usage, Contributing, Data pipeline and retrieval index, Installation, Key properties, lexhybrid (+8 more)

### Community 78 - "parametrize"
Cohesion: 0.13
Nodes (15): parametrize, An array wrapper picks its work by SLURM_ARRAY_TASK_ID and logs each task on…, check_env.py reads uv.lock through `uv`; a missing uv failed with 'uv not found…, The Qwen3-8B-Base teacher fits only with gradient checkpointing., An ARCH line without an expected token, or with a forbidden one, stops the job…, The batch partition is preemptible. A training job is requeued, and its log is…, test_array_wrappers_log_per_task(), test_body() (+7 more)

### Community 79 - "eval_retrieval.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, eval_retrieval.sh script

### Community 80 - "test_rsync_rules.py"
Cohesion: 0.19
Nodes (13): parametrize, skipif, `.rsync-exclude` does what `scripts/sync_to_cluster.sh` needs. rsync is first-…, An unanchored `data/probes/` also dropped `lexhybrid/data/probes/`. Mirror…, `sync_to_cluster.sh` runs `rsync --delete`; excluded paths are protected on the…, rules(), test_data_excludes_are_anchored_to_the_root(), test_delete_keeps_what_the_cluster_owns() (+5 more)

### Community 82 - "run"
Cohesion: 0.13
Nodes (16): collate(), Dataset, One split (``train.parquet`` / ``dev.parquet``) as ``{input_ids, labels}``…, Right-pad ``input_ids`` (with 0: no real position sees a pad) and ``labels``…, SFTSplit, build_callbacks(), build_model(), build_module() (+8 more)

### Community 83 - "MambaBlock"
Cohesion: 0.19
Nodes (9): MambaBlock, Tensor, Re-apply the Mamba dt init after the model's global weight pass (no-op unless…, (batch, seq_len, dim) -> (batch, seq_len, dim); ``doc_ids`` resets state per…, Non-fast-path scan: the same implementation as ``selective_scan`` with chunk…, Per-(row, segment) wrapper that resets the SSM state at document boundaries., Mamba-1 mixer: input projection, depthwise causal conv, selective SSM, gated…, The declared capability and the real signature must not drift apart. (+1 more)

### Community 84 - "dpo_module.py"
Cohesion: 0.11
Nodes (23): answer_logps(), collate_pairs(), dpo_loss(), DPOLightningModule, kto_loss(), _pad(), Tensor, Preference optimisation of an SFT model: DPO and KTO against a frozen reference… (+15 more)

### Community 85 - "build_shards.py"
Cohesion: 0.13
Nodes (23): cache_dir(), encode_document(), load_tokenizer(), main(), ``data/hf/hub`` unless the environment names a Hugging Face cache., Qwen3's tokenizer, with the pointer specials unless ``with_specials=False``.…, Token ids of raw document text; special-token strings inside it are split into…, build() (+15 more)

### Community 86 - "test_repo_hygiene.py"
Cohesion: 0.23
Nodes (12): make_tree(), parametrize, Path, scripts/check_repo_hygiene.py on synthetic trees., test_a_missing_critical_file_is_reported(), test_an_ignored_analysis_document_is_reported(), test_case_collisions_are_caught(), test_clean_tree_passes() (+4 more)

### Community 87 - "German-law pretraining corpus: tokens, deduplication, scrub"
Cohesion: 0.29
Nodes (6): Budget rule, Discussion, German-law pretraining corpus: tokens, deduplication, scrub, Limitations, Reproduction, Summary

### Community 88 - "eval_gold_passages.py"
Cohesion: 0.12
Nodes (23): gold_relation(), ``(doc_id, copies) -> bool``: a chunk is a gold passage or a copy of one when…, build_sets(), ``(sets, counts)``: the prompt sets of every item with a gold passage (see the…, read_chunks(), _sets_of(), easy(), related() (+15 more)

### Community 89 - "build_index.py"
Cohesion: 0.13
Nodes (15): copy_groups(), Path, ``{document id: the other documents of its copy group, sorted}`` for every…, read_dedup_logs(), BgeM3Encoder, bge-m3's dense vectors: the [CLS] state of the last layer, L2-normalised, in…, build(), build_dense() (+7 more)

### Community 90 - "build_sft_data.py"
Cohesion: 0.09
Nodes (33): _unit(), question_key(), A hash of the question with its form normalised (NFC, case, whitespace, final…, Hex SHA-256 of the UTF-8 text: the identity dedup and the manifests key on., sha256_text(), German sentences of a passage: split after ``.``, ``!`` or ``?`` before an…, split_sentences(), build() (+25 more)

### Community 91 - "probe_ckpt_size.sh"
Cohesion: 0.15
Nodes (12): BATCH_SIZE, CKPT_EVERY, DATASET_CONFIG, DISTILL_CFG, EXPERIMENT, EXTRA_OVERRIDES, MAX_STEPS, MODEL_CONFIG (+4 more)

### Community 92 - "_CaptureLogger"
Cohesion: 0.12
Nodes (7): _CaptureLogger, _ddp_fit(), _RankRows, Under DDP (two CPU ranks, gloo) each rank reads its own rows, r, r + 2, ...;…, Keeps every ``log_metrics`` call in memory: ``rows = [(step, {tag: value})]``., Writes the row indices each DDP rank trains on to <dir>/rank<r>.txt., test_schedule_sampler_splits_ddp_ranks_and_reads_on_after_a_resume()

### Community 93 - "check_repo_hygiene.py"
Cohesion: 0.17
Nodes (19): check_critical(), check_files(), list_files(), load_ignore(), main(), Path, Critical files must exist; locally they must also survive ``.gitignore``. In…, Repository hygiene: the tree that would be (or was) pushed is clean. Two modes,… (+11 more)

### Community 94 - "screen_table.py"
Cohesion: 0.29
Nodes (12): apply_rule(), arm_means(), build_rows(), _f(), main(), parse_run(), The architecture screen's seed table, per-arm means and the decision rule.…, read_eval() (+4 more)

### Community 95 - "Profile ladder, chunk size and distillation memory"
Cohesion: 0.18
Nodes (10): Chunk size (training, forward+backward, slab loss, packed, doc_len 1,000, L=4,096), Discussion, Distillation memory probe, Hypotheses and criteria, Profile ladder, chunk size and distillation memory, Reproduction, Results, Sanity points (L=16,384, forward, bf16, batch 4) (+2 more)

### Community 96 - "README.md"
Cohesion: 0.32
Nodes (4): Corpus licence register, Evaluation sets (never trained on), Models the pipeline runs, Training sources (collectors)

### Community 97 - "Architecture screen: ablations of the hybrid backbone"
Cohesion: 0.33
Nodes (5): Architecture screen: ablations of the hybrid backbone, Discussion, Hypotheses and selection rule, Reproduction, Summary

### Community 98 - "BM25Index"
Cohesion: 0.16
Nodes (20): BM25Index, Path, A ``bm25s`` index over chunks, queried by text., Index ``chunks`` (anything with ``id`` and ``text``:…, build(), faithfulness_items(), _item(), main() (+12 more)

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

### Community 106 - "PointerFSM"
Cohesion: 0.11
Nodes (23): Decoding, PointerFSM, process(), Tensor, Pointer-constrained decoding. The legal model never generates statute text: it…, The pointer grammar for one retrieved context. Args: specials: the special-…, The state after ``token``; raises ``ValueError`` if the grammar forbids it., Boolean (vocab,) mask of the tokens the grammar allows next. (+15 more)

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

### Community 111 - "argparse"
Cohesion: 0.12
Nodes (39): argparse, collections, add_judge_args(), answer_text(), correctness_prompt(), faithfulness_prompt(), judge_from_args(), judge_prompts() (+31 more)

### Community 112 - "_doc"
Cohesion: 0.08
Nodes (32): admits(), build_corpus(), Corpus assembly gate: which documents may enter which training arm. Two arms:…, Whether ``doc`` (already licence-checked) belongs in ``arm``., Yield the documents of ``arm``; raise ``LicenceError`` on the first licence…, check_document(), is_commercial_safe(), Licence (+24 more)

### Community 113 - "evaluate_lm.py"
Cohesion: 0.14
Nodes (13): evaluate_perplexity(), load_model(), main(), measure_throughput(), ParquetRows, Dataset, no_grad, Tensor (+5 more)

### Community 114 - "eval_system_ablation.py"
Cohesion: 0.16
Nodes (19): gold_chunk_ids(), Each item's gold-relevant chunk ids: its ``gold_chunk_ids`` the index holds,…, chunk_dict(), copy_top(), main(), prompt_tokens(), Path, Append each arm's missing records to ``<out>/<arm>.jsonl``: A1-A4 per question,… (+11 more)

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
Nodes (34): OpenAICompatible, ``POST {base_url}/chat/completions``: one user message, temperature 0; returns…, requests, fake_post(), post(), FakeResponse, _qa_row(), The judge's HTTP client, its CLI, and its calibration items and scoring (all… (+26 more)

### Community 119 - "Tokenizer fertility on DACH legal text"
Cohesion: 0.25
Nodes (7): Discussion, Hypotheses and method, Limitations, Reproduction, Results, Summary, Tokenizer fertility on DACH legal text

### Community 120 - "_Paras"
Cohesion: 0.25
Nodes (3): _Paras, HTMLParser, The ``<div class="para">`` blocks as ``(bold, text)``: ``bold`` is the text of…

### Community 121 - "Document"
Cohesion: 0.05
Nodes (53): Change, changes_since(), main(), Path, The currency changelog: which provisions changed since a date. A provision's…, write_changelog(), parse_gii_xml(), parse_toc() (+45 more)

### Community 122 - "train_sft_hf.py"
Cohesion: 0.24
Nodes (12): ``lm`` frozen but for rank-``rank`` LoRA adapters on ``LORA_TARGETS`` and the…, wrap_lora(), adapt(), load_lm(), main(), DictConfig, main, The Hugging Face baseline of supervised fine-tuning: ``sft.hf_model``… (+4 more)

### Community 123 - "Backbone results: the hybrid and a Transformer baseline at 8,192 tokens"
Cohesion: 0.18
Nodes (10): Backbone results: the hybrid and a Transformer baseline at 8,192 tokens, Discussion, Handoff, Limitations, Predictions, Probes on the stage-B checkpoints, by context length, Reproduction, Results (+2 more)

### Community 124 - "test_gpu.py"
Cohesion: 0.10
Nodes (25): Summed CE and summed masked KL for one slab (recomputed in the backward pass)., Slab-wise ``(1 - alpha) * CE + alpha * T^2 * KL`` without (B, L, V) logits.…, slab_ce_kl(), _slab_terms(), _packed_doc_ids(), parametrize, Tensor, CUDA checks. Marked ``cuda``: they collect everywhere and skip, with the… (+17 more)

### Community 125 - "test_http_cache_replays_responses_without_the_network"
Cohesion: 0.17
Nodes (10): HttpCache, Path, Successful responses on disk, keyed by the request (method, URL, query, JSON…, Serve every ``http_get``/``http_post`` of this process through a cache under…, set_http_cache(), --http-cache: a restarted collection is served what it already fetched -- GET…, test_http_cache_replays_responses_without_the_network(), get() (+2 more)

### Community 126 - "build_corpus_array.sh"
Cohesion: 0.33
Nodes (5): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_corpus_array.sh script

### Community 127 - "pack_corpus.sh"
Cohesion: 0.33
Nodes (5): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, pack_corpus.sh script, TOKENIZERS_PARALLELISM

### Community 128 - "sft_module.py"
Cohesion: 0.23
Nodes (9): HeadFn, Mean over depths of each depth's boundary-masked cross-entropy. Args: hidden:…, Tensor, Cross-entropy over a large vocabulary without materialising ``(B, L, V)``…, Summed cross-entropy of ``head_fn(hidden)`` against ``targets`` (-100 ignored),…, _slab_ce_sum(), slab_cross_entropy(), Supervised fine-tuning: cross-entropy on the answer tokens only, for the hybrid… (+1 more)

### Community 129 - "measure_fertility.py"
Cohesion: 0.29
Nodes (11): collect(), main(), measure(), n_words(), pooled(), Path, Tokenizer fertility on DACH legal text (results in…, Whole documents in order until ``words`` words, skipping any longer than… (+3 more)

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

### Community 137 - "Chunk"
Cohesion: 0.05
Nodes (64): concurrent_futures, dataclasses, date, The gold-passage sanity check: with retrieval assumed solved, does the model…, Lexical retrieval with BM25 over the retrieval chunks (``bm25s``). Every…, Chunk, estimate_tokens(), Retrieval chunks: the units the retriever indexes and a prompt quotes. A chunk… (+56 more)

### Community 138 - "test_diag_spike_reruns_the_arm_observed_in_its_own_directory"
Cohesion: 0.40
Nodes (5): _compose(), The fake interpreter recorded `distill=null` happily; real Hydra refused it…, Loss-spike rerun: S1-s42 with the screen settings, seed and schedule unchanged…, test_diag_spike_reruns_the_arm_observed_in_its_own_directory(), test_every_train_argument_list_composes_under_real_hydra()

### Community 139 - "test_imports.py"
Cohesion: 0.24
Nodes (10): importlib, pkgutil, _exists_with_exact_case(), _internal_targets(), AST, parametrize, Path, Every module imports, and every internal import names the file with its exact… (+2 more)

### Community 140 - "ByteTokenizer"
Cohesion: 0.09
Nodes (29): _jsonl(), _qa_file(), _qa_row(), One evaluation-only set's manifest: the raw questions' hashes, as the…, QA rows with the writer's ``info.json`` beside them, free for training unless…, build_sft_data writes train and dev splits in which every target ends in…, QA items are all answerable: when they hold the share below 0.2, a seeded…, A QA file without info.json, or one that does not free its items for training;… (+21 more)

### Community 141 - "record_metrics"
Cohesion: 0.23
Nodes (11): doc_of(), _mean(), _ratio(), One summary per (system, seed, arm), in order of first appearance (see the…, The document of a chunk id (``<document id>#<n>``)., One record's metrics (see the module docstring)., record_metrics(), summarise() (+3 more)

### Community 142 - "parse_rii_xml"
Cohesion: 0.18
Nodes (10): _iso(), parse_rii_xml(), Element, Paragraph text: ``<p>``s become lines, ``<br/>`` breaks, whitespace normalised., One decision as a ``Document`` (None if it has no text)., read_zip_xml(), _text(), walk() (+2 more)

### Community 143 - "_executable"
Cohesion: 0.40
Nodes (5): _executable(), Path, The log is the job's own stdout, and GNU grep refuses to read its output file…, test_fetch_hf_models_override_fetches_only_those_at_main_unless_pinned(), test_kd_probe_summary_reads_its_own_log()

### Community 162 - "Judge"
Cohesion: 0.10
Nodes (21): Complete, average_precision(), first_json_object(), Judge, _parse(), Path, The first ``{...}`` in ``text`` that parses as a JSON object (code fences and…, A ``Complete`` behind the reply cache; ``calls`` counts the completions… (+13 more)

### Community 167 - "test_utils.py"
Cohesion: 0.10
Nodes (28): detect_prefix(), _first_segments(), infer_architecture(), infer_layer_types(), InferredArchitecture, load_every_key(), load_state_dict_guarded(), _mixer_params() (+20 more)

### Community 168 - "SourceShards"
Cohesion: 0.24
Nodes (6): Path, Where ``source``'s ``row_len``-token rows come from, and how many sub-rows a…, The rows of one source's split, read shard by shard on first use. With…, shard_directory(), SourceShards, flush()

### Community 169 - "corpus_counts.py"
Cohesion: 0.31
Nodes (9): functools, count(), main(), manifest_counts(), Path, Per-source corpus counts from the collection manifests. .venv/bin/python…, (jurisdiction, licence) -> documents and characters over one manifest., jurisdiction -> sampled documents, characters, tokens and tokens per character… (+1 more)

### Community 170 - "dump_reference_parity.py"
Cohesion: 0.31
Nodes (9): block_cases(), build_all(), import_reference(), inputs(), main(), model_cases(), Register the reference package without running its __init__ and without…, Dump the reference-parity fixtures -- run once, while Reference/ still exists.… (+1 more)

### Community 171 - "MDebertaNLI"
Cohesion: 0.22
Nodes (9): MDebertaNLI, mDeBERTa XNLI as an ``NLI`` callable: softmax over the classes, the entailment…, How it answers, calibrate(), main(), Path, Calibrate the verifier's entailment threshold on hand-labelled premise/claim…, read_pairs() (+1 more)

### Community 172 - "PretrainLightningModule"
Cohesion: 0.10
Nodes (18): no_grad, Tensor, ``(position_ids, attention_mask)`` for a Hugging Face decoder on packed rows.…, The teacher's final hidden states (after its final norm), (B, L, D_teacher),…, teacher_forward_packed(), teacher_inputs_packed(), PretrainLightningModule, Tensor (+10 more)

### Community 173 - "build_source_shards_stream"
Cohesion: 0.24
Nodes (9): build_source_shards(), build_source_shards_stream(), The validation document ids: ``max(ceil(fraction * n), minimum)`` of them, by…, Split one source's documents, pack both sides, write the shards and…, ``build_source_shards`` for a source that does not fit in memory: ONE pass over…, split_by_document(), At scale: one pass, each document tokenised once, packed at two row lengths…, test_streaming_build_equals_the_in_memory_build_at_every_row_len() (+1 more)

### Community 174 - "test_tfla_wide_chunks_factorise_per_block_exactly"
Cohesion: 0.33
Nodes (7): 256 positions (two chunks of 128), two rows, boundaries inside and on chunk…, Trained forget gates widened a 128-chunk's log-decay past fp32 (half-range 43.2…, In fp32 the per-block path forms no inf, forward or backward (a masked inf…, test_tfla_per_block_path_is_finite_and_close_in_fp32(), test_tfla_wide_chunks_factorise_per_block_exactly(), _tfla_grads(), _wide_tfla_inputs()

### Community 175 - "SFTLightningModule"
Cohesion: 0.28
Nodes (6): Args: those of ``HybridLightningModule`` (``learning_rate`` is the plain value…, ``(mean CE over the supervised targets, their count)``; position t predicts…, SFTLightningModule, The answer loss never uses the MTP head; DDP would wait for gradients it never…, test_sft_freezes_an_mtp_head_outside_its_loss(), __init__()

### Community 176 - "test_hf_sampling_through_its_cache_matches_the_uncached_reference"
Cohesion: 0.67
Nodes (3): Same logits, same generator: the KV-cached sampler and the uncached ``sample``…, test_hf_sampling_through_its_cache_matches_the_uncached_reference(), sample()

### Community 177 - "lightning_module.py"
Cohesion: 0.09
Nodes (27): LambdaLR, PyTorch Lightning wrapper for causal-LM training (ported base class of the…, Apply the beta2 schedule when WSD and ``beta2_schedule`` are both active., configure_optimizer(), get_parameter_groups(), Any, Module, Optimizer (+19 more)

### Community 178 - "FakeQwen"
Cohesion: 0.22
Nodes (4): FakeQwen, Just enough of a Hugging Face tokenizer for ``load_tokenizer``'s checks., test_tokenizer_cache_and_cli(), test_tokenizer_loader_checks_eos_and_vocabulary()

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
Cohesion: 0.26
Nodes (17): CompletedProcess, stat, calls(), _load_tree_hash(), make_tree(), parametrize, Path, `scripts/sync_to_cluster.sh` and `scripts/sync_from_cluster.sh`. Each script… (+9 more)

### Community 183 - "test_packaging.py"
Cohesion: 0.25
Nodes (5): subprocess, tarfile, dists(), fixture, The built distributions contain the package and nothing else (CI `package`…

### Community 184 - "train_dpo_4gpu.sh"
Cohesion: 0.29
Nodes (7): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, opt(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, train_dpo_4gpu.sh script

### Community 185 - "PairSchedule"
Cohesion: 0.29
Nodes (4): PairSchedule, Dataset, ``num_samples`` pairs on a seeded schedule (one permutation per pass), or every…, test_pair_schedule_is_seeded_and_reads_every_pair_once_per_pass()

### Community 186 - "test_eval_gold_passages_wrapper_keeps_runs_apart_and_reuses_sets"
Cohesion: 0.29
Nodes (7): Two runs of one system and seed (SFT and DPO) get their own outputs, so the…, test_eval_gold_passages_wrapper_keeps_runs_apart_and_reuses_sets(), run(), value(), Exit 1 from the audit (a rate above 0) is reported and the job succeeds, any…, test_ablation_wrapper_fails_on_an_audit_crash_but_not_on_truncation(), wrapper()

### Community 187 - "Results"
Cohesion: 0.33
Nodes (6): Deduplication (drop within a source, keep and link across), Efficiency, Estimate vs exact, Results, Scrub (`flair/ner-german-legal`, tags PER/RR/AN/STR), Tokens per source, jurisdiction, licence and arm (exact)

### Community 188 - "Results"
Cohesion: 0.33
Nodes (6): Efficiency, Paired comparisons, Per-seed calls (where the seeds disagree), Results, Seed table, first run (superseded: two runs collapsed), Seed table, second run (the measurement)

### Community 189 - "html_blocks"
Cohesion: 0.33
Nodes (5): html_parser, html_blocks(), HTML to text blocks for collectors that receive HTML fragments (OLDP). Stdlib…, The non-empty text blocks of ``html``, in document order., test_html_blocks()

### Community 191 - "setup_vllm.sh"
Cohesion: 0.33
Nodes (5): PYTHONUNBUFFERED, setup_vllm.sh script, UV_CACHE_DIR, UV_LINK_MODE, UV_PROJECT_ENVIRONMENT

### Community 192 - "_tfla_model"
Cohesion: 0.40
Nodes (5): The shipped configuration runs exact TFLA on its factorised path at init. With…, The record behind the fp32 range limit: forget bias 0 gives f = 0.5 and a half-…, test_tfla_fast_path_at_init(), test_tfla_reference_forget_bias_factorises_per_block(), _tfla_model()

### Community 193 - "test_param_bands"
Cohesion: 0.50
Nodes (3): Parameter counts of the legal model configs, The predicted counts and bands, and analysis/param_counts.md is current., test_param_bands()

### Community 194 - "test_packing_with_the_qwen3_tokenizer"
Cohesion: 0.50
Nodes (4): The documents back out of rows: real tokens concatenated and split after each…, The real tokenizer: every document decodes back exactly from 4,096-token rows., test_packing_with_the_qwen3_tokenizer(), _unpack()

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
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1776 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **30 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HybridLanguageModel` connect `HybridLanguageModel` to `test_data.py`, `test_retrieval.py`, `performance_profile.py`, `test_dpo.py`, `test_models.py`, `test_numerics_fp64.py`, `train_pretrain.py`, `evaluate_val_ppl.py`, `test_decoding.py`, `SyntheticPackedDataset`, `check_operator_equivalence.py`, `torch`, `test_scripts.py`, `HybridBlock`, `HybridConfig`, `datasets.py`, `screen_arms.py`, `test_training.py`, `test_utils.py`, `evaluate_probes.py`, `PretrainLightningModule`, `sandbox`, `lightning_module.py`, `sft_format.py`, `hybrid_lm.py`, `test_multigpu.py`, `test_slurm_wrappers.py`, `test_config_invariants.py`, `HybridLightningModule`, `_tfla_model`, `load_model_config`, `test_reference_parity.py`, `train_dpo.py`, `run`, `dpo_module.py`, `eval_gold_passages.py`, `Architecture`, `evaluate_lm.py`, `Document`, `test_gpu.py`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `Document` connect `Document` to `packing.py`, `test_data.py`, `pathlib`, `test_retrieval.py`, `Chunk`, `parse_rii_xml`, `index.py`, `test_collect_offline.py`, `http_get`, `parse_citation`, `fedlex.py`, `datasets.py`, `dedup.py`, `corpus_counts.py`, `evaluate_probes.py`, `build_source_shards_stream`, `test_verifier.py`, `eurlex.py`, `scrub_ler.py`, `build_shards.py`, `build_index.py`, `build_sft_data.py`, `Architecture`, `_doc`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `HybridConfig` connect `HybridConfig` to `HybridLanguageModel`, `test_data.py`, `test_retrieval.py`, `test_layers.py`, `test_dpo.py`, `test_models.py`, `test_numerics_fp64.py`, `train_pretrain.py`, `test_decoding.py`, `SyntheticPackedDataset`, `test_scripts.py`, `HybridBlock`, `datasets.py`, `screen_arms.py`, `test_training.py`, `test_utils.py`, `evaluate_probes.py`, `lightning_module.py`, `hybrid_lm.py`, `test_config_invariants.py`, `_tfla_model`, `load_model_config`, `test_reference_parity.py`, `train_dpo.py`, `dpo_module.py`, `eval_gold_passages.py`, `check_configs.py`, `Architecture`, `Document`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 57 inferred relationships involving `Document` (e.g. with `Data pipeline` and `Document schema`) actually correct?**
  _`Document` has 57 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `HybridLanguageModel` (e.g. with `Model` and `__getattr__()`) actually correct?**
  _`HybridLanguageModel` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 60 inferred relationships involving `Chunk` (e.g. with `answer_words()` and `cited_passages()`) actually correct?**
  _`Chunk` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `HybridConfig` (e.g. with `Model` and `load_model_config()`) actually correct?**
  _`HybridConfig` has 6 INFERRED edges - model-reasoned connections that need verification._