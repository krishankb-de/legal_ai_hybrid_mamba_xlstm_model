# Graph Report - legal_ai_hybrid_mamba_xlstm_model  (2026-10-28)

## Corpus Check
- 218 files · ~198,634 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 59 file(s) not represented in the graph (top: .jsonl 29, .xml 12, (none) 6)

## Summary
- 3224 nodes · 7162 edges · 167 communities (142 shown, 25 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 322 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7553031c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- torch
- HybridLanguageModel
- test_data.py
- collectors/base.py
- test_retrieval.py
- test_layers.py
- evalsets/base.py
- test_verifier.py
- performance_profile.py
- verifier.py
- Chunk
- analyze_generation_diversity.py
- json
- test_training.py
- test_models.py
- index.py
- test_numerics_fp64.py
- generate.py
- test_collect_offline.py
- train_pretrain.py
- Document
- evaluate_lm.py
- test_decoding.py
- http_get
- ris.py
- ssd_chunked_scan
- tfla_forward_parallel
- AttentionBlock
- load_model_config
- test_scripts.py
- parse_citation
- dataclasses
- scrub_ler.py
- HybridConfig
- SpikeDiagnosticsCallback
- _oldp_docs
- fedlex.py
- packing.py
- dedup.py
- datasets.py
- lightning_module.py
- Mamba3Block
- SyntheticPackedDataset
- statute_recall.py
- mLSTMBlock
- PointerFSM
- verify
- sandbox
- encode_document
- mqar.py
- check_env.py
- sft_format.py
- chunking.py
- HybridLightningModule
- LayerSplit
- test_config_invariants.py
- test_slurm_wrappers.py
- HybridBlock
- test_check_env.py
- pytest
- scrub_document
- StopAtStepCallback
- _stub_transformers
- code
- test_sync_scripts.py
- build_corpus
- measure_fertility.py
- greedy
- SpecialTokens
- test_code_hygiene.py
- mamba_block.py
- build_doc_block_mask
- test_ci_workflow.py
- test_reference_parity.py
- eurlex.py
- cumulative_angles
- screen_arms.py
- lexhybrid
- directives
- FineWeb2DECollector
- test_rsync_rules.py
- _BlockParser
- ScheduleSampler
- MambaBlock
- SignalCheckpointCallback
- PretrainLightningModule
- test_repo_hygiene.py
- German-law pretraining corpus: tokens, deduplication, scrub
- filters.py
- check_operator_equivalence.py
- scrub_ner_worker.py
- probe_ckpt_size.sh
- _CaptureLogger
- check_repo_hygiene.py
- screen_table.py
- Profile ladder, chunk size and distillation memory
- README.md
- Architecture screen: ablations of the hybrid backbone
- test_imports.py
- gpu_tests.sh
- multigpu_tests.sh
- train_pretrain_1gpu.sh
- Contributing
- check_configs.py
- changelog.py
- StepStatsCallback
- metrics.py
- dump_reference_parity.py
- equivalence.sh
- kd_memory_probe.sh
- Architecture
- OLDPCollector
- .__init__
- inspect_checkpoint.py
- screen_metrics.py
- preflight.sh
- profile.sh
- validate.sh
- test_packaging.py
- Tokenizer fertility on DACH legal text
- _Paras
- paragraphs
- .__init__
- ParquetRows
- test_tfla_wide_chunk_factorises_per_block_on_gpu
- render
- build_corpus_array.sh
- pack_corpus.sh
- _FakeSession
- CrossEncoderScorer
- corpus_counts.sh
- diag_compile.sh
- eval_probes.sh
- fetch_hf.sh
- scrub_array.sh
- setup_env.sh
- train_pretrain_4gpu.sh
- test_ssd_tfla_under_bf16_autocast_match_fp32
- test_diag_spike_reruns_the_arm_observed_in_its_own_directory
- test_param_bands
- admits
- _text
- _scan_cases
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
- test_train_wrapper_stops_before_step_0
- lexhybrid
- lexhybrid-scrub

## God Nodes (most connected - your core abstractions)
1. `Document` - 132 edges
2. `HybridLanguageModel` - 98 edges
3. `HybridConfig` - 64 edges
4. `Chunk` - 61 edges
5. `Section` - 41 edges
6. `load_model_config()` - 40 edges
7. `sandbox()` - 33 edges
8. `http_get()` - 32 edges
9. `AttentionBlock` - 32 edges
10. `parse_citation()` - 31 edges

## Surprising Connections (you probably didn't know these)
- `Data pipeline` --references--> `Document`  [INFERRED]
  docs/ARCHITECTURE.md → lexhybrid/data/schema.py
- `Citation ids` --references--> `parse_citation()`  [INFERRED]
  docs/DOCUMENT_SCHEMA.md → lexhybrid/data/schema.py
- `Hypotheses and method` --references--> `encode_document()`  [INFERRED]
  analysis/tokenizer_fertility.md → lexhybrid/data/tokenizer.py
- `Tokenizer and pointer vocabulary` --references--> `encode_document()`  [INFERRED]
  docs/ARCHITECTURE.md → lexhybrid/data/tokenizer.py
- `Model` --references--> `HybridConfig`  [INFERRED]
  docs/ARCHITECTURE.md → lexhybrid/config/hybrid_config.py

## Import Cycles
- None detected.

## Communities (167 total, 25 thin omitted)

### Community 0 - "torch"
Cohesion: 0.05
Nodes (54): HeadFn, LayerType, Gold-answer likelihood: a continuous probe score that an exact-match floor…, forward_fill_from_starts(), pad_segments(), device, Tensor, Document segments inside packed rows, shared by the SSD and TFLA kernels. A… (+46 more)

### Community 1 - "HybridLanguageModel"
Cohesion: 0.04
Nodes (46): argparse, __getattr__(), lexhybrid -- a DACH legal, citation-faithful, retrieval-gated hybrid Mamba-3 /…, CausalLMOutput, HybridLanguageModel, no_grad, Tensor, Language-model forward pass. Args: input_ids: (B, L) token ids; mutually… (+38 more)

### Community 2 - "test_data.py"
Cohesion: 0.04
Nodes (64): act_code(), The citation code of an act: its short name, else ``VO (EU) YYYY/N`` or ``RL…, latest_versions(), SPARQL: the German XML consolidations of the act in force with SR number ``sr``…, ``{sr: (date, file)}`` from SPARQL result bindings: per SR number the latest…, versions_query(), court_abbreviation(), ``Oberlandesgericht Nürnberg`` -> ``OLG Nürnberg``; ``Hessischer… (+56 more)

### Community 3 - "collectors/base.py"
Cohesion: 0.07
Nodes (55): collections_abc, datetime, hashlib, io, Corpus assembly gate: which documents may enter which training arm. Two arms:…, licence_flags(), main(), What every collector shares: the protocol, polite HTTP, and the CLI. A… (+47 more)

### Community 4 - "test_retrieval.py"
Cohesion: 0.07
Nodes (48): chunk_document(), estimate_tokens(), Statutes and regulations by provision, decisions by window; other types are not…, Ranked, Cross-encoder reranking of fused candidates into the prompt's passages. The…, The first ``limit`` candidates re-scored and sorted best first (ties keep the…, Best first, while there are fewer than ``max_chunks`` and the passages fit…, The step from the fused top 50 to the 3-8 passages (<= 4K tokens) of the prompt. (+40 more)

### Community 5 - "test_layers.py"
Cohesion: 0.07
Nodes (51): _bc_bias_pair(), _continue_with_steps(), _drive_lambda_to_one(), m3_block(), no_grad_params(), parametrize, Block-level behaviour of the four mixers and HybridBlock. Ported from the…, Isolation plus correctness: document B equals running document B on its own. (+43 more)

### Community 6 - "evalsets/base.py"
Cohesion: 0.09
Nodes (47): gzip, cli(), EvalSetInfo, Passage, Path, QAItem, QASet, Query (+39 more)

### Community 7 - "test_verifier.py"
Cohesion: 0.07
Nodes (48): entity_hash(), Answer verification: pointers, quotes, entailment and scrub leaks., candidate_forms(), leaked_entities(), Scrub leaks into the index: no scrubbed name in a retrievable chunk. The scrub…, Each capitalised word of ``text`` and each run of 2..``MAX_RUN`` capitalised…, ``(chunk id, entity_hmac)`` of every logged entity a chunk still holds, once…, _balanced_body() (+40 more)

### Community 8 - "performance_profile.py"
Cohesion: 0.07
Nodes (48): contextlib, csv, boundary_masked_labels(), Next-token targets ``labels[:, 1:]`` with every cross-document prediction set…, Summed CE and summed masked KL for one slab (recomputed in the backward pass)., Slab-wise ``(1 - alpha) * CE + alpha * T^2 * KL`` without (B, L, V) logits.…, slab_ce_kl(), _slab_terms() (+40 more)

### Community 9 - "verifier.py"
Cohesion: 0.11
Nodes (46): pointer_text(), The pointer as the model wrote it: ``<|q|><|c2|><|s1|><|s2|>``,…, CitePointer, parse_answer(), text(), pointers(), PointerSyntaxError, Segment (+38 more)

### Community 10 - "Chunk"
Cohesion: 0.09
Nodes (44): _check_sentences(), _labels(), Segment, ValueError, The renderer: the model's pointer answer as the text a reader sees. The answer…, The rendered answer of parsed segments against the retrieved passages (prompt…, The rendered answer of the model's raw answer text (``<|endoftext|>`` allowed…, Every source holding the passage: the chunk's own first, then its copies'… (+36 more)

### Community 11 - "analyze_generation_diversity.py"
Cohesion: 0.07
Nodes (42): collections, corpus_bleu(), _lcs_length(), _ngram_counts(), Counter, Sentence ROUGE-L and corpus BLEU, stdlib only. Ported verbatim from the…, Sentence-level ROUGE-L F-measure (Lin, 2004)., Corpus-level BLEU-N with brevity penalty, one reference per hypothesis. (+34 more)

### Community 12 - "json"
Cohesion: 0.08
Nodes (39): json, detect_prefix(), _first_segments(), infer_architecture(), infer_layer_types(), InferredArchitecture, _mixer_params(), Any (+31 more)

### Community 13 - "test_training.py"
Cohesion: 0.07
Nodes (41): no_grad, Tensor, ``(position_ids, attention_mask)`` for a Hugging Face decoder on packed rows.…, The teacher's final hidden states (after its final norm), (B, L, D_teacher),…, teacher_forward_packed(), teacher_inputs_packed(), WSD learning-rate multiplier at ``step``. warmup: linear from…, wsd_factor() (+33 more)

### Community 14 - "test_models.py"
Cohesion: 0.08
Nodes (43): copy, _delta_at_init(), _delta_model(), linux_only, parametrize, slow, Model-level behaviour of HybridLanguageModel., The model's weight pass zeroed both gate biases; `post_model_init` restores the… (+35 more)

### Community 15 - "index.py"
Cohesion: 0.08
Nodes (39): document_digest(), DocumentDigest, file_sha256(), indexed_documents(), IndexVersion, latest(), Path, Index versions: a registry of the retrieval indexes that have been built.… (+31 more)

### Community 16 - "test_numerics_fp64.py"
Cohesion: 0.10
Nodes (43): _audit_inputs(), _clamp_fire_fraction(), _measure(), parametrize, Tensor, Operator-level numerical correctness against float64 sequential oracles. Ported…, ``use_fast_path=False`` shares the one implementation (the reference once had a…, The legacy error is governed entirely by whether f_cum underflows the 1e-6… (+35 more)

### Community 17 - "generate.py"
Cohesion: 0.12
Nodes (39): Advance, apply_repetition_penalty(), _beam_core(), beam_search(), beam_search_cached(), beam_search_uncached(), best_of_n(), _check_one_sample() (+31 more)

### Community 18 - "test_collect_offline.py"
Cohesion: 0.08
Nodes (29): api_key(), GIICollector, RIICollector, FakeQwen, FakeWeb, The collectors' and fetchers' network loops, offline (CI has no network). Each…, ``get``/``post`` answering from ``routes`` (regex on the URL -> function(url,…, Just enough of a Hugging Face tokenizer for ``load_tokenizer``'s checks. (+21 more)

### Community 19 - "train_pretrain.py"
Cohesion: 0.08
Nodes (38): Build a config from a Hydra ``cfg.model`` node, keeping every field the…, Training callbacks: checkpoint-on-signal (ported from the reference) and step…, Loss-spike diagnostics for one training run. Two screening runs collapsed to…, main, math, pathlib, pytorch_lightning, pytorch_lightning_callbacks (+30 more)

### Community 20 - "Document"
Cohesion: 0.07
Nodes (34): cli(), Collector, parse_limit(), Collect up to ``limit`` documents (all when None) into ``out_dir/<name>.jsonl``…, ``--limit``: a positive number of documents, or ``all`` (every document the…, A public source of documents., run_collector(), manifest_record() (+26 more)

### Community 21 - "evaluate_lm.py"
Cohesion: 0.07
Nodes (33): Probes scored on any checkpoint: MQAR, statute recall, multi-hop., Seeded random-token rows with packed-document ids, for smoke tests and…, load_state_dict_guarded(), Module, Normalise ``model._orig_mod.`` / ``_orig_mod.model.`` / ``_orig_mod.`` /…, ``load_state_dict(strict=False)`` that refuses a load missing more than…, strip_prefixes(), evaluate_perplexity() (+25 more)

### Community 22 - "test_decoding.py"
Cohesion: 0.08
Nodes (37): filter_logits(), Temperature, then top-k, then nucleus (top-p) filtering; filtered entries…, Shared sampling filter (``lexhybrid.decoding.filter_logits``)., cached_lm(), family_lm(), _prefill_then_greedy(), parametrize, Decoding. Cached decode must be an equivalence with full recomputation. The… (+29 more)

### Community 23 - "http_get"
Cohesion: 0.08
Nodes (29): _cached(), http_get(), http_post(), HttpCache, Path, RateLimiter, Successful responses on disk, keyed by the request (method, URL, query, JSON…, Serve every ``http_get``/``http_post`` of this process through a cache under… (+21 more)

### Community 24 - "ris.py"
Cohesion: 0.08
Nodes (30): FetchError, RuntimeError, ``fetch(url)``'s response, or None when it raised ``FetchError``., A request failed after every retry., Fetch one document, or skip it (logged) when its fetch fails after every retry,…, SkipFailedDocuments, _as_list(), _clean() (+22 more)

### Community 25 - "ssd_chunked_scan"
Cohesion: 0.08
Nodes (36): Mamba-2/3 state space dual (SSD) kernels., device, Tensor, Lower-triangular segment sums: ``out[..., t, s] = sum_{i=s+1}^{t} v_i`` for ``s…, Monotone segment index per position (see ``lexhybrid.kernels.segments``)., Chunk-parallel SSD scan. h_t = exp(dt_t A) h_{t-1} + sum_terms coeff_t (B_t…, _segment_ids(), segsum() (+28 more)

### Community 26 - "tfla_forward_parallel"
Cohesion: 0.08
Nodes (36): TFLA (tiled linear attention) kernels for the mLSTM., apply_tfla(), _blockwise_scores(), _intra_chunk_sequential(), RuntimeError, Tensor, Chunk-parallel TFLA (tiled linear attention) for the mLSTM. Forward is O(L/C)…, Chunk-parallel TFLA. Args: q, k, v: (B, H, L, D) i_gate: input gates (B, H, L,… (+28 more)

### Community 27 - "AttentionBlock"
Cohesion: 0.08
Nodes (30): apply_rope(), AttentionBlock, build_rope_cache(), compiled_flex_attention(), positions_within_documents(), device, dtype, Tensor (+22 more)

### Community 28 - "load_model_config"
Cohesion: 0.08
Nodes (30): Configuration dataclass for the hybrid decoder. Ported from the reference…, Model configuration and the yaml loader every script uses., available_model_configs(), load_model_config(), load_model_yaml(), Any, Path, Load a model yaml into a ``HybridConfig`` -- the one path every script uses. A… (+22 more)

### Community 29 - "test_scripts.py"
Cohesion: 0.06
Nodes (24): fixture, The ported command-line scripts run end to end on CPU with tiny models., A tiny model yaml and a Lightning-style checkpoint for it., Every arm builds through HybridConfig.from_hydra at reduced width, runs a…, Full size on `meta` with the bands: every arm passes; S5 carries +2.36% as a…, Guard: if an arm's lever never reaches the model, its ARCH line lacks the token., The method: whole documents until the word budget, documents over the cap…, Write ``{tag: [(step, value), ...]}`` as TensorBoard events where… (+16 more)

### Community 30 - "parse_citation"
Cohesion: 0.08
Nodes (34): Limitations, docket(), ``7Ob143/26v`` -> ``7 Ob 143/26v`` (the spaced form courts print)., build_items(), numbered_once(), MultihopItem, no_grad, Two-hop cross-reference probe: the answer needs both passages. An item takes a… (+26 more)

### Community 31 - "dataclasses"
Cohesion: 0.09
Nodes (25): dataclasses, BM25Index, Hit, Path, Lexical retrieval with BM25 over the retrieval chunks (``bm25s``). Every…, The index's tokens of each text (lower-cased, German stopwords out, stemmed)., A ``bm25s`` index over chunks, queried by text., Index ``chunks`` (anything with ``id`` and ``text``:… (+17 more)

### Community 32 - "scrub_ler.py"
Cohesion: 0.09
Nodes (30): hmac, itertools, iter_part(), main(), part_name(), Path, The German legal-entity scrub: names out before training. Entities come from…, Each document with its entities, in order, from ONE worker process: the… (+22 more)

### Community 33 - "HybridConfig"
Cohesion: 0.08
Nodes (25): HybridConfig, Validate and auto-compute derived parameters., The per-mixer arguments for layer ``layer_idx`` (a readable summary; not used…, Every architectural knob of the hybrid decoder. One dataclass serves every…, Convert the config to a plain dictionary., Create a config from a dictionary produced by ``to_dict``., Write ``config.json`` into ``save_directory`` and return its path., HybridEmbedding (+17 more)

### Community 34 - "SpikeDiagnosticsCallback"
Cohesion: 0.11
Nodes (14): Callback, Tensor, Trainer, SpikeDiagnosticsCallback, hook(), hook(), hook(), hook() (+6 more)

### Community 35 - "_oldp_docs"
Cohesion: 0.09
Nodes (32): effective_weights(), mixture_from_config(), PackedMixture, Dataset, Tensor, Each admitted source's share of the rows: its group's share, split by weight…, Rows of the admitted sources as training items ``{input_ids, labels,…, ``SourceSpec``s from a dataset yaml's ``sources`` block. (+24 more)

### Community 36 - "fedlex.py"
Cohesion: 0.11
Nodes (27): act_code(), act_meta(), acts_query(), article_sections(), FedlexCollector, _inline(), _is_text(), _local() (+19 more)

### Community 37 - "packing.py"
Cohesion: 0.10
Nodes (23): Fields, main(), pack(), Packer, Path, EOS-delimited packing into fixed-length rows and parquet shards. Every document…, Parquet shards ``shard-00000.parquet``... of at most ``rows_per_shard`` rows,…, Parquet shards ``shard-00000.parquet``... of at most ``rows_per_shard`` rows. (+15 more)

### Community 38 - "dedup.py"
Cohesion: 0.09
Nodes (30): deduplicate(), deduplicate_signatures(), deduplicate_stream(), DedupLog, keep_order(), minhash(), Deduplication of the training corpus: exact, then near-duplicate. 1. **Exact.**…, ``deduplicate_signatures`` over documents streamed once, their MinHashes… (+22 more)

### Community 39 - "datasets.py"
Cohesion: 0.11
Nodes (23): build_source_shards(), build_source_shards_stream(), Path, Pretraining data over packed shards: the split, the mixture and the arms.…, Where ``source``'s ``row_len``-token rows come from, and how many sub-rows a…, The rows of one source's split, read shard by shard on first use. With…, The validation document ids: ``max(ceil(fraction * n), minimum)`` of them, by…, Split one source's documents, pack both sides, write the shards and… (+15 more)

### Community 40 - "lightning_module.py"
Cohesion: 0.10
Nodes (26): LambdaLR, PyTorch Lightning wrapper for causal-LM training (ported base class of the…, configure_optimizer(), get_parameter_groups(), Any, Module, Optimizer, Optimizer construction with weight-decay parameter groups (ported from the… (+18 more)

### Community 41 - "Mamba3Block"
Cohesion: 0.10
Nodes (18): heavy_tail_activation(), Mamba3Block, Tensor, (Re-)apply the SSM initializations after the parent model's global weight pass.…, Per-head state transition, strictly negative. Shape (nheads,) or (batch, len,…, (batch, seqlen, dim) -> (batch, seqlen, dim). Argument order is load-bearing:…, SSD scan, plus the trapezoidal term and the complex-state rotation when…, The last ``conv_size - 1`` pre-convolution columns, (batch, channels, conv_size… (+10 more)

### Community 42 - "SyntheticPackedDataset"
Cohesion: 0.10
Nodes (24): Dataset, Tensor, ``num_rows`` rows of ``row_len`` random tokens, each packed from several…, SyntheticPackedDataset, _cpu_trainer(), _diag_fit(), Stage B ends at step 18,037 with checkpoints every 1,000: without…, STEPSTATS: one timing per optimizer step, not per micro-batch; a line every… (+16 more)

### Community 43 - "statute_recall.py"
Cohesion: 0.11
Nodes (25): gold_nll(), no_grad, Mean per-token NLL of ``target_ids`` following ``prompt_ids`` (natural log).…, build_items(), char_f1(), exact_match(), _norm(), Counter (+17 more)

### Community 44 - "mLSTMBlock"
Cohesion: 0.10
Nodes (18): mLSTMBlock, Tensor, (Re-)apply the gate-bias inits after the parent model's global weight pass.…, Soft-capped gates: exp input gate (computed in fp32), sigmoid forget and output…, (batch, seq_len, dim) -> (batch, seq_len, dim). Args: x: input. cache: an EMPTY…, mLSTM decode state: the matrix cell ``C`` and the normalizer ``n``, both zeros,…, Advance one token. ``(batch, dim)`` or ``(batch, 1, dim)`` in, ``(batch, dim)``…, Soft cap via tanh: maps x to ``tanh(x / cap) * cap``, range (-cap, cap). (+10 more)

### Community 45 - "PointerFSM"
Cohesion: 0.12
Nodes (18): Decoding, PointerFSM, process(), Tensor, Pointer-constrained decoding. The legal model never generates statute text: it…, The pointer grammar for one retrieved context. Args: specials: the special-…, The state after ``token``; raises ``ValueError`` if the grammar forbids it., Boolean (vocab,) mask of the tokens the grammar allows next. (+10 more)

### Community 46 - "verify"
Cohesion: 0.10
Nodes (22): asks_for_party(), ClaimCheck, pooled_citation_precision(), _precision(), The verifier log line: no answer text, so it can be kept beside the manifests., Passed cited claims over cited claims, pooled over many answers (``None`` if…, The question asks who a party of a case is (German or English patterns)., The verified, rendered answer of the model's raw ``answer`` (see the module… (+14 more)

### Community 47 - "sandbox"
Cohesion: 0.08
Nodes (24): fixture, The ARCH line of the real base model (S1), built on meta from the screen's own…, A repository copy with the wrappers, screen_arms.py and a fake interpreter;…, One sequence length per process, a fresh Inductor cache per point, the…, Task i scrubs part i % PARTS of source i / PARTS; every task uses the same HMAC…, Loss-spike inspection: by default the two collapsed screen runs and a healthy…, The array index picks the arm; screen_arms.py resolves its config and screen2_…, The login node runs no rm: approved deletions go through cleanup.sh. Only… (+16 more)

### Community 48 - "encode_document"
Cohesion: 0.13
Nodes (22): functools, cache_dir(), encode_document(), load_tokenizer(), main(), The student's tokenizer: Qwen3's own, plus the SFT pointer specials.…, ``data/hf/hub`` unless the environment names a Hugging Face cache., Qwen3's tokenizer, with the pointer specials unless ``with_specials=False``.… (+14 more)

### Community 49 - "mqar.py"
Cohesion: 0.11
Nodes (15): generate(), MQARConfig, MQARItem, no_grad, Multi-query associative recall, MQAR, with the upstream reference's…, Accuracy over all queries; the logits are computed at the query positions only., score(), numpy (+7 more)

### Community 50 - "check_env.py"
Cohesion: 0.13
Nodes (22): importlib_metadata, packaging_markers, packaging_requirements, packaging_utils, packaging_version, Requirement, check_cuda_build(), check_editable() (+14 more)

### Community 51 - "sft_format.py"
Cohesion: 0.17
Nodes (20): Cite, encode(), Example, parse(), Passage, Quote, The retrieval-grounded SFT format: one prompt, pointer answers. The prompt is…, The ``Example`` back from ``prompt + target`` (as ``render`` writes them). (+12 more)

### Community 52 - "chunking.py"
Cohesion: 0.13
Nodes (22): German sentences of a passage: split after ``.``, ``!`` or ``?`` before an…, split_sentences(), act_of(), _anchor(), chunk_case_law(), chunk_statute(), _common(), _label() (+14 more)

### Community 53 - "HybridLightningModule"
Cohesion: 0.14
Nodes (9): HybridLightningModule, Optimizer, Tensor, Token-weighted sums for the epoch's ``val/perplexity``. Logged per batch with…, Clip gradients and log the pre-clip norm (manual, so foreach AdamW and AMP…, A WSD scheduler with an absolute warmup; records the decay window for the beta2…, Apply the beta2 schedule when WSD and ``beta2_schedule`` are both active., Training loop around a ``HybridLanguageModel``. Args: model: the model to… (+1 more)

### Community 54 - "LayerSplit"
Cohesion: 0.11
Nodes (11): _delta_ms(), LayerSplit, hook(), hook(), _mark(), Start/stop marker: a CUDA event on GPU, a wall clock on CPU., Per-`HybridBlock` timing, aggregated by mixer type. Markers are queued on the…, {mixer type: mean ms per forward}, plus a per-layer-index breakdown. (+3 more)

### Community 55 - "test_config_invariants.py"
Cohesion: 0.11
Nodes (16): counts(), fixture, parametrize, slow, Configuration invariants: the living spec of what every yaml guarantees., One factor at a time: each screen yaml equals the base in every field but its…, S5 (legacy recipe): the <= 1% target did not hold (+2.36%); it is reported as a…, Structural parity with the reference: same modules, same shapes, to the… (+8 more)

### Community 56 - "test_slurm_wrappers.py"
Cohesion: 0.13
Nodes (19): SLURM wrappers under `scripts/slurm/`. Static checks on every `*.sh` there: the…, The Qwen3-8B-Base teacher fits only with gradient checkpointing., A requeued run finds its last.ckpt and passes it on., The array index picks the arm; screen_arms.py resolves it ON THE NODE; the…, The wrappers the pipeline names are all here., Only the named shapes are rerun: the 1.7B shapes are rerun alone; the 8B shapes…, Stage B: INIT_FROM (stage A's last.ckpt) while the stage has none of its own;…, test_every_planned_wrapper_exists() (+11 more)

### Community 57 - "HybridBlock"
Cohesion: 0.10
Nodes (17): HybridBlock, Tensor, (batch, seq_len, dim) -> (batch, seq_len, dim). Argument order matches…, Per-layer decode state, or None if this mixer has no step (Mamba-1)., One token through mixer + MLP with residuals, mirroring ``forward``'s topology…, One decoder block: ``x + mixer(norm1(x))``, then the MLP (pre-norm, or…, Under `hybrid`, block >= 1 computes norm2(x + mlp(x)); block 0 stays pre-norm., The reference filtered on unprefixed names, so every `mlstm_*` option fell back… (+9 more)

### Community 58 - "test_check_env.py"
Cohesion: 0.10
Nodes (15): env(), skipif, slow, scripts/check_env.py: the environment-equals-lock check., The CI condition, reproduced: the real uv, FORCE_COLOR=1, the real lock., Run from the main environment, `--scrub` must fail: no flair, transformers 5,…, The live venv equals the lock (skipped in fast mode; CI and --full run it)., FORCE_COLOR=1 made uv colour its "# via" lines, and they failed to parse. (+7 more)

### Community 59 - "pytest"
Cohesion: 0.13
Nodes (18): importlib_util, pytest, reference, load_script(), fixture, pytest_collection_modifyitems(), Shared test configuration. Markers are declared in pyproject.toml under…, Every test starts from the same RNG state, so failures reproduce. (+10 more)

### Community 60 - "scrub_document"
Cohesion: 0.15
Nodes (20): Entity, _pattern(), placeholders(), Surface form -> placeholder, numbered per tag in order of first appearance., Replace every occurrence of the propagating forms (longest first, whole words…, The tagged spans replaced exactly (right to left), then the propagating forms., The document with its people replaced, and its scrub-log lines. A section found…, The model tags whitespace tokens, so ``Haag,`` or ``(Müller`` carries… (+12 more)

### Community 61 - "StopAtStepCallback"
Cohesion: 0.11
Nodes (13): End training after optimizer step ``stop_at_step`` while the LR schedule keeps…, StopAtStepCallback, _ddp_fit(), _IndexRows, Under DDP (two CPU ranks, gloo) each rank reads its own rows, r, r + 2, ...;…, Row i carries its index, so a run's data order can be read back., Lightning restarts a plain loader at row 0 on resume (screen run S5's…, Stage B: resumed from stage A's last step onto a shorter schedule of its own… (+5 more)

### Community 62 - "_stub_transformers"
Cohesion: 0.12
Nodes (13): entailment_index(), MDebertaNLI, Entailment of a claim by its cited sources: the NLI step of the verifier. An…, The class index whose label is ``entailment`` (any case); ``ValueError`` unless…, mDeBERTa XNLI as an ``NLI`` callable: softmax over the classes, the entailment…, How it answers, SimpleNamespace, Stand-in tokenizer and model for ``MDebertaNLI``: one token per word, three… (+5 more)

### Community 63 - "code"
Cohesion: 0.10
Nodes (20): code(), The two shell lookups in fetch_hf.sh must find the constants the loaders use., check_env.py reads uv.lock through `uv`; a missing uv failed with 'uv not found…, The command as documented, the same check on the legal base, the CUDA smoke., The 4-GPU header, and the body it delegates to passes last.ckpt as…, ARM is resolved inside the job by `screen_arms.py env`, EXPERIMENT from the…, fetch_hf.sh fetched Qwen3-1.7B-Base at the tokenizer's pinned commit, which…, The scrub array and the counts read the same German-market sources; the pack… (+12 more)

### Community 64 - "test_sync_scripts.py"
Cohesion: 0.26
Nodes (17): CompletedProcess, stat, calls(), _load_tree_hash(), make_tree(), parametrize, Path, `scripts/sync_to_cluster.sh` and `scripts/sync_from_cluster.sh`. Each script… (+9 more)

### Community 65 - "build_corpus"
Cohesion: 0.15
Nodes (18): build_corpus(), Yield the documents of ``arm``; raise ``LicenceError`` on the first licence…, check_document(), is_commercial_safe(), Licence, LicenceError, ValueError, Licence register in code. ``docs/CORPUS_LICENCE_REGISTER.md`` is its prose… (+10 more)

### Community 66 - "measure_fertility.py"
Cohesion: 0.17
Nodes (15): DIPCollector, ``types``: the Drucksache types, collected in turn, newest first;…, collect(), _collectors(), load_tokenizers(), main(), measure(), n_words() (+7 more)

### Community 67 - "greedy"
Cohesion: 0.12
Nodes (17): greedy(), Uncached greedy decoding: the full forward is recomputed for every new token…, Bigram, log_table(), ``logits[b, t] = table[ids[b, t]]``: the next-token scores given the last token., Rows of probabilities -> log-probabilities (so a beam's score is a sum of known…, Finished hypotheses leave the beam with their EOS and are ranked by…, The reference's `length_penalty` divided every live beam by the same length, so… (+9 more)

### Community 68 - "SpecialTokens"
Cohesion: 0.15
Nodes (10): J for ``<|sJ|>``, else None., Ids of the 87 SFT-only special tokens (never used in distillation)., The token strings, in registry order (the order ``from_offset`` assigns ids)., Consecutive ids from ``first_id`` -- tests, and any vocabulary that appends…, Ids from a tokenizer that has the tokens added (they are added to the Qwen3…, K for ``<|cK|>``, else None., SpecialTokens, _specials() (+2 more)

### Community 69 - "test_code_hygiene.py"
Cohesion: 0.18
Nodes (17): _code_files(), git_calls_in_python(), git_calls_in_shell(), _is_sys_path_edit(), _names_git(), AST, parametrize, Path (+9 more)

### Community 70 - "mamba_block.py"
Cohesion: 0.20
Nodes (15): einops, Mamba-1 selective scan (legacy ablation only)., Tensor, Mamba-1 selective scan (legacy incumbent; used only by the…, Exact selective scan -- the recurrence the Mamba-1 block is specified to…, Exact chunked selective scan -- no division, no clamp. Flips which axis is…, Public selective-scan entry point, computed in fp32 and cast back.…, Legacy chunk-parallel selective scan (divide-and-clamp; see module docstring).… (+7 more)

### Community 71 - "build_doc_block_mask"
Cohesion: 0.12
Nodes (16): Limitations, disable, build_doc_block_mask(), build_doc_boundary_attn_mask(), Additive causal mask that also blocks attention across documents. Args:…, FlexAttention ``BlockMask`` for causal attention inside each document of a…, _packed_doc_ids(), linux_only (+8 more)

### Community 72 - "test_ci_workflow.py"
Cohesion: 0.16
Nodes (12): fixture, The CI workflow and the local validation script stay in step. PyYAML parses the…, Fast excludes slow/cuda/multigpu/network; --full and --ci exclude…, steps_text(), test_every_installing_job_uses_the_lock_and_never_pip(), test_hygiene_runs_on_a_pristine_checkout(), test_lint_runs_every_static_tool(), test_package_job_imports_the_wheel_outside_the_checkout() (+4 more)

### Community 73 - "test_reference_parity.py"
Cohesion: 0.20
Nodes (14): inspect, _block_defaults(), block_params(), _config_defaults(), load(), model_params(), _per_document(), parametrize (+6 more)

### Community 74 - "eurlex.py"
Cohesion: 0.22
Nodes (13): EURLexCollector, _find(), _inline(), _local(), oj_date(), parse_eurlex_xhtml(), Element, EUR-Lex: EU regulations and directives in German. Source: the Publications… (+5 more)

### Community 75 - "cumulative_angles"
Cohesion: 0.17
Nodes (14): apply_rotary(), cumulative_angles(), cumulative_angles_fp64(), Tensor, Data-dependent rotary embeddings for the complex-valued SSM. Mamba-3 Sec 3.2…, Theta_t = sum_{i<=t} dt_i * theta_i, accumulated in float64 and wrapped to [0,…, ``cumulative_angles`` before the final cast: float64, wrapped to [0, 2pi).…, Forward-fill each segment's start value across the segment. (+6 more)

### Community 76 - "screen_arms.py"
Cohesion: 0.19
Nodes (15): NamedTuple, Arm, build_config(), check_tokens(), env_lines(), _fmt(), hydra_overrides(), _inference_non_embedding() (+7 more)

### Community 77 - "lexhybrid"
Cohesion: 0.12
Nodes (16): Acknowledgements and data attribution, Architecture at a glance, Cluster usage, Contributing, Data pipeline and retrieval index, Installation, Key properties, lexhybrid (+8 more)

### Community 78 - "directives"
Cohesion: 0.16
Nodes (16): directives(), parametrize, An array wrapper picks its work by SLURM_ARRAY_TASK_ID and logs each task on…, A `--gpus` directive is not overridden by an environment variable: a multi-GPU…, `#SBATCH --key=value` lines before the first command, trailing comments dropped., The batch partition is preemptible. A training job is requeued, and its log is…, test_array_wrappers_log_per_task(), test_body() (+8 more)

### Community 79 - "FineWeb2DECollector"
Cohesion: 0.14
Nodes (8): FineWeb2DECollector, iter_rows(), MultiLegalPileCollector, The rows of one ``.jsonl.xz`` file, decompressed as they arrive., Each subset in turn; with a limit, an equal share of it from each., subset_name(), test_fineweb_and_multilegalpile_collectors_read_their_rows(), load_dataset()

### Community 80 - "test_rsync_rules.py"
Cohesion: 0.19
Nodes (13): parametrize, skipif, `.rsync-exclude` does what `scripts/sync_to_cluster.sh` needs. rsync is first-…, An unanchored `data/probes/` also dropped `lexhybrid/data/probes/`. Mirror…, `sync_to_cluster.sh` runs `rsync --delete`; excluded paths are protected on the…, rules(), test_data_excludes_are_anchored_to_the_root(), test_delete_keeps_what_the_cluster_owns() (+5 more)

### Community 81 - "_BlockParser"
Cohesion: 0.19
Nodes (7): html_parser, _BlockParser, html_blocks(), HTMLParser, HTML to text blocks for collectors that receive HTML fragments (OLDP). Stdlib…, The non-empty text blocks of ``html``, in document order., test_html_blocks()

### Community 82 - "ScheduleSampler"
Cohesion: 0.19
Nodes (6): _rank_world(), A training mixture's rows for one rank, from where the run stands. Sample ``i``…, Every row of an evaluation set across the ranks, each rank the same count…, ScheduleSampler, ShardSampler, test_schedule_sampler_splits_ranks_evenly_and_skips_what_was_trained()

### Community 83 - "MambaBlock"
Cohesion: 0.19
Nodes (9): MambaBlock, Tensor, Re-apply the Mamba dt init after the model's global weight pass (no-op unless…, (batch, seq_len, dim) -> (batch, seq_len, dim); ``doc_ids`` resets state per…, Non-fast-path scan: the same implementation as ``selective_scan`` with chunk…, Per-(row, segment) wrapper that resets the SSM state at document boundaries., Mamba-1 mixer: input projection, depthwise causal conv, selective SSM, gated…, The declared capability and the real signature must not drift apart. (+1 more)

### Community 84 - "SignalCheckpointCallback"
Cohesion: 0.16
Nodes (6): FinalCheckpointCallback, Callback, Write ``<checkpoint_dir>/last.ckpt`` when training ends normally.…, Write ``interrupt.ckpt`` when the process is signalled; a no-op after the first…, SignalCheckpointCallback, test_signal_callback_saves_once()

### Community 85 - "PretrainLightningModule"
Cohesion: 0.21
Nodes (6): PretrainLightningModule, Tensor, Clip and log the gradient norm; a non-finite norm clears the gradients (step…, Args: those of ``HybridLightningModule`` plus teacher: optional frozen Hugging…, The late-25% PPL: a predicted token counts as late when at least late_from…, test_late_context_ppl_counts_document_context_and_matches_training_val_ce()

### Community 86 - "test_repo_hygiene.py"
Cohesion: 0.23
Nodes (12): make_tree(), parametrize, Path, scripts/check_repo_hygiene.py on synthetic trees., test_a_missing_critical_file_is_reported(), test_an_ignored_analysis_document_is_reported(), test_case_collisions_are_caught(), test_clean_tree_passes() (+4 more)

### Community 87 - "German-law pretraining corpus: tokens, deduplication, scrub"
Cohesion: 0.15
Nodes (12): Budget rule, Deduplication (drop within a source, keep and link across), Discussion, Efficiency, Estimate vs exact, German-law pretraining corpus: tokens, deduplication, scrub, Limitations, Reproduction (+4 more)

### Community 88 - "filters.py"
Cohesion: 0.31
Nodes (12): date, _allowed(), _bound(), _day(), filter_chunks(), _parse(), Retrieval filters: jurisdiction, validity date and document type.…, A ``date`` from a date, a datetime (its day) or an ISO date string. (+4 more)

### Community 89 - "check_operator_equivalence.py"
Cohesion: 0.23
Nodes (11): check_model(), check_operator(), main(), _operands(), Model level: do the logits move? This is what a decoded token actually sees., The equivalence gate every operator or speed change has to pass. Ported from…, max |a-b| / max(|b|, eps), computed in fp64., Operator level: does changing chunk_size change the function the scan computes? (+3 more)

### Community 90 - "scrub_ner_worker.py"
Cohesion: 0.22
Nodes (12): cap(), chunks(), default_cache_dir(), load_tagger(), main(), flush(), Entities of every document, the sentences of all ``docs`` tagged in ONE…, German legal-entity recognition for the LER scrub. Runs in the scrub… (+4 more)

### Community 91 - "probe_ckpt_size.sh"
Cohesion: 0.15
Nodes (12): BATCH_SIZE, CKPT_EVERY, DATASET_CONFIG, DISTILL_CFG, EXPERIMENT, EXTRA_OVERRIDES, MAX_STEPS, MODEL_CONFIG (+4 more)

### Community 92 - "_CaptureLogger"
Cohesion: 0.15
Nodes (4): _CaptureLogger, _RankRows, Keeps every ``log_metrics`` call in memory: ``rows = [(step, {tag: value})]``., Writes the row indices each DDP rank trains on to <dir>/rank<r>.txt.

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
Cohesion: 0.22
Nodes (6): Corpus licence register, Evaluation sets (never trained on), Models the pipeline runs, Training sources (collectors), Citation ids, Document schema

### Community 97 - "Architecture screen: ablations of the hybrid backbone"
Cohesion: 0.18
Nodes (11): Architecture screen: ablations of the hybrid backbone, Discussion, Efficiency, Hypotheses and selection rule, Paired comparisons, Per-seed calls (where the seeds disagree), Reproduction, Results (+3 more)

### Community 98 - "test_imports.py"
Cohesion: 0.24
Nodes (10): importlib, pkgutil, _exists_with_exact_case(), _internal_targets(), AST, parametrize, Path, Every module imports, and every internal import names the file with its exact… (+2 more)

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
Cohesion: 0.20
Nodes (9): Cluster jobs, Continuous integration, Contributing, Dependencies, Lint and style, Repository hygiene, Setup, Tests (+1 more)

### Community 103 - "check_configs.py"
Cohesion: 0.27
Nodes (9): hydra, hydra_core_global_hydra, check_model(), compose_model(), main(), model_names(), validate.sh gate 3: every model yaml composes through Hydra and pins what it…, Problems with ``configs/model/<name>.yaml``; empty when it passes. (+1 more)

### Community 104 - "changelog.py"
Cohesion: 0.31
Nodes (9): Change, changes_since(), main(), Path, The currency changelog: which provisions changed since a date. A provision's…, write_changelog(), _fedlex(), RIS versions (linked to the version they replaced), Fedlex articles by their… (+1 more)

### Community 106 - "metrics.py"
Cohesion: 0.27
Nodes (9): compute_accuracy(), compute_bits_per_token(), compute_mqar_accuracy(), Tensor, Training and evaluation metrics (ported from the reference)., Cross-entropy in bits per token (``loss / ln 2``). The reference named this…, Token-level argmax accuracy over positions whose label is not ``ignore_index``., Accuracy at the query positions of a multi-query associative-recall sequence. (+1 more)

### Community 107 - "dump_reference_parity.py"
Cohesion: 0.31
Nodes (9): block_cases(), build_all(), import_reference(), inputs(), main(), model_cases(), Register the reference package without running its __init__ and without…, Dump the reference-parity fixtures -- run once, while Reference/ still exists.… (+1 more)

### Community 108 - "equivalence.sh"
Cohesion: 0.22
Nodes (9): CUDA_LAUNCH_BLOCKING, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, equivalence.sh script, step() (+1 more)

### Community 109 - "kd_memory_probe.sh"
Cohesion: 0.22
Nodes (9): DATASET_CONFIG, GRAD_CKPT, MAX_STEPS, MODEL_CONFIG, NUM_GPUS, probe(), SAVE_TOP_K, kd_memory_probe.sh script (+1 more)

### Community 110 - "Architecture"
Cohesion: 0.22
Nodes (9): Answer pipeline, Architecture, Data pipeline, Model, Renderer, Retrieval, Tokenizer and pointer vocabulary, Training (+1 more)

### Community 111 - "OLDPCollector"
Cohesion: 0.22
Nodes (5): iter_rows(), OLDPCollector, The dump's rows in shard order, one parquet row group in memory at a time., The OLDP API stopped paging past page 10, so the collector reads the bulk dump…, test_oldp_collector_reads_the_bulk_dump()

### Community 112 - ".__init__"
Cohesion: 0.25
Nodes (8): _mamba3_params(), _mlstm_params(), _prefixed_kwargs(), The ``mamba3_*`` options this dispatcher forwards, read off the block's…, Every ``mLSTMBlock`` parameter, read off its signature like ``_mamba3_params``., Kwargs for a prefixed mixer: ``shared`` unprefixed names first, then…, _signature_params(), test_the_dispatcher_whitelist_is_derived_not_listed()

### Community 113 - "inspect_checkpoint.py"
Cohesion: 0.33
Nodes (8): _f(), inspect(), main(), optimizer_names(), part_of(), Module, Per-parameter weights and AdamW state of a training checkpoint.…, Parameter names in the order AdamW's state indexes them (group by group).

### Community 114 - "screen_metrics.py"
Cohesion: 0.39
Nodes (8): _fmt(), latest_version(), main(), Path, Validation and training curves of the screen runs, read from their TensorBoard…, ``<run>/logs/<exp>/version_<n>`` with the highest n that holds an events file., read_scalars(), report()

### Community 115 - "preflight.sh"
Cohesion: 0.25
Nodes (8): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PATH, PYTHONUNBUFFERED, preflight.sh script, step(), TORCHINDUCTOR_CACHE_DIR

### Community 116 - "profile.sh"
Cohesion: 0.25
Nodes (8): CUDA_LAUNCH_BLOCKING, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, point(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, profile.sh script

### Community 117 - "validate.sh"
Cohesion: 0.39
Nodes (8): fail(), FAILED, pass(), PASSED, run_gate(), validate.sh script, warn(), WARNED

### Community 118 - "test_packaging.py"
Cohesion: 0.22
Nodes (6): shutil, subprocess, tarfile, dists(), fixture, The built distributions contain the package and nothing else (CI `package`…

### Community 119 - "Tokenizer fertility on DACH legal text"
Cohesion: 0.25
Nodes (7): Discussion, Hypotheses and method, Limitations, Reproduction, Results, Summary, Tokenizer fertility on DACH legal text

### Community 120 - "_Paras"
Cohesion: 0.25
Nodes (3): _Paras, HTMLParser, The ``<div class="para">`` blocks as ``(bold, text)``: ``bold`` is the text of…

### Community 121 - "paragraphs"
Cohesion: 0.33
Nodes (6): is_heading(), paragraphs(), A page's footnotes interrupt the paragraph that runs on to the next page ("...…, The paper's paragraphs as ``(is_heading, text)``, cleaned of the PDF's layout…, _rejoin_across_footnotes(), test_dip_paragraph_rules()

### Community 122 - ".__init__"
Cohesion: 0.38
Nodes (4): MTPModule, Module, Tensor, One prediction depth: two norms, a 2D -> D projection, and one ``HybridBlock``.

### Community 123 - "ParquetRows"
Cohesion: 0.29
Nodes (4): ParquetRows, Dataset, Tensor, Rows of a packed parquet shard: ``input_ids`` (and ``doc_ids`` when present),…

### Community 124 - "test_tfla_wide_chunk_factorises_per_block_on_gpu"
Cohesion: 0.33
Nodes (7): _packed_doc_ids(), parametrize, Tensor, On the GPU, a 128-chunk whose log-decay overflows one factorisation (forget…, CUDA variant of tests/test_layers.py::test_flex_block_mask_equals_dense: the…, test_flex_block_mask_equals_dense_on_gpu(), test_tfla_wide_chunk_factorises_per_block_on_gpu()

### Community 125 - "render"
Cohesion: 0.33
Nodes (6): Element, Text of a gii-norm content element: list items on their own lines, whitespace…, The date of the last amendment in the header's ``Stand`` comment, else the…, render(), walk(), stand_date()

### Community 126 - "build_corpus_array.sh"
Cohesion: 0.33
Nodes (5): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_corpus_array.sh script

### Community 127 - "pack_corpus.sh"
Cohesion: 0.33
Nodes (5): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, pack_corpus.sh script, TOKENIZERS_PARALLELISM

### Community 128 - "_FakeSession"
Cohesion: 0.33
Nodes (3): _FakeResponse, _FakeSession, post()

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

### Community 137 - "test_ssd_tfla_under_bf16_autocast_match_fp32"
Cohesion: 0.40
Nodes (5): The fused flash kernel and the math reference agree in bf16 on causal attention., SSD and TFLA keep fp32 state under bf16 autocast; outputs stay within bf16…, _rel(), test_sdpa_flash_equals_math_bf16(), test_ssd_tfla_under_bf16_autocast_match_fp32()

### Community 138 - "test_diag_spike_reruns_the_arm_observed_in_its_own_directory"
Cohesion: 0.40
Nodes (5): _compose(), The fake interpreter recorded `distill=null` happily; real Hydra refused it…, Loss-spike rerun: S1-s42 with the screen settings, seed and schedule unchanged…, test_diag_spike_reruns_the_arm_observed_in_its_own_directory(), test_every_train_argument_list_composes_under_real_hydra()

### Community 139 - "test_param_bands"
Cohesion: 0.50
Nodes (3): Parameter counts of the legal model configs, The predicted counts and bands, and analysis/param_counts.md is current., test_param_bands()

### Community 140 - "admits"
Cohesion: 0.50
Nodes (4): admits(), Whether ``doc`` (already licence-checked) belongs in ``arm``., Every Multi Legal Pile document is research_only and never commercial-safe (CC…, test_multilegalpile_rows_are_research_only()

### Community 141 - "_text"
Cohesion: 0.50
Nodes (4): Element, Paragraph text: ``<p>``s become lines, ``<br/>`` breaks, whitespace normalised., _text(), walk()

### Community 142 - "_scan_cases"
Cohesion: 0.50
Nodes (4): _delta_cases(), Measured: chunk=8 rescues delta=0.1 but nothing rescues delta>=0.3 on the…, _scan_cases(), _xfail_if()

### Community 143 - "_executable"
Cohesion: 0.50
Nodes (4): _executable(), Path, The log is the job's own stdout, and GNU grep refuses to read its output file…, test_kd_probe_summary_reads_its_own_log()

## Knowledge Gaps
- **180 isolated node(s):** `lexhybrid-scrub`, `lexhybrid`, `build_corpus_array.sh script`, `HF_HOME`, `HF_HUB_OFFLINE` (+175 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1381 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HybridLanguageModel` connect `HybridLanguageModel` to `torch`, `test_data.py`, `performance_profile.py`, `json`, `test_training.py`, `test_models.py`, `test_numerics_fp64.py`, `train_pretrain.py`, `evaluate_lm.py`, `test_decoding.py`, `load_model_config`, `test_scripts.py`, `HybridConfig`, `_oldp_docs`, `lightning_module.py`, `SyntheticPackedDataset`, `statute_recall.py`, `sandbox`, `HybridLightningModule`, `test_config_invariants.py`, `test_slurm_wrappers.py`, `HybridBlock`, `test_reference_parity.py`, `screen_arms.py`, `PretrainLightningModule`, `check_operator_equivalence.py`, `Architecture`, `inspect_checkpoint.py`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Why does `Document` connect `Document` to `test_data.py`, `collectors/base.py`, `test_retrieval.py`, `test_verifier.py`, `admits`, `index.py`, `test_collect_offline.py`, `evaluate_lm.py`, `http_get`, `ris.py`, `parse_citation`, `dataclasses`, `scrub_ler.py`, `fedlex.py`, `packing.py`, `dedup.py`, `datasets.py`, `statute_recall.py`, `encode_document`, `chunking.py`, `scrub_document`, `build_corpus`, `measure_fertility.py`, `eurlex.py`, `FineWeb2DECollector`, `README.md`, `changelog.py`, `Architecture`, `OLDPCollector`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `HybridConfig` connect `HybridConfig` to `torch`, `HybridLanguageModel`, `test_data.py`, `test_layers.py`, `performance_profile.py`, `json`, `test_training.py`, `test_models.py`, `test_numerics_fp64.py`, `train_pretrain.py`, `test_decoding.py`, `load_model_config`, `test_scripts.py`, `_oldp_docs`, `lightning_module.py`, `SyntheticPackedDataset`, `statute_recall.py`, `test_config_invariants.py`, `HybridBlock`, `test_reference_parity.py`, `screen_arms.py`, `check_configs.py`, `Architecture`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 57 inferred relationships involving `Document` (e.g. with `Data pipeline` and `Document schema`) actually correct?**
  _`Document` has 57 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `HybridLanguageModel` (e.g. with `Model` and `__getattr__()`) actually correct?**
  _`HybridLanguageModel` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `HybridConfig` (e.g. with `Model` and `load_model_config()`) actually correct?**
  _`HybridConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 40 inferred relationships involving `Chunk` (e.g. with `_check_sentences()` and `_labels()`) actually correct?**
  _`Chunk` has 40 INFERRED edges - model-reasoned connections that need verification._