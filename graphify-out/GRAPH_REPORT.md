# Graph Report - legal_ai_hybrid_mamba_xlstm_model  (2026-11-06)

## Corpus Check
- 266 files · ~266,362 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 62 file(s) not represented in the graph (top: .jsonl 30, .xml 12, (none) 6)

## Summary
- 4205 nodes · 9756 edges · 209 communities (179 shown, 30 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 510 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0b19b6f0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Packer
- test_verifier.py
- test_data.py
- Document
- HybridConfig
- test_layers.py
- evalsets/base.py
- dedup.py
- performance_profile.py
- test_pipeline.py
- hybrid_lm.py
- parse_citation
- train_dpo.py
- http_post
- test_models.py
- index.py
- ssd_chunked_scan
- generate.py
- Resp
- train_pretrain.py
- test_numerics_fp64.py
- evaluate_val_ppl.py
- family_lm
- FedlexCollector
- PretrainLightningModule
- bootstrap_compare.py
- rel_max_err
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
- gerlerb.py
- scrub_document
- test_training.py
- Mamba3Block
- eval_retrieval.py
- mqar.py
- mLSTMBlock
- SpecialTokens
- generate_synthetic_qa.py
- sandbox
- teacher
- check_operator_equivalence.py
- check_env.py
- sft_format.py
- quotes.py
- filter_logits
- sequential_mlstm_reference
- filters.py
- test_slurm_wrappers.py
- verifier.py
- test_check_env.py
- test_config_invariants.py
- scrub_file
- segment_ids
- test_dpo.py
- code
- HybridLightningModule
- test_renderer.py
- load_model_config
- test_decoding.py
- build_dpo_pairs.py
- test_code_hygiene.py
- test_eval.py
- NLI verifier threshold on 100 hand-built statute pairs
- test_ci_workflow.py
- screen_metrics.py
- dpo_module.py
- gold_passages.py
- build_dense_index.sh
- lexhybrid
- parametrize
- eval_retrieval.sh
- test_rsync_rules.py
- _BlockParser
- train_sft.py
- MambaBlock
- DPOLightningModule
- measure_fertility.py
- test_repo_hygiene.py
- German-law pretraining corpus: tokens, deduplication, scrub
- tfla_forward_parallel
- screen_arms.py
- build_sft_data.py
- probe_ckpt_size.sh
- _CaptureLogger
- check_repo_hygiene.py
- screen_table.py
- Profile ladder, chunk size and distillation memory
- README.md
- Architecture screen: ablations of the hybrid backbone
- test_retrieval.py
- gpu_tests.sh
- multigpu_tests.sh
- train_pretrain_1gpu.sh
- Contributing
- check_configs.py
- strip_prefixes
- SignalCheckpointCallback
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
- statute_recall.py
- judge_answers.py
- Backbone results: the hybrid and a Transformer baseline at 8,192 tokens
- test_gpu.py
- ScheduleSampler
- build_corpus_array.sh
- pack_corpus.sh
- _s1_fingerprint
- write_run_metadata
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
- ByteTokenizer
- test_eval_probes_scores_named_runs_at_another_length
- test_reference_parity.py
- .__init__
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
- RateLimiter
- Retrieval on GerLeRB and GerDaLIR
- MDebertaNLI
- citation_metrics.py
- pytest
- _schedule_fit
- _pretrain_state
- train_sft_hf.py
- test_hf_sampling_through_its_cache_matches_the_uncached_reference
- StepStatsCallback
- changelog.py
- synth_qa.sh
- train_sft_4gpu.sh
- gerdalir.py
- test_sync_scripts.py
- shutil
- train_dpo_4gpu.sh
- FakeQwen
- write
- configure_optimizer
- _bger
- paragraphs
- setup_arm_env.sh
- setup_vllm.sh
- write_manifest
- test_tfla_wide_chunks_factorise_per_block_exactly
- pathlib
- build_dpo_pairs.sh
- build_sft_data.sh
- eval_gold_passages.sh
- eval_system_ablation.sh
- vllm_server.sh
- render
- read_samples
- test_fineweb_and_multilegalpile_collectors_read_their_rows
- lexhybrid-arm
- test_compile_forward_chunk128
- _text
- lexhybrid-vllm
- test_ablation_appends_each_record_and_resumes_after_preemption
- test_eval_retrieval_rerank_shards_and_a_killed_run_leave_only_the_missing_queries

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

## Communities (209 total, 30 thin omitted)

### Community 0 - "Packer"
Cohesion: 0.12
Nodes (14): Citation ids, Document schema, Fields, Packer, Path, Parquet shards ``shard-00000.parquet``... of at most ``rows_per_shard`` rows,…, Parquet shards ``shard-00000.parquet``... of at most ``rows_per_shard`` rows., Rows of ``row_len`` tokens from one source's documents, fed one tokenised… (+6 more)

### Community 1 - "test_verifier.py"
Cohesion: 0.05
Nodes (54): entity_hash(), Answer verification: pointers, quotes, entailment and scrub leaks., candidate_forms(), leaked_entities(), Scrub leaks into the index: no scrubbed name in a retrievable chunk. The scrub…, Each capitalised word of ``text`` and each run of 2..``MAX_RUN`` capitalised…, ``(chunk id, entity_hmac)`` of every logged entity a chunk still holds, once…, entailment_index() (+46 more)

### Community 2 - "test_data.py"
Cohesion: 0.05
Nodes (51): BGerCollector, EURLexCollector, FineWeb2DECollector, GIICollector, MultiLegalPileCollector, OLDPCollector, parse_toc(), Every ``<item>`` of ``rii-toc.xml`` as a dict (court, date, docket, link). (+43 more)

### Community 3 - "Document"
Cohesion: 0.04
Nodes (106): collections_abc, datetime, io, cli(), Collector, http_get(), licence_flags(), main() (+98 more)

### Community 4 - "HybridConfig"
Cohesion: 0.08
Nodes (28): HybridConfig, Validate and auto-compute derived parameters., The per-mixer arguments for layer ``layer_idx`` (a readable summary; not used…, Every architectural knob of the hybrid decoder. One dataclass serves every…, Build a config from a Hydra ``cfg.model`` node, keeping every field the…, Convert the config to a plain dictionary., Create a config from a dictionary produced by ``to_dict``., Write ``config.json`` into ``save_directory`` and return its path. (+20 more)

### Community 5 - "test_layers.py"
Cohesion: 0.05
Nodes (66): HybridBlock, Tensor, (batch, seq_len, dim) -> (batch, seq_len, dim). Argument order matches…, Per-layer decode state, or None if this mixer has no step (Mamba-1)., One token through mixer + MLP with residuals, mirroring ``forward``'s topology…, One decoder block: ``x + mixer(norm1(x))``, then the MLP (pre-norm, or…, _bc_bias_pair(), _continue_with_steps() (+58 more)

### Community 6 - "evalsets/base.py"
Cohesion: 0.16
Nodes (22): html, cli(), EvalSetInfo, QAItem, QASet, What every evaluation set shares: records, the info block, writers, the CLI., A question with its reference answers; ``context`` is the passage an extractive…, bgb_citation() (+14 more)

### Community 7 - "dedup.py"
Cohesion: 0.15
Nodes (20): deduplicate(), deduplicate_signatures(), deduplicate_stream(), DedupLog, keep_order(), minhash(), Deduplication of the training corpus: exact, then near-duplicate. 1. **Exact.**…, ``deduplicate_signatures`` over documents streamed once, their MinHashes… (+12 more)

### Community 8 - "performance_profile.py"
Cohesion: 0.05
Nodes (50): contextlib, csv, attention_backend(), available_configs(), build_model(), compile_for(), _delta_ms(), effective_chunk_sizes() (+42 more)

### Community 9 - "test_pipeline.py"
Cohesion: 0.07
Nodes (28): Two runs of one system and seed (SFT and DPO) get their own outputs, so the…, test_eval_gold_passages_wrapper_keeps_runs_apart_and_reuses_sets(), run(), fixture, End-to-end answer pipeline on CPU: index -> retrieve -> prompt -> constrained…, With best-of-n's samples scripted (every token one the FSM allows), the…, Half of the samples or more being the model's own refusal decides, though a…, A passage over the token budget can be the fused top but is never selected: A1… (+20 more)

### Community 10 - "hybrid_lm.py"
Cohesion: 0.06
Nodes (34): HeadFn, boundary_masked_labels(), CausalLMOutput, ``HybridLanguageModel``: embedding -> N x HybridBlock -> RMSNorm -> LM head.…, Language-model forward pass. Args: input_ids: (B, L) token ids; mutually…, Embedding and every block: the residual stream (B, L, D) BEFORE the final norm.…, Output of a causal language model forward pass., Next-token targets ``labels[:, 1:]`` with every cross-document prediction set… (+26 more)

### Community 11 - "parse_citation"
Cohesion: 0.06
Nodes (34): Limitations, court_abbreviation(), ``Oberlandesgericht Nürnberg`` -> ``OLG Nürnberg``; ``Hessischer…, docket(), ``7Ob143/26v`` -> ``7 Ob 143/26v`` (the spaced form courts print)., numbered_once(), parse_citation(), The parts of a citation id; raises ``ValueError`` if it matches neither… (+26 more)

### Community 12 - "train_dpo.py"
Cohesion: 0.12
Nodes (25): copy, check_run_settings(), Path, One output directory, one run. A job that finds ``last.ckpt`` there resumes it,…, build_callbacks(), build_dataloaders(), build_module(), build_policy() (+17 more)

### Community 13 - "http_post"
Cohesion: 0.14
Nodes (15): _cached(), http_post(), HttpCache, Path, Successful responses on disk, keyed by the request (method, URL, query, JSON…, POST a JSON body (a search API's query), with the same rate limit and retries…, ``fetch()`` through the process's ``HttpCache`` when one is set., _with_retries() (+7 more)

### Community 14 - "test_models.py"
Cohesion: 0.09
Nodes (38): _delta_at_init(), _delta_model(), parametrize, Model-level behaviour of HybridLanguageModel., The model's weight pass zeroed both gate biases; `post_model_init` restores the…, A `.to(bfloat16)` model without autocast once died in the SSD einsums., A document's last token (its EOS) does not predict the next document's first;…, Init then tie; `non_embedding = total - embedding - (lm_head if untied)`, so… (+30 more)

### Community 15 - "index.py"
Cohesion: 0.08
Nodes (40): document_digest(), DocumentDigest, file_sha256(), indexed_documents(), IndexVersion, latest(), Path, Index versions: a registry of the retrieval indexes that have been built.… (+32 more)

### Community 16 - "ssd_chunked_scan"
Cohesion: 0.08
Nodes (36): Mamba-2/3 state space dual (SSD) kernels., device, Tensor, Lower-triangular segment sums: ``out[..., t, s] = sum_{i=s+1}^{t} v_i`` for ``s…, Monotone segment index per position (see ``lexhybrid.kernels.segments``)., Chunk-parallel SSD scan. h_t = exp(dt_t A) h_{t-1} + sum_terms coeff_t (B_t…, _segment_ids(), segsum() (+28 more)

### Community 17 - "generate.py"
Cohesion: 0.12
Nodes (39): Advance, apply_repetition_penalty(), _beam_core(), beam_search(), beam_search_cached(), beam_search_uncached(), best_of_n(), _check_one_sample() (+31 more)

### Community 18 - "Resp"
Cohesion: 0.08
Nodes (25): FakeWeb, ``get``/``post`` answering from ``routes`` (regex on the URL -> function(url,…, One document's connection dropped five times and the whole RII crawl (3.8 h,…, Resp, _ris_page(), test_bger_collector_pages_by_search_after_and_skips_blocked_decisions(), html(), search() (+17 more)

### Community 19 - "train_pretrain.py"
Cohesion: 0.10
Nodes (31): Training callbacks: checkpoint-on-signal (ported from the reference) and step…, Loss-spike diagnostics for one training run. Two screening runs collapsed to…, pytorch_lightning, pytorch_lightning_callbacks, pytorch_lightning_strategies, build_dataloaders(), build_model(), build_module() (+23 more)

### Community 20 - "test_numerics_fp64.py"
Cohesion: 0.12
Nodes (24): cumulative_angles(), Theta_t = sum_{i<=t} dt_i * theta_i, accumulated in float64 and wrapped to [0,…, test_cumulative_angles_restart_per_document(), _clamp_fire_fraction(), _delta_cases(), _measure(), Operator-level numerical correctness against float64 sequential oracles. Ported…, The shipped configuration runs exact TFLA on its factorised path at init. With… (+16 more)

### Community 21 - "evaluate_val_ppl.py"
Cohesion: 0.21
Nodes (11): document_offsets(), late_context_sums(), main(), no_grad, Path, Tensor, Validation PPL of a checkpoint, overall and late in context (the late-25% PPL).…, (B, L): how many tokens of the same document precede each position in its row. (+3 more)

### Community 22 - "family_lm"
Cohesion: 0.11
Nodes (20): family_lm(), _prefill_then_greedy(), parametrize, The one-pass prefill leaves every cache exactly where token-by-token decoding…, In bf16 the two paths round differently, so the claim is argmax identity…, A packed prompt: decoding continues each row's last document as if it were…, Cached greedy equals the uncached reference token for token, batch 3 with rows…, With EOS, length normalisation and a repetition penalty, cached beam search… (+12 more)

### Community 23 - "FedlexCollector"
Cohesion: 0.13
Nodes (15): act_code(), The citation code of an act: its short name, else ``VO (EU) YYYY/N`` or ``RL…, acts_query(), FedlexCollector, latest_versions(), SPARQL: the SR number of every act in force., SPARQL: the German XML consolidations of the act in force with SR number ``sr``…, ``{sr: (date, file)}`` from SPARQL result bindings: per SR number the latest… (+7 more)

### Community 24 - "PretrainLightningModule"
Cohesion: 0.06
Nodes (35): Dataset, Tensor, ``num_rows`` rows of ``row_len`` random tokens, each packed from several…, SyntheticPackedDataset, PretrainLightningModule, Tensor, Clip and log the gradient norm; a non-finite norm clears the gradients (step…, Args: those of ``HybridLightningModule`` plus teacher: optional frozen Hugging… (+27 more)

### Community 25 - "bootstrap_compare.py"
Cohesion: 0.05
Nodes (59): collections, corpus_bleu(), _lcs_length(), _ngram_counts(), Counter, Sentence ROUGE-L and corpus BLEU, stdlib only. Ported verbatim from the…, Sentence-level ROUGE-L F-measure (Lin, 2004)., Corpus-level BLEU-N with brevity penalty, one reference per hypothesis. (+51 more)

### Community 26 - "rel_max_err"
Cohesion: 0.15
Nodes (22): _audit_inputs(), parametrize, Tensor, ``use_fast_path=False`` shares the one implementation (the reference once had a…, The legacy error is governed entirely by whether f_cum underflows the 1e-6…, The overflow guard: re-centring the whole chunk produced NaN here; re-centring…, The slow path is a configured choice. A too-wide chunk is factorised per row…, float64 ground truth for the Mamba-1 (S6) recurrence, written independently of… (+14 more)

### Community 27 - "AttentionBlock"
Cohesion: 0.05
Nodes (49): Limitations, disable, Document segments inside packed rows, shared by the SSD and TFLA kernels. A…, (batch, seqlen) bool, True where a new document begins; position 0 is False., segment_starts(), apply_rope(), AttentionBlock, build_doc_block_mask() (+41 more)

### Community 28 - "torch"
Cohesion: 0.07
Nodes (41): einops, Gold-answer likelihood: a continuous probe score that an exact-match floor…, Mamba-1 selective scan (legacy ablation only)., Tensor, Mamba-1 selective scan (legacy incumbent; used only by the…, Exact selective scan -- the recurrence the Mamba-1 block is specified to…, Exact chunked selective scan -- no division, no clamp. Flips which axis is…, Public selective-scan entry point, computed in fp32 and cast back.… (+33 more)

### Community 29 - "test_scripts.py"
Cohesion: 0.05
Nodes (30): fixture, The ported command-line scripts run end to end on CPU with tiny models., A tiny model yaml and a Lightning-style checkpoint for it., Every arm builds through HybridConfig.from_hydra at reduced width, runs a…, The learning-rate arms are S0 and S1 with only `model.learning_rate` changed,…, Full size on `meta` with the bands: every arm passes; S5 carries +2.36% as a…, Guard: if an arm's lever never reaches the model, its ARCH line lacks the token., The method: whole documents until the word budget, documents over the cap… (+22 more)

### Community 30 - "Chunk"
Cohesion: 0.06
Nodes (48): German sentences of a passage: split after ``.``, ``!`` or ``?`` before an…, split_sentences(), Path, read_chunks(), _anchor(), Chunk, chunk_case_law(), chunk_document() (+40 more)

### Community 31 - "test_synth_qa.py"
Cohesion: 0.14
Nodes (20): http_server, socket, stat, generator(), _executable(), _free_port(), index(), jobs() (+12 more)

### Community 32 - "CausalLMAdapter"
Cohesion: 0.16
Nodes (7): CausalLMAdapter, Module, Tensor, A Hugging Face causal LM behind the hybrid's interfaces: the…, Args: those of ``HybridLightningModule`` (``learning_rate`` is the plain value…, ``(mean CE over the supervised targets, their count)``; position t predicts…, SFTLightningModule

### Community 33 - "HybridLanguageModel"
Cohesion: 0.06
Nodes (26): __getattr__(), lexhybrid -- a DACH legal, citation-faithful, retrieval-gated hybrid Mamba-3 /…, HybridLanguageModel, no_grad, Tensor, One-line summary of what was actually built; see…, Final norm and LM head: residual stream -> logits., Uncached sampling (``lexhybrid.decoding.sample``): the full forward per new… (+18 more)

### Community 34 - "SpikeDiagnosticsCallback"
Cohesion: 0.11
Nodes (14): Callback, Tensor, Trainer, SpikeDiagnosticsCallback, hook(), hook(), hook(), hook() (+6 more)

### Community 35 - "FetchError"
Cohesion: 0.12
Nodes (17): FetchError, RuntimeError, ``fetch(url)``'s response, or None when it raised ``FetchError``., A request failed after every retry., test_skip_failed_documents_stops_on_an_outage(), fail(), --http-cache: a restarted collection is served what it already fetched -- GET…, An act CELLAR does not serve as German XHTML (HTTP 404, e.g. 31993L0013) is… (+9 more)

### Community 36 - "fedlex.py"
Cohesion: 0.11
Nodes (33): act_code(), act_meta(), article_sections(), _inline(), _is_text(), _local(), _note_dates(), _numbered_ps() (+25 more)

### Community 37 - "datasets.py"
Cohesion: 0.05
Nodes (55): build_source_shards(), build_source_shards_stream(), effective_weights(), mixture_from_config(), PackedMixture, Dataset, Path, Tensor (+47 more)

### Community 38 - "gerlerb.py"
Cohesion: 0.19
Nodes (19): gzip, Passage, Query, RetrievalSet, fetch(), gii_id(), iter_corpus(), parse_qrels() (+11 more)

### Community 39 - "scrub_document"
Cohesion: 0.15
Nodes (20): Entity, _pattern(), placeholders(), Surface form -> placeholder, numbered per tag in order of first appearance., Replace every occurrence of the propagating forms (longest first, whole words…, The tagged spans replaced exactly (right to left), then the propagating forms., The document with its people replaced, and its scrub-log lines. A section found…, The model tags whitespace tokens, so ``Haag,`` or ``(Müller`` carries… (+12 more)

### Community 40 - "test_training.py"
Cohesion: 0.04
Nodes (63): LambdaLR, no_grad, Tensor, ``(position_ids, attention_mask)`` for a Hugging Face decoder on packed rows.…, The teacher's final hidden states (after its final norm), (B, L, D_teacher),…, teacher_forward_packed(), teacher_inputs_packed(), apply_beta2_schedule() (+55 more)

### Community 41 - "Mamba3Block"
Cohesion: 0.08
Nodes (24): heavy_tail_activation(), Mamba3Block, Tensor, (Re-)apply the SSM initializations after the parent model's global weight pass.…, Per-head state transition, strictly negative. Shape (nheads,) or (batch, len,…, (batch, seqlen, dim) -> (batch, seqlen, dim). Argument order is load-bearing:…, SSD scan, plus the trapezoidal term and the complex-state rotation when…, The last ``conv_size - 1`` pre-convolution columns, (batch, channels, conv_size… (+16 more)

### Community 42 - "eval_retrieval.py"
Cohesion: 0.14
Nodes (22): CrossEncoderScorer, ``bge-reranker-v2-m3`` as a ``Scorer``: one logit per (question, passage) pair,…, Path, The file's non-empty whole lines; a last line left without its newline is cut…, read_complete_lines(), dense_index(), load_set(), main() (+14 more)

### Community 43 - "mqar.py"
Cohesion: 0.18
Nodes (12): generate(), MQARConfig, MQARItem, no_grad, Multi-query associative recall, MQAR, with the upstream reference's…, Accuracy over all queries; the logits are computed at the query positions only., score(), run_probes() (+4 more)

### Community 44 - "mLSTMBlock"
Cohesion: 0.11
Nodes (16): mLSTMBlock, Tensor, (Re-)apply the gate-bias inits after the parent model's global weight pass.…, Soft-capped gates: exp input gate (computed in fp32), sigmoid forget and output…, (batch, seq_len, dim) -> (batch, seq_len, dim). Args: x: input. cache: an EMPTY…, mLSTM decode state: the matrix cell ``C`` and the normalizer ``n``, both zeros,…, Advance one token. ``(batch, dim)`` or ``(batch, 1, dim)`` in, ``(batch, dim)``…, Soft cap via tanh: maps x to ``tanh(x / cap) * cap``, range (-cap, cap). (+8 more)

### Community 45 - "SpecialTokens"
Cohesion: 0.09
Nodes (21): generated_tokens(), The answer part of a decoded row: after the prompt, up to and including the…, K for ``<|cK|>``, else None., J for ``<|sJ|>``, else None., Ids of the 87 SFT-only special tokens (never used in distillation)., The token strings, in registry order (the order ``from_offset`` assigns ids)., Consecutive ids from ``first_id`` -- tests, and any vocabulary that appends…, Ids from a tokenizer that has the tokens added (they are added to the Qwen3… (+13 more)

### Community 46 - "generate_synthetic_qa.py"
Cohesion: 0.07
Nodes (48): concurrent_futures, Exception, estimate_tokens(), Ranked, Cross-encoder reranking of fused candidates into the prompt's passages. The…, Best first, while there are fewer than ``max_chunks`` and the passages fit…, select(), add_passages() (+40 more)

### Community 47 - "sandbox"
Cohesion: 0.07
Nodes (30): _executable(), fixture, Path, The default sources' scrubbed parts go to the bm25 build with the pack's dedup…, A repository copy with the wrappers, screen_arms.py and a fake interpreter;…, The log is the job's own stdout, and GNU grep refuses to read its output file…, One sequence length per process, a fresh Inductor cache per point, the…, Task i scrubs part i % PARTS of source i / PARTS; every task uses the same HMAC… (+22 more)

### Community 48 - "teacher"
Cohesion: 0.15
Nodes (18): nli(), Through the CLI: 40 rows that round-trip through sft_format.parse; answerable…, Quotas met: no teacher call, same rows. A log cut mid-line by a kill loses only…, Rounds smaller than the limit add up: 32 errors in a row stop the run, whatever…, Attempts run out with too few answerable rows: a seeded sample of refusals…, Rows are capped kind by kind at the new count's quotas, with no teacher call. A…, Pure, so threads may call it in any order. A question names its source's…, _rows() (+10 more)

### Community 49 - "check_operator_equivalence.py"
Cohesion: 0.23
Nodes (11): check_model(), check_operator(), main(), _operands(), Model level: do the logits move? This is what a decoded token actually sees., The equivalence gate every operator or speed change has to pass. Ported from…, max |a-b| / max(|b|, eps), computed in fp64., Operator level: does changing chunk_size change the function the scan computes? (+3 more)

### Community 50 - "check_env.py"
Cohesion: 0.07
Nodes (41): importlib, importlib_metadata, importlib_util, packaging_markers, packaging_requirements, packaging_utils, packaging_version, pkgutil (+33 more)

### Community 51 - "sft_format.py"
Cohesion: 0.08
Nodes (53): Cite, encode(), Example, parse(), Passage, prompt_ids(), Quote, The retrieval-grounded SFT format: one prompt, pointer answers. The prompt is… (+45 more)

### Community 52 - "quotes.py"
Cohesion: 0.09
Nodes (35): check_pointer(), check_pointers(), PointerIssue, Segment, Pointer resolution: every pointer of an answer names a passage and sentences…, The failures of one pointer (segment ``index``) against the retrieved passages., Every failing pointer of the answer, in segment order; ``[]`` when all resolve., _balanced_body() (+27 more)

### Community 53 - "filter_logits"
Cohesion: 0.33
Nodes (5): filter_logits(), Temperature, then top-k, then nucleus (top-p) filtering; filtered entries…, choose(), Shared sampling filter (``lexhybrid.decoding.filter_logits``)., test_filter_logits()

### Community 54 - "sequential_mlstm_reference"
Cohesion: 0.12
Nodes (18): dtype, Tensor, float64 sequential oracle for the recurrence ``tfla_forward_parallel``…, Plain-loop mLSTM recurrence; inputs (B, H, L, D), output (B, H, L, D) in…, sequential_mlstm_reference(), _packed_tfla_inputs(), Two rows with different boundary sets; doc_ids increase at each listed position., Resets inside the kernel equal running every document alone, and equal the fp64… (+10 more)

### Community 55 - "filters.py"
Cohesion: 0.21
Nodes (14): date, _allowed(), _bound(), _day(), filter_chunks(), _parse(), Retrieval filters: jurisdiction, validity date and document type.…, A ``date`` from a date, a datetime (its day) or an ISO date string. (+6 more)

### Community 56 - "test_slurm_wrappers.py"
Cohesion: 0.15
Nodes (17): SLURM wrappers under `scripts/slurm/`. Static checks on every `*.sh` there: the…, A requeued run finds its last.ckpt and passes it on., The array index picks the arm; screen_arms.py resolves it ON THE NODE; the…, The wrappers the pipeline names are all here., Only the named shapes are rerun: the 1.7B shapes are rerun alone; the 8B shapes…, Stage B: INIT_FROM (stage A's last.ckpt) while the stage has none of its own;…, test_every_planned_wrapper_exists(), test_four_gpu_wrapper_runs_ddp_through_the_same_body() (+9 more)

### Community 57 - "verifier.py"
Cohesion: 0.16
Nodes (18): Sentences ``J1..Jn`` of the chunk joined by one space: what a quote of them…, source_text(), check_claim(), ClaimCheck, _hypothesis(), _party_named(), _precision(), _premise() (+10 more)

### Community 58 - "test_check_env.py"
Cohesion: 0.10
Nodes (15): env(), skipif, slow, scripts/check_env.py: the environment-equals-lock check., The CI condition, reproduced: the real uv, FORCE_COLOR=1, the real lock., Run from the main environment, `--scrub` must fail: no flair, transformers 5,…, The live venv equals the lock (skipped in fast mode; CI and --full run it)., FORCE_COLOR=1 made uv colour its "# via" lines, and they failed to parse. (+7 more)

### Community 59 - "test_config_invariants.py"
Cohesion: 0.11
Nodes (16): counts(), fixture, parametrize, slow, Configuration invariants: the living spec of what every yaml guarantees., One factor at a time: each screen yaml equals the base in every field but its…, S5 (legacy recipe): the <= 1% target did not hold (+2.36%); it is reported as a…, Structural parity with the reference: same modules, same shapes, to the… (+8 more)

### Community 60 - "scrub_file"
Cohesion: 0.13
Nodes (21): iter_part(), main(), part_name(), Path, Each document with its entities, in order, from ONE worker process: the…, Entities per document id from the NER worker in the scrub environment, and its…, The documents on lines ``part, part + parts, part + 2 * parts, ...`` of…, ``<source>.jsonl``, or ``<source>.part-<k>-of-<n>.jsonl`` for one part of a… (+13 more)

### Community 61 - "segment_ids"
Cohesion: 0.15
Nodes (10): device, Monotone segment index per position, (batch, seqlen) long; all zeros without…, segment_ids(), MTPHead, MTPModule, Module, Tensor, One prediction depth: two norms, a 2D -> D projection, and one ``HybridBlock``. (+2 more)

### Community 62 - "test_dpo.py"
Cohesion: 0.05
Nodes (63): load_sft_model(), The SFT model of ``checkpoint``: the hybrid of ``config``, or a…, _argv(), _dpo_overrides(), _fake_trainer(), hf_run(), _pair_rows(), _pairs() (+55 more)

### Community 63 - "code"
Cohesion: 0.09
Nodes (25): code(), directives(), A `--gpus` directive is not overridden by an environment variable: a multi-GPU…, The two shell lookups in fetch_hf.sh must find the constants the loaders use., The command as documented, the same check on the legal base, the CUDA smoke., The 4-GPU header, and the body it delegates to passes last.ckpt as…, ARM is resolved inside the job by `screen_arms.py env`, EXPERIMENT from the…, `#SBATCH --key=value` lines before the first command, trailing comments dropped. (+17 more)

### Community 64 - "HybridLightningModule"
Cohesion: 0.10
Nodes (17): HybridLightningModule, Optimizer, Tensor, Token-weighted sums for the epoch's ``val/perplexity``. Logged per batch with…, Clip gradients and log the pre-clip norm (manual, so foreach AdamW and AMP…, A WSD scheduler with an absolute warmup; records the decay window for the beta2…, Apply the beta2 schedule when WSD and ``beta2_schedule`` are both active., Training loop around a ``HybridLanguageModel``. Args: model: the model to… (+9 more)

### Community 65 - "test_renderer.py"
Cohesion: 0.08
Nodes (61): _check_sentences(), _labels(), pointer_text(), Segment, ValueError, The renderer: the model's pointer answer as the text a reader sees. The answer…, The rendered answer of parsed segments against the retrieved passages (prompt…, The rendered answer of the model's raw answer text (``<|endoftext|>`` allowed… (+53 more)

### Community 66 - "load_model_config"
Cohesion: 0.11
Nodes (25): Configuration dataclass for the hybrid decoder. Ported from the reference…, Model configuration and the yaml loader every script uses., available_model_configs(), load_model_config(), load_model_yaml(), Any, Path, Load a model yaml into a ``HybridConfig`` -- the one path every script uses. A… (+17 more)

### Community 67 - "test_decoding.py"
Cohesion: 0.10
Nodes (26): Bigram, cached_lm(), log_table(), Decoding. Cached decode must be an equivalence with full recomputation. The…, All-or-nothing: Mamba-1 has no step(), so the legacy stack cannot decode with a…, Beams reorder the KV cache row-wise, and cached beam search over a stack with…, ``logits[b, t] = table[ids[b, t]]``: the next-token scores given the last token., Rows of probabilities -> log-probabilities (so a beam's score is a sum of known… (+18 more)

### Community 68 - "build_dpo_pairs.py"
Cohesion: 0.15
Nodes (20): decode_answer(), Answer tokens as the pointer markup: special tokens by name, so no tokenizer…, Pointers, answer_pointers(), balance(), build_pairs(), draw_samples(), is_good() (+12 more)

### Community 69 - "test_code_hygiene.py"
Cohesion: 0.18
Nodes (17): _code_files(), git_calls_in_python(), git_calls_in_shell(), _is_sys_path_edit(), _names_git(), AST, parametrize, Path (+9 more)

### Community 70 - "test_eval.py"
Cohesion: 0.09
Nodes (34): outcome(), What one decoded set shows: the gold position (1-based), whether the first…, One summary per (system, seed, arm), in order of first appearance (see the…, score(), _share(), Path, read_records(), evaluate() (+26 more)

### Community 71 - "NLI verifier threshold on 100 hand-built statute pairs"
Cohesion: 0.22
Nodes (8): NLI verifier threshold on 100 hand-built statute pairs, Not licensed, Open limitations, Prediction and outcome, Reproduction, Results, Summary, What this licenses

### Community 72 - "test_ci_workflow.py"
Cohesion: 0.16
Nodes (12): fixture, The CI workflow and the local validation script stay in step. PyYAML parses the…, Fast excludes slow/cuda/multigpu/network; --full and --ci exclude…, steps_text(), test_every_installing_job_uses_the_lock_and_never_pip(), test_hygiene_runs_on_a_pristine_checkout(), test_lint_runs_every_static_tool(), test_package_job_imports_the_wheel_outside_the_checkout() (+4 more)

### Community 73 - "screen_metrics.py"
Cohesion: 0.39
Nodes (8): _fmt(), latest_version(), main(), Path, Validation and training curves of the screen runs, read from their TensorBoard…, ``<run>/logs/<exp>/version_<n>`` with the highest n that holds an events file., read_scalars(), report()

### Community 74 - "dpo_module.py"
Cohesion: 0.13
Nodes (12): Seeded random-token rows with packed-document ids, for smoke tests and…, PairSchedule, Dataset, Preference optimisation of an SFT model: DPO and KTO against a frozen reference…, ``num_samples`` pairs on a seeded schedule (one permutation per pass), or every…, Supervised fine-tuning: cross-entropy on the answer tokens only, for the hybrid…, ``lm`` frozen but for rank-``rank`` LoRA adapters on ``LORA_TARGETS`` and the…, wrap_lora() (+4 more)

### Community 75 - "gold_passages.py"
Cohesion: 0.17
Nodes (15): Pointer-constrained decoding. The legal model never generates statute text: it…, gold_relation(), ``(doc_id, copies) -> bool``: a chunk is a gold passage or a copy of one when…, build_sets(), gold_chunk_ids(), The gold-passage sanity check: with retrieval assumed solved, does the model…, Each item's gold-relevant chunk ids: its ``gold_chunk_ids`` the index holds,…, ``(sets, counts)``: the prompt sets of every item with a gold passage (see the… (+7 more)

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

### Community 81 - "_BlockParser"
Cohesion: 0.19
Nodes (7): html_parser, _BlockParser, html_blocks(), HTMLParser, HTML to text blocks for collectors that receive HTML fragments (OLDP). Stdlib…, The non-empty text blocks of ``html``, in document order., test_html_blocks()

### Community 82 - "train_sft.py"
Cohesion: 0.11
Nodes (22): best_val_checkpoint(), Dataset, The ``save_top_k`` lowest-``val/loss`` checkpoints, ``best-step<N>.ckpt``,…, One split (``train.parquet`` / ``dev.parquet``) as ``{input_ids, labels}``…, SFTSplit, ModelCheckpoint, omegaconf, pytorch_lightning_loggers (+14 more)

### Community 83 - "MambaBlock"
Cohesion: 0.19
Nodes (9): MambaBlock, Tensor, Re-apply the Mamba dt init after the model's global weight pass (no-op unless…, (batch, seq_len, dim) -> (batch, seq_len, dim); ``doc_ids`` resets state per…, Non-fast-path scan: the same implementation as ``selective_scan`` with chunk…, Per-(row, segment) wrapper that resets the SSM state at document boundaries., Mamba-1 mixer: input projection, depthwise causal conv, selective SSM, gated…, The declared capability and the real signature must not drift apart. (+1 more)

### Community 84 - "DPOLightningModule"
Cohesion: 0.09
Nodes (24): Tests, answer_logps(), collate_pairs(), dpo_loss(), DPOLightningModule, kto_loss(), _pad(), Module (+16 more)

### Community 85 - "measure_fertility.py"
Cohesion: 0.05
Nodes (57): functools, main(), pack(), EOS-delimited packing into fixed-length rows and parquet shards. Every document…, Rows of ``row_len`` tokens from the documents of one source, in order., read_rows(), Probes scored on any checkpoint: MQAR, statute recall, multi-hop., cache_dir() (+49 more)

### Community 86 - "test_repo_hygiene.py"
Cohesion: 0.23
Nodes (12): make_tree(), parametrize, Path, scripts/check_repo_hygiene.py on synthetic trees., test_a_missing_critical_file_is_reported(), test_an_ignored_analysis_document_is_reported(), test_case_collisions_are_caught(), test_clean_tree_passes() (+4 more)

### Community 87 - "German-law pretraining corpus: tokens, deduplication, scrub"
Cohesion: 0.15
Nodes (12): Budget rule, Deduplication (drop within a source, keep and link across), Discussion, Efficiency, Estimate vs exact, German-law pretraining corpus: tokens, deduplication, scrub, Limitations, Reproduction (+4 more)

### Community 88 - "tfla_forward_parallel"
Cohesion: 0.13
Nodes (23): forward_fill_from_starts(), pad_segments(), Tensor, Extend ``seg`` by ``pad`` positions that join the LAST real segment. Padding…, For each position, ``values`` at the most recent start at or before it along…, TFLA (tiled linear attention) kernels for the mLSTM., apply_tfla(), _blockwise_scores() (+15 more)

### Community 89 - "screen_arms.py"
Cohesion: 0.19
Nodes (15): NamedTuple, Arm, build_config(), check_tokens(), env_lines(), _fmt(), hydra_overrides(), _inference_non_embedding() (+7 more)

### Community 90 - "build_sft_data.py"
Cohesion: 0.12
Nodes (27): question_key(), A hash of the question with its form normalised (NFC, case, whitespace, final…, build(), evalset_guard(), is_eval(), Item, main(), Path (+19 more)

### Community 91 - "probe_ckpt_size.sh"
Cohesion: 0.15
Nodes (12): BATCH_SIZE, CKPT_EVERY, DATASET_CONFIG, DISTILL_CFG, EXPERIMENT, EXTRA_OVERRIDES, MAX_STEPS, MODEL_CONFIG (+4 more)

### Community 92 - "_CaptureLogger"
Cohesion: 0.12
Nodes (7): _CaptureLogger, _RankRows, The answer loss never uses the MTP head; DDP would wait for gradients it never…, Keeps every ``log_metrics`` call in memory: ``rows = [(step, {tag: value})]``., Writes the row indices each DDP rank trains on to <dir>/rank<r>.txt., test_sft_freezes_an_mtp_head_outside_its_loss(), __init__()

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
Cohesion: 0.20
Nodes (7): Parameter counts of the legal model configs, Corpus licence register, Evaluation sets (never trained on), Models the pipeline runs, Training sources (collectors), The predicted counts and bands, and analysis/param_counts.md is current., test_param_bands()

### Community 97 - "Architecture screen: ablations of the hybrid backbone"
Cohesion: 0.17
Nodes (11): Architecture screen: ablations of the hybrid backbone, Discussion, Efficiency, Hypotheses and selection rule, Paired comparisons, Per-seed calls (where the seeds disagree), Reproduction, Results (+3 more)

### Community 98 - "test_retrieval.py"
Cohesion: 0.04
Nodes (75): follows(), parse_bger(), Whether consideration ``cur`` can follow ``prev``: the first child (2 -> 2.1),…, One decision from its HTML and its search-API record (``id``, ``date``,…, fused_retriever(), retrieve(), Encoder, The index's ranking: BM25, fused with ``dense/`` when it exists and ``encode``… (+67 more)

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

### Community 103 - "check_configs.py"
Cohesion: 0.27
Nodes (9): hydra, hydra_core_global_hydra, check_model(), compose_model(), main(), model_names(), validate.sh gate 3: every model yaml composes through Hydra and pins what it…, Problems with ``configs/model/<name>.yaml``; empty when it passes. (+1 more)

### Community 104 - "strip_prefixes"
Cohesion: 0.17
Nodes (15): Normalise ``model._orig_mod.`` / ``_orig_mod.model.`` / ``_orig_mod.`` /…, strip_prefixes(), load_model(), The model from its yaml with the checkpoint's weights (the guarded loader), in…, _f(), inspect(), main(), optimizer_names() (+7 more)

### Community 105 - "SignalCheckpointCallback"
Cohesion: 0.13
Nodes (9): FinalCheckpointCallback, Callback, End training after optimizer step ``stop_at_step`` while the LR schedule keeps…, Write ``<checkpoint_dir>/last.ckpt`` when training ends normally.…, Write ``interrupt.ckpt`` when the process is signalled; a no-op after the first…, SignalCheckpointCallback, StopAtStepCallback, build_callbacks() (+1 more)

### Community 106 - "PointerFSM"
Cohesion: 0.11
Nodes (22): Decoding, PointerFSM, process(), Tensor, The pointer grammar for one retrieved context. Args: specials: the special-…, The state after ``token``; raises ``ValueError`` if the grammar forbids it., Boolean (vocab,) mask of the tokens the grammar allows next., Set every forbidden logit to ``-inf``, row by row; ``generated`` (B, T) are the… (+14 more)

### Community 107 - "conftest.py"
Cohesion: 0.13
Nodes (18): reference, load_script(), fixture, pytest_collection_modifyitems(), Shared test configuration. Markers are declared in pyproject.toml under…, Every test starts from the same RNG state, so failures reproduce., Import ``scripts/<name>.py`` as a module without touching sys.path., Fixture form of ``load_script``: ``script("evaluate_lm").main([...])``. (+10 more)

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
Cohesion: 0.13
Nodes (32): answer_text(), correctness_prompt(), faithfulness_prompt(), judge_prompts(), _listed(), _numbered(), _parse(), precision_prompt() (+24 more)

### Community 112 - "_doc"
Cohesion: 0.08
Nodes (32): admits(), build_corpus(), Corpus assembly gate: which documents may enter which training arm. Two arms:…, Whether ``doc`` (already licence-checked) belongs in ``arm``., Yield the documents of ``arm``; raise ``LicenceError`` on the first licence…, check_document(), is_commercial_safe(), Licence (+24 more)

### Community 113 - "Judge calibration and validation on gold labels"
Cohesion: 0.13
Nodes (14): 1. Validation half, prompt version 2 (accuracy / κ / precision / recall; unparsed in brackets), 2. Accuracy by variant, validation half, 3. Calibration half, prompt version 2, with the check judges (accuracy / κ), Bar and outcome, Judge calibration and validation on gold labels, Not licensed, Open limitations, Reproduction (+6 more)

### Community 114 - "AnswerRecord"
Cohesion: 0.08
Nodes (33): AnswerRecord, chunk_dict(), write_records(), copy_top(), main(), prompt_tokens(), Path, Append each arm's missing records to ``<out>/<arm>.jsonl``: A1-A4 per question,… (+25 more)

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
Nodes (34): first_json_object(), OpenAICompatible, The first ``{...}`` in ``text`` that parses as a JSON object (code fences and…, ``POST {base_url}/chat/completions``: one user message, temperature 0; returns…, requests, fake_post(), post(), FakeResponse (+26 more)

### Community 119 - "Tokenizer fertility on DACH legal text"
Cohesion: 0.25
Nodes (7): Discussion, Hypotheses and method, Limitations, Reproduction, Results, Summary, Tokenizer fertility on DACH legal text

### Community 120 - "_Paras"
Cohesion: 0.25
Nodes (3): _Paras, HTMLParser, The ``<div class="para">`` blocks as ``(bold, text)``: ``bold`` is the text of…

### Community 121 - "statute_recall.py"
Cohesion: 0.09
Nodes (34): gold_nll(), no_grad, Mean per-token NLL of ``target_ids`` following ``prompt_ids`` (natural log).…, build_items(), MultihopItem, no_grad, Two-hop cross-reference probe: the answer needs both passages. An item takes a…, Exact match of the first line of a greedy continuation against the title, and… (+26 more)

### Community 122 - "judge_answers.py"
Cohesion: 0.21
Nodes (13): add_judge_args(), judge_from_args(), judge_record(), ArgumentParser, ``(scores, details)``: a score per metric (``None`` when not applicable or not…, Namespace, check_validated(), main() (+5 more)

### Community 123 - "Backbone results: the hybrid and a Transformer baseline at 8,192 tokens"
Cohesion: 0.18
Nodes (10): Backbone results: the hybrid and a Transformer baseline at 8,192 tokens, Discussion, Handoff, Limitations, Predictions, Probes on the stage-B checkpoints, by context length, Reproduction, Results (+2 more)

### Community 124 - "test_gpu.py"
Cohesion: 0.15
Nodes (19): _packed_doc_ids(), parametrize, Tensor, CUDA checks. Marked ``cuda``: they collect everywhere and skip, with the…, On the GPU, a 128-chunk whose log-decay overflows one factorisation (forget…, CUDA variant of the compiled legal base check: the compiled legal base equals…, Under an outer torch.compile, a block mask traced inside the graph was wrong…, CUDA variant of tests/test_layers.py::test_flex_block_mask_equals_dense: the… (+11 more)

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

### Community 129 - "write_run_metadata"
Cohesion: 0.18
Nodes (14): Any, DictConfig, Path, sha256 over the sorted (relative path, file sha256) pairs of the source tree., Contents of ``.sync_stamp`` (date + tree hash of the last sync to the cluster),…, The checked-out commit id, read from ``.git/HEAD`` and its ref file; "" if…, Persist ``<output_dir>/run_metadata.json`` and return its path., read_git_head() (+6 more)

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

### Community 140 - "ByteTokenizer"
Cohesion: 0.14
Nodes (19): _jsonl(), _qa_file(), _qa_row(), One evaluation-only set's manifest: the raw questions' hashes, as the…, QA rows with the writer's ``info.json`` beside them, free for training unless…, build_sft_data writes train and dev splits in which every target ends in…, QA items are all answerable: when they hold the share below 0.2, a seeded…, A QA file without info.json, or one that does not free its items for training;… (+11 more)

### Community 142 - "test_reference_parity.py"
Cohesion: 0.20
Nodes (14): inspect, _block_defaults(), block_params(), _config_defaults(), load(), model_params(), _per_document(), parametrize (+6 more)

### Community 143 - ".__init__"
Cohesion: 0.17
Nodes (12): LayerType, create_hybrid_blocks(), _mamba3_params(), _mlstm_params(), _prefixed_kwargs(), ``num_layers`` blocks cycling through ``layer_pattern``; block 0 is marked…, The ``mamba3_*`` options this dispatcher forwards, read off the block's…, Every ``mLSTMBlock`` parameter, read off its signature like ``_mamba3_params``. (+4 more)

### Community 162 - "Judge"
Cohesion: 0.12
Nodes (15): Complete, average_precision(), Judge, Path, A ``Complete`` behind the reply cache; ``calls`` counts the completions…, RAGAS context precision: precision@k averaged over the positions k of the…, The judge reads the first JSON object of a reply and retries an unparseable one…, test_judge_parses_scores_and_caches() (+7 more)

### Community 167 - "test_utils.py"
Cohesion: 0.11
Nodes (27): detect_prefix(), _first_segments(), infer_architecture(), infer_layer_types(), InferredArchitecture, load_every_key(), load_state_dict_guarded(), _mixer_params() (+19 more)

### Community 168 - "RateLimiter"
Cohesion: 0.15
Nodes (7): RateLimiter, At most one request per ``min_interval`` seconds per host., _FakeResponse, _FakeSession, post(), test_http_get_retries_transient_errors_and_honours_retry_after(), test_rate_limiter_spaces_requests_per_host()

### Community 169 - "Retrieval on GerLeRB and GerDaLIR"
Cohesion: 0.18
Nodes (10): GerDaLIR (12,234 queries), GerLeRB (367 queries), Not licensed, Open limitations, Prediction and outcome, Reproduction, Results, Retrieval on GerLeRB and GerDaLIR (+2 more)

### Community 170 - "MDebertaNLI"
Cohesion: 0.22
Nodes (9): MDebertaNLI, mDeBERTa XNLI as an ``NLI`` callable: softmax over the classes, the entailment…, How it answers, calibrate(), main(), Path, Calibrate the verifier's entailment threshold on hand-labelled premise/claim…, read_pairs() (+1 more)

### Community 171 - "citation_metrics.py"
Cohesion: 0.14
Nodes (22): answer_words(), cited_passages(), doc_of(), kept_segments(), _mean(), Segment, _ratio(), Citation metrics of answer records (``lexhybrid.eval.records``), per question… (+14 more)

### Community 172 - "pytest"
Cohesion: 0.17
Nodes (11): pytest, linux_only, slow, Checks that need Linux: they run in CI and are skipped, with the reason, on…, torch.compile on the Inductor CPU backend (gcc) reproduces the eager logits.…, test_inductor_cpu_compile_matches_eager(), _ddp_worker(), Multi-GPU checks. Marked ``multigpu``: they collect everywhere and skip, with… (+3 more)

### Community 173 - "_schedule_fit"
Cohesion: 0.17
Nodes (8): _IndexRows, Row i carries its index, so a run's data order can be read back., Lightning restarts a plain loader at row 0 on resume (screen run S5's…, Stage B: resumed from stage A's last step onto a shorter schedule of its own…, _schedule_fit(), _SeenRows, test_a_resumed_run_reads_on_through_the_schedule_and_matches_an_unbroken_one(), test_a_stage_on_a_new_schedule_starts_it_at_row_0_and_a_requeue_reads_on()

### Community 174 - "_pretrain_state"
Cohesion: 0.20
Nodes (11): _compose_sft(), _pretrain_state(), What a pretraining run's checkpoint holds for the model of ``cfg``, keys as…, train_sft.py starts the hybrid from a pretraining checkpoint and trains on…, A pretraining checkpoint loads exactly; one missing or unexpected key stops the…, A scripted val/loss of 3, 1, 2, 2.5 over 4 steps, validating after each: the…, Rows reach 8,192 tokens, and stage-B pretraining ran micro-batch 4 at that…, test_sft_best_checkpoint_holds_the_weights_of_the_best_dev_loss() (+3 more)

### Community 175 - "train_sft_hf.py"
Cohesion: 0.29
Nodes (10): adapt(), load_lm(), main(), DictConfig, main, The Hugging Face baseline of supervised fine-tuning: ``sft.hf_model``…, A sibling script as a module: ``scripts/`` is not a package., The model with room for every pointer special, behind the hybrid's… (+2 more)

### Community 176 - "test_hf_sampling_through_its_cache_matches_the_uncached_reference"
Cohesion: 0.67
Nodes (3): Same logits, same generator: the KV-cached sampler and the uncached ``sample``…, test_hf_sampling_through_its_cache_matches_the_uncached_reference(), sample()

### Community 178 - "changelog.py"
Cohesion: 0.36
Nodes (8): Change, changes_since(), main(), Path, The currency changelog: which provisions changed since a date. A provision's…, write_changelog(), RIS versions (linked to the version they replaced), Fedlex articles by their…, test_changelog_from_fixture_versions()

### Community 179 - "synth_qa.sh"
Cohesion: 0.20
Nodes (8): scripts_slurm_lib_vllm_server_sh, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, synth_qa.sh script, HF_HOME, HF_HUB_OFFLINE, vllm_smoke.sh script

### Community 180 - "train_sft_4gpu.sh"
Cohesion: 0.22
Nodes (9): HF_DATASETS_CACHE, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, opt(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, train_sft_4gpu.sh script (+1 more)

### Community 181 - "gerdalir.py"
Cohesion: 0.36
Nodes (8): fetch(), _lines(), parse_passage(), parse_qrel(), parse_query(), GerDaLIR, the German Dataset for Legal Information Retrieval: case-law…, select(), test_gerdalir_select_on_fixture()

### Community 182 - "test_sync_scripts.py"
Cohesion: 0.24
Nodes (18): CompletedProcess, counting(), decode(), calls(), _load_tree_hash(), make_tree(), parametrize, Path (+10 more)

### Community 183 - "shutil"
Cohesion: 0.22
Nodes (6): shutil, subprocess, tarfile, dists(), fixture, The built distributions contain the package and nothing else (CI `package`…

### Community 184 - "train_dpo_4gpu.sh"
Cohesion: 0.29
Nodes (7): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, opt(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, train_dpo_4gpu.sh script

### Community 185 - "FakeQwen"
Cohesion: 0.22
Nodes (4): FakeQwen, Just enough of a Hugging Face tokenizer for ``load_tokenizer``'s checks., test_tokenizer_cache_and_cli(), test_tokenizer_loader_checks_eos_and_vocabulary()

### Community 186 - "write"
Cohesion: 0.32
Nodes (6): Path, The set's files under ``out_dir`` and its manifest (ids and hashes, no text);…, write(), _write_jsonl(), Hex SHA-256 of the UTF-8 text: the identity dedup and the manifests key on., sha256_text()

### Community 187 - "configure_optimizer"
Cohesion: 0.32
Nodes (8): configure_optimizer(), get_parameter_groups(), Any, Module, Optimizer, Split trainable parameters into a decayed and a non-decayed group by name…, Build an optimizer over ``get_parameter_groups``; ``foreach`` is used only on…, test_parameter_groups_exclude_norms_biases_and_embeddings_from_decay()

### Community 188 - "_bger"
Cohesion: 0.25
Nodes (8): _bger(), Before ~2007 the numbers are not bold: a number opens a consideration only when…, An order (Verfügung): unnumbered "in Erwägung, dass ..." considerations, then…, One document per decision; the heading, facts, considerations (``E. 4.4.2``),…, test_bger_parser_on_an_order(), test_bger_parser_on_fixtures(), test_bger_parser_on_the_unmarked_older_layout(), test_packing_keeps_one_source_per_pack()

### Community 189 - "paragraphs"
Cohesion: 0.33
Nodes (6): is_heading(), paragraphs(), A page's footnotes interrupt the paragraph that runs on to the next page ("...…, The paper's paragraphs as ``(is_heading, text)``, cleaned of the PDF's layout…, _rejoin_across_footnotes(), test_dip_paragraph_rules()

### Community 190 - "setup_arm_env.sh"
Cohesion: 0.33
Nodes (5): PYTHONUNBUFFERED, setup_arm_env.sh script, UV_CACHE_DIR, UV_LINK_MODE, UV_PROJECT_ENVIRONMENT

### Community 191 - "setup_vllm.sh"
Cohesion: 0.33
Nodes (5): PYTHONUNBUFFERED, setup_vllm.sh script, UV_CACHE_DIR, UV_LINK_MODE, UV_PROJECT_ENVIRONMENT

### Community 192 - "write_manifest"
Cohesion: 0.33
Nodes (7): Path, Write the manifest lines for ``docs`` to ``path`` (replacing it); return the…, read_manifest(), write_manifest(), The manifest keeps every schema field but the text, one line per document., test_collector_cli_writes_documents_and_manifest(), test_manifest_roundtrip()

### Community 193 - "test_tfla_wide_chunks_factorise_per_block_exactly"
Cohesion: 0.33
Nodes (7): 256 positions (two chunks of 128), two rows, boundaries inside and on chunk…, Trained forget gates widened a 128-chunk's log-decay past fp32 (half-range 43.2…, In fp32 the per-block path forms no inf, forward or backward (a masked inf…, test_tfla_per_block_path_is_finite_and_close_in_fp32(), test_tfla_wide_chunks_factorise_per_block_exactly(), _tfla_grads(), _wide_tfla_inputs()

### Community 194 - "pathlib"
Cohesion: 0.06
Nodes (54): argparse, dataclasses, hashlib, hmac, itertools, json, The German legal-entity scrub: names out before training. Entities come from…, The student's tokenizer: Qwen3's own, plus the SFT pointer specials.… (+46 more)

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

### Community 200 - "render"
Cohesion: 0.33
Nodes (6): Element, Text of a gii-norm content element: list items on their own lines, whitespace…, The date of the last amendment in the header's ``Stand`` comment, else the…, render(), walk(), stand_date()

### Community 201 - "read_samples"
Cohesion: 0.40
Nodes (5): Path, The rows of an SFT builder's split; anything the builder's manifest does not…, The records a previous attempt finished, by id. Every record ends in a newline,…, read_items(), read_samples()

### Community 204 - "test_compile_forward_chunk128"
Cohesion: 0.40
Nodes (4): linux_only, slow, Hybrid_legal_base at reduced width -- chunk 128 for both Mamba-3 and mLSTM,…, test_compile_forward_chunk128()

### Community 205 - "_text"
Cohesion: 0.50
Nodes (4): Element, Paragraph text: ``<p>``s become lines, ``<br/>`` breaks, whitespace normalised., _text(), walk()

### Community 207 - "test_ablation_appends_each_record_and_resumes_after_preemption"
Cohesion: 0.67
Nodes (3): Records reach the arm files as they are produced, so a run cut mid-question…, test_ablation_appends_each_record_and_resumes_after_preemption(), flaky()

## Knowledge Gaps
- **267 isolated node(s):** `lexhybrid-arm`, `lexhybrid-scrub`, `lexhybrid-vllm`, `lexhybrid`, `build_corpus_array.sh script` (+262 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1820 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **30 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HybridLanguageModel` connect `HybridLanguageModel` to `_s1_fingerprint`, `test_data.py`, `HybridConfig`, `test_layers.py`, `performance_profile.py`, `test_pipeline.py`, `hybrid_lm.py`, `test_models.py`, `test_reference_parity.py`, `train_pretrain.py`, `test_numerics_fp64.py`, `evaluate_val_ppl.py`, `PretrainLightningModule`, `torch`, `test_scripts.py`, `datasets.py`, `test_utils.py`, `test_training.py`, `mqar.py`, `pytest`, `_pretrain_state`, `check_operator_equivalence.py`, `sft_format.py`, `filter_logits`, `test_slurm_wrappers.py`, `test_config_invariants.py`, `configure_optimizer`, `segment_ids`, `test_dpo.py`, `HybridLightningModule`, `load_model_config`, `pathlib`, `test_decoding.py`, `dpo_module.py`, `test_compile_forward_chunk128`, `train_sft.py`, `DPOLightningModule`, `measure_fertility.py`, `screen_arms.py`, `strip_prefixes`, `Architecture`, `statute_recall.py`, `test_gpu.py`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `Document` connect `Document` to `Packer`, `test_verifier.py`, `test_data.py`, `dedup.py`, `test_pipeline.py`, `index.py`, `FedlexCollector`, `Chunk`, `fedlex.py`, `datasets.py`, `scrub_document`, `generate_synthetic_qa.py`, `changelog.py`, `sft_format.py`, `write`, `scrub_file`, `write_manifest`, `pathlib`, `measure_fertility.py`, `test_retrieval.py`, `Architecture`, `_doc`, `statute_recall.py`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `Chunk` connect `Chunk` to `test_verifier.py`, `ByteTokenizer`, `Judge`, `fedlex.py`, `citation_metrics.py`, `SpecialTokens`, `generate_synthetic_qa.py`, `sft_format.py`, `quotes.py`, `filters.py`, `verifier.py`, `test_dpo.py`, `test_renderer.py`, `pathlib`, `build_dpo_pairs.py`, `test_eval.py`, `gold_passages.py`, `test_retrieval.py`, `AnswerRecord`, `test_judge.py`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 57 inferred relationships involving `Document` (e.g. with `Data pipeline` and `Document schema`) actually correct?**
  _`Document` has 57 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `HybridLanguageModel` (e.g. with `Model` and `__getattr__()`) actually correct?**
  _`HybridLanguageModel` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 60 inferred relationships involving `Chunk` (e.g. with `answer_words()` and `cited_passages()`) actually correct?**
  _`Chunk` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `HybridConfig` (e.g. with `Model` and `load_model_config()`) actually correct?**
  _`HybridConfig` has 6 INFERRED edges - model-reasoned connections that need verification._