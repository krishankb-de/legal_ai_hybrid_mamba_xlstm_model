# Graph Report - legal_ai_hybrid_mamba_xlstm_model  (2026-11-12)

## Corpus Check
- 267 files · ~272,433 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 62 file(s) not represented in the graph (top: .jsonl 30, .xml 12, (none) 6)

## Summary
- 4261 nodes · 9894 edges · 203 communities (172 shown, 31 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 529 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e0ed6a09`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- ris.py
- packing.py
- test_data.py
- Document
- HybridLanguageModel
- test_layers.py
- test_eval.py
- load_model_config
- performance_profile.py
- collectors/base.py
- answer.py
- family_lm
- gold_passages.py
- Mamba3Block
- test_models.py
- index.py
- tfla_forward_parallel
- generate.py
- ByteTokenizer
- train_pretrain.py
- build_sft_data.py
- evaluate_val_ppl.py
- PointerFSM
- eurlex.py
- eval_retrieval.py
- analyze_generation_diversity.py
- generate_synthetic_qa.py
- AttentionBlock
- torch
- test_scripts.py
- test_retrieval.py
- fedlex.py
- Chunk
- train_dpo.py
- SpikeDiagnosticsCallback
- test_numerics_fp64.py
- HybridConfig
- dedup.py
- train_sft.py
- _compose
- datasets.py
- ssd_sequential_reference
- build_dpo_pairs.py
- teacher
- test_dpo.py
- rel_max_err
- sft_format.py
- sandbox
- test_synth_qa.py
- ScanSplit
- check_env.py
- test_training.py
- test_verifier.py
- filter_logits
- load_tokenizer
- os
- test_slurm_wrappers.py
- eval_system_ablation.py
- test_check_env.py
- test_config_invariants.py
- scrub_ler.py
- inspect_checkpoint.py
- changelog.py
- code
- HybridBlock
- test_renderer.py
- paragraphs
- test_decoding.py
- slab_ce_kl
- test_code_hygiene.py
- bootstrap_compare.py
- NLI verifier threshold on 100 hand-built statute pairs
- Architecture
- screen_metrics.py
- ParquetRows
- evalsets/base.py
- build_dense_index.sh
- lexhybrid
- parametrize
- eval_retrieval.sh
- test_rsync_rules.py
- _BlockParser
- SyntheticPackedDataset
- run_metadata.py
- DPOLightningModule
- cumulative_angles
- test_repo_hygiene.py
- German-law pretraining corpus: tokens, deduplication, scrub
- conftest.py
- test_sync_scripts.py
- PretrainLightningModule
- probe_ckpt_size.sh
- _CaptureLogger
- check_repo_hygiene.py
- screen_table.py
- Profile ladder, chunk size and distillation memory
- build_shards.py
- Architecture screen: ablations of the hybrid backbone
- test_http_teacher_retries_and_the_reply_parsers_are_strict
- gpu_tests.sh
- multigpu_tests.sh
- train_pretrain_1gpu.sh
- Contributing
- tfla_interface.py
- test_ci_workflow.py
- _oldp_docs
- SourceShards
- fused_retriever
- equivalence.sh
- kd_memory_probe.sh
- scan_interface.py
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
- _text
- Judge
- Backbone results: the hybrid and a Transformer baseline at 8,192 tokens
- test_gpu.py
- LayerSplit
- build_corpus_array.sh
- pack_corpus.sh
- _s1_fingerprint
- scrub_ner_worker.py
- corpus_counts.sh
- diag_compile.sh
- eval_probes.sh
- fetch_hf.sh
- scrub_array.sh
- setup_env.sh
- train_pretrain_4gpu.sh
- .__init__
- test_diag_spike_reruns_the_arm_observed_in_its_own_directory
- test_verify_handoff_checks_every_run_and_fails_if_one_does
- .forward
- test_eval_probes_scores_named_runs_at_another_length
- test_reference_parity.py
- StepStatsCallback
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
- test_sft_requeue_resumes_with_every_validation_ranked
- lexhybrid
- lexhybrid-scrub
- pathlib
- StopAtStepCallback
- Retrieval on GerLeRB and GerDaLIR
- metrics.py
- test_hf_sampling_through_its_cache_matches_the_uncached_reference
- Post-training results: supervised fine-tuning
- slab_cross_entropy
- README.md
- .from_hydra
- CausalLMAdapter
- SignalCheckpointCallback
- parse_citation
- synth_qa.sh
- train_sft_4gpu.sh
- teacher_forward_packed
- test_tfla_wide_chunks_factorise_per_block_exactly
- _tfla_model
- train_dpo_4gpu.sh
- test_collect_offline.py
- test_param_bands
- lightning_module.py
- Document schema
- ._gates
- setup_arm_env.sh
- setup_vllm.sh
- HybridLightningModule
- test_tfla_chunk_size_reaches_kernel
- measure_fertility.py
- build_dpo_pairs.sh
- build_sft_data.sh
- eval_gold_passages.sh
- eval_system_ablation.sh
- vllm_server.sh
- test_pipeline.py
- lexhybrid-arm
- lexhybrid-vllm

## God Nodes (most connected - your core abstractions)
1. `Document` - 132 edges
2. `HybridLanguageModel` - 109 edges
3. `Chunk` - 100 edges
4. `HybridConfig` - 72 edges
5. `AnswerRecord` - 49 edges
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
- `Limitations` --references--> `build_doc_block_mask()`  [INFERRED]
  analysis/profile_results.md → lexhybrid/layers/attention_block.py

## Import Cycles
- None detected.

## Communities (203 total, 31 thin omitted)

### Community 0 - "ris.py"
Cohesion: 0.09
Nodes (26): FetchError, RuntimeError, ``fetch(url)``'s response, or None when it raised ``FetchError``., A request failed after every retry., Fetch one document, or skip it (logged) when its fetch fails after every retry,…, SkipFailedDocuments, _as_list(), _clean() (+18 more)

### Community 1 - "packing.py"
Cohesion: 0.13
Nodes (18): main(), pack(), Packer, Path, EOS-delimited packing into fixed-length rows and parquet shards. Every document…, Parquet shards ``shard-00000.parquet``... of at most ``rows_per_shard`` rows,…, Parquet shards ``shard-00000.parquet``... of at most ``rows_per_shard`` rows., Rows of ``row_len`` tokens from one source's documents, fed one tokenised… (+10 more)

### Community 2 - "test_data.py"
Cohesion: 0.04
Nodes (71): BGerCollector, FineWeb2DECollector, GIICollector, parse_toc(), ``(slug, xml.zip URL)`` for every law in ``gii-toc.xml``., read_zip_xml(), MultiLegalPileCollector, court_abbreviation() (+63 more)

### Community 3 - "Document"
Cohesion: 0.05
Nodes (73): datetime, html_parser, io, licence_flags(), main(), ``licence``, ``commercial_safe`` and ``research_only`` as the register defines…, A collector module's entry point: ``cli``, then end the process at once with…, utc_now() (+65 more)

### Community 4 - "HybridLanguageModel"
Cohesion: 0.06
Nodes (32): HybridLanguageModel, no_grad, Tensor, Language-model forward pass. Args: input_ids: (B, L) token ids; mutually…, Embedding and every block: the residual stream (B, L, D) BEFORE the final norm.…, Final norm and LM head: residual stream -> logits., Uncached sampling (``lexhybrid.decoding.sample``): the full forward per new…, True when every mixer has an O(1) ``step``; all-or-nothing by design, because… (+24 more)

### Community 5 - "test_layers.py"
Cohesion: 0.06
Nodes (61): mLSTMBlock, (Re-)apply the gate-bias inits after the parent model's global weight pass.…, mLSTM decode state: the matrix cell ``C`` and the normalizer ``n``, both zeros,…, mLSTM mixer: per-dimension exponential input gate, sigmoid forget gate, matrix…, _bc_bias_pair(), _continue_with_steps(), _drive_lambda_to_one(), m3_block() (+53 more)

### Community 6 - "test_eval.py"
Cohesion: 0.07
Nodes (40): outcome(), What one decoded set shows: the gold position (1-based), whether the first…, One summary per (system, seed, gold arm), in order of first appearance (see the…, score(), _share(), evaluate(), ndcg_at_k(), Ranking metrics over qrels, as trec_eval computes them: nDCG@k (gain =… (+32 more)

### Community 7 - "load_model_config"
Cohesion: 0.09
Nodes (34): Configuration dataclass for the hybrid decoder. Ported from the reference…, Model configuration and the yaml loader every script uses., available_model_configs(), load_model_config(), load_model_yaml(), Any, Path, Load a model yaml into a ``HybridConfig`` -- the one path every script uses. A… (+26 more)

### Community 8 - "performance_profile.py"
Cohesion: 0.09
Nodes (34): contextlib, csv, attention_backend(), available_configs(), build_model(), compile_for(), effective_chunk_sizes(), fit_log_slope() (+26 more)

### Community 9 - "collectors/base.py"
Cohesion: 0.05
Nodes (52): _cached(), cli(), Collector, http_get(), http_post(), HttpCache, parse_limit(), Path (+44 more)

### Community 10 - "answer.py"
Cohesion: 0.06
Nodes (41): decode_answer(), Answer tokens as the pointer markup: special tokens by name, so no tokenizer…, Pointer-constrained decoding. The legal model never generates statute text: it…, K for ``<|cK|>``, else None., J for ``<|sJ|>``, else None., Ids of the 87 SFT-only special tokens (never used in distillation)., The token strings, in registry order (the order ``from_offset`` assigns ids)., Consecutive ids from ``first_id`` -- tests, and any vocabulary that appends… (+33 more)

### Community 11 - "family_lm"
Cohesion: 0.11
Nodes (20): family_lm(), _prefill_then_greedy(), parametrize, The one-pass prefill leaves every cache exactly where token-by-token decoding…, In bf16 the two paths round differently, so the claim is argmax identity…, A packed prompt: decoding continues each row's last document as if it were…, Cached greedy equals the uncached reference token for token, batch 3 with rows…, With EOS, length normalisation and a repetition penalty, cached beam search… (+12 more)

### Community 12 - "gold_passages.py"
Cohesion: 0.09
Nodes (46): answer_words(), cited_passages(), doc_of(), gold_relation(), kept_segments(), _mean(), Segment, _ratio() (+38 more)

### Community 13 - "Mamba3Block"
Cohesion: 0.06
Nodes (34): _mamba3_params(), _mlstm_params(), _prefixed_kwargs(), The ``mamba3_*`` options this dispatcher forwards, read off the block's…, Every ``mLSTMBlock`` parameter, read off its signature like ``_mamba3_params``., Kwargs for a prefixed mixer: ``shared`` unprefixed names first, then…, _signature_params(), heavy_tail_activation() (+26 more)

### Community 14 - "test_models.py"
Cohesion: 0.08
Nodes (43): copy, _delta_at_init(), _delta_model(), linux_only, parametrize, slow, Model-level behaviour of HybridLanguageModel., The model's weight pass zeroed both gate biases; `post_model_init` restores the… (+35 more)

### Community 15 - "index.py"
Cohesion: 0.09
Nodes (37): document_digest(), DocumentDigest, file_sha256(), indexed_documents(), IndexVersion, latest(), Path, Index versions: a registry of the retrieval indexes that have been built.… (+29 more)

### Community 16 - "tfla_forward_parallel"
Cohesion: 0.06
Nodes (43): Limitations, TFLA (tiled linear attention) kernels for the mLSTM., apply_tfla(), _blockwise_scores(), _intra_chunk_sequential(), RuntimeError, Tensor, Chunk-parallel TFLA. Args: q, k, v: (B, H, L, D) i_gate: input gates (B, H, L,… (+35 more)

### Community 17 - "generate.py"
Cohesion: 0.11
Nodes (41): Advance, apply_repetition_penalty(), _beam_core(), beam_search(), beam_search_cached(), beam_search_uncached(), best_of_n(), _check_one_sample() (+33 more)

### Community 18 - "ByteTokenizer"
Cohesion: 0.14
Nodes (19): _jsonl(), _qa_file(), _qa_row(), One evaluation-only set's manifest: the raw questions' hashes, as the…, QA rows with the writer's ``info.json`` beside them, free for training unless…, build_sft_data writes train and dev splits in which every target ends in…, QA items are all answerable: when they hold the share below 0.2, a seeded…, A QA file without info.json, or one that does not free its items for training;… (+11 more)

### Community 19 - "train_pretrain.py"
Cohesion: 0.09
Nodes (33): Training callbacks: checkpoint-on-signal (ported from the reference) and step…, Loss-spike diagnostics for one training run. Two screening runs collapsed to…, math, pytorch_lightning, pytorch_lightning_callbacks, pytorch_lightning_strategies, build_callbacks(), build_dataloaders() (+25 more)

### Community 20 - "build_sft_data.py"
Cohesion: 0.09
Nodes (32): question_key(), A hash of the question with its form normalised (NFC, case, whitespace, final…, German sentences of a passage: split after ``.``, ``!`` or ``?`` before an…, split_sentences(), ``(grain, text)`` for one Absatz: its sentences (``Satz 2``) and its items…, statute_units(), build(), evalset_guard() (+24 more)

### Community 21 - "evaluate_val_ppl.py"
Cohesion: 0.18
Nodes (13): document_offsets(), late_context_sums(), load_model(), main(), no_grad, Path, Tensor, Validation PPL of a checkpoint, overall and late in context (the late-25% PPL).… (+5 more)

### Community 22 - "PointerFSM"
Cohesion: 0.09
Nodes (27): Decoding, PointerFSM, process(), Tensor, The pointer grammar for one retrieved context. Args: specials: the special-…, The state after ``token``; raises ``ValueError`` if the grammar forbids it., Boolean (vocab,) mask of the tokens the grammar allows next., Set every forbidden logit to ``-inf``, row by row; ``generated`` (B, T) are the… (+19 more)

### Community 23 - "eurlex.py"
Cohesion: 0.15
Nodes (18): act_code(), EURLexCollector, _find(), _inline(), _local(), oj_date(), parse_eurlex_xhtml(), Element (+10 more)

### Community 24 - "eval_retrieval.py"
Cohesion: 0.05
Nodes (56): hashlib, BM25Index, Hit, Path, Lexical retrieval with BM25 over the retrieval chunks (``bm25s``). Every…, The index's tokens of each text (lower-cased, German stopwords out, stemmed)., A ``bm25s`` index over chunks, queried by text., Index ``chunks`` (anything with ``id`` and ``text``:… (+48 more)

### Community 25 - "analyze_generation_diversity.py"
Cohesion: 0.16
Nodes (18): analyse(), distinct_n(), duplicate_clusters(), main(), _ngram_counts(), Mean BLEU-4 of each text against a sample of the OTHERS. Higher = more…, Render the comparison table plus the verdict., Boilerplate / duplicate-template analysis for generated answers, with controls.… (+10 more)

### Community 26 - "generate_synthetic_qa.py"
Cohesion: 0.06
Nodes (48): concurrent_futures, Exception, Ranked, Cross-encoder reranking of fused candidates into the prompt's passages. The…, The first ``limit`` candidates re-scored and sorted best first (ties keep the…, Best first, while there are fewer than ``max_chunks`` and the passages fit…, The step from the fused top 50 to the 3-8 passages (<= 4K tokens) of the prompt., rerank() (+40 more)

### Community 27 - "AttentionBlock"
Cohesion: 0.06
Nodes (40): disable, apply_rope(), AttentionBlock, build_doc_block_mask(), build_doc_boundary_attn_mask(), build_rope_cache(), compiled_flex_attention(), positions_within_documents() (+32 more)

### Community 28 - "torch"
Cohesion: 0.07
Nodes (37): einops, Gold-answer likelihood: a continuous probe score that an exact-match floor…, exponential_activation(), Tensor, Activation functions used by the mixers., SiLU (swish) activation., Exponential gate activation of the mLSTM input gate., silu_activation() (+29 more)

### Community 29 - "test_scripts.py"
Cohesion: 0.06
Nodes (27): The ported command-line scripts run end to end on CPU with tiny models., Every arm builds through HybridConfig.from_hydra at reduced width, runs a…, The learning-rate arms are S0 and S1 with only `model.learning_rate` changed,…, Full size on `meta` with the bands: every arm passes; S5 carries +2.36% as a…, Guard: if an arm's lever never reaches the model, its ARCH line lacks the token., The method: whole documents until the word budget, documents over the cap…, Write ``{tag: [(step, value), ...]}`` as TensorBoard events where…, Val PPL is only in TensorBoard; the reader takes the run's highest version (the… (+19 more)

### Community 30 - "test_retrieval.py"
Cohesion: 0.04
Nodes (78): _anchor(), chunk_case_law(), chunk_document(), chunk_statute(), _common(), estimate_tokens(), _label(), provision_heading() (+70 more)

### Community 31 - "fedlex.py"
Cohesion: 0.09
Nodes (33): act_code(), act_meta(), acts_query(), article_sections(), FedlexCollector, _inline(), _is_text(), latest_versions() (+25 more)

### Community 32 - "Chunk"
Cohesion: 0.10
Nodes (26): collections_abc, date, Chunk, One retrievable passage., What the lexical and dense indexes see: the header, the hierarchy, the…, _allowed(), _bound(), _day() (+18 more)

### Community 33 - "train_dpo.py"
Cohesion: 0.10
Nodes (31): Seeded random-token rows with packed-document ids, for smoke tests and…, load_sft_model(), Preference optimisation of an SFT model: DPO and KTO against a frozen reference…, The SFT model of ``checkpoint``: the hybrid of ``config``, or a…, best_val_checkpoint(), Supervised fine-tuning: cross-entropy on the answer tokens only, for the hybrid…, The ``save_top_k`` lowest-``val/loss`` checkpoints, ``best-step<N>.ckpt``,…, ``lm`` frozen but for rank-``rank`` LoRA adapters on ``LORA_TARGETS`` and the… (+23 more)

### Community 34 - "SpikeDiagnosticsCallback"
Cohesion: 0.11
Nodes (14): Callback, Tensor, Trainer, SpikeDiagnosticsCallback, hook(), hook(), hook(), hook() (+6 more)

### Community 35 - "test_numerics_fp64.py"
Cohesion: 0.12
Nodes (29): Legacy chunk-parallel selective scan (divide-and-clamp; see module docstring).…, selective_scan_parallel(), _audit_inputs(), _clamp_fire_fraction(), _delta_cases(), _measure(), Operator-level numerical correctness against float64 sequential oracles. Ported…, ``use_fast_path=False`` shares the one implementation (the reference once had a… (+21 more)

### Community 36 - "HybridConfig"
Cohesion: 0.07
Nodes (32): HybridConfig, Validate and auto-compute derived parameters., The per-mixer arguments for layer ``layer_idx`` (a readable summary; not used…, Every architectural knob of the hybrid decoder. One dataclass serves every…, Convert the config to a plain dictionary., Create a config from a dictionary produced by ``to_dict``., Write ``config.json`` into ``save_directory`` and return its path., __getattr__() (+24 more)

### Community 37 - "dedup.py"
Cohesion: 0.17
Nodes (18): deduplicate(), deduplicate_signatures(), deduplicate_stream(), DedupLog, keep_order(), minhash(), Deduplication of the training corpus: exact, then near-duplicate. 1. **Exact.**…, ``deduplicate_signatures`` over documents streamed once, their MinHashes… (+10 more)

### Community 38 - "train_sft.py"
Cohesion: 0.06
Nodes (33): Checkpoint, importlib_util, _rank_world(), A training mixture's rows for one rank, from where the run stands. Sample ``i``…, Every row of an evaluation set across the ranks, each rank the same count…, ScheduleSampler, ShardSampler, Write ``<checkpoint_dir>/last.ckpt`` at the end of each validation, after a… (+25 more)

### Community 39 - "_compose"
Cohesion: 0.07
Nodes (38): _argv(), _dpo_overrides(), _fake_trainer(), _pair_rows(), parametrize, 4 ranks x 2 pairs need 8 pairs for one optimizer step (and twice that with 2…, KTO's mismatched rows need two pairs: its dev loader drops a short last batch…, The SFT model's FSM-constrained samples, verified (stub NLI), give one pair per… (+30 more)

### Community 40 - "datasets.py"
Cohesion: 0.12
Nodes (23): build_source_shards(), effective_weights(), mixture_from_config(), PackedMixture, Dataset, Tensor, Pretraining data over packed shards: the split, the mixture and the arms.…, Each admitted source's share of the rows: its group's share, split by weight… (+15 more)

### Community 41 - "ssd_sequential_reference"
Cohesion: 0.15
Nodes (15): Mamba-2/3 state space dual (SSD) kernels., device, Tensor, Lower-triangular segment sums: ``out[..., t, s] = sum_{i=s+1}^{t} v_i`` for ``s…, Monotone segment index per position (see ``lexhybrid.kernels.segments``)., _segment_ids(), segsum(), _heads_per_group() (+7 more)

### Community 42 - "build_dpo_pairs.py"
Cohesion: 0.06
Nodes (49): argparse, collections, prompt_ids(), The prompt's token ids, up to and including ``<|answer|>`` (the positions…, The student's tokenizer: Qwen3's own, plus the SFT pointer specials.…, entailment_index(), MDebertaNLI, Entailment of a claim by its cited sources: the NLI step of the verifier. An… (+41 more)

### Community 43 - "teacher"
Cohesion: 0.13
Nodes (19): nli(), parametrize, Quotas met: no teacher call, same rows. A log cut mid-line by a kill loses only…, The teacher answers over the prompt's passages in the normal format. A refusal…, Rounds smaller than the limit add up: 32 errors in a row stop the run, whatever…, Attempts run out with too few answerable rows: a seeded sample of refusals…, Rows are capped kind by kind at the new count's quotas, with no teacher call. A…, Pure, so threads may call it in any order. A question names its source's… (+11 more)

### Community 44 - "test_dpo.py"
Cohesion: 0.06
Nodes (39): _unit(), PairSchedule, Dataset, ``num_samples`` pairs on a seeded schedule (one permutation per pass), or every…, hf_run(), _pairs(), fixture, Preference optimisation: pair selection and balance, the answer-token log-… (+31 more)

### Community 45 - "rel_max_err"
Cohesion: 0.15
Nodes (24): Chunk-parallel SSD scan. h_t = exp(dt_t A) h_{t-1} + sum_terms coeff_t (B_t…, ssd_chunked_scan(), Term, parametrize, Tensor, Resets are boolean masks inside the scan; the on-chunk-edge case is where an…, Perturbing document A must leave document B bit-identical, not merely close., The state after the last token, including a partial last chunk -- where the… (+16 more)

### Community 46 - "sft_format.py"
Cohesion: 0.09
Nodes (48): Cite, encode(), Example, parse(), Passage, Quote, The retrieval-grounded SFT format: one prompt, pointer answers. The prompt is…, ``(prompt, target)`` as strings with the special-token markup. (+40 more)

### Community 47 - "sandbox"
Cohesion: 0.07
Nodes (30): _executable(), fixture, Path, The default sources' scrubbed parts go to the bm25 build with the pack's dedup…, A repository copy with the wrappers, screen_arms.py and a fake interpreter;…, The log is the job's own stdout, and GNU grep refuses to read its output file…, One sequence length per process, a fresh Inductor cache per point, the…, Task i scrubs part i % PARTS of source i / PARTS; every task uses the same HMAC… (+22 more)

### Community 48 - "test_synth_qa.py"
Cohesion: 0.16
Nodes (18): http_server, socket, generator(), _executable(), _free_port(), index(), jobs(), run() (+10 more)

### Community 49 - "ScanSplit"
Cohesion: 0.18
Nodes (5): _delta_ms(), {mixer type: mean ms per forward}, plus a per-layer-index breakdown., Time every `ssd_chunked_scan` call inside the Mamba-3 blocks. `mamba3_block.py`…, ScanSplit, timed()

### Community 50 - "check_env.py"
Cohesion: 0.06
Nodes (54): importlib, importlib_metadata, packaging_markers, packaging_requirements, packaging_utils, packaging_version, pkgutil, Requirement (+46 more)

### Community 51 - "test_training.py"
Cohesion: 0.07
Nodes (34): LambdaLR, Optimizer, WSD learning-rate multiplier at ``step``. warmup: linear from…, Warmup-Stable-Decay scheduler. Ratios must sum to 1; an absolute…, wsd_factor(), WSDScheduler, _lambda(), _full_kd() (+26 more)

### Community 52 - "test_verifier.py"
Cohesion: 0.05
Nodes (79): dataclasses, Answer verification: pointers, quotes, entailment and scrub leaks., _balanced_body(), check_quote(), check_quotes(), extract_quotes(), _find_quote(), normalise() (+71 more)

### Community 53 - "filter_logits"
Cohesion: 0.33
Nodes (5): filter_logits(), Temperature, then top-k, then nucleus (top-p) filtering; filtered entries…, choose(), Shared sampling filter (``lexhybrid.decoding.filter_logits``)., test_filter_logits()

### Community 54 - "load_tokenizer"
Cohesion: 0.16
Nodes (19): functools, cache_dir(), encode_document(), load_tokenizer(), main(), ``data/hf/hub`` unless the environment names a Hugging Face cache., Qwen3's tokenizer, with the pointer specials unless ``with_specials=False``.…, Token ids of raw document text; special-token strings inside it are split into… (+11 more)

### Community 55 - "os"
Cohesion: 0.15
Nodes (17): os, adapt(), load_lm(), main(), DictConfig, main, The Hugging Face baseline of supervised fine-tuning: ``sft.hf_model``…, A sibling script as a module: ``scripts/`` is not a package. (+9 more)

### Community 56 - "test_slurm_wrappers.py"
Cohesion: 0.15
Nodes (17): SLURM wrappers under `scripts/slurm/`. Static checks on every `*.sh` there: the…, A requeued run finds its last.ckpt and passes it on., The array index picks the arm; screen_arms.py resolves it ON THE NODE; the…, The wrappers the pipeline names are all here., Only the named shapes are rerun: the 1.7B shapes are rerun alone; the 8B shapes…, Stage B: INIT_FROM (stage A's last.ckpt) while the stage has none of its own;…, test_every_planned_wrapper_exists(), test_four_gpu_wrapper_runs_ddp_through_the_same_body() (+9 more)

### Community 57 - "eval_system_ablation.py"
Cohesion: 0.13
Nodes (23): gold_chunk_ids(), Each item's gold-relevant chunk ids: its ``gold_chunk_ids`` the index holds,…, chunk_dict(), copy_top(), main(), prompt_tokens(), Path, Append each arm's missing records to ``<out>/<arm>.jsonl``: A1-A4 per question,… (+15 more)

### Community 58 - "test_check_env.py"
Cohesion: 0.10
Nodes (15): env(), skipif, slow, scripts/check_env.py: the environment-equals-lock check., The CI condition, reproduced: the real uv, FORCE_COLOR=1, the real lock., Run from the main environment, `--scrub` must fail: no flair, transformers 5,…, The live venv equals the lock (skipped in fast mode; CI and --full run it)., FORCE_COLOR=1 made uv colour its "# via" lines, and they failed to parse. (+7 more)

### Community 59 - "test_config_invariants.py"
Cohesion: 0.11
Nodes (16): counts(), fixture, parametrize, slow, Configuration invariants: the living spec of what every yaml guarantees., One factor at a time: each screen yaml equals the base in every field but its…, S5 (legacy recipe): the <= 1% target did not hold (+2.36%); it is reported as a…, Structural parity with the reference: same modules, same shapes, to the… (+8 more)

### Community 60 - "scrub_ler.py"
Cohesion: 0.06
Nodes (54): hmac, Entity, entity_hash(), iter_part(), main(), part_name(), _pattern(), placeholders() (+46 more)

### Community 61 - "inspect_checkpoint.py"
Cohesion: 0.33
Nodes (8): _f(), inspect(), main(), optimizer_names(), part_of(), Module, Per-parameter weights and AdamW state of a training checkpoint.…, Parameter names in the order AdamW's state indexes them (group by group).

### Community 62 - "changelog.py"
Cohesion: 0.11
Nodes (21): Change, changes_since(), main(), Path, The currency changelog: which provisions changed since a date. A provision's…, write_changelog(), docket(), ``7Ob143/26v`` -> ``7 Ob 143/26v`` (the spaced form courts print). (+13 more)

### Community 63 - "code"
Cohesion: 0.09
Nodes (25): code(), directives(), A `--gpus` directive is not overridden by an environment variable: a multi-GPU…, The two shell lookups in fetch_hf.sh must find the constants the loaders use., The command as documented, the same check on the legal base, the CUDA smoke., The 4-GPU header, and the body it delegates to passes last.ckpt as…, ARM is resolved inside the job by `screen_arms.py env`, EXPERIMENT from the…, `#SBATCH --key=value` lines before the first command, trailing comments dropped. (+17 more)

### Community 64 - "HybridBlock"
Cohesion: 0.11
Nodes (16): HybridBlock, Tensor, (batch, seq_len, dim) -> (batch, seq_len, dim). Argument order matches…, Per-layer decode state, or None if this mixer has no step (Mamba-1)., One token through mixer + MLP with residuals, mirroring ``forward``'s topology…, One decoder block: ``x + mixer(norm1(x))``, then the MLP (pre-norm, or…, Under `hybrid`, block >= 1 computes norm2(x + mlp(x)); block 0 stays pre-norm., The reference filtered on unprefixed names, so every `mlstm_*` option fell back… (+8 more)

### Community 65 - "test_renderer.py"
Cohesion: 0.08
Nodes (49): _check_sentences(), _labels(), pointer_text(), Segment, ValueError, The renderer: the model's pointer answer as the text a reader sees. The answer…, The rendered answer of parsed segments against the retrieved passages (prompt…, The rendered answer of the model's raw answer text (``<|endoftext|>`` allowed… (+41 more)

### Community 66 - "paragraphs"
Cohesion: 0.33
Nodes (6): is_heading(), paragraphs(), A page's footnotes interrupt the paragraph that runs on to the next page ("...…, The paper's paragraphs as ``(is_heading, text)``, cleaned of the PDF's layout…, _rejoin_across_footnotes(), test_dip_paragraph_rules()

### Community 67 - "test_decoding.py"
Cohesion: 0.08
Nodes (33): generated_tokens(), The answer part of a decoded row: after the prompt, up to and including the…, Bigram, cached_lm(), log_table(), Decoding. Cached decode must be an equivalence with full recomputation. The…, All-or-nothing: Mamba-1 has no step(), so the legacy stack cannot decode with a…, The KV-cache step reproduces the full forward, from an empty cache and after a… (+25 more)

### Community 68 - "slab_ce_kl"
Cohesion: 0.12
Nodes (17): boundary_masked_labels(), Next-token targets ``labels[:, 1:]`` with every cross-document prediction set…, Summed CE and summed masked KL for one slab (recomputed in the backward pass)., Slab-wise ``(1 - alpha) * CE + alpha * T^2 * KL`` without (B, L, V) logits.…, slab_ce_kl(), _slab_terms(), Pretraining Lightning module: slab-wise CE, optional logit KD, MTP, and a real…, measure_point() (+9 more)

### Community 69 - "test_code_hygiene.py"
Cohesion: 0.18
Nodes (17): _code_files(), git_calls_in_python(), git_calls_in_shell(), _is_sys_path_edit(), _names_git(), AST, parametrize, Path (+9 more)

### Community 70 - "bootstrap_compare.py"
Cohesion: 0.10
Nodes (27): corpus_bleu(), _lcs_length(), _ngram_counts(), Counter, Sentence ROUGE-L and corpus BLEU, stdlib only. Ported verbatim from the…, Sentence-level ROUGE-L F-measure (Lin, 2004)., Corpus-level BLEU-N with brevity penalty, one reference per hypothesis., rouge_l_score() (+19 more)

### Community 71 - "NLI verifier threshold on 100 hand-built statute pairs"
Cohesion: 0.22
Nodes (8): NLI verifier threshold on 100 hand-built statute pairs, Not licensed, Open limitations, Prediction and outcome, Reproduction, Results, Summary, What this licenses

### Community 72 - "Architecture"
Cohesion: 0.22
Nodes (9): Answer pipeline, Architecture, Data pipeline, Model, Renderer, Retrieval, Tokenizer and pointer vocabulary, Training (+1 more)

### Community 73 - "screen_metrics.py"
Cohesion: 0.39
Nodes (8): _fmt(), latest_version(), main(), Path, Validation and training curves of the screen runs, read from their TensorBoard…, ``<run>/logs/<exp>/version_<n>`` with the highest n that holds an events file., read_scalars(), report()

### Community 74 - "ParquetRows"
Cohesion: 0.29
Nodes (4): ParquetRows, Dataset, Tensor, Rows of a packed parquet shard: ``input_ids`` (and ``doc_ids`` when present),…

### Community 75 - "evalsets/base.py"
Cohesion: 0.08
Nodes (54): gzip, html, cli(), EvalSetInfo, Passage, Path, QAItem, QASet (+46 more)

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

### Community 82 - "SyntheticPackedDataset"
Cohesion: 0.11
Nodes (23): Dataset, Tensor, ``num_rows`` rows of ``row_len`` random tokens, each packed from several…, SyntheticPackedDataset, _cpu_trainer(), _diag_fit(), Stage B ends at step 18,037 with checkpoints every 1,000: without…, STEPSTATS: one timing per optimizer step, not per micro-batch; a line every… (+15 more)

### Community 83 - "run_metadata.py"
Cohesion: 0.16
Nodes (16): Any, DictConfig, Path, ``run_metadata.json``: what produced a checkpoint, written at the start of…, sha256 over the sorted (relative path, file sha256) pairs of the source tree., Contents of ``.sync_stamp`` (date + tree hash of the last sync to the cluster),…, The checked-out commit id, read from ``.git/HEAD`` and its ref file; "" if…, Persist ``<output_dir>/run_metadata.json`` and return its path. (+8 more)

### Community 84 - "DPOLightningModule"
Cohesion: 0.11
Nodes (22): answer_logps(), collate_pairs(), dpo_loss(), DPOLightningModule, kto_loss(), _pad(), Tensor, (prompt, answer) rows as right-padded ``input_ids`` and ``labels`` (-100 off… (+14 more)

### Community 85 - "cumulative_angles"
Cohesion: 0.16
Nodes (15): SSD scan, plus the trapezoidal term and the complex-state rotation when…, apply_rotary(), cumulative_angles(), cumulative_angles_fp64(), Tensor, Data-dependent rotary embeddings for the complex-valued SSM. Mamba-3 Sec 3.2…, Theta_t = sum_{i<=t} dt_i * theta_i, accumulated in float64 and wrapped to [0,…, ``cumulative_angles`` before the final cast: float64, wrapped to [0, 2pi).… (+7 more)

### Community 86 - "test_repo_hygiene.py"
Cohesion: 0.23
Nodes (12): make_tree(), parametrize, Path, scripts/check_repo_hygiene.py on synthetic trees., test_a_missing_critical_file_is_reported(), test_an_ignored_analysis_document_is_reported(), test_case_collisions_are_caught(), test_clean_tree_passes() (+4 more)

### Community 87 - "German-law pretraining corpus: tokens, deduplication, scrub"
Cohesion: 0.15
Nodes (12): Budget rule, Deduplication (drop within a source, keep and link across), Discussion, Efficiency, Estimate vs exact, German-law pretraining corpus: tokens, deduplication, scrub, Limitations, Reproduction (+4 more)

### Community 88 - "conftest.py"
Cohesion: 0.13
Nodes (18): reference, load_script(), fixture, pytest_collection_modifyitems(), Shared test configuration. Markers are declared in pyproject.toml under…, Every test starts from the same RNG state, so failures reproduce., Import ``scripts/<name>.py`` as a module without touching sys.path., Fixture form of ``load_script``: ``script("evaluate_lm").main([...])``. (+10 more)

### Community 89 - "test_sync_scripts.py"
Cohesion: 0.07
Nodes (39): CompletedProcess, SimpleNamespace, stat, `--docs-per-batch` puts the sentences of several documents into ONE predict…, test_scrub_worker_tags_a_batch_of_documents_in_one_predict_call(), __init__(), Two runs of one system and seed (SFT and DPO) get their own outputs, so the…, counting() (+31 more)

### Community 90 - "PretrainLightningModule"
Cohesion: 0.15
Nodes (9): PretrainLightningModule, Tensor, Clip and log the gradient norm; a non-finite norm clears the gradients (step…, Args: those of ``HybridLightningModule`` plus teacher: optional frozen Hugging…, slow, The entry point end to end on CPU: 2 steps with last.ckpt written every step,…, The late-25% PPL: a predicted token counts as late when at least late_from…, test_late_context_ppl_counts_document_context_and_matches_training_val_ce() (+1 more)

### Community 91 - "probe_ckpt_size.sh"
Cohesion: 0.15
Nodes (12): BATCH_SIZE, CKPT_EVERY, DATASET_CONFIG, DISTILL_CFG, EXPERIMENT, EXTRA_OVERRIDES, MAX_STEPS, MODEL_CONFIG (+4 more)

### Community 92 - "_CaptureLogger"
Cohesion: 0.15
Nodes (5): _CaptureLogger, The answer loss never uses the MTP head; DDP would wait for gradients it never…, Keeps every ``log_metrics`` call in memory: ``rows = [(step, {tag: value})]``., test_sft_freezes_an_mtp_head_outside_its_loss(), __init__()

### Community 93 - "check_repo_hygiene.py"
Cohesion: 0.33
Nodes (11): check_critical(), check_files(), list_files(), load_ignore(), main(), Path, Critical files must exist; locally they must also survive ``.gitignore``. In…, Repository hygiene: the tree that would be (or was) pushed is clean. Two modes,… (+3 more)

### Community 94 - "screen_table.py"
Cohesion: 0.33
Nodes (11): apply_rule(), arm_means(), build_rows(), _f(), main(), parse_run(), The architecture screen's seed table, per-arm means and the decision rule.…, read_eval() (+3 more)

### Community 95 - "Profile ladder, chunk size and distillation memory"
Cohesion: 0.17
Nodes (11): Chunk size (training, forward+backward, slab loss, packed, doc_len 1,000, L=4,096), Discussion, Distillation memory probe, Hypotheses and criteria, Limitations, Profile ladder, chunk size and distillation memory, Reproduction, Results (+3 more)

### Community 96 - "build_shards.py"
Cohesion: 0.18
Nodes (15): itertools, build_source_shards_stream(), The validation document ids: ``max(ceil(fraction * n), minimum)`` of them, by…, ``build_source_shards`` for a source that does not fit in memory: ONE pass over…, split_by_document(), build(), iter_documents(), main() (+7 more)

### Community 97 - "Architecture screen: ablations of the hybrid backbone"
Cohesion: 0.17
Nodes (11): Architecture screen: ablations of the hybrid backbone, Discussion, Efficiency, Hypotheses and selection rule, Paired comparisons, Per-seed calls (where the seeds disagree), Reproduction, Results (+3 more)

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
Cohesion: 0.17
Nodes (10): Cluster jobs, Continuous integration, Contributing, Dependencies, Lint and style, Repository hygiene, Setup, Tests (+2 more)

### Community 103 - "tfla_interface.py"
Cohesion: 0.20
Nodes (13): forward_fill_from_starts(), pad_segments(), device, Tensor, Document segments inside packed rows, shared by the SSD and TFLA kernels. A…, (batch, seqlen) bool, True where a new document begins; position 0 is False., Monotone segment index per position, (batch, seqlen) long; all zeros without…, Extend ``seg`` by ``pad`` positions that join the LAST real segment. Padding… (+5 more)

### Community 104 - "test_ci_workflow.py"
Cohesion: 0.16
Nodes (12): fixture, The CI workflow and the local validation script stay in step. PyYAML parses the…, Fast excludes slow/cuda/multigpu/network; --full and --ci exclude…, steps_text(), test_every_installing_job_uses_the_lock_and_never_pip(), test_hygiene_runs_on_a_pristine_checkout(), test_lint_runs_every_static_tool(), test_package_job_imports_the_wheel_outside_the_checkout() (+4 more)

### Community 105 - "_oldp_docs"
Cohesion: 0.15
Nodes (15): _fineweb_docs(), _oldp_docs(), Signatures (no text, 512 bytes of MinHash) computed in worker processes from a…, The documents back out of rows: real tokens concatenated and split after each…, The real tokenizer: every document decodes back exactly from 4,096-token rows., 0.1% of the documents or the minimum (2,000 by default), by seeded hash,…, At scale: one pass, each document tokenised once, packed at two row lengths…, The pack step (scripts/build_shards.py): a whole scrubbed file and a source in… (+7 more)

### Community 106 - "SourceShards"
Cohesion: 0.21
Nodes (8): Path, Where ``source``'s ``row_len``-token rows come from, and how many sub-rows a…, The rows of one source's split, read shard by shard on first use. With…, shard_directory(), SourceShards, flush(), The screen arms train on 2,048-token rows and the shard builder packs 4,096 and…, test_mixture_cuts_stored_rows_for_a_shorter_row_len()

### Community 107 - "fused_retriever"
Cohesion: 0.21
Nodes (13): build_sets(), fused_retriever(), retrieve(), Encoder, Path, ``(sets, counts)``: the prompt sets of every item with a gold passage (see the…, The index's ranking: BM25, fused with ``dense/`` when it exists and ``encode``…, _sets_of() (+5 more)

### Community 108 - "equivalence.sh"
Cohesion: 0.22
Nodes (9): CUDA_LAUNCH_BLOCKING, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, equivalence.sh script, step() (+1 more)

### Community 109 - "kd_memory_probe.sh"
Cohesion: 0.22
Nodes (9): DATASET_CONFIG, GRAD_CKPT, MAX_STEPS, MODEL_CONFIG, NUM_GPUS, probe(), SAVE_TOP_K, kd_memory_probe.sh script (+1 more)

### Community 110 - "scan_interface.py"
Cohesion: 0.29
Nodes (9): Mamba-1 selective scan (legacy ablation only)., Tensor, Mamba-1 selective scan (legacy incumbent; used only by the…, Exact selective scan -- the recurrence the Mamba-1 block is specified to…, Exact chunked selective scan -- no division, no clamp. Flips which axis is…, Public selective-scan entry point, computed in fp32 and cast back.…, selective_scan(), selective_scan_exact() (+1 more)

### Community 111 - "judge.py"
Cohesion: 0.11
Nodes (41): add_judge_args(), answer_text(), correctness_prompt(), faithfulness_prompt(), judge_from_args(), judge_prompts(), judge_record(), _listed() (+33 more)

### Community 112 - "_doc"
Cohesion: 0.08
Nodes (31): admits(), build_corpus(), Corpus assembly gate: which documents may enter which training arm. Two arms:…, Whether ``doc`` (already licence-checked) belongs in ``arm``., Yield the documents of ``arm``; raise ``LicenceError`` on the first licence…, check_document(), is_commercial_safe(), Licence (+23 more)

### Community 113 - "Judge calibration and validation on gold labels"
Cohesion: 0.13
Nodes (14): 1. Validation half, prompt version 2 (accuracy / κ / precision / recall; unparsed in brackets), 2. Accuracy by variant, validation half, 3. Calibration half, prompt version 2, with the check judges (accuracy / κ), Bar and outcome, Judge calibration and validation on gold labels, Not licensed, Open limitations, Reproduction (+6 more)

### Community 114 - "AnswerRecord"
Cohesion: 0.09
Nodes (33): as_is_sets(), A QA item from an SFT row (``id``, ``chunk_ids`` in prompt order, ``text`` as…, Each SFT row's own prompt as a record without system or answer (arm ``as_is``):…, read_chunks(), sft_item(), AnswerRecord, Path, One system's answer to one evaluation question: the JSONL record every… (+25 more)

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

### Community 121 - "_text"
Cohesion: 0.50
Nodes (4): Element, Paragraph text: ``<p>``s become lines, ``<br/>`` breaks, whitespace normalised., _text(), walk()

### Community 122 - "Judge"
Cohesion: 0.10
Nodes (21): Complete, average_precision(), first_json_object(), Judge, _parse(), Path, The first ``{...}`` in ``text`` that parses as a JSON object (code fences and…, A ``Complete`` behind the reply cache; ``calls`` counts the completions… (+13 more)

### Community 123 - "Backbone results: the hybrid and a Transformer baseline at 8,192 tokens"
Cohesion: 0.17
Nodes (11): Backbone results: the hybrid and a Transformer baseline at 8,192 tokens, Discussion, Handoff, Learning-rate sensitivity at screen scale, Limitations, Predictions, Probes on the stage-B checkpoints, by context length, Reproduction (+3 more)

### Community 124 - "test_gpu.py"
Cohesion: 0.15
Nodes (19): _packed_doc_ids(), parametrize, Tensor, CUDA checks. Marked ``cuda``: they collect everywhere and skip, with the…, On the GPU, a 128-chunk whose log-decay overflows one factorisation (forget…, CUDA variant of the compiled legal base check: the compiled legal base equals…, Under an outer torch.compile, a block mask traced inside the graph was wrong…, CUDA variant of tests/test_layers.py::test_flex_block_mask_equals_dense: the… (+11 more)

### Community 125 - "LayerSplit"
Cohesion: 0.25
Nodes (6): LayerSplit, hook(), hook(), _mark(), Start/stop marker: a CUDA event on GPU, a wall clock on CPU., Per-`HybridBlock` timing, aggregated by mixer type. Markers are queued on the…

### Community 126 - "build_corpus_array.sh"
Cohesion: 0.33
Nodes (5): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_corpus_array.sh script

### Community 127 - "pack_corpus.sh"
Cohesion: 0.33
Nodes (5): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, pack_corpus.sh script, TOKENIZERS_PARALLELISM

### Community 128 - "_s1_fingerprint"
Cohesion: 0.33
Nodes (4): The eval guard reads manifests of the full sets, fetched online into a…, The ARCH line of the real base model (S1), built on meta from the screen's own…, _s1_fingerprint(), test_build_sft_data_guards_with_complete_manifests_fetched_once()

### Community 129 - "scrub_ner_worker.py"
Cohesion: 0.22
Nodes (12): cap(), chunks(), default_cache_dir(), load_tagger(), main(), flush(), Entities of every document, the sentences of all ``docs`` tagged in ONE…, German legal-entity recognition for the LER scrub. Runs in the scrub… (+4 more)

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

### Community 137 - ".__init__"
Cohesion: 0.22
Nodes (7): LayerType, create_hybrid_blocks(), ``num_layers`` blocks cycling through ``layer_pattern``; block 0 is marked…, HybridEmbedding, One-line summary of what was actually built; see…, Token embedding followed by dropout. There is no positional embedding: position…, ModuleList

### Community 138 - "test_diag_spike_reruns_the_arm_observed_in_its_own_directory"
Cohesion: 0.40
Nodes (5): _compose(), The fake interpreter recorded `distill=null` happily; real Hydra refused it…, Loss-spike rerun: S1-s42 with the screen settings, seed and schedule unchanged…, test_diag_spike_reruns_the_arm_observed_in_its_own_directory(), test_every_train_argument_list_composes_under_real_hydra()

### Community 140 - ".forward"
Cohesion: 0.38
Nodes (4): Tensor, (batch, seq_len, dim) -> (batch, seq_len, dim); ``doc_ids`` resets state per…, Non-fast-path scan: the same implementation as ``selective_scan`` with chunk…, Per-(row, segment) wrapper that resets the SSM state at document boundaries.

### Community 142 - "test_reference_parity.py"
Cohesion: 0.20
Nodes (14): inspect, _block_defaults(), block_params(), _config_defaults(), load(), model_params(), _per_document(), parametrize (+6 more)

### Community 162 - "test_sft_requeue_resumes_with_every_validation_ranked"
Cohesion: 0.10
Nodes (18): _compose_sft(), train_sft.py starts the hybrid from a pretraining checkpoint and trains on…, Two LoRA runs with the same seed start from the same adapters, whatever the RNG…, A pretraining checkpoint loads exactly; one missing or unexpected key stops the…, A scripted val/loss of 3, 1, 2, 2.5 over 4 steps, validating after each: the…, A 6-step run validating every 2 steps with a scripted val/loss of 2, 1, 3 is…, Rows reach 8,192 tokens, and stage-B pretraining ran micro-batch 4 at that…, A scripted val/loss of 3, 1, 2, 2.5 over 4 steps, validating after each: the… (+10 more)

### Community 167 - "pathlib"
Cohesion: 0.07
Nodes (46): json, Probes scored on any checkpoint: MQAR, statute recall, multi-hop., Cross-source copies of a document, from the pack's dedup log. A document whose…, detect_prefix(), _first_segments(), infer_architecture(), infer_layer_types(), InferredArchitecture (+38 more)

### Community 168 - "StopAtStepCallback"
Cohesion: 0.09
Nodes (15): End training after optimizer step ``stop_at_step`` while the LR schedule keeps…, StopAtStepCallback, _ddp_fit(), _IndexRows, _RankRows, Writes the row indices each DDP rank trains on to <dir>/rank<r>.txt., Under DDP (two CPU ranks, gloo) each rank reads its own rows, r, r + 2, ...;…, Row i carries its index, so a run's data order can be read back. (+7 more)

### Community 169 - "Retrieval on GerLeRB and GerDaLIR"
Cohesion: 0.18
Nodes (10): GerDaLIR (12,234 queries), GerLeRB (367 queries), Not licensed, Open limitations, Prediction and outcome, Reproduction, Results, Retrieval on GerLeRB and GerDaLIR (+2 more)

### Community 170 - "metrics.py"
Cohesion: 0.27
Nodes (9): compute_accuracy(), compute_bits_per_token(), compute_mqar_accuracy(), Tensor, Training and evaluation metrics (ported from the reference)., Cross-entropy in bits per token (``loss / ln 2``). The reference named this…, Token-level argmax accuracy over positions whose label is not ``ignore_index``., Accuracy at the query positions of a multi-query associative-recall sequence. (+1 more)

### Community 171 - "test_hf_sampling_through_its_cache_matches_the_uncached_reference"
Cohesion: 0.67
Nodes (3): Same logits, same generator: the KV-cached sampler and the uncached ``sample``…, test_hf_sampling_through_its_cache_matches_the_uncached_reference(), sample()

### Community 172 - "Post-training results: supervised fine-tuning"
Cohesion: 0.22
Nodes (8): Citation, refusal and truncation on the dev split, Learning-rate selection, Limitations, Post-training results: supervised fine-tuning, Reproduction, Summary, Three seeds at the chosen learning rate, Throughput

### Community 173 - "slab_cross_entropy"
Cohesion: 0.13
Nodes (13): HeadFn, MTPHead, MTPModule, Module, Tensor, One prediction depth: two norms, a 2D -> D projection, and one ``HybridBlock``., ``depths`` sequential ``MTPModule``s; ``blocks`` are built by the model's block…, Mean over depths of each depth's boundary-masked cross-entropy. Args: hidden:… (+5 more)

### Community 174 - "README.md"
Cohesion: 0.28
Nodes (4): Corpus licence register, Evaluation sets (never trained on), Models the pipeline runs, Training sources (collectors)

### Community 175 - ".from_hydra"
Cohesion: 0.13
Nodes (16): hydra, hydra_core_global_hydra, Build a config from a Hydra ``cfg.model`` node, keeping every field the…, check_model(), compose_model(), main(), model_names(), validate.sh gate 3: every model yaml composes through Hydra and pins what it… (+8 more)

### Community 176 - "CausalLMAdapter"
Cohesion: 0.12
Nodes (12): CausalLMAdapter, Module, Tensor, A Hugging Face causal LM behind the hybrid's interfaces: the…, Args: those of ``HybridLightningModule`` (``learning_rate`` is the plain value…, ``(mean CE over the supervised targets, their count)``; position t predicts…, SFTLightningModule, collate right-pads (labels -100), and the SFT loss is the CE of the answer… (+4 more)

### Community 177 - "SignalCheckpointCallback"
Cohesion: 0.16
Nodes (6): FinalCheckpointCallback, Callback, Write ``<checkpoint_dir>/last.ckpt`` when training ends normally.…, Write ``interrupt.ckpt`` when the process is signalled; a no-op after the first…, SignalCheckpointCallback, test_signal_callback_saves_once()

### Community 178 - "parse_citation"
Cohesion: 0.05
Nodes (56): gold_nll(), no_grad, Mean per-token NLL of ``target_ids`` following ``prompt_ids`` (natural log).…, generate(), MQARConfig, MQARItem, no_grad, Multi-query associative recall, MQAR, with the upstream reference's… (+48 more)

### Community 179 - "synth_qa.sh"
Cohesion: 0.20
Nodes (8): scripts_slurm_lib_vllm_server_sh, HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, synth_qa.sh script, HF_HOME, HF_HUB_OFFLINE, vllm_smoke.sh script

### Community 180 - "train_sft_4gpu.sh"
Cohesion: 0.22
Nodes (9): HF_DATASETS_CACHE, HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, opt(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, train_sft_4gpu.sh script (+1 more)

### Community 181 - "teacher_forward_packed"
Cohesion: 0.28
Nodes (9): no_grad, Tensor, ``(position_ids, attention_mask)`` for a Hugging Face decoder on packed rows.…, The teacher's final hidden states (after its final norm), (B, L, D_teacher),…, teacher_forward_packed(), teacher_inputs_packed(), A packed row reaches the teacher with per-document position ids and a block-…, test_teacher_position_ids_isolation() (+1 more)

### Community 182 - "test_tfla_wide_chunks_factorise_per_block_exactly"
Cohesion: 0.33
Nodes (7): 256 positions (two chunks of 128), two rows, boundaries inside and on chunk…, Trained forget gates widened a 128-chunk's log-decay past fp32 (half-range 43.2…, In fp32 the per-block path forms no inf, forward or backward (a masked inf…, test_tfla_per_block_path_is_finite_and_close_in_fp32(), test_tfla_wide_chunks_factorise_per_block_exactly(), _tfla_grads(), _wide_tfla_inputs()

### Community 183 - "_tfla_model"
Cohesion: 0.40
Nodes (5): The shipped configuration runs exact TFLA on its factorised path at init. With…, The record behind the fp32 range limit: forget bias 0 gives f = 0.5 and a half-…, test_tfla_fast_path_at_init(), test_tfla_reference_forget_bias_factorises_per_block(), _tfla_model()

### Community 184 - "train_dpo_4gpu.sh"
Cohesion: 0.29
Nodes (7): HF_DATASETS_OFFLINE, HF_HOME, HF_HUB_OFFLINE, opt(), PYTHONUNBUFFERED, PYTORCH_CUDA_ALLOC_CONF, train_dpo_4gpu.sh script

### Community 185 - "test_collect_offline.py"
Cohesion: 0.07
Nodes (31): api_key(), DIPCollector, ``types``: the Drucksache types, collected in turn, newest first;…, FakeQwen, FakeWeb, The collectors' and fetchers' network loops, offline (CI has no network). Each…, The OLDP API stopped paging past page 10, so the collector reads the bulk dump…, ``get``/``post`` answering from ``routes`` (regex on the URL -> function(url,… (+23 more)

### Community 186 - "test_param_bands"
Cohesion: 0.50
Nodes (3): Parameter counts of the legal model configs, The predicted counts and bands, and analysis/param_counts.md is current., test_param_bands()

### Community 187 - "lightning_module.py"
Cohesion: 0.13
Nodes (19): PyTorch Lightning wrapper for causal-LM training (ported base class of the…, Apply the beta2 schedule when WSD and ``beta2_schedule`` are both active., configure_optimizer(), get_parameter_groups(), Any, Module, Optimizer, Optimizer construction with weight-decay parameter groups (ported from the… (+11 more)

### Community 188 - "Document schema"
Cohesion: 0.50
Nodes (3): Citation ids, Document schema, Fields

### Community 189 - "._gates"
Cohesion: 0.27
Nodes (6): Tensor, Soft-capped gates: exp input gate (computed in fp32), sigmoid forget and output…, (batch, seq_len, dim) -> (batch, seq_len, dim). Args: x: input. cache: an EMPTY…, Advance one token. ``(batch, dim)`` or ``(batch, 1, dim)`` in, ``(batch, dim)``…, Soft cap via tanh: maps x to ``tanh(x / cap) * cap``, range (-cap, cap)., _tanh_soft_cap()

### Community 190 - "setup_arm_env.sh"
Cohesion: 0.33
Nodes (5): PYTHONUNBUFFERED, setup_arm_env.sh script, UV_CACHE_DIR, UV_LINK_MODE, UV_PROJECT_ENVIRONMENT

### Community 191 - "setup_vllm.sh"
Cohesion: 0.33
Nodes (5): PYTHONUNBUFFERED, setup_vllm.sh script, UV_CACHE_DIR, UV_LINK_MODE, UV_PROJECT_ENVIRONMENT

### Community 192 - "HybridLightningModule"
Cohesion: 0.16
Nodes (8): HybridLightningModule, Optimizer, Tensor, Token-weighted sums for the epoch's ``val/perplexity``. Logged per batch with…, Clip gradients and log the pre-clip norm (manual, so foreach AdamW and AMP…, A WSD scheduler with an absolute warmup; records the decay window for the beta2…, Training loop around a ``HybridLanguageModel``. Args: model: the model to…, compute_perplexity()

### Community 194 - "measure_fertility.py"
Cohesion: 0.29
Nodes (11): collect(), main(), measure(), n_words(), pooled(), Path, Tokenizer fertility on DACH legal text (results in…, Whole documents in order until ``words`` words, skipping any longer than… (+3 more)

### Community 195 - "build_dpo_pairs.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_dpo_pairs.sh script

### Community 196 - "build_sft_data.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, build_sft_data.sh script

### Community 197 - "eval_gold_passages.sh"
Cohesion: 0.40
Nodes (5): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, score(), eval_gold_passages.sh script

### Community 198 - "eval_system_ablation.sh"
Cohesion: 0.40
Nodes (4): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, eval_system_ablation.sh script

### Community 201 - "test_pipeline.py"
Cohesion: 0.07
Nodes (25): pytest, subprocess, tarfile, dists(), fixture, The built distributions contain the package and nothing else (CI `package`…, End-to-end answer pipeline on CPU: index -> retrieve -> prompt -> constrained…, With best-of-n's samples scripted (every token one the FSM allows), the… (+17 more)

## Knowledge Gaps
- **273 isolated node(s):** `lexhybrid-arm`, `lexhybrid-scrub`, `lexhybrid-vllm`, `lexhybrid`, `build_corpus_array.sh script` (+268 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1852 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **31 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HybridLanguageModel` connect `HybridLanguageModel` to `_s1_fingerprint`, `test_data.py`, `load_model_config`, `performance_profile.py`, `.__init__`, `Mamba3Block`, `test_models.py`, `test_reference_parity.py`, `train_pretrain.py`, `evaluate_val_ppl.py`, `torch`, `test_scripts.py`, `test_retrieval.py`, `train_dpo.py`, `test_numerics_fp64.py`, `HybridConfig`, `train_sft.py`, `pathlib`, `datasets.py`, `test_dpo.py`, `slab_cross_entropy`, `.from_hydra`, `parse_citation`, `test_training.py`, `filter_logits`, `os`, `_tfla_model`, `test_slurm_wrappers.py`, `lightning_module.py`, `test_config_invariants.py`, `inspect_checkpoint.py`, `HybridBlock`, `HybridLightningModule`, `test_decoding.py`, `slab_ce_kl`, `Architecture`, `test_pipeline.py`, `SyntheticPackedDataset`, `DPOLightningModule`, `PretrainLightningModule`, `test_gpu.py`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `Document` connect `Document` to `ris.py`, `packing.py`, `test_data.py`, `collectors/base.py`, `index.py`, `eurlex.py`, `eval_retrieval.py`, `test_retrieval.py`, `fedlex.py`, `dedup.py`, `pathlib`, `datasets.py`, `parse_citation`, `test_verifier.py`, `load_tokenizer`, `test_collect_offline.py`, `scrub_ler.py`, `Document schema`, `changelog.py`, `Architecture`, `build_shards.py`, `_doc`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `HybridConfig` connect `HybridConfig` to `test_data.py`, `HybridLanguageModel`, `test_layers.py`, `load_model_config`, `.__init__`, `test_models.py`, `test_reference_parity.py`, `train_pretrain.py`, `torch`, `test_scripts.py`, `test_retrieval.py`, `train_dpo.py`, `test_numerics_fp64.py`, `train_sft.py`, `pathlib`, `datasets.py`, `test_dpo.py`, `slab_cross_entropy`, `.from_hydra`, `parse_citation`, `test_training.py`, `_tfla_model`, `test_config_invariants.py`, `lightning_module.py`, `test_decoding.py`, `slab_ce_kl`, `Architecture`, `test_pipeline.py`, `SyntheticPackedDataset`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 57 inferred relationships involving `Document` (e.g. with `Data pipeline` and `Document schema`) actually correct?**
  _`Document` has 57 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `HybridLanguageModel` (e.g. with `Model` and `__getattr__()`) actually correct?**
  _`HybridLanguageModel` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 61 inferred relationships involving `Chunk` (e.g. with `answer_words()` and `cited_passages()`) actually correct?**
  _`Chunk` has 61 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `HybridConfig` (e.g. with `Model` and `load_model_config()`) actually correct?**
  _`HybridConfig` has 6 INFERRED edges - model-reasoned connections that need verification._