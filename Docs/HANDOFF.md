# Handoff — branch `p4-offline-backlog`

> **Updated:** 2026-10-27. **Read this right after** `python3 scripts/plan_state.py resume`. Where the two disagree, the state file (`legal_build_state.json`, via `resume`) wins: it holds the live job ids, ticks and next action. **Keep this file current:** at the end of any session that changes the goal, a decision, what is done or the next step, update the dated sections below and commit them with the work (the user decides every commit and push).

## 1. What we are building

A citation-faithful, retrieval-gated legal language model for the **German market** (`lexhybrid`): a corrected Mamba-3 + mLSTM + sparse-attention decoder, trained from scratch with logit distillation from Qwen3-8B-Base, that cites by pointer into retrieved passages and is gated by a verifier. The plan of record is `LEGAL_BUILD_PLAN.md` (phases P0–P11), executed one phase at a time on the HPI aisc cluster (H100s) from this Mac.

## 2. What this branch is for

Opened for the P4 work (cluster bring-up, corpus at scale, profiling), it now carries **P4 and P5 complete** (the pre-registered screen chose the backbone) and the **start of P6** (full pretraining) and P7. It has not been merged to `main`. Merging is the user's call; P5-Z has passed.

## 3. Where things stand (2026-10-27)

| phase | state |
|---|---|
| P0–P4 | done (P4-Z passed 2026-10-03: `validate.sh --full` exit 0) |
| P5 screen | **done** (P5-Z 2026-10-20, `validate.sh --full` exit 0). S1 `hybrid_legal_base` with AdamW β2 0.95 advances (`decisions.backbone`) |
| **P6 pretraining** | **10/15** (A–J). **All training is finished**: four stage-A and four stage-B runs, no restarts, every stage B at step 18,037. Left: P6-K (8,192 val for s43_B and research_B), P6-L (handoff for s43_B and research_B), P6-M (probes at 8,192), then P6-N and P6-Z. K, L and M need the cluster; the user holds those until the queue frees |
| **P7** (current phase) | **9/14** (A, B, C1, C, E, F, G, H, J). Fusion and rerank, filters, the index registry and resync stub, the renderer (every source of a copied passage named, `decisions.cite_every_copy`), and the pointer, quote, NLI, party and leak checks, all tested and reviewed. Left: D and K (GPU), I (NLI calibration), L (answer CLI), Z |
| P8–P11 | not started |

**Open jobs:** none (2026-10-27). P6 results so far (commercial-safe validation; late-25% = tokens with at least 75% of the row length of their own document before them):

| checkpoint | val PPL @4,096 | late-25% @4,096 | val PPL @8,192 | late-25% @8,192 |
|---|---|---|---|---|
| hybrid s42 A | 3.9775 | 3.7814 | 3.8638 | 3.7886 |
| hybrid s42 **B** | | | **3.7861** | **3.6533** (−3.57%) |
| hybrid s43 A | 3.9724 | 3.7727 | 3.8635 | 3.8107 |
| Transformer s42 A | 4.0236 | 3.8240 | 8.2270 | 119.93 (no extrapolation past 4,096) |
| Transformer s42 **B** | | | **3.8337** | **3.7156** |
| research hybrid s42 A | 4.0365 | 3.8441 | 3.9206 | 3.8682 |

Jobs: training 2724164–67 (A) and 2764152–55 (B); evaluation 2764157, 2792760, 2792761; reader 2792759; handoff 2792762 (s42_B and Transformer_B PASS).

**Decisions taken on this branch** (all in `decisions.*` of the state file):

| decision | value | when / who |
|---|---|---|
| `teacher` | Qwen3-8B-Base with gradient checkpointing (43.18 GB, 6.10 s/step at 4,096 on 4×H100) | 2026-10-01, user |
| `market` | Germany only: ris (AT), fedlex and bger (CH), and multilegalpile's Swiss rows out | 2026-10-02, user |
| `eurlex_training` | eurlex out of training (565 docs cannot give the ≥ 2,000-doc validation split); kept for retrieval | 2026-10-03, user |
| `p6_token_budget` | 5 B tokens = min(5 B, 6.09 B commercial-safe) | 2026-10-03, rule |
| `tfla_overflow` | exact mLSTM kernel keeps chunk 128; a too-wide chunk is factorised per row block | 2026-10-09, user |
| `p5e_submit_before_p5d` | P5-E arms submitted before S1 finished, with no dependency; if S1 fails on an infrastructure error, `scancel 2629409 2629410`, fix, resubmit S1 and P5-E | 2026-10-09, user |
| `p5_divergence` | S1-s42 and S3-s43 diverged: cancel S5, re-run S1-s42 once with diagnostics (checkpoints every 250 steps) to find the cause, then apply one stability fix to every arm and re-run the whole screen (about 150 GPU-hours) | 2026-10-11, user |
| `p5_fix` | Screen attempt 2: AdamW β2 0.95 in every legal model yaml (`adam_beta2`); S6 loads no teacher when alpha is 0; runs named `screen2_*` | 2026-10-11, user |
| `late25_ppl` | Late-25% PPL counts predicted tokens with at least 1,536 tokens of their own document before them in the row (`late_doc`); the row-position variant (`late_row`) is reported beside it | 2026-10-13, user (the recommended option) |
| `p5_rule_scores` | Statute recall by exact match (char-F1 beside it), multi-hop by exact match; "lose by more than the bar" means more than 0.10 on the 0–1 scale; fixed before any probe result | 2026-10-13, Claude, recorded |
| `backbone` | S1 = `hybrid_legal_base` with AdamW β2 0.95, by the P5 rule (bar 0.10 PPL) | 2026-10-20, rule |
| `nc_arm` | yes: a research-arm hybrid also trains (P6-F) | 2026-10-20, user |
| `p6_source_weights` | Square-root sampling within legal, 4-pass cap. Weights gii 138.0, rii 699.7, oldp 1809.2, dip 853.1, multilegalpile 1216.8. Commercial-safe passes: gii 4.00, rii 1.53, oldp 0.59, dip 1.26 | 2026-10-20, user (the recommended option) |
| `resume_data` | The training loader reads on from the restored batch count (`ScheduleSampler`); `trainer.stop_at_step`; `INIT_FROM` for stage B | 2026-10-20 |
| `p7_rebuild` | Rebuild the P7 chunks and BM25 index at P7's start (gii ids and citation grammar changed) | 2026-10-20, user (the recommended option) |
| `p6_probe_scores` | The P6 exact-match prediction is unchanged. Char-F1 and a gold-answer NLL are pre-registered as secondary scores before any P6 result, and the NLL scorer must exist before P6-M | 2026-10-20, user (the recommended option) |

## 4. What was done (with the numbers' sources)

- **Corpus** (`analysis/corpus_manifest.md`, pack job 2599735): 6 packed sources at 4,096 and 8,192 tokens. Commercial-safe **6,092,655,089** training tokens (gii, rii, oldp, dip, fineweb2_de); research arm **7,474,326,564** (adds multilegalpile's German rows). Dedup: 30,597 dropped within a source, 241,242 cross-source copies kept and linked. Scrub: 15,960,418 names replaced, about 83 H100-hours.
- **Fixes found at corpus scale** (each with a test): scrub crashed on whitespace-only entity spans and OOM'd on 33k-character DIP lines (4,096-character pieces); 137 gii ids named several provisions (articles that number their own §§): ids, citations (`BGBEG Art. 229 §6`) and URLs (`art_229__6.html`) are now article-aware; dedup refuses repeated ids up front; the training mixture listed sources without shards.
- **Profiling and the KD probe** (`analysis/profile_results.md`): chunk 128 kept; the teacher was decided by the P4-U rule.
- **Screen attempt 1** (jobs 2626202, 2629409; `analysis/screen_results.md` §3.1 and §7; val PPL = exp(val loss) at step 11,999, read by `scripts/screen_metrics.py`, jobs 2636358 and 2636379): S0 6.196/6.226, S1 **1,184.8**/5.563, S2 5.571/5.481, S3 5.464/**1,186.4**, S4 5.473/5.494, S6 5.270/5.280 (seed 42/43). The two bold runs collapsed after one gradient spike: S1-s42 at steps 3,500–3,749 (grad norm up to 93), S3-s43 at 6,250–6,499 (up to 542), then CE about 7.4, roughly what a unigram model scores, for the rest of the run. The S1 bar would be max(2 × 833.8, 0.10) = 1,667.6 PPL. Spikes that recovered occur in every KD run, against at most 3.9 in the no-KD S6 runs (a hint, not a finding). A run takes 10–11 h at 2.73–3.14 s per optimizer step, with peak 41.4–41.7 GB.
- **Why attempt 1 collapsed** (`analysis/screen_results.md` §7):
  - **Abrupt:** each collapse happened within 25 steps of one pre-clip gradient norm of 93 or 542 (jobs 2637389, 2637390).
  - **Batch-triggered, KD-amplified:** spikes land on the same steps across arms (job 2637413). No-KD runs peak at 3.9; KD runs reach 10–75.
  - **Unrecoverable:** afterwards the lower blocks got almost no gradient. AdamW's update size fell about tenfold per block going down, against about 0.2 everywhere in the healthy runs (job 2637405, `scripts/inspect_checkpoint.py`).
  - **Seeds:** both seeds share the data order (`dataset.seed: 0`).
  - **Fixed alongside:** `val/perplexity` is now logged as exp(token-weighted epoch CE).
- **Screen attempt 2** (`analysis/screen_results.md` §3.1). val PPL s42/s43:

  | arm | s42 | s43 |
  |---|---|---|
  | S0 | 6.024 | 6.043 |
  | S1 | 5.363 | 5.353 |
  | S2 | 5.361 | 5.326 |
  | S3 | 5.292 | 5.309 |
  | S4 | 5.359 | 5.357 |
  | S6 | 5.418 | 5.397 |

  There was no collapse; spikes up to 50.8 all recovered. P5-D: SD 0.0074, so **bar = 0.10 PPL** (the floor binds). S6 without the teacher runs at 1.20 s/step and 12.5 GB.
- **P5 outcome** (`analysis/screen_results.md`; RESULTS block in the plan's P5 section).
  - Two-seed mean val PPL: S1 5.3582, S3 5.3007, S2 5.3434, S4 5.3578, S6 5.4070, S0 6.0336, S5 6.6394. No lever beats S1 by more than 0.10, so **S1 advances**.
  - Late-25% PPL: S3 4.950 is best and S1 5.001.
  - MQAR: S6 0.193, S1 0.137, S0 0.007, S5 0.000.
  - **Statute-recall and multi-hop exact match are 0.000 in all 14 runs.** That is a floor at 220M parameters and 393M tokens, so the rule's recall clause decided nothing.
  - Predictions: S2, S4 and S5 confirmed; S1-beats-S0 and S6 refuted; S3 partial (better, not worse, on late-25% PPL).
- **The data loader now resumes** (2026-10-20). Before this, Lightning restarted the training schedule at row 0 on every resume: screen S5's continuation re-read its first rows (recorded in `analysis/screen_results.md` §7; the decision is unchanged). `ScheduleSampler` now skips exactly the micro-batches Lightning restored as done. Tests show:
  - a resume gives weights bit-identical to an unbroken run;
  - stage B starts its schedule at row 0 and reads on across a requeue;
  - two-rank CPU DDP splits the rows and resumes correctly.
- **Late-25% PPL** (`eval_2663318`, 2-seed means): S0 5.885, S1 5.001, S2 4.984, S3 4.950, S4 5.002, S6 5.106. The evaluator's overall PPL equals the training reader's within 0.0001 on every run.
- **Citation grammar fix (2026-10-15).** The probes crashed because `parse_citation` took only one-token act codes, and 12.7% of gii's provisions failed (`SGB 5 §20`, `13. BImSchV §3`, `1-DM-GoldmünzG §4`). A further 111 were mis-read as court decisions. The code is now everything before the provision: all 90,255 gii ids parse, and every old statute parse is unchanged. This also matters later for retrieval chunking (P7) and the citation verifier.
- **Diagnostic run 2636405 ran with β2 0.95, not 0.999.** SLURM copied `diag_spike.sh` at submission, before my edit; the yamls were read when it started. It is a third S1-s42 run under the fix. It survived to step 7,000 through 108 spikes (largest 69.7 at step 2,948, mostly L0–L4), all of which recovered.
- **mLSTM overflow** (S1 attempt 2599890, both seeds FAILED after 419–838 steps, see `analysis/screen_results.md` §7): trained forget gates widened a chunk's log-decay past what one fp32 factorisation holds. Fixed in `lexhybrid/kernels/tfla/tfla_interface.py` (`_row_block`, `_blockwise_scores`). It is exact: forward and backward equal the fp64 recurrence within 1e-12 and 1e-10; in fp32 no inf or NaN. The arms' ARCH line is unchanged. Port map §4 records it.

## 5. Next steps, in order

1. **When the cluster queue is free** (the user's call), submit three jobs. The exact commands are in `next_action` (`resume`).
   - `verify_handoff.sh` for s43_B and research_B, then tick **P6-L**.
   - `eval_probes.sh ROW_LEN=8192 PROBES=0` for s43_B and research_B, then **P6-K**.
   - `eval_probes.sh CONTEXT=8192 VALPPL=0` on the three stage-B checkpoints, then **P6-M**.

   Then **P6-N** (`analysis/backbone_results.md`) and **P6-Z** (`--force` from P7).
2. **P7 locally** (the current phase), in this order:
   - **P7-L** `scripts/answer.py` (CPU, tiny checkpoint). First reconcile the pointer FSM with the training format. `PointerFSM` requires a sentence after `<|cite|><|cK|>`, but `sft_format.Cite` writes none. The shared grammar `lexhybrid/renderer/segments.py` already accepts both. Recommended: let the FSM end a citation after `<|cK|>`, and add the test.
   - **P7-I:** a hand-built 100-pair calibration set (`data/manifests/nli_calibration.jsonl`, to be reviewed by the user) and `scripts/calibrate_nli.py`. This needs the mDeBERTa model, so the cluster or an HF download. The threshold of 0.5 is already pre-registered.
   - **P7-D, P7-K** need a GPU: `build_dense_index.sh` (`bge-m3`) and `eval_retrieval.sh` (GerLeRB, GerDaLIR).
   - **Known limits** the reviewers recorded (state note, 2026-10-27):
     - free-text quotation marks are checked only by NLI;
     - the leak check misses names with lower-case particles or of 4 or more words;
     - the party regex misses some phrasings;
     - claim splitting can merge two sentences at "1." or "12.";
     - the date filter drops a decision before its decision date.

## 6. Checks while processing

- **Every screen log:** `ARCH CHECK OK` before step 0; **no `TFLAFallbackError`**; a `STEPSTATS … final=1` line (s per optimizer step, peak GB); `TRAIN DONE`. The log holds no validation numbers. Read them with `screen_metrics.sh`, and look at its `CURVE` rows for a jump in loss or grad norm: a run can finish with exit 0 and still have diverged.
- **Val PPL** is exp(`val/loss`). The logged `val/perplexity` is a batch-mean of exp(CE) and overstates the PPL (S0-s43 step 7,999: 31.69 against 8.19). Fix that logging before P6.
- **Resume:** only `last.ckpt` resumes a run (`CKPT_EVERY` 1,000). The `interrupt.ckpt` files (2.6 GB each) left in `outputs/screen_S1_s4{2,3}/checkpoints/` by attempt 2599890 are never read; they can only be deleted from inside a job.
- **What the numbers mean:** val PPL pools every validation row once, and about 61% of those tokens are dip (long Bundestag papers). The mixture gives gii, the smallest source, 20% of rows: about 2.3 passes in a screen run. Both are the same for every arm.
- **Before P6:** decide the within-legal source weights (at 5 B tokens gii would be seen about 29 times) and rebuild the P7 retrieval chunks and BM25 index (1,132 gii ids changed on 2026-10-03).
- **Cluster rules** (`CLAUDE.md` e–g): the login node runs only `sbatch`, `squeue`, `sacct`, `scancel`, `cat`, `tail`, `grep`, `ls`, `du`, `df`, `mkdir`, so no `rm` and no `python`. `bash scripts/sync_to_cluster.sh` before every submission; pull only `logs/` and `analysis/`. To count things in scratch data, `grep -o` the fields and tally them on the Mac.
- **Results discipline:** a failed gate or measurement is a result. Never re-run to change a number or move a bar; never write a number without a job id and a log path.
- **Local quirks:** `validate.sh --ci` fails on the Mac only because `shellcheck` is not installed (CI runs it). If the shell reports `Operation not permitted` under `~/Desktop`, the terminal app lost macOS Desktop access: grant Desktop Folder (or Full Disk Access) in System Settings → Privacy & Security, then restart the app.

## 7. Open questions for the user

- `graphify-out/` dated snapshot folders (`2026-10-02/`, `2026-10-03/`) are tracked in git; keep them or ignore them.
- When to merge `p4-offline-backlog` into `main`.

## 8. Where to look

| what | where |
|---|---|
| live state, jobs, next action | `python3 scripts/plan_state.py resume` (`legal_build_state.json`) |
| the current phase's plan | the P5 section that `resume` prints (`LEGAL_BUILD_PLAN.md`) |
| results | `analysis/corpus_manifest.md`, `analysis/profile_results.md`, `analysis/screen_results.md` |
| operator spec | `Docs/REFERENCE_PORT_MAP.md` (§4 TFLA) |
| cluster wrappers | `scripts/slurm/` (`screen_array.sh`, `gpu_tests.sh`, `pack_corpus.sh`, …) |
| pulled logs (not in git) | `cluster/logs/` |
