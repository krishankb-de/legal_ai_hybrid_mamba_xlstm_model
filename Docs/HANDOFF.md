# Handoff — branch `p4-offline-backlog`

> **Updated:** 2026-10-09. **Read this right after** `python3 scripts/plan_state.py resume`. Where the two disagree, the state file (`legal_build_state.json`, via `resume`) wins: it holds the live job ids, ticks and next action. **Keep this file current:** at the end of any session that changes the goal, a decision, what is done or the next step, update the dated sections below and commit them with the work (the user decides every commit and push).

## 1. What we are building

A citation-faithful, retrieval-gated legal language model for the **German market** (`lexhybrid`): a corrected Mamba-3 + mLSTM + sparse-attention decoder, trained from scratch with logit distillation from Qwen3-8B-Base, that cites by pointer into retrieved passages and is gated by a verifier. The plan of record is `LEGAL_BUILD_PLAN.md` (phases P0–P11), executed one phase at a time on the HPI aisc cluster (H100s) from this Mac.

## 2. What this branch is for

Opened for the P4 work (cluster bring-up, corpus at scale, profiling), it now carries **P4 complete** and the **start of P5**, the pre-registered architecture screen that picks the backbone configuration by measurement. It has not been merged to `main`. Merging is the user's call, sensibly after P5-Z.

## 3. Where things stand (2026-10-09)

| phase | state |
|---|---|
| P0–P4 | done (P4-Z passed 2026-10-03: `validate.sh --full` exit 0) |
| **P5 screen** | **3/10**: P5-A, B (prediction, wrappers), C (S1 submitted). S1 is the control arm `hybrid_legal_base`, seeds 42 and 43 |
| P6 | 1/15 (P6-A done ahead of phase) |
| P7 | 4/14 (A, B, C1, C done ahead of phase; **must be rebuilt**, see §6) |
| P8–P11 | not started |

**Open jobs** (check with `sacct`): `2626201` = `gpu_tests.sh` (all `cuda` tests, including the new TFLA one); `2626202` = S1 retry (`_0` = S1-s42, `_1` = S1-s43), queued `afterok:2626201`, so a GPU-test failure cancels it.

**Decisions taken on this branch** (all in `decisions.*` of the state file):

| decision | value | when / who |
|---|---|---|
| `teacher` | Qwen3-8B-Base with gradient checkpointing (43.18 GB, 6.10 s/step at 4,096 on 4×H100) | 2026-10-01, user |
| `market` | Germany only: ris (AT), fedlex and bger (CH), and multilegalpile's Swiss rows out | 2026-10-02, user |
| `eurlex_training` | eurlex out of training (565 docs cannot give the ≥ 2,000-doc validation split); kept for retrieval | 2026-10-03, user |
| `p6_token_budget` | 5 B tokens = min(5 B, 6.09 B commercial-safe) | 2026-10-03, rule |
| `tfla_overflow` | exact mLSTM kernel keeps chunk 128; a too-wide chunk is factorised per row block | 2026-10-09, user |

## 4. What was done (with the numbers' sources)

- **Corpus** (`analysis/corpus_manifest.md`, pack job 2599735): 6 packed sources at 4,096 and 8,192 tokens. Commercial-safe **6,092,655,089** training tokens (gii, rii, oldp, dip, fineweb2_de); research arm **7,474,326,564** (adds multilegalpile's German rows). Dedup: 30,597 dropped within a source, 241,242 cross-source copies kept and linked. Scrub: 15,960,418 names replaced, about 83 H100-hours.
- **Fixes found at corpus scale** (each with a test): scrub crashed on whitespace-only entity spans and OOM'd on 33k-character DIP lines (4,096-character pieces); 137 gii ids named several provisions (articles that number their own §§): ids, citations (`BGBEG Art. 229 §6`) and URLs (`art_229__6.html`) are now article-aware; dedup refuses repeated ids up front; the training mixture listed sources without shards.
- **Profiling and the KD probe** (`analysis/profile_results.md`): chunk 128 kept; the teacher was decided by the P4-U rule.
- **mLSTM overflow** (S1 attempt 2599890, both seeds FAILED after 419–838 steps, see `analysis/screen_results.md` §7): trained forget gates widened a chunk's log-decay past what one fp32 factorisation holds. Fixed in `lexhybrid/kernels/tfla/tfla_interface.py` (`_row_block`, `_blockwise_scores`). It is exact: forward and backward equal the fp64 recurrence within 1e-12 and 1e-10; in fp32 no inf or NaN. The arms' ARCH line is unchanged. Port map §4 records it.

## 5. Next steps, in order

1. `sacct -j 2626201,2626202 --format=JobID,State,Elapsed,MaxRSS,ExitCode -P -X`.
2. **gpu_tests FAILED** → `bash scripts/sync_from_cluster.sh logs`, read `cluster/logs/gpu_tests_2626201.log`, fix, add the test, resubmit both (S1 was auto-cancelled).
3. **S1 RUNNING** → `tail` its logs for speed and errors (§6), record `job update … --status RUNNING`, stop.
4. **S1 COMPLETED** → pull logs → **P5-D**: val PPL per seed, two-seed SD, `bar_value = max(2 × SD, 0.10)` into the P5 section and `pre_registered.P5.bar_value` (say if the floor binds).
5. **P5-E**: S0, S2, S3, S4, S6 × seeds 42/43 (`ARMS='S0-s42 S0-s43 …' sbatch --array=0-9`) and S5 × 2 separately (`--time=24:00:00`). Pick the other arms' walltime from S1's measured s/step (`STEPSTATS … final=1`). Attempt 2599890's progress bar showed 838 of 12,000 after 20 min (0.70 it/s); whether that counts optimizer steps or micro-batches (accum 2) is unconfirmed, so a run is somewhere around 5–10 h.
6. P5-F (per-arm PPL, s/step, peak GB into `analysis/screen_results.md`), P5-G/H (probes), P5-I (apply the decision rule verbatim), P5-Z (gate) → P6.

## 6. Checks while processing

- **Every screen log:** `ARCH CHECK OK` before step 0; a steady rate on the progress bar; **no `TFLAFallbackError`**; validation at steps 4,000 / 8,000 / 12,000; a `STEPSTATS … final=1` line (s/step, peak GB); `TRAIN DONE`. The `_row_block` path is silent; a large slowdown against attempt 2599890's 0.70 it/s would point to small blocks.
- **Resume:** only `last.ckpt` resumes a run (`CKPT_EVERY` 1,000). The `interrupt.ckpt` files (2.6 GB each) left in `outputs/screen_S1_s4{2,3}/checkpoints/` by attempt 2599890 are never read; they can only be deleted from inside a job.
- **What the numbers mean:** val PPL pools every validation row once, and about 61% of those tokens are dip (long Bundestag papers). The mixture gives gii, the smallest source, 20% of rows: about 2.3 passes in a screen run. Both are the same for every arm.
- **Before P6:** decide the within-legal source weights (at 5 B tokens gii would be seen about 29 times) and rebuild the P7 retrieval chunks and BM25 index (1,132 gii ids changed on 2026-10-03).
- **Cluster rules** (`CLAUDE.md` e–g): the login node runs only `sbatch`, `squeue`, `sacct`, `scancel`, `cat`, `tail`, `grep`, `ls`, `du`, `df`, `mkdir`, so no `rm` and no `python`. `bash scripts/sync_to_cluster.sh` before every submission; pull only `logs/` and `analysis/`. To count things in scratch data, `grep -o` the fields and tally them on the Mac.
- **Results discipline:** a failed gate or measurement is a result. Never re-run to change a number or move a bar; never write a number without a job id and a log path.
- **Local quirks:** `validate.sh --ci` fails on the Mac only because `shellcheck` is not installed (CI runs it). If the shell reports `Operation not permitted` under `~/Desktop`, the terminal app lost macOS Desktop access: grant Desktop Folder (or Full Disk Access) in System Settings → Privacy & Security, then restart the app.

## 7. Open questions for the user

- P6 mixture: the within-legal weights (gii, rii, oldp, dip).
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
