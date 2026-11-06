# Retrieval on GerLeRB and GerDaLIR

How well BM25, dense, fused and reranked retrieval find the gold passages of two German legal retrieval benchmarks, each searched over its own corpus.

**Setup.** `scripts/slurm/eval_retrieval.sh` → `scripts/eval_retrieval.py`. Every arm ranks the top 100 passages per judged query:

- **bm25**: `bm25s` with the German Snowball stemmer;
- **dense**: `BAAI/bge-m3` in fp16, passages and queries truncated at 512 tokens;
- **fused**: reciprocal rank fusion of bm25 and dense;
- **reranked**: the fused top 50 re-scored by `BAAI/bge-reranker-v2-m3` (fp32, pairs truncated at 1,024 tokens), followed by fused ranks 51–100.

Metrics: nDCG@10, R@5/10/20, MRR (`lexhybrid.eval.retrieval_metrics`). Both sets are German law (federal statutes; decisions of German courts), jurisdiction DE only, so there is one row per arm.

| set | queries | corpus | what a query is | what a passage is |
|---|---|---|---|---|
| GerLeRB (Zenodo 15745124, CC BY 4.0) | 367 | 90,917 rows → 90,218 passages | a question about a statute | one statute paragraph |
| GerDaLIR (`mteb/GerDaLIRSmall` @ b199f38, MIT) | 12,234 | 9,969 passages | a passage of a decision, its citation masked as `[REF]` | a whole cited decision |

GerLeRB's corpus repeats 460 docnos (89,851 distinct): an exact repeat is dropped, and a repeated id with different text stays searchable as `<id>#<k>`. No judged id repeats, so the gold set is unchanged. GerDaLIR here is the small MTEB corpus (9,969 decisions), not the full GerDaLIR collection, so its numbers are not comparable to results on the full corpus.

**Hardware.** GerLeRB ran on one H100 (x86_64). GerDaLIR's corpus and query vectors were also computed on H100s. Most of its reranking ran on 2× L40 (aarch64), split into two shards of 6,117 queries. The scoring run reused a saved reranked list wherever its fused top 50 held the same passages, and reranked on the H100 the 731 queries (6.0%) whose set had changed. All machines used the same versions of torch (2.11.0+cu128), transformers (5.17.0), tokenizers (0.23.2) and bm25s (0.3.11), and the reranker ran in fp32 on both.

## Summary

1. **GerLeRB (statute questions): the full pipeline is best.** Reranked nDCG@10 is 0.6615, against dense 0.5620, fused 0.5223 and BM25 0.3515. The reranker adds 0.139 over the fused list it starts from. Fusing with the weaker BM25 costs 0.040 against dense alone.
2. **GerDaLIR (whole decisions): BM25 alone is best, by a wide margin.** BM25 reaches nDCG@10 0.5923, fused 0.3822, reranked 0.3056 and dense 0.1635. The full pipeline loses 0.287 against BM25 alone, and reranking lowers the fused list by 0.077.
3. **The prediction holds for GerLeRB and fails for GerDaLIR** (0.3056 < 0.35), so the outcome is partial.
4. **A likely cause (not tested here):** each GerDaLIR passage is a whole decision. BM25 indexes all of its text, while the dense encoder reads only its first 512 tokens and the reranker the first 1,024 tokens of the query–passage pair.

## Prediction and outcome

| Prediction (stated before the run) | Outcome | Verdict |
|---|---|---|
| Hybrid BM25 + dense + rerank reaches nDCG@10 ≥ 0.45 on GerLeRB | reranked 0.6615 | Confirmed |
| … and ≥ 0.35 on GerDaLIR | reranked 0.3056 (BM25 alone 0.5923, fused 0.3822) | Refuted |

Overall: **partial**.

## Results

### GerLeRB (367 queries)

| arm | nDCG@10 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|
| bm25 | 0.3515 | 0.4114 | 0.5123 | 0.5777 | 0.3103 |
| dense | 0.5620 | 0.6458 | 0.7193 | 0.7902 | 0.5192 |
| fused | 0.5223 | 0.5940 | 0.6975 | 0.7684 | 0.4755 |
| reranked | **0.6615** | **0.7439** | **0.8011** | **0.8365** | **0.6205** |

### GerDaLIR (12,234 queries)

| arm | nDCG@10 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|
| bm25 | **0.5923** | **0.6528** | **0.7107** | **0.7555** | **0.5716** |
| dense | 0.1635 | 0.1857 | 0.2327 | 0.2817 | 0.1562 |
| fused | 0.3822 | 0.4474 | 0.5782 | 0.7021 | 0.3437 |
| reranked | 0.3056 | 0.3525 | 0.4616 | 0.5894 | 0.2822 |

## What this licenses

- For questions over statute paragraphs, BM25 + dense + fusion + rerank, as configured, is the best of the four arms. It leads the next arm by 0.10 nDCG@10.
- For a corpus of whole court decisions (not split into windows), BM25 alone beats every other arm as configured. Dense retrieval and reranking at 512 and 1,024 tokens should not rank whole decisions.

## Not licensed

- A comparison with published GerLeRB or GerDaLIR numbers. Those use other corpora (GerDaLIR's full collection), other truncation or other models.
- Claims about Austrian, Swiss or EU law. Both sets are German only.
- Retrieval quality over the project's own statute and case-law index. Each set was searched over its own corpus, and the project's case-law index splits decisions into 300–500-token windows, which this setup does not.
- That truncation is what causes the GerDaLIR result. It is likely, but no windowed or longer-context arm was run.
- Differences under about 0.02 nDCG@10. There is one run per arm and no confidence interval.

## Open limitations

- GerDaLIR passages are whole decisions, but the dense encoder reads only the first 512 tokens and the reranker only the first 1,024 tokens of each query–passage pair. Splitting decisions into windows, as the case-law chunker does for the index, would likely change the dense and reranked numbers. How much it would recover (300–500-token windows, each decision scored by its best window) was not measured.
- 94% of GerDaLIR's reranked lists come from the L40 node, which lost a GPU about two minutes after both shards had finished. Checked afterwards: the two shards hold interleaved halves of the query set, and their saved lists score nDCG@10 0.3080 and 0.3034 (6,117 queries each; difference 0.005, standard error of the difference about 0.007). The 731 lists reranked on the H100 score 0.3075. There is no sign of a faulty GPU.
- The software is the same on both GPU types, but fp32 results can still differ in the last digits between them.
- No bootstrap confidence intervals.

## Reproduction

`sbatch scripts/slurm/eval_retrieval.sh` (`SETS="gerlerb gerdalir"` by default). A set missing from `$SCRATCH_ROOT/data/evalsets/` is fetched first. For a long rerank: `SETS=gerdalir SHARDS=2 sbatch --array=0-1 scripts/slurm/eval_retrieval.sh`, then the same command without `SHARDS` to score. On the aarch64 node the wrapper uses `.venv-arm` (`scripts/slurm/setup_arm_env.sh`). Results: `logs/eval_retrieval_<job>/results.json`, one `RETRIEVAL` line per set and arm.
