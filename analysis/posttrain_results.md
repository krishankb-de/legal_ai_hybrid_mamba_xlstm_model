# Post-training results: supervised fine-tuning

The hybrid, a matching Transformer and Qwen3-0.6B-Base were fine-tuned on the same supervised data at three learning rates each, and a LoRA ceiling on Qwen3-8B-Base at two. The hybrid and the Transformer were then trained at two more seeds at their chosen rate. This page records the learning-rate choice, the dev loss of every seed, and citation, refusal and truncation on the dev split.

**Setup.** 4×H100 DDP, bf16-mixed, 2,000 optimizer steps of 4 rows per GPU (16 rows per step), AdamW (weight decay 0.1), 50 warmup steps, then cosine to 0.1× the peak, gradient clip 1.0. The data are synthetic German legal QA over retrieved passages in the pointer-citation format: 19,076 training and 924 dev items, split by question, with a quarter unanswerable. The loss covers answer tokens only. The hybrid and the Transformer start from their seed-42 pretrained checkpoints (after the 8,192-token stage). Qwen3-0.6B-Base is fully fine-tuned. The LoRA ceiling trains rank-16 adapters on every attention and MLP projection of a frozen Qwen3-8B-Base, plus the pointer-token rows of the embedding and output layer. All systems share one tokenizer (Qwen3's, plus the pointer specials). The learning-rate sweep uses seed 42. Seeds 43 and 44 also start from the seed-42 pretrained checkpoint, so they vary only the fine-tuning run's randomness (such as data order), not pretraining.

## Summary

The rule chose 4e-5 for the hybrid and the Transformer, 2e-5 for Qwen3-0.6B, and 2e-4 for the LoRA ceiling.

**Every run was still improving at step 2,000.** Dev loss fell at each of the eight validations in all eleven runs, so the best checkpoint is always the final one.

**Over three seeds the hybrid's dev loss is lower than the Transformer's on every seed:** 0.4965 ± 0.0008 against 0.5140 ± 0.0019, a gap of 0.0176 (3.4%), about nine Transformer seed SDs. Dev loss is fit to the synthetic answer distribution. The pre-stated comparison is on citation precision and refusal, where the two are within one seed SD of each other (below), so the dev-loss gap is descriptive.

**On the dev split's own prompts, citation precision is 0.533 ± 0.046 (hybrid) and 0.519 ± 0.038 (Transformer), short of the 0.85 target; Qwen3-0.6B reaches 0.787 and the LoRA ceiling 0.874.** No system decodes without truncation (target 0%): 0.2–6.9% of decodes run out of tokens, mostly in repetition loops.

## Learning-rate selection

The rule was written before the first run. Per system, take the lowest dev loss over the run's validations. If it is within 1% (relative) of the 1× run's (4e-5, the default), keep 1×. The LoRA ceiling takes the lower dev loss of 1e-4 and 2e-4. Dev loss is the token-weighted mean cross-entropy on the dev split's answer tokens. Dev PPL is its exp.

| system | LR | dev loss | vs 4e-5 | dev PPL | chosen |
|---|---|---|---|---|---|
| hybrid | 2e-5 | 0.5121 | +3.04% | 1.6688 | |
| hybrid | 4e-5 | **0.4970** | | 1.6438 | **yes** (lowest) |
| hybrid | 8e-5 | 0.4988 | +0.36% | 1.6467 | |
| Transformer | 2e-5 | 0.5303 | +3.25% | 1.6994 | |
| Transformer | 4e-5 | **0.5136** | | 1.6713 | **yes** (8e-5 is within 1%) |
| Transformer | 8e-5 | 0.5127 | −0.18% | 1.6698 | |
| Qwen3-0.6B-Base | 2e-5 | **0.3592** | −2.47% | 1.4322 | **yes** (lowest, beyond 1%) |
| Qwen3-0.6B-Base | 4e-5 | 0.3683 | | 1.4453 | |
| Qwen3-0.6B-Base | 8e-5 | 0.4090 | +11.05% | 1.5053 | |
| LoRA, Qwen3-8B-Base | 1e-4 | 0.2023 | | 1.2242 | |
| LoRA, Qwen3-8B-Base | 2e-4 | **0.2010** | −0.64% vs 1e-4 | 1.2226 | **yes** (lowest) |

- **The hybrid and the Transformer are flat between 4e-5 and 8e-5** (0.36% and 0.18% apart) and both lose about 3% at 2e-5.
- **Qwen3-0.6B prefers the lowest LR tried.** The grid does not bracket its optimum, so an LR below 2e-5 might score lower still. Its 8e-5 run is 11% worse.
- **The two LoRA rates differ by 0.64%.**

### Three seeds at the chosen learning rate

| system | seed 42 | seed 43 | seed 44 | mean ± SD | dev PPL (mean) |
|---|---|---|---|---|---|
| hybrid, 4e-5 | 0.4970 | 0.4956 | 0.4968 | **0.4965 ± 0.0008** | 1.6429 |
| Transformer, 4e-5 | 0.5136 | 0.5124 | 0.5161 | **0.5140 ± 0.0019** | 1.6720 |

- **The hybrid is lower on all three seeds,** by 0.0166, 0.0168 and 0.0193 (mean 0.0176).
- **Again the lowest dev loss is at the final validation** in all four new runs. Wall-clock per run is the same as in the sweep (42–45 and 32 min).

### Throughput

| system | s/step | peak allocated GB | wall-clock per run |
|---|---|---|---|
| hybrid | 1.165–1.168 | 46.9 | 42 min |
| Transformer | 0.853–0.858 | 22.3 | 32 min |
| Qwen3-0.6B-Base | 1.406–1.414 | 72.5 | 52 min |
| LoRA, Qwen3-8B-Base | 3.884–3.899 | 32.8 | 2 h 25 min |

s/step is the median over the last 1,995 steps; peak memory is the largest of the four ranks (GB = 10⁹ bytes). Full fine-tuning of Qwen3-0.6B without gradient checkpointing reserved up to 80.9 GB of the H100's 85.9 GB.

## Citation, refusal and truncation on the dev split

Each of the 924 dev rows is decoded on its own prompt as it stands: greedy, under the pointer constraint, up to 256 new tokens, then checked by the mDeBERTa NLI verifier. 181 rows should be refused; 743 have a cited gold passage. The hybrid, the Transformer and Qwen3-0.6B decode in fp32. The LoRA ceiling decodes with its weights (adapters included) in bf16 and RoPE frequencies in fp32. All decodes ran on H100s.

| system | seed | dev loss | citation precision (passed / cited) | citation recall | refusal F1 | refused / 924 | truncated |
|---|---|---|---|---|---|---|---|
| hybrid | 42 | 0.4970 | 0.4892 (452/924) | 0.2168 | 0.3984 | 547 | 57 (6.17%) |
| hybrid | 43 | 0.4956 | 0.5296 (456/861) | 0.2042 | 0.4222 | 539 | 24 (2.60%) |
| hybrid | 44 | 0.4968 | 0.5804 (397/684) | 0.1887 | 0.4053 | 574 | 38 (4.11%) |
| Transformer | 42 | 0.5136 | 0.4791 (424/885) | 0.1974 | 0.3990 | 591 | 64 (6.93%) |
| Transformer | 43 | 0.5124 | 0.5233 (427/816) | 0.2027 | 0.4103 | 560 | 60 (6.49%) |
| Transformer | 44 | 0.5161 | 0.5554 (401/722) | 0.2036 | 0.4187 | 569 | 47 (5.09%) |
| Qwen3-0.6B-Base | 42 | 0.3592 | 0.7871 (780/991) | 0.5996 | 0.5820 | 362 | 36 (3.90%) |
| LoRA, Qwen3-8B-Base | 42 | 0.2010 | 0.8741 (958/1096) | 0.7600 | 0.7032 | 257 | 2 (0.22%) |

Over three seeds: citation precision hybrid 0.533 ± 0.046, Transformer 0.519 ± 0.038; refusal F1 0.409 ± 0.012 and 0.409 ± 0.010; recall 0.203 ± 0.014 and 0.201 ± 0.003.

- **The hybrid and the Transformer are within one seed SD of each other on citation and refusal.** Their precision differs by 0.014 against seed SDs of 0.046 and 0.038, and refusal F1 by 0.001; three seeds that share one pretrained checkpoint cannot separate them. The hybrid's lower dev loss (3.4%) does not carry over to these metrics.
- **The stated target of citation precision ≥ 0.85 after fine-tuning is missed by the hybrid (0.53), the Transformer (0.52) and Qwen3-0.6B (0.79).** Only the 8B LoRA ceiling clears it (0.874).
- **Both small systems over-refuse.** They refuse 539–591 of 924 rows against 181 that should be refused (refusal recall 0.80–0.87, refusal precision 0.26–0.28).
- **Precision partly tracks how much a model answers.** From seed 42 to 44 the number of cited claims falls (hybrid 924 → 684, Transformer 885 → 722) while precision rises and recall stays near 0.20.
- **Truncation is not 0% for any system (target: 0%).** Every truncated decode used all 256 tokens. For the hybrid, the Transformer and Qwen3-0.6B, 306 of the 326 truncated decodes end in a short unit repeated three or more times: a loop of quote pointers or of a phrase. The same rows recur across systems, and their references are short. The LoRA's two truncated decodes show no loop; their references are about 1,100 characters, and both are among the estimated 5 of the 924 references that exceed 256 tokens.
- **Quote fidelity is 1.0 everywhere,** because quotes are copied spans.

Citation precision here is system-internal: the training data were filtered with the same verifier.


## Limitations

- **One seed per learning rate.** The 0.18% and 0.36% gaps above are likely inside seed noise. The rule keeps 1× in exactly that case. The seed SD at 4e-5 (0.0008 and 0.0019) is of that size.
- **The three seeds share one pretrained checkpoint per system,** so their spread omits pretraining variance.
- **The dev split comes from the same synthetic generator as the training data.** Dev loss measures fit to that distribution, not citation precision or refusal quality. The verifier that scores citation precision also filtered the training data.
- **The 256-token budget truncates a few long reference answers by construction** (estimated 5 of 924, from character counts, not the tokenizer).
- **The LoRA ceiling decodes in bf16, the other systems in fp32.** In a separate gold-passage check, the 47 decodes available in both precisions (eight questions under six distractor conditions, none reaching the token budget) were identical in 44; the effect of bf16 on truncation at the budget is not measured.
- **No run converged.** A longer schedule would likely lower every dev loss. The budget is the same for all systems.
- **Qwen3-0.6B's chosen LR is at the edge of the grid.**

## Reproduction

- **Training:** `scripts/slurm/train_sft_4gpu.sh` with `LR` and `SEED`. Hybrid: `INIT_FROM=<hybrid stage-B last.ckpt>`. Transformer: also `MODEL_CONFIG=transformer_legal_base`. Qwen3-0.6B: `HF_BASELINE=1`. LoRA ceiling: `LORA=1`.
- **Dev loss per validation:** `scripts/slurm/screen_metrics.sh` with `PREFIX=sft_`.
- **Citation, refusal and truncation on the dev split:** `scripts/slurm/eval_gold_passages.sh` with `SET=sft_dev AS_IS=1` and `RUNS` naming the checkpoints (the LoRA ceiling also `DTYPE=bfloat16`). It decodes each dev row's own prompt and scores the run with `scripts/evaluate_citation.py` and `scripts/audit_truncation.py`.
- **Checkpoints (cluster-only):** `$SCRATCH_ROOT/outputs/<run>/checkpoints/best-step002000.ckpt`, with `<run>` = `sft_hybrid_legal_base_s42_lr4e-5`, `sft_transformer_legal_base_s42_lr4e-5`, `sft_qwen3_0p6b_s42_lr2e-5` and `sft_lora_qwen3_8b_s42_lr2e-4`; seeds 43 and 44: `sft_{hybrid,transformer}_legal_base_s{43,44}_lr4e-5`.
