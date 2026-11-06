# Judge calibration and validation on gold labels

How well three LLM judges, scoring with one fixed prompt set, agree with gold labels on six item types, measured before any of them scores a system.

**Setup.** Judges: `gemma-4-31b`, `ministral-3-14b` and `llama-3-3-70b` on the AISC gateway (`https://api.aisc.hpi.de`, LiteLLM hosted at HPI). Temperature 0, replies capped at 2,048 tokens, every reply cached by prompt hash. Prompts: `lexhybrid.eval.judge`, prompt version 2; scoring: `scripts/score_judge_calibration.py`.

Two check judges, `gpt-oss-120b` and `qwen3-8-27b`, ran on the calibration half only. No prompt was tuned on them, and neither scores system answers: `gpt-oss-120b` is itself one of the systems compared, and `qwen3` shares a model family with the model that wrote the synthetic training data. They test whether version 2 fits the three judges it was tuned with rather than the task.

**Items.** `scripts/build_judge_calibration.py` over LegalQuAD, GerLayQA, GerLeRB, GerDaLIR and the 100 reviewed NLI pairs (`data/manifests/nli_calibration.jsonl`). The items are split once, by question group, into two halves:

- **calibration**, used only to tune prompts, scored at up to 200 items per type;
- **validation**, reported, scored once at the final prompt version, at up to 500 items per type.

Both halves are sampled as whole groups (an item with its negatives) in hash order.

| type | source | label 1 | label 0 | predicted 1 when |
|---|---|---|---|---|
| correctness | GerLayQA, LegalQuAD | the gold answer | another question's gold answer; the gold answer with one § changed, one other number changed, or negated | verdict "correct" |
| answer relevancy | GerLayQA, LegalQuAD | the gold answer | another question's gold answer | rating ≥ 0.5 |
| context precision | LegalQuAD (with the reference answer) | the gold context | the best BM25 match among other contexts that does not hold the answer | the passage is useful |
| context recall | LegalQuAD | the gold answer against its gold context | … against a random context that does not hold it | ≥ half the reference attributed |
| context relevance | GerLeRB (statute questions) | a qrel-relevant passage | the best BM25 match not judged relevant | the passage is useful |
| faithfulness | NLI pairs | entailed claim | not entailed | every statement supported |

A reply that does not parse counts as a wrong prediction.

**Bar,** fixed before any judge call: per judge and per type on the validation half, Cohen's κ ≥ 0.6 **and** accuracy ≥ 0.85. A type that misses is unvalidated for that judge and licenses no claim. A claim about systems stands only if it holds under every judge validated for that metric.

**Context relevance on GerLeRB only.** In the first calibration pass, the judges scored near chance on the mixed context relevance sample, which is 97% GerDaLIR: accuracy 0.630 for gemma and 0.545 for llama. So the bar for this type was moved to the GerLeRB items before any validation call. The mixed sample is still scored and reported beside it, with no bar, so that a prompt fit to GerLeRB alone would show.

**Prompt history.** Prompts were revised on the calibration half only. Version 2 was fixed as final after the check judges' run and before the first validation call.

## Summary

1. **`gemma-4-31b` meets the bar on all six types.** Its lowest is correctness, at accuracy 0.872 and κ 0.691.
2. **`llama-3-3-70b` meets it on five.** Correctness misses on accuracy (0.844 < 0.85; κ 0.638 passes).
3. **`ministral-3-14b` meets it on five.** GerLeRB context relevance misses on accuracy (0.834 < 0.85; κ 0.669 passes).
4. **Validated judges per metric:**
   - answer relevancy, context precision, context recall, faithfulness: all three;
   - correctness: gemma and ministral;
   - context relevance: gemma and llama.
5. **The judges' common weakness is a changed number.** On correctness, a gold answer with one number changed is caught only 54–60% of the time, and a negated one 74–75%. All three accept every gold answer (recall 1.000), so their precision on "correct" is only 0.59–0.63 on this mix (110 right answers, 390 wrong).
6. **Context relevance on the mixed sample stays below the bar for every judge:** accuracy 0.726, 0.556 and 0.594 on validation (482 GerDaLIR + 18 GerLeRB items). On calibration, all five judges scored 0.515–0.620.
7. **The check judges land in the same range as the three tuned judges** (table 3), so version 2 is not fit to those three alone. On GerLeRB context relevance, `qwen3-8-27b` meets the bar (0.855 / 0.711) and `gpt-oss-120b` misses on accuracy (0.831 / 0.662).

## Bar and outcome

| judge | types meeting the bar (validation) | missed |
|---|---|---|
| gemma-4-31b | 6 / 6 | none |
| ministral-3-14b | 5 / 6 | context relevance (GerLeRB): accuracy 0.834 |
| llama-3-3-70b | 5 / 6 | correctness: accuracy 0.844 |

## Results

### 1. Validation half, prompt version 2 (accuracy / κ / precision / recall; unparsed in brackets)

| type | n | gemma-4-31b | ministral-3-14b | llama-3-3-70b |
|---|---|---|---|---|
| answer relevancy | 500 | 0.990 / 0.980 / 1.000 / 0.980 | 0.988 / 0.976 / 0.996 / 0.980 | 0.996 / 0.992 / 0.992 / 1.000 |
| context precision | 193 | 0.964 / 0.927 / 0.933 / 1.000 | 0.886 / 0.772 / 0.844 / 0.948 (8) | 0.912 / 0.824 / 0.926 / 0.897 (2) |
| context recall | 194 | 0.985 / 0.969 / 0.990 / 0.979 | 0.948 / 0.897 / 0.958 / 0.938 (7) | 0.964 / 0.928 / 0.979 / 0.948 (1) |
| context relevance, GerLeRB | 326 | 0.896 / 0.791 / 0.869 / 0.933 | **0.834** / 0.669 / 0.777 / 0.939 | 0.887 / 0.773 / 0.850 / 0.939 |
| correctness | 500 | 0.872 / 0.691 / 0.632 / 1.000 | 0.858 / 0.664 / 0.608 / 1.000 | **0.844** / 0.638 / 0.585 / 1.000 |
| faithfulness | 49 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| *context relevance, mixed (no bar)* | 500 | *0.726 / 0.452 / 0.681 / 0.852* | *0.556 / 0.112 / 0.534 / 0.888 (8)* | *0.594 / 0.188 / 0.562 / 0.852 (2)* |

### 2. Accuracy by variant, validation half

| type | variant | n | gemma | ministral | llama |
|---|---|---|---|---|---|
| correctness | gold | 110 | 1.000 | 1.000 | 1.000 |
| | other question's answer | 110 | 0.982 | 0.973 | 0.964 |
| | § changed | 104 | 0.971 | 0.894 | 0.827 |
| | negated | 104 | 0.750 | 0.750 | 0.740 |
| | number changed | 72 | 0.542 | 0.569 | 0.597 |
| context relevance, GerLeRB | relevant | 163 | 0.933 | 0.939 | 0.939 |
| | BM25 negative | 163 | 0.859 | 0.730 | 0.834 |
| context precision | gold context | 97 | 1.000 | 0.948 | 0.897 |
| | BM25 negative | 96 | 0.927 | 0.823 | 0.927 |

Faithfulness: every variant 1.000 for all three judges (entailed: quote 5, paraphrase 12, subset 6; not entailed: detail 10, negation 4, framing 4, extra 3, unrelated 5).

### 3. Calibration half, prompt version 2, with the check judges (accuracy / κ)

| type | n | gemma | ministral | llama | gpt-oss-120b (check) | qwen3-8-27b (check) |
|---|---|---|---|---|---|---|
| answer relevancy | 200 | 0.980 / 0.960 | 0.995 / 0.990 | 0.995 / 0.990 | 0.965 / 0.930 | 0.975 / 0.950 |
| context precision | 199 | 0.970 / 0.940 | 0.899 / 0.799 | 0.945 / 0.889 | 0.950 / 0.899 | 0.980 / 0.960 |
| context recall | 200 | 1.000 / 1.000 | 0.965 / 0.930 | 0.990 / 0.980 | 1.000 / 1.000 | 0.970 / 0.940 |
| context relevance, GerLeRB | 408 | 0.860 / 0.721 | 0.819 / 0.637 | 0.843 / 0.686 | 0.831 / 0.662 | 0.855 / 0.711 |
| correctness | 200 | 0.865 / 0.674 | 0.880 / 0.704 | 0.860 / 0.665 | 0.890 / 0.725 | 0.940 / 0.839 |
| faithfulness | 51 | 1.000 / 1.000 | 0.824 / 0.642 | 0.961 / 0.922 | 0.922 / 0.844 | 0.980 / 0.961 |
| context relevance, mixed | 200 | 0.620 / 0.240 | 0.515 / 0.030 | 0.525 / 0.050 | 0.540 / 0.080 | 0.565 / 0.130 |

`qwen3-8-27b` left 29 of the 200 mixed context relevance replies unparsed, and at most 5 elsewhere.

## What this licenses

- Scoring answer relevancy, context precision, context recall and faithfulness with any of the three judges, and correctness with gemma or ministral. Context relevance over statute passages (GerLeRB-like) may be scored with gemma or llama.
- Any claim about systems on a metric holds only under every judge validated for it. For correctness that is gemma and ministral; for context relevance, gemma and llama.
- A judge's "incorrect" on correctness is reliable: no gold answer was rejected. Its "correct" accepts a fair share of answers with one wrong number or a dropped negation.

## Not licensed

- **Context relevance on whole court decisions** (GerDaLIR-like passages), under any judge. All of them are far below the bar there.
- **Correctness under llama, and GerLeRB context relevance under ministral.** Each missed its bar by 0.006–0.016 accuracy, about one standard error at these sizes. That still counts as a miss: the bar is not rerun or relaxed.
- **Agreement with jurists.** That is a separate measurement, not run yet.
- **That the judges catch a wrong figure in an answer.** A changed number gets through 40–46% of the time.

## Open limitations

- **Faithfulness rests on 49 validation items, four of them negations.** ministral missed faithfulness on calibration (0.824; it accepted all five negated claims) and scored 49 / 49 on validation. That fits small-sample noise as well as a real fix. Its faithfulness passes the bar as written, but its verdicts on negated claims are the weakest evidence in this table.
- **The correctness negatives are built by rule:** another answer, one § or number changed, a negation. Errors a real system makes may be harder or easier to catch.
- **Why the mixed context relevance sample fails is not established.** Its BM25 negatives are passages from whole decisions not marked relevant, and some may be relevant but unjudged. No items were re-read.
- **One run per judge,** no confidence intervals. At n = 500 the standard error of accuracy is about 0.016; at n = 326, about 0.019.
- **The gateway serves models by name.** The cache keeps the replies, not the model weights, so a later run against an updated model may differ.
- **Weak spots on correctness, all on its negatives:** §-changed answers 0.83–0.97, negated answers about 0.75, changed numbers 0.54–0.60.

## Reproduction

```
.venv/bin/python scripts/build_judge_calibration.py --qa-dir data/evalsets/legalquad \
    --qa-dir data/evalsets/gerlayqa --retrieval-dir data/evalsets/gerlerb \
    --retrieval-dir data/evalsets/gerdalir --nli-pairs data/manifests/nli_calibration.jsonl \
    --out data/evalsets/judge_calibration.jsonl
.venv/bin/python scripts/build_judge_calibration.py --qa-dir data/evalsets/legalquad \
    --qa-dir data/evalsets/gerlayqa --retrieval-dir data/evalsets/gerlerb \
    --nli-pairs data/manifests/nli_calibration.jsonl --out data/evalsets/judge_calibration_gerlerb.jsonl
JUDGE_API_KEY=... .venv/bin/python scripts/score_judge_calibration.py \
    --items data/evalsets/judge_calibration_gerlerb.jsonl --split validation --max-per-type 500 \
    --base-url https://api.aisc.hpi.de --model <judge> --api-key-env JUDGE_API_KEY \
    --min-kappa 0.6 --min-accuracy 0.85 --out logs/judge/valid_<judge>_v2.json
```

The mixed context relevance figures come from the same command on the `context_relevance` rows of `judge_calibration.jsonl`. Each report is `logs/judge/valid_<judge>_v2*.json`, with one `JUDGE-CAL` line per type. Calibration runs use `--split calibration --max-per-type 200`.
