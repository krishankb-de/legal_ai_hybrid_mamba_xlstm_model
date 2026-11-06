# NLI verifier threshold on 100 hand-built statute pairs

How well the verifier's entailment model separates claims a statute passage supports from claims it does not, at the threshold fixed before the run.

**Setup.** `scripts/calibrate_nli.py` with `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (CPU, fp32, pairs up to 512 tokens, the premise truncated first) on `data/manifests/nli_calibration.jsonl`. The file holds 100 premise/claim pairs, 50 entailed and 50 not, in eight kinds:

- entailed: quote 10, paraphrase 25, subset 15;
- not entailed: contradicting detail 15, negation 10, framing 8, neutral extra 9, unrelated 8.

Premises are German statute wording. Claims take the form the verifier builds: inline quotes in „…“ after a framing sentence.

**Labels.** A person reviewed every label before the model ran. The rule: label 1 only if the premise text alone entails the whole claim; legal knowledge from outside the premise does not count. The review changed 14 rows and kept the 50/50 split:

- five quote claims were rewritten because they named a party or source the premise does not (Verbraucher, Arbeitnehmer, GmbH, Grundgesetz);
- one paraphrase got a corrected reason;
- one shortened premise got its exception back, along with the negative that shares it;
- three negatives moved from neutral to contradicting;
- four negatives that real law partly supports got values with no legal basis.

**Threshold.** 0.5 on the entailment probability, fixed before the script existed. It ships unless its precision is below 0.85.

## Summary

1. At 0.5: precision 0.878 (43 of 49 accepted claims are entailed), recall 0.860 (43 of 50), F1 0.869. **0.5 ships** (0.878 ≥ 0.85).
2. Precision stays under the predicted 0.90.
3. Four of the six false accepts are claims with a wrong number:
   - 14 Tage vs "einen Monat";
   - 25,000 vs 50,000 €;
   - 72 hours vs "zwölf Stunden";
   - the GmbH's 25,000 € vs the AG's 50,000 €.

   The other two: "Geschwister" accepted for "Verwandte in gerader Linie", and a framing sentence that contradicts its own quote.
4. Five of the seven false rejects are quote claims in the verifier's own form (framing sentence plus „quote“). The quote is the premise word for word, yet the model scores the pair 0.07–0.44. Quote claims are right only half the time (5/10), against 0.80–1.00 for every other kind.

## Prediction and outcome

| Prediction (stated before the run) | Outcome | Verdict |
|---|---|---|
| The NLI verifier at the pre-registered threshold reaches ≥ 0.90 precision on the 100-pair set | 0.878 | Refuted |

Ship rule as written before the run: the fixed threshold ships unless its precision is below 0.85. 0.878 is above it, so 0.5 ships.

## Results

| threshold | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.5 (fixed before the run) | 0.878 | 0.860 | 0.869 | 43 | 6 | 7 |
| 0.154 (best F1, chosen on the same 100 pairs) | 0.891 | 0.980 | 0.933 | 49 | 6 | 1 |

The best threshold is chosen in-sample, so its numbers are optimistic. It is reported, not used.

| kind | n | accuracy at 0.5 |
|---|---|---|
| entail_quote | 10 | 0.50 |
| entail_paraphrase | 25 | 0.96 |
| entail_subset | 15 | 0.93 |
| contradict_detail | 15 | 0.80 |
| contradict_negation | 10 | 1.00 |
| contradict_framing | 8 | 0.88 |
| neutral_extra | 9 | 0.89 |
| neutral_unrelated | 8 | 0.88 |

## What this licenses

- The verifier runs at 0.5. On statute-style claims about 1 in 8 accepted claims is not entailed, and about 1 in 7 entailed claims is rejected.

## Not licensed

- Precision on generated answers, on case law or on long passages: these are 100 short, hand-written statute pairs.
- That 0.154 is better: it was chosen on the same pairs it is scored on.
- Fine differences between kinds: each holds 8–25 pairs.

## Open limitations

- Wrong numbers pass the model. A check that every number in a claim appears in its passage would have caught four of the six false accepts, but it was not part of the fixed verifier and is untested on fresh pairs.
- The verifier's own quote form is the weakest kind. The quote is already checked word for word by the quote check, so scoring only the framing sentence against the passage may help. That is also untested.

## Reproduction

`.venv/bin/python scripts/calibrate_nli.py --pairs data/manifests/nli_calibration.jsonl --out logs/nli_calibration.json` (about 1.1 GB model download on the first run; minutes on a CPU). The script refuses a pairs file whose `review` is not `done`.
