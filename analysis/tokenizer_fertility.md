# P3 results — tokenizer fertility on DACH legal text

> **Written:** 2026-09-27, plan item P3-D. **Plan:** `LEGAL_BUILD_PLAN.md` · **State:** `legal_build_state.json`.
> **Measured:** 2026-09-27 on the Mac (no cluster job; CPU tokenization only). Command, log and sample manifest in §8.
> **Precedence:** where a summary line elsewhere disagrees with this document, the tables below are the record.

## 1. Summary

1. Qwen3's tokenizer needs **2.280 tokens per word** on the pooled 1.04M-word DACH legal sample: the
   pre-registered "< 1.8" is **refuted**. The prediction is left as written above.
2. GPT-2 needs 2.872 tokens per word, so Qwen3 uses 21% fewer tokens than GPT-2 for the same legal
   text (2,367,295 vs 2,982,096); the GPT-2 prediction ("> 2.2") is confirmed.
3. Per source, Qwen3 ranges from 2.117 (Fedlex statutes) to 2.462 (Federal Supreme Court decisions);
   statutes (GII 2.168, EUR-Lex 2.178, Fedlex 2.117, RIS 2.259) are cheaper than decisions
   (RII 2.335, OLDP 2.335, BGer 2.462) and Bundestag papers (2.361).
4. General German web text (FineWeb-2, reference, not in the legal sample) costs 2.033 tokens per word
   with Qwen3: legal text is 12% more expensive than general German for the same tokenizer.

## 2. What was pre-registered, and what happened

The prediction and the method below were written on 2026-09-27 **before** any fertility was measured
(rule R4). The outcome columns are filled in afterwards; nothing above them is changed.

| Prediction (written 2026-09-27, before measuring) | Outcome | Verdict |
|---|---|---|
| Qwen3 < 1.8 tokens/word on a 1M-word DACH legal sample | **2.280** tokens/word (1,038,327 words, 2,367,295 tokens) | **REFUTED** |
| GPT-2 > 2.2 tokens/word on the same sample | **2.872** tokens/word (2,982,096 tokens) | **CONFIRMED** |

**Method, fixed before measuring.**
- **Sample.** The eight legal collectors of P3-F..M (`gii`, `rii`, `oldp`, `eurlex`, `ris`, `fedlex`,
  `bger`, `dip`), each run live on 2026-09-27 in its own order (newest first where it has one;
  RIS takes norms and decisions half and half), whole documents until the source has at least
  125,000 words, so the pooled sample is about 1,000,000 words. A single document longer than
  50,000 words is skipped, so that one Bundestag paper (some exceed 300,000 words) cannot dominate a
  source. A source that yields fewer words within its run is reported as it is. The P3-F..M
  test fixtures are reported separately; FineWeb-2 German (`fineweb2_de`, 125,000 words) is reported
  as a general-German reference and is **not** part of the legal sample.
- **Words.** Whitespace-separated tokens of `Document.text` (punctuation stays attached, so
  `§ 573 Abs. 2 BGB.` is five words).
- **Tokens.** Qwen3: `lexhybrid.data.tokenizer.encode_document` (Qwen/Qwen3-1.7B-Base at revision
  `ea980cb`; the pointer specials cannot occur). GPT-2: `openai-community/gpt2`, `add_special_tokens=False`.
- **Fertility.** Total tokens / total words, per source and pooled over the legal sample
  (a word-weighted mean). The verdicts use the pooled numbers.

## 3. Results

Tokens/word = tokens / whitespace words (method in §2). Live sample collected 2026-09-27.

| source | documents | words | Qwen3 tokens | Qwen3 tokens/word | GPT-2 tokens | GPT-2 tokens/word |
|---|---:|---:|---:|---:|---:|---:|
| gii (German statutes) | 1,267 | 125,049 | 271,127 | 2.168 | 353,069 | 2.823 |
| rii (German federal courts) | 94 | 126,892 | 296,332 | 2.335 | 353,418 | 2.785 |
| oldp (German courts, all instances) | 49 | 130,094 | 303,806 | 2.335 | 373,484 | 2.871 |
| eurlex (EU acts, German) | 516 | 125,006 | 272,318 | 2.178 | 368,726 | 2.950 |
| ris (Austrian norms and OGH decisions) | 916 | 127,925 | 288,956 | 2.259 | 359,862 | 2.813 |
| fedlex (Swiss federal statutes) | 1,845 | 125,189 | 264,975 | 2.117 | 350,194 | 2.797 |
| bger (Swiss Federal Supreme Court) | 103 | 128,728 | 316,910 | 2.462 | 380,844 | 2.959 |
| dip (Bundestag printed papers) | 13 | 149,444 | 352,871 | 2.361 | 442,499 | 2.961 |
| **pooled legal** | **4,803** | **1,038,327** | **2,367,295** | **2.280** | **2,982,096** | **2.872** |
| fineweb2_de (general German, reference) | 328 | 125,509 | 255,124 | 2.033 | 337,382 | 2.688 |
| P3 fixtures, pooled legal (32 documents) | 32 | 27,579 | 65,374 | 2.370 | 80,332 | 2.913 |

## 5. What this licenses

- Qwen3's tokenizer is markedly more efficient than GPT-2's on DACH legal text (2.28 vs 2.87 tokens per
  whitespace word, every source in the same direction), which supports decision 6 (the Qwen3
  tokenizer) over a GPT-2-style vocabulary.
- A token budget converts to roughly 1 / 2.28 = 0.44 legal words per token for planning; P6's budget is
  stated in tokens and does not change.

## 6. Not licensed

- No claim that Qwen3 is "efficient" in absolute terms for German legal text: the pre-registered
  bound (< 1.8) failed by a wide margin.
- No comparison with other tokenizers (German-specific or larger multilingual vocabularies): none were
  measured.

## 7. Open limitations

- Whitespace words keep punctuation attached (`BGB.`, `(1)`, `§`), which raises tokens/word against
  word definitions that split punctuation off; the method was fixed before measuring and is not
  changed after it.
- The sample is each collector's first documents on one day (newest decisions, first statutes), not a
  random sample of each source; a source's full corpus may differ.
- DIP's sample is 13 papers (the 50,000-word cap skipped the longest ones).

## 8. Reproduction

    .venv/bin/python scripts/measure_fertility.py --collect   # collect (kept per source) and measure
    .venv/bin/python scripts/measure_fertility.py             # measure the existing sample again

Log of the measured run: `analysis/tokenizer_fertility.log` (2026-09-27, exit 0, 19 min wall, most of
it the collectors' one-request-per-second rate limit). Sample manifest: `analysis/tokenizer_fertility_sample.jsonl`
(5,131 documents: id, SHA-256 and word count; the texts stay in `data/cache/fertility/`, git-ignored).
Tokenizers: `Qwen/Qwen3-1.7B-Base` at `ea980cb` (`lexhybrid.data.tokenizer`), `openai-community/gpt2`.
Two collector defects this run found were fixed before the measurement (EUR-Lex act without German
XHTML skipped; RIS versions that never applied skipped); neither changes a measured text.
