# German-law pretraining corpus: tokens, deduplication, scrub

Exact token counts for the packed pretraining corpus, with the deduplication and entity-scrub statistics behind them.

**Setup.** Collection, counts and packing ran on CPU nodes; the scrub ran on 1×H100 80 GB per task. The pack step ran on one node with 32 workers (1 h 44 min, 22.9 GB peak RSS). Token counts use the Qwen3 tokenizer. The exact counts below (from `build_summary.json` of the pack) replace an earlier estimate taken before deduplication and scrub; the comparison is in "Estimate vs exact".

## Summary

1. **Commercial-safe training tokens: 6,092,655,089** (5 sources, after scrub and deduplication; 75,026,585 more in validation). Of these 4,225,013,660 are legal text and 1,867,641,429 general German (fineweb2_de).
2. **Research arm: 7,474,326,564 training tokens**: the commercial-safe set plus multilegalpile's German rows (1,381,671,475, CC BY-NC-SA 4.0).
3. **Pretraining token budget = min(5 B, 6.09 B) = 5,000,000,000 tokens.** Both parts of the planned mix are covered: 3.5 B legal (70%) from 4.23 B available, 1.5 B general German (30%) from 1.87 B.
4. **Deduplication** over 2,885,936 documents: 30,597 within-source duplicates dropped (1.06%; dip the highest at 12.3%), 241,242 cross-source copies kept and linked (8.36%). 73% of rii decisions and 74% of multilegalpile rows have a copy in another source, mostly oldp; both copies are trained on.
5. **Scrub:** 15,960,418 person/judge/lawyer/street entities replaced across 2,886,503 documents of 7 sources, about 83 H100-hours over the completed parts.
6. **Scope:** Germany only. ris (AT), fedlex and bger (CH) and multilegalpile's 325,593 Swiss rows are out of training because pretraining covers German law only. eurlex is collected and scrubbed but out of training (565 documents cannot give the minimum 2,000-document validation split). All remain available to retrieval.

## Budget rule

The budget rule was fixed before the numbers: token budget = `min(5B, available commercial-safe)`. Available commercial-safe training tokens are 6,092,655,089 > 5 B, so **budget = 5 B**.

## Results

### Tokens per source, jurisdiction, licence and arm (exact)

Train and validation are split by document (0.1%, at least 2,000 documents per source, seed 0). Both row lengths come from one tokenisation, so the token counts are identical at 4,096 and 8,192.

| source | jurisdiction | licence | arm | train docs | train tokens | val docs | val tokens | rows @4,096 | rows @8,192 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| gii (federal statutes) | DE | DE-UrhG-5 | commercial-safe | 83,650 | 34,500,954 | 2,000 | 823,607 | 8,424 | 4,212 |
| rii (federal courts) | DE | DE-UrhG-5 | commercial-safe | 78,473 | 456,813,003 | 2,000 | 11,582,466 | 111,527 | 55,764 |
| oldp (courts, all instances) | DE | ODbL-1.0 | commercial-safe | 411,588 | 3,054,572,288 | 2,000 | 15,217,411 | 745,746 | 372,873 |
| dip (Bundestag papers) | DE | DE-UrhG-5 | commercial-safe | 27,297 | 679,127,415 | 2,000 | 45,542,409 | 165,803 | 82,902 |
| fineweb2_de (general German) | DE | ODC-By-1.0 | commercial-safe | 1,997,381 | 1,867,641,429 | 2,000 | 1,860,692 | 455,968 | 227,984 |
| **commercial-safe total** | | | | **2,598,389** | **6,092,655,089** | **10,000** | **75,026,585** | **1,487,468** | **743,735** |
| multilegalpile (German rows) | DE | CC-BY-NC-SA-4.0 | research only | 244,950 | 1,381,671,475 | 2,000 | 11,092,158 | 337,323 | 168,662 |
| **research total** | | | | **2,843,339** | **7,474,326,564** | **12,000** | **86,118,743** | **1,824,791** | **912,397** |

Not in training: eurlex (EU, CC-BY-4.0; 567 collected, 565 after its own deduplication, ~0.3 M tokens estimated) because 565 documents cannot fill the 2,000-document validation split; ris (AT), fedlex and bger (CH) and multilegalpile's CH rows because pretraining covers German law only. All remain available to retrieval.

### Deduplication (drop within a source, keep and link across)

2,885,936 documents entered (the six packed sources); 30,597 dropped, 241,242 linked. Collected minus kept equals the dropped count for every source.

| source | collected | kept | dropped within source | % | exact / near |
|---|---:|---:|---:|---:|---|
| gii | 90,255 | 85,650 | 4,605 | 5.10 | 2,209 / 2,396 |
| rii | 84,347 | 80,473 | 3,874 | 4.59 | 406 / 3,468 |
| oldp | 423,363 | 413,588 | 9,775 | 2.31 | 1,138 / 8,637 |
| dip | 33,398 | 29,297 | 4,101 | 12.28 | 1,930 / 2,171 |
| fineweb2_de | 2,000,000 | 1,999,381 | 619 | 0.03 | 156 / 463 |
| multilegalpile (DE) | 254,573 | 246,950 | 7,623 | 2.99 | 3,022 / 4,601 |

Cross-source copies, kept in both sources and linked in `data/shards/<L>/dedup.jsonl` (the later source in visiting order points to the earlier one):

| copy in | matches | exact | near | total |
|---|---|---:|---:|---:|
| multilegalpile | oldp | 62,715 | 90,768 | 153,483 |
| rii | oldp | 1,287 | 57,519 | 58,806 |
| multilegalpile | gii | 0 | 27,077 | 27,077 |
| multilegalpile | rii | 1 | 1,510 | 1,511 |
| oldp | fineweb2_de | 0 | 185 | 185 |
| multilegalpile | fineweb2_de | 37 | 58 | 95 |
| gii | fineweb2_de | 0 | 71 | 71 |
| rii | fineweb2_de | 0 | 13 | 13 |
| fineweb2_de | dip | 0 | 1 | 1 |

### Scrub (`flair/ner-german-legal`, tags PER/RR/AN/STR)

Per source, summed over its 8 parts from the final line of each completed task's log (failed and cancelled attempts excluded). All 7 scrubbed sources are listed, eurlex included.

| source | documents | entities replaced | per document | H100-hours |
|---|---:|---:|---:|---:|
| gii | 90,255 | 2,775 | 0.03 | 0.4 |
| rii | 84,347 | 553,572 | 6.56 | 3.8 |
| oldp | 423,363 | 3,872,345 | 9.15 | 27.3 |
| eurlex | 567 | 0 | 0.00 | 0.1 |
| dip | 33,398 | 719,961 | 21.56 | 19.2 |
| fineweb2_de | 2,000,000 | 9,047,139 | 4.52 | 18.9 |
| multilegalpile (DE) | 254,573 | 1,764,626 | 6.93 | 13.3 |
| **total** | **2,886,503** | **15,960,418** | | **83.0** |

### Estimate vs exact

The earlier estimate multiplied each source's collected characters by the tokens per character of a 2,000-document stride sample, before scrub and deduplication. Exact = train + validation tokens of the pack.

| source | estimate | exact | exact / estimate |
|---|---:|---:|---:|
| gii | 35,986,771 | 35,324,561 | 0.982 |
| rii | 494,546,077 | 468,395,469 | 0.947 |
| oldp | 3,151,931,553 | 3,069,789,699 | 0.974 |
| dip | 827,614,079 | 724,669,824 | 0.876 |
| fineweb2_de | 1,857,435,129 | 1,869,502,121 | 1.006 |
| multilegalpile (DE) | 1,425,721,823 | 1,392,763,633 | 0.977 |

Where a source drops many documents the gap follows its within-source drop rate (dip 12.3% → 0.876, rii 4.59% → 0.947), but not everywhere (gii drops 5.10% yet sits at 0.982); the sample estimator itself was within about 1-3% for the other sources.

### Efficiency

| step | resource | wall / cost |
|---|---|---|
| scrub, 56 parts | 1×H100 per part | 83.0 H100-hours (completed parts); largest part (oldp) ≈ 3.5 h |
| pack (dedup + split + pack at two lengths) | 32 CPUs | 1 h 44 min, 22.9 GB peak RSS; 12 GB of shards per row length |

## Discussion

- A 5 B-token pretraining budget can be trained on commercial-safe data alone, at the 70/30 legal/general mix. The square-root source weights repeat the small sources (`configs/dataset/mixture_pretrain.yaml`: gii 4.00 passes, rii 1.53, dip 1.26; oldp 0.59 and fineweb2_de 0.80 are below one).
- Every packed document was scrubbed (`meta.json` records `scrubbed: true` for all six sources), and document ids are unique within each source (the deduplication refuses repeated ids).
- multilegalpile is a shard set of its own, so the commercial-safe arm never reads CC BY-NC-SA text.

## Limitations

- No claim about **unique** legal content: 73% of rii and 74% of multilegalpile documents have a near or exact copy in oldp (or gii), and both copies are counted in the tokens above. The commercial-safe legal total therefore overstates distinct text by roughly rii's linked share.
- No precision or recall claim for the scrub at corpus scale: the counts are what the tagger replaced, not a measured miss rate. The fineweb2_de count (4.5 entities per page) includes whatever the legal NER tags in web text.
- No claim that EU law is in the pretraining data: eurlex is out of training.
- The scrub counts were not broken down by tag (PER/RR/AN/STR) here; the per-part scrub logs on the cluster (under `$SCRATCH_ROOT/data/manifests/`) hold one row per entity for that.
- Some documents are longer than the tokenizer's 131,072-token context (transformers warned five times during the pack). Packing cuts them into rows, so this is harmless for training; it matters only to anything that feeds a whole document to Qwen3.
- gii ids changed after the counts above were first collected (1,132 provisions inside articles now carry `art<n>:` ids). Retrieval chunks and a BM25 index built before that change must be rebuilt.
- fineweb2_de is capped at 2,000,000 pages (collection limit), so its 1.87 B tokens are a sampling choice, not the source's size.

## Reproduction

Wrappers: `scripts/slurm/build_corpus_array.sh`, `scripts/slurm/corpus_counts.sh`, `scripts/slurm/scrub_array.sh` (`SOURCES`, `PARTS=8`, `JURISDICTIONS="DE EU"`), `scripts/slurm/pack_corpus.sh` (`SOURCES` without eurlex). Fixes made along the way: whitespace-only spans and 4,096-character pieces in the scrub, gii rescrubbed with `FORCE=1`; earlier pack attempts failed on repeated gii ids and on eurlex falling below the validation minimum. Shards (cluster-only): `$SCRATCH_ROOT/data/shards/{4096,8192}/<source>/{train,val}/`, `build_summary.json`, `<L>/dedup.jsonl`.
