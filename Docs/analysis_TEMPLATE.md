# <Phase> results — <one-line title>

> **Written:** <YYYY-MM-DD>, plan item <Pn-X>. **Plan:** `LEGAL_BUILD_PLAN.md` · **State:** `legal_build_state.json`.
> **Measured:** <node(s)>, <N>×H100 80 GB, bf16, <batch / rows / steps>, jobs <id, id, id>. Logs under `cluster/logs/<name>_<id>.log`.
> **Precedence:** where a summary line elsewhere disagrees with this document, the seed tables below are the record. Where an earlier number in this repo disagrees, the reason is stated here rather than the older number quietly replaced.

## 1. Summary

1. <Finding, with the number and the job id.> (job <id>)
2. <Finding.>
3. <Null result, stated as a null.>

## 2. What was pre-registered, and what happened

| Prediction (written <date>, before submission) | Outcome | Verdict |
|---|---|---|
| <copy the italic line from the plan> | <measured> | CONFIRMED / REFUTED / PARTIAL |

Bar / decision rule as written before the numbers: <rule>. Bar value: <n> (<measured SD or floor>).

## 3. Results

### 3.1 Seed table

| arm | seed | val PPL | late-25% PPL | MQAR | statute-recall | multi-hop | s/step | peak GB | job |
|---|---|---|---|---|---|---|---|---|---|
| | 42 | | | | | | | | |
| | 43 | | | | | | | | |
| **mean ± SD** | | | | | | | | | |

### 3.2 Paired comparisons

Decision rule (decision 16): paired mean > one baseline seed SD **and** sign at ≥ 2/3 seeds; bootstrap CIs beside, never instead.

| metric | A | B | diff | 95% CI (1,000 resamples by question) | seed SD | verdict |
|---|---|---|---|---|---|---|

### 3.3 Per-seed calls (to show where seeds disagree)

| metric | seed 42 | seed 43 | seed 44 |
|---|---|---|---|

## 4. Efficiency (if measured)

| L | ours (ms / GB) | Transformer (ms / GB) | ratio | job |
|---|---|---|---|---|

One length per process, one Inductor cache per point; `effective_chunk_size` checked in every row.

## 5. What this licenses

- <claim the numbers support, at the width they support it>

## 6. Not licensed

- <claim the numbers do not support, and why>

## 7. Open limitations

- <what was not measured; what would change the verdict>

## 8. Reproduction

Key job ids: <list>. Wrappers: `scripts/slurm/<name>.sh` with env `<levers>`. Configs: `configs/model/<yaml>`. Checkpoints (cluster-only): `<artifact keys from legal_build_state.json>`. Every number above has a job id and a log path (R12).
