# Parameter counts of the legal model configs

> Computed from the model definitions by `scripts/param_counts.py` (models built on the `meta`
> device; no job, no measurement). `tests/test_config_invariants.py::test_param_bands` fails if
> this file is stale and asserts the plan's bands (P2-W). Counts follow decision 11:
> non-embedding = total - embedding - (lm_head if untied). The MTP head is training-only and is
> reported in its own column, outside the inference counts.

| config | role | layers | non-embedding | total | MTP head (training only) | Δ non-embedding vs base |
|---|---|---|---:|---:|---:|---:|
| `hybrid_legal_base` | base; screen arm S1 (control) | mamba3x7, attentionx2, mlstmx3 | 104,066,872 | 220,753,720 | — | +0.00% |
| `transformer_legal_base` | matched Transformer (plan §4) | attentionx15 | 106,192,128 | 222,878,976 | — | +2.04% |
| `hybrid_legal_legacy` | legacy-operator ablation (decision 17; arm S5) | mlstmx3, mambax9 | 106,527,072 | 223,213,920 | — | +2.36% |
| `hybrid_legal_attn0` | screen arm S0: no attention | mamba3x9, mlstmx3 | 106,997,448 | 223,684,296 | — | +2.82% |
| `hybrid_legal_attn4` | screen arm S2: four attention layers | mamba3x5, attentionx4, mlstmx3 | 101,136,296 | 217,823,144 | — | -2.82% |
| `hybrid_legal_ds64` | screen arm S3: Mamba-3 d_state 64 | mamba3x7, attentionx2, mlstmx3 | 103,287,352 | 219,974,200 | — | -0.75% |
| `hybrid_legal_mtp` | screen arm S4: multi-token prediction | mamba3x7, attentionx2, mlstmx3 | 104,066,872 | 220,753,720 | 9,726,024 | +0.00% |

Bands (P2-W): Transformer within ±2.5% of the base; each screen variant within ±3%, its delta
reported. The legacy ablation (S5) was pre-registered at <= 1% but measures +2.36% for the
9 x mamba + 3 x mlstm shape; by the user's decision of 2026-09-27 (plan §14) it is reported as a
stated difference inside ±2.5%, not reshaped: S5 measures the old recipe, PPL/probe-only, and
the difference favours legacy, so a hybrid win is conservative and a legacy win inside the
margin is inconclusive.
