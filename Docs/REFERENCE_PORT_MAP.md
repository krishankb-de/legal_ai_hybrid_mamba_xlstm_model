# REFERENCE_PORT_MAP — what to port from `Reference/hybrid_model_mamba_xlstm`, and what changes

> Written 2026-09-27 from the reference at branch `h100_efficiency`, HEAD `900acee` (the `h100_mamba3_v2` branch plus the closed efficiency plan). Every path below is relative to `Reference/hybrid_model_mamba_xlstm/` (source) or the new repo root (destination). Line numbers are those of the reference at that commit. **After P1-Z the reference is gone; this file is the record.** Where a block is marked *verbatim*, copy it as written; where marked *change*, the change is spelled out. The 18 defects are in §15 and are fixed in P2, not P1 (the P1 parity test needs the faithful port).

## 0. Source → destination table (read by `tests/test_port_map.py`)

Paths in the Source column are relative to `Reference/hybrid_model_mamba_xlstm/` (the reference package is `hybrid_xmamba/`); paths in the Destination column are relative to the repo root. The Box column names the plan box that creates the destination: `tests/test_port_map.py` requires every P1 destination to exist, and every later destination once its box is ticked. While `Reference/` exists it also requires every source to exist.

| Source | Destination | Box | Action |
|---|---|---|---|
| `hybrid_xmamba/models/configuration_hybrid.py` | `lexhybrid/config/hybrid_config.py` | P1-C | port; drop `slstm_*`, `pooling_strategy`, `bidirectional_encode`, `proj_head_dropout`, `use_cache`, `use_tfla` |
| (new) | `lexhybrid/config/loading.py` | P1-R | the one yaml -> `HybridConfig` path every script uses (FM5) |
| `hybrid_xmamba/layers/normalization.py`, `hybrid_xmamba/layers/activations.py` | `lexhybrid/layers/normalization.py`, `lexhybrid/layers/activations.py` | P1-D | port |
| `hybrid_xmamba/layers/rotary.py` | `lexhybrid/layers/rotary.py` | P1-E | port; rename the document-id argument to `doc_ids` |
| `hybrid_xmamba/kernels/ssd/{ssd_interface,ssd_reference,__init__}.py` | `lexhybrid/kernels/ssd/{ssd_interface,ssd_reference,__init__}.py` | P1-F | port verbatim; `doc_ids` |
| `hybrid_xmamba/kernels/tfla/tfla_interface.py`, `tests/test_mamba3_numerics.py` | `lexhybrid/kernels/tfla/{tfla_interface,tfla_reference,__init__}.py` | P1-G | port; the fp64 oracle moves into the package |
| `hybrid_xmamba/kernels/selective_scan/scan_interface.py` | `lexhybrid/kernels/selective_scan/{scan_interface,__init__}.py` | P1-H | port (legacy Mamba-1; ablation only) |
| `hybrid_xmamba/layers/mamba3_block.py` | `lexhybrid/layers/mamba3_block.py` | P1-I | port; `doc_ids` |
| `hybrid_xmamba/layers/mlstm_block.py` | `lexhybrid/layers/mlstm_block.py` | P1-J | port; delete `_slow_forward` and `use_tfla` |
| `hybrid_xmamba/layers/attention_block.py` | `lexhybrid/layers/attention_block.py` | P1-K | port |
| `hybrid_xmamba/layers/mamba_block.py` | `lexhybrid/layers/mamba_block.py` | P1-L | port |
| `hybrid_xmamba/layers/hybrid_block.py` | `lexhybrid/layers/hybrid_block.py` | P1-M | port; delete the `slstm` branch and import |
| `hybrid_xmamba/models/hybrid_lm.py` | `lexhybrid/models/hybrid_lm.py`, `lexhybrid/utils/arch_fingerprint.py` | P1-N | port lines 1–600; drop the image-prefix arguments, `AttentionPooling`, `HybridTextEncoder` |
| `scripts/evaluate_report_generation.py` | `lexhybrid/decoding/generate.py` | P1-N | port `beam_search_decode` as `beam_search_uncached` (no image prefix): the reference for the cached beam |
| `hybrid_xmamba/training/{optimizer,schedulers,metrics}.py` | `lexhybrid/training/{optimizer,schedulers,metrics}.py` | P1-O | port; `compute_bits_per_byte` -> `compute_bits_per_token` |
| `hybrid_xmamba/training/signal_callbacks.py`, `hybrid_xmamba/training/lightning_module.py` | `lexhybrid/training/callbacks.py`, `lexhybrid/training/lightning_module.py` | P1-P | port `SignalCheckpointCallback` and the base `HybridLightningModule` (lines 19–283) |
| `hybrid_xmamba/utils/run_metadata.py`, `hybrid_xmamba/utils/checkpoint_arch.py`, `scripts/evaluate_lm.py`, `scripts/train_report_generation.py` | `lexhybrid/utils/run_metadata.py`, `lexhybrid/utils/checkpoint.py` | P1-Q | port; `git_*` -> `sync_stamp`, `tree_hash`, `git_head`; prefix stripping (`evaluate_lm.py:223-262`) and the >50% guard (`train_report_generation.py:271-287`) join the arch sniffer |
| `scripts/evaluate_lm.py`, `scripts/verify_stage0_handoff.py` | `scripts/evaluate_lm.py`, `scripts/verify_handoff.py` | P1-R | port; yaml via `from_hydra`, guarded load, parquet shards |
| `scripts/performance_profile.py`, `scripts/check_operator_equivalence.py` | `scripts/performance_profile.py`, `scripts/check_operator_equivalence.py` | P1-S | port |
| `scripts/evaluate_report_generation.py`, `scripts/bootstrap_compare.py`, `scripts/analyze_generation_diversity.py` | `lexhybrid/eval/text_metrics.py`, `scripts/bootstrap_compare.py`, `scripts/analyze_generation_diversity.py` | P1-T | ROUGE-L and corpus BLEU verbatim (stdlib); bootstrap without CheXbert, with per-item metric files |
| `tests/test_mamba3_numerics.py`, `tests/test_scan_correctness.py` | `tests/test_numerics_fp64.py` | P1-U | new file, same structure (§10) |
| `configs/model/hybrid_150m_m3.yaml`, `configs/model/transformer_150m_baseline.yaml`, `configs/trainer/h100_single_gpu.yaml`, `configs/trainer/h100_multi_ddp.yaml`, `configs/callbacks/default.yaml`, `configs/config.yaml` | `configs/model/ref_hybrid_m3.yaml`, `configs/model/ref_transformer.yaml`, `configs/trainer/h100_single_gpu.yaml`, `configs/trainer/h100_multi_ddp.yaml`, `configs/trainer/cpu_debug.yaml`, `configs/callbacks/default.yaml`, `configs/config.yaml`, `configs/dataset/synthetic.yaml`, `lexhybrid/data/synthetic.py`, `tests/test_config_invariants.py` | P1-V | port with the changes in §11.6; reference replicas keep GPT-2 vocab and exact parameter counts |
| `scripts/validate.sh` | `scripts/validate.sh`, `scripts/check_configs.py`, `scripts/smoke_model.py`, `tests/test_plan_state.py` | P1-W | port; gates per §13; the inline heredoc checks become files |
| (new) | `scripts/check_env.py`, `tests/test_check_env.py` | P1-W1 | new (plan §8.5) |
| (new) | `scripts/check_repo_hygiene.py`, `tests/test_repo_hygiene.py` | P1-W2 | new (plan §8.1) |
| (new) | `.github/workflows/ci.yml` | P1-W3 | new (plan §8.4) |
| (new) | `tests/test_ci_workflow.py` | P1-W4 | new |
| (new) | `scripts/dump_reference_parity.py`, `tests/fixtures/reference_parity/{mamba3,mlstm,attention,mamba,model}.pt`, `tests/test_reference_fixtures_fresh.py` | P1-X | new; the only file allowed to name the reference package |
| (new) | `tests/test_reference_parity.py`, `tests/test_port_map.py`, `tests/test_plan_references.py` | P1-Y | new |
| `.gitignore` | `.gitignore`, `.rsync-exclude`, `tests/test_rsync_rules.py`, `tests/test_packaging.py` | P1-A | port the guards in §12; the rsync rules include `tests/fixtures/**` before excluding `*.pt` |
| `scripts/mamba3_arms.py` | `scripts/screen_arms.py` | P2-X | new file, same pattern (§11.5) |
| `scripts/setup_env_h100.sh` | `scripts/slurm/setup_env.sh` | P4-E | new file from §11.1/§11.3 |
| `scripts/preflight_mamba3_h100.sh` | `scripts/slurm/preflight.sh` | P4-I | new file from §11.1 |
| `scripts/profile_e1_confirm_h100.sh` | `scripts/slurm/profile.sh` | P4-P | new file from §11.1/§11.8 |
| `scripts/mamba3_watch.sh` | `scripts/slurm/watch.sh`, `tests/test_slurm_wrappers.py` | P4-V | new file from §11.7 |
| `scripts/train_stage0_h100.sh`, `scripts/screen_arms_h100.sh` | `scripts/slurm/train_pretrain_1gpu.sh`, `scripts/slurm/screen_array.sh` | P5-B | new files from §11.1/§11.4 |
| `scripts/train_stage0_150m_h100.sh` | `scripts/slurm/train_pretrain_4gpu.sh` | P6-A | new file from §11.1/§11.4 |
| `scripts/verify_stage0_handoff.sh` | `scripts/slurm/verify_handoff.sh` | P6-L | new file from §16 |

**Dropped, never ported:** `models/prefix_mapper.py`, `models/vision_hybrid.py`, `layers/slstm_block.py`, `training/moco_queue.py`, `training/contrastive_eval_callback.py`, every Lightning module other than `HybridLightningModule`, `utils/registry.py` (imports the vision model; nothing needs it), `utils/generation.py` (placeholder; the real decode is on the model), `utils/initialization.py` (unused by the model), `scripts/train_contrastive.py`, `train_report_generation.py`, `evaluate_report_generation.py`, `evaluate_retrieval.py`, `evaluate_cxr_retrieval.py`, `evaluate_sts.py`, `score_chexbert_standalone.py`, `mimic_cxr_vendor/`, `build_mimic_cxr_local.*`, every `*_rrg.yaml`, every `cxr_*`/`mimic_*`/`indiana_*` dataset yaml, every `distill/*.yaml` (the KD they describe was never implemented), the Colab notebooks, the root-level `*.py` debug scripts, `v1_scripts/`, `Research_docs/`, `hpi_results_logs/`.

## 1. `HybridConfig` (`models/configuration_hybrid.py`)

Dataclass fields with defaults (lines 49–171), grouped. Keep every one not struck through.

```
# core
vocab_size: int = 50257            # legal: 151936
dim: int = 768
num_layers: int = 12
layer_pattern: List[Literal["mamba","mamba3","mlstm","slstm","attention"]] = ["mamba","mamba","mlstm"]   # drop "slstm"
# Mamba-1
state_size: int = 16 ; conv_size: int = 4 ; expand_factor: int = 2 ; dt_rank: Optional[int] = None ; use_fast_path: bool = True
scan_impl: str = "legacy"          # "legacy" | "exact"
dt_init_strategy: str = "none"     # "none" | "mamba"
dt_min: float = 1e-3 ; dt_max: float = 1e-1
tfla_impl: str = "legacy"          # "legacy" | "exact"
# Mamba-3 (every flag defaults to the Mamba-2 reduction)
mamba3_d_state: int = 128 ; mamba3_head_dim: int = 64 ; mamba3_ngroups: int = 1 ; mamba3_chunk_size: int = 64
mamba3_use_conv: bool = True ; mamba3_conv_size: int = 4 ; mamba3_use_trapezoid: bool = False ; mamba3_use_rope: bool = False
mamba3_rope_fraction: float = 0.5 ; mamba3_theta_max: float = 1.0 ; mamba3_dt_min: float = 1e-3 ; mamba3_dt_max: float = 1e-1
mamba3_dt_init_floor: float = 1e-4 ; mamba3_bc_bias: str = "none" ; mamba3_mimo_rank: int = 1 ; mamba3_a_mode: str = "static"
mamba3_a_floor: float = 1e-4 ; mamba3_dt_limit: float = 1.0 ; mamba3_use_outproj_norm: bool = False
# (P2-A adds) mamba3_expand_factor: int = 2
# mLSTM
head_dim: int = 64 ; num_heads: Optional[int] = None ; proj_factor: int = 2
mlstm_gate_soft_cap: float = 15.0 ; mlstm_input_gate_bias_init: float = -10.0 ; mlstm_forget_gate_bias_init: float = 0.0
# (drop) use_tfla: bool = True      (P2-A adds) mlstm_chunk_size: int = 128
# attention
attn_dropout: float = 0.0 ; rope_theta: float = 10000.0        # legal: 500000
# (drop) slstm_hidden_dim, slstm_num_heads, use_exponential_gate
# shared
norm_type: str = "rms" ; norm_topology: str = "pre_rms"   # "pre_rms" | "hybrid" | "hybrid_bc"
use_mlp: bool = True ; mlp_ratio: float = 4.0
max_position_embeddings: int = 2048 ; dropout: float = 0.1 ; initializer_range: float = 0.02
# (drop) use_cache ; tie_word_embeddings: bool = False   # legal default True (P2-M)
use_gradient_checkpointing: bool = False
# (drop) pooling_strategy, bidirectional_encode, proj_head_dropout
model_type: str = "hybrid_xmamba"                          # legal: "lexhybrid"
```

`__post_init__` (173–220) validates `layer_pattern` entries, `norm_topology`, `scan_impl`, `tfla_impl`, `mamba3_bc_bias`, `mamba3_a_mode`, `dt_init_strategy`; derives `num_heads = max(1, dim // head_dim)` and `dt_rank = dim // 16` when `None`. `get_layer_config(layer_type, layer_idx)` (222–284) maps prefixed keys per type and raises on unknown types.

`from_hydra` (286–313), *verbatim in spirit*:

```python
@classmethod
def from_hydra(cls, model_cfg, **overrides) -> "HybridConfig":
    valid = {f.name for f in dataclasses.fields(cls)}
    raw = {k: v for k, v in dict(model_cfg).items() if k in valid and k != "model_type"}
    raw.update({k: v for k, v in overrides.items() if k in valid})
    if "layer_pattern" in raw and raw["layer_pattern"] is not None:
        raw["layer_pattern"] = [str(x) for x in raw["layer_pattern"]]
    return cls(**raw)
```

Every training entry point builds the config through `from_hydra`; a hand-written kwarg list is how three fields were silently dropped (job 2513007).

## 2. `RMSNorm`, activations (`layers/normalization.py`, `layers/activations.py`)

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):  # computes in x.dtype -- defect 10 (P2-E adds an fp32 upcast)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

`activations.py`: `exponential_activation = torch.exp` (used by mLSTM's input gate), `silu_activation`, `swish_activation`, module wrappers.

## 3. SSD kernels (`kernels/ssd/`)

Shapes: `x (B,L,H,P)`, `dt (B,L,H)`, `A (H,)` strictly negative (or `(B,L,H)` data-dependent), `B/C (B,L,G,N)`, `D (H,)`, state `(B,H,P,N)`.

```python
def segsum(v):                       # out[..., t, s] = sum_{i=s+1..t} v_i ; -inf above the diagonal ; diagonal 0
def _segment_ids(doc_ids, batch, seqlen, device):   # monotone segment index; zeros when doc_ids is None
def ssd_chunked_scan(x, dt, A, B, C, D=None, chunk_size=64, doc_ids=None, coeff=None, extra_terms=None) -> y
def ssd_step(x_t, dt_t, A, B_t, C_t, state, D=None, coeff=None, extra_terms=None) -> (y_t, new_state)
def ssd_sequential_reference(x, dt, A, B, C, D=None, doc_ids=None, coeff=None, extra_terms=None, dtype=torch.float64) -> y
```

Inside `ssd_chunked_scan` (verbatim mechanics; keep the comments' intent):
- `chunk_size = max(1, min(chunk_size, seqlen))`; `terms = [(dt if coeff is None else coeff, B, x)] + extra_terms`; `dA = dt * A`.
- Padding to a multiple of `chunk_size`: `dA`, `C`, every term padded with zeros; `seg` padded with `int(seg.max().item()) + 1` — **defect 7**: this is why the final state is zero on partial chunks and why prefill was token-by-token. P2-F pads `seg` with the last real id instead: `torch.cat([seg, seg[:, -1:].expand(-1, pad)], 1)`; padded `dt` is 0 so decay is 1 and padded terms are 0, so `y[:, :seqlen]` is unchanged.
- Three boolean masks: `same = seg_c[..., :, None] == seg_c[..., None, :]` (cs×cs), `carry_ok = seg_c == prev_seg[..., None]` where `prev_seg[:, ci] = seg_c[:, ci-1, -1]` (−1 for the first chunk), `end_ok = seg_c == seg_c[:, :, -1:]`.
- `mask = exp(segsum(dA_c.permute(0,1,3,2))) * same.unsqueeze(2)`; intra-chunk `y += einsum("bchts,bcsh,bcshp->bcthp", (mask*CB).to(cdtype), co.to(cdtype), x.to(cdtype))` with `CB = einsum("bctgn,bcsgn->bcgts", C_c, B_c).repeat_interleave(rep, dim=2)`.
- Inter-chunk loop over `nc`: `offset_ci = einsum("bhpn,bthn->bthp", state, C_h[:, ci]) * exp(A_cum_ci) * carry_ok[:, ci]`; `state = exp(A_cum_ci[:, -1]) * carry_ok[:, ci, -1] * state + einsum("bth,bth,bthp,bthn->bhpn", exp(A_cum_ci[:, -1:] - A_cum_ci) * end_ok[:, ci], co[:, ci], x[:, ci], B_h[:, ci])`. The state after the last chunk is what `step()` needs (P2-F returns it).
- `cdtype = x.dtype`; every einsum is handed operands cast to `cdtype` because einsum does not promote (job 2560261). `state` is allocated in `x.dtype` and becomes fp32 by promotion — **defect 16**: P2-E allocates it `float32`.
- `y = y + D.view(1,1,-1,1) * x`; return `y.to(cdtype)[:, :seqlen]`.

`ssd_sequential_reference` zeroes the state wherever `doc_ids[:, t] != doc_ids[:, t-1]`, expects `A (H,)`, runs a plain loop.

## 4. TFLA (`kernels/tfla/tfla_interface.py`) and the mLSTM oracle

```python
_EXP_SAFE = 40.0
def _intra_chunk_sequential(q_c, k_gated, v_c, f_c) -> (num, den)      # exact fallback, depth = chunk_size
def tfla_forward_parallel(q, k, v, i_gate, f_gate, chunk_size=64, tfla_impl="legacy") -> (B,H,L,D)
def apply_tfla(q, k, v, i_gate, f_gate, tfla_impl="legacy")             # chunk size hard-wired by L: 32 if L<=128, 64 if L<=2048, else 128 -- defect 9
```

Mechanics to keep: `log_f = log(f.clamp(min=1e-6))`, `f_cum = cumsum(log_f)` per chunk; `legacy` computes `q*f_cum` and `k_gated / f_cum.clamp(1e-6)` (the clamp defect, kept only for the `hybrid_legal_legacy` arm); `exact` re-centres on `m = 0.5*log_f_cum[-1]` and, if `half_range = 0.5*max|log_f_cum[-1]| >= _EXP_SAFE`, falls back to `_intra_chunk_sequential` (a `.max()` host sync and a data-dependent Python branch). Inter-chunk loop carries `C_state (B,H,D,D)`, `n_state (B,H,D)` with `total_f_last` decay and `decay_to_end` gating; computes an `m_state` that is never applied (dead code, delete in P2-C); output `num / clamp(den, min=1.0)` (signed clamp). Padding: `f` padded with 1.0 (line 103). P2-D adds the three doc masks (§3) at: intra-chunk `scores` after the causal `masked_fill` (line 175), `carry_ok[:, ci]` on the carried numerator/denominator (206–212), `carry_ok[:, ci, -1]` on the `C_state`/`n_state` decay (219–220), `end_ok` on `decay_to_end` (223–225).

**At init with `forget_gate_bias_init 0.0` and chunk 128, `half_range ≈ 44.4 > 40`: the exact path starts on the sequential fallback** (defect 9). Decision 10 sets the bias to 3.0 (f ≈ 0.95, half-range ≈ 3.1).

fp64 oracle, from `tests/test_mamba3_numerics.py:287-314`, moves into `tfla_reference.py`:

```python
def sequential_mlstm_fp64(q, k, v, i, f, doc_ids=None):   # (B,H,L,D) each; matrix memory C, normalizer n, denominator clamp(min=1)
    C = zeros(B,H,D,D); n = zeros(B,H,D); ys = []
    for t in range(L):
        if doc_ids is not None and t > 0: reset C, n where doc_ids[:, t] != doc_ids[:, t-1]       # P2-D adds this
        f_t = f[:,:,t].clamp(min=1e-6)            # (B,H,D)
        ki = i[:,:,t] * k[:,:,t]
        C = f_t[..., None] * C + einsum("bhd,bhe->bhde", ki, v[:,:,t]);  n = f_t * n + ki
        y = einsum("bhde,bhd->bhe", C, q[:,:,t]) / clamp((n * q[:,:,t]).sum(-1, keepdim=True), min=1.0)
        ys.append(y)
    return stack(ys, dim=2)
```

## 5. `Mamba3Block` (`layers/mamba3_block.py`)

```python
class Mamba3Block(nn.Module):
    supports_cu_seqlens = True          # rename supports_doc_ids
    supports_step = True
    def __init__(self, dim, d_state=128, head_dim=64, expand_factor=2, ngroups=1, chunk_size=64, use_conv=True,
                 conv_size=4, use_trapezoid=False, use_rope=False, rope_fraction=0.5, bc_bias="none", mimo_rank=1,
                 a_mode="static", a_floor=1e-4, dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4, dt_limit=1.0,
                 theta_max=1.0, use_outproj_norm=False, use_hybrid_norm=False, **unused)
```

Guards: `bc_bias ∈ {none, zero_init, one_init}`, `a_mode ∈ {static, data_dependent}`, `rope_fraction ∈ {0.5, 1.0}`, `mimo_rank != 1 → NotImplementedError`. Sizes: `inner_dim = dim*expand_factor`, `nheads = inner_dim // head_dim`, `bc_dim = d_state*ngroups`, `n_rope_angles = int(d_state*rope_fraction)//2*ngroups`. `in_proj = Linear(dim, 2*inner + 2*bc_dim + 3*nheads + n_rope_angles, bias=False)`, split order `[z, xs, B, C, dt_raw, a_raw, trap_raw, angles]` (every slice exists whatever the flags, so arms are parameter-matched). Parameters: depthwise `conv1d` over `inner + 2*bc_dim` channels (`padding=k-1`, bias), `B_norm`/`C_norm = RMSNorm(d_state)` always, `B_bias`/`C_bias (nheads, d_state)` only when `bc_bias != none`, `trap_bias (nheads,)` only when trapezoid, `A_log (nheads,)` only when static, `dt_bias (nheads,)`, `D = ones(nheads)`, `out_norm` optional, `out_proj = Linear(inner, dim, bias=False)`.

`post_model_init()` (216–234): `dt ~ exp(U * (log dt_max − log dt_min) + log dt_min)`, clamped at `dt_init_floor`; `dt_bias = dt + log(−expm1(−dt))`; `A_log = log(U[1,16])`.

`forward(x, cache=None, doc_ids=None)` (242–284): in_proj → split → conv over `cat[xs,B,C]` (+ SiLU; with `doc_ids`, `_mask_conv_across_documents` recomputes the first `k−1` positions of each document) → `dt = softplus(dt_raw.float() + dt_bias).clamp(max=dt_limit)` → `A = −exp(A_log)` or `clamp(−heavy_tail(a_raw), max=−a_floor)` → `B_norm`, `C_norm` on `(B,L,G,N)` → `_scan` (RoPE via `cumulative_angles` + `apply_rotary` first; then bias; then `_trapezoid_terms`; then `ssd_chunked_scan(..., chunk_size=self.chunk_size, doc_ids, coeff, extra_terms)`) → `out_proj(y * silu(z))`. **Argument order is load-bearing**: `hybrid_lm.py` passes `(x, cache, doc_ids)` positionally through `torch.utils.checkpoint.checkpoint`. `cache` is ignored in forward (P2-F writes it).

`allocate_inference_cache(batch_size, device=None, dtype=float32)` (380–416): `ssm_state (B,H,P,N)`, `seen = 0`, `conv_state (B, inner+2*bc_dim, k−1)` if conv, `angle_state (B,1,n_rope_angles)` in **fp64** if rope, `B_prev (B, H or G, N)` + `x_prev (B,H,P)` if trapezoid.

`step(x_t, cache)` (418–493): accepts `(B,D)` or `(B,1,D)`; conv as an elementwise window product with `cache["conv_state"] = window[..., 1:]` (pre-activation columns); dt/A as in forward; B/C normed and `.float()`; rope increments `angle_state` in fp64 mod 2π; bias; trapezoid `lam = sigmoid(trap_raw + trap_bias)`, `coeff = lam*dt`, `beta = (1−lam)*dt*exp(dt*A)` on `B_prev`/`x_prev` when `seen > 0`; `y, cache["ssm_state"] = ssd_step(...)`; `seen += 1`; state stays fp32 regardless of autocast.

## 6. `mLSTMBlock` (`layers/mlstm_block.py`)

```python
class mLSTMBlock(nn.Module):
    supports_cu_seqlens = True ; supports_step = True
    def __init__(self, dim, head_dim=64, num_heads=None, use_tfla=True, tfla_impl="legacy", proj_factor=2,
                 gate_soft_cap=15.0, input_gate_bias_init=-10.0, forget_gate_bias_init=0.0, use_hybrid_norm=False)
```

Layers: `in_proj dim→inner*proj_factor` (no bias); `q_proj, k_proj, v_proj inner→inner` (no bias); `i_gate_proj, f_gate_proj, o_gate_proj inner→inner` (bias; per-dimension gates); `q_norm, k_norm = RMSNorm(head_dim)`; `v_norm` only under hybrid norm; `out_proj inner→dim`; gate biases set with `nn.init.constant_` at 123–124 — **defect 3**: `HybridLanguageModel._init_weights` zeroes every Linear bias afterwards and this block has no `post_model_init` (P2-B adds one). Forward (126–209): `x_inner, x_gate = in_proj(x).chunk(2)`; heads `(B,H,L,D)`; `_tanh_soft_cap(x, cap) = tanh(x/cap)*cap`; `i = exp(capped_logit.float())`, `f = sigmoid(capped)`, `o = sigmoid`; `apply_tfla(q,k,v,i,f,tfla_impl)`; `h*o` → merge heads → `* sigmoid(x_gate)` → `out_proj`. With `doc_ids` it calls `_forward_segmented` (367–395): Python loop over rows and segments, `torch.nonzero(...).tolist()` per row — **defect 15** (P2-D deletes it). `_slow_forward` (211–279) computes a different function (LSE-stabilised, |·| clamp) — deleted in P1-J together with `use_tfla`.

`allocate_inference_cache` (284–296): `{"C": (B,H,D,D), "n": (B,H,D), "seen": 0}`; no `m`. `step` (298–365): `f_eff = f.clamp(min=1e-6)`; `ki = i*k`; `C = f_eff[...,None]*C + ki ⊗ v`; `n = f_eff*n + ki`; `h = (qᵀC) / clamp((n·q).sum(-1), min=1.0)` (signed clamp); `cache["C"].to(v.dtype)` — **defect 16** casts the state to bf16 (P2-E keeps fp32).

## 7. `AttentionBlock` (`layers/attention_block.py`)

```python
def build_rope_cache(head_dim, max_seq_len, theta=10000.0, device=None, dtype=float32) -> (cos, sin)   # (L, head_dim), rotate-half layout
def apply_rope(x, cos, sin)                                   # x (B,H,L,Dh)
def build_doc_boundary_attn_mask(doc_ids) -> (B,1,L,L) float  # 0 / -inf ; same_doc & causal   -- dense; defect 18
class AttentionBlock(nn.Module):
    supports_cu_seqlens = True                                # no supports_step, no step(), no cache -- defect 5
    def __init__(self, dim, num_heads=None, head_dim=64, attn_dropout=0.0, rope_theta=10000.0, max_position_embeddings=1024, use_hybrid_norm=False)
```

`self.head_dim = dim // num_heads` (the `head_dim` argument only defaults `num_heads`); `qkv_proj = Linear(dim, 3*dim, bias=False)`, `out_proj`; QK-norm only under hybrid norm; RoPE tables as non-persistent buffers; `_rope_tables` rebuilds with the **default θ, not `rope_theta`** (defect 8; also positions are absolute within the packed row). Forward: `is_causal=True` without `doc_ids`, else the dense additive mask with `is_causal=False`. `cache` accepted and ignored.

## 8. `HybridBlock` and factory (`layers/hybrid_block.py`)

```python
@functools.lru_cache(maxsize=1)
def _mamba3_params() -> frozenset:    # inspect.signature(Mamba3Block.__init__) minus self, dim, **unused
class HybridBlock(nn.Module):
    def __init__(self, dim, layer_type="mamba", norm_type="rms", use_mlp=True, mlp_ratio=4.0, norm_topology="pre_rms", is_first_block=False, **layer_kwargs)
def create_hybrid_blocks(dim, num_layers, layer_pattern, norm_topology="pre_rms", **kwargs) -> nn.ModuleList   # cycles the pattern; is_first_block=(i==0)
```

Topology: `_use_hybrid_norm = norm_topology in ("hybrid","hybrid_bc")`; `_ffn_post_norm = _use_hybrid_norm and not is_first_block`; `layer_kwargs["use_hybrid_norm"] = _use_hybrid_norm`; `setdefault("use_dt_norm", norm_topology != "hybrid_bc")`. Prefix guard: any kwarg starting with `mamba3_`/`mlstm_`/`slstm_` whose suffix is not a known parameter raises; unprefixed unknowns are dropped silently. Dispatch: `mamba` allow-list `{state_size, conv_size, expand_factor, dt_rank, use_fast_path, use_hybrid_norm, use_dt_norm, scan_impl, dt_init_strategy, dt_min, dt_max}`; `mamba3` strips the `mamba3_` prefix against `_mamba3_params()` (later keys win in dataclass order, so `mamba3_conv_size`/`dt_min`/`dt_max` beat the Mamba-1 ones; `head_dim` forced from `mamba3_head_dim`; `expand_factor` shared — **defect 4**); `mlstm` allow-list `{head_dim, num_heads, use_tfla, tfla_impl, proj_factor, gate_soft_cap, input_gate_bias_init, forget_gate_bias_init, use_hybrid_norm}` **without prefix stripping** so `mlstm_*` never arrives — **defect 2**; `attention` allow-list `{num_heads, head_dim, attn_dropout, rope_theta, max_position_embeddings, use_hybrid_norm}`. `_mixer_takes_cu_seqlens = getattr(mixer, "supports_cu_seqlens", False)`. MLP `Linear(dim, dim*mlp_ratio, bias=False) → GELU → Linear(bias=False)`.

```python
def forward(self, x, cache=None, doc_ids=None):
    x = x + self.mixer(self.norm1(x), cache=cache, doc_ids=doc_ids)     # doc_ids only if the mixer takes it
    if self.use_mlp:
        x = self.norm2(x + self.mlp(x)) if self._ffn_post_norm else x + self.mlp(self.norm2(x))
    return x
def step(self, x_t, cache):    # same topology, (B,D) or (B,1,D); NotImplementedError if the mixer lacks supports_step
def allocate_inference_cache(self, batch_size, device=None, dtype=float32)   # None if the mixer has no allocator
```

## 9. `HybridLanguageModel` (`models/hybrid_lm.py`)

Construction (94–151): `HybridEmbedding` (token embedding + dropout; no positional embedding); `layer_kwargs = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}` minus the reserved `dim, num_layers, layer_pattern, norm_type, norm_topology, use_mlp, mlp_ratio`; `create_hybrid_blocks(...)`; `final_norm = RMSNorm(dim)`; `lm_head = Linear(dim, vocab, bias=False)`; **tying before `self.apply(_init_weights)`** (defect 17); then `mixer.post_model_init()` for mixers that have it; `logger.info(self.architecture_fingerprint())`.

`_init_weights`: Linear weight `normal_(0, initializer_range)`, bias zero; Embedding `normal_`. Conv1d and RMSNorm untouched.

ARCH fingerprint (153–184), *verbatim format*:

```
"ARCH " + " | ".join([
  f"layers=[{', '.join(f'{k}x{types.count(k)}' for k in sorted(set(types)))}]",
  f"norm_topology={cfg.norm_topology}", f"scan_impl={cfg.scan_impl}", f"tfla_impl={cfg.tfla_impl}", f"dt_init={cfg.dt_init_strategy}",
  "mamba3(d_state=…, head_dim=…, ngroups=…, conv=…, trapezoid=…, rope=…, bc_bias=…, a_mode=…, mimo_rank=…)"   # only if "mamba3" in types
  f"params={total:,}" ])
# legal port adds: params_nonembed=…, vocab=…, attn(n=…, rope_theta=…, kv_cache=…), mlstm(chunk_size=…, forget_bias=…), mtp=…
```

`forward(input_ids=None, inputs_embeds=None, labels=None, attention_mask=None, doc_ids=None, output_hidden_states=False, return_dict=True)` (195–284): exactly one of ids/embeds; `attention_mask` only multiplies the embeddings; gradient checkpointing `checkpoint(layer, h, None, doc_ids, use_reentrant=False)`; loss = shift-by-one `CrossEntropyLoss()` with `ignore_index=-100` — **defect 6** (no boundary mask); returns `CausalLMOutput(loss, logits, hidden_states, attentions=None)`.

Cache API (388–572): `supports_cached_decode()` = all mixers have `supports_step`; `allocate_inference_cache(batch_size, device=None, dtype=float32)` → list of per-layer dicts; `step_logits(hidden_t (B,D), caches) → (B,V)`; `prefill(hidden (B,L,D), caches)` token-by-token — **defect 7**; `_filter_logits(logits, temperature, top_k, top_p)`; `generate_cached(input_ids, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None)`; `reorder_cache(caches, index)` staticmethod: `index_select(0, index)` on tensor values, non-tensors copied; `beam_search_cached(input_ids, beam_size=3, max_new_tokens=100, length_penalty=1.0)`: batch 1, beams on the batch axis, prefill once replicated, `scores = [0, -inf, …]`, `ranked = total / length**length_penalty`, `topk` over the flat view, `reorder_cache(caches, beam_idx)`, no EOS — **defect 11**. `get_num_params(non_embedding=True)` subtracts only the token embedding (so an untied `lm_head` still counts — P2-M redefines). `get_layer_types()`.

`checkpoint_arch.FINGERPRINTS` (first name after `mixer.` that only one family owns): `mamba3 → ("dt_bias","B_bias")`, `mamba → ("dt_proj",)`, `mlstm → ("i_gate_proj",)`, `attention → ("qkv_proj",)`; `detect_prefix` reads `<prefix>layers.<i>.mixer.` off the keys; `infer_architecture` recovers `layer_pattern`, `norm_topology` (hybrid if `dt_norm`/`v_norm`/`q_norm` present), tensor-derived sizes; cannot recover `scan_impl`, `tfla_impl`, `dt_init_strategy`.

Prefix stripping (`evaluate_lm.py:223-262`): remove `model._orig_mod.`, `_orig_mod.model.`, `_orig_mod.`, `model.` prefixes; drop `projection_head.*`, `logit_scale`; then remove `lm.`. Guarded load (`train_report_generation.py:271-287`): `missing_frac = len(missing)/n_keys`; `> 0.5 → RuntimeError`, `> 0.05 → warning`; "`Missing keys: 0` is not evidence" when the model was built from the wrong yaml.

## 10. Oracle test structure (`tests/test_mamba3_numerics.py`)

```python
DELTAS = [1e-3, 1e-2, 1e-1, 0.3, 0.705, 1.0] ; CHUNKS = [8, 64] ; TOL = 1e-6
def _xfail_if(condition, reason): return [pytest.mark.xfail(strict=True, reason=reason)] if condition else []
SCAN_IMPLS = ("legacy", "exact")
def _scan_cases():   # legacy is broken where delta >= 0.3 or (delta >= 0.1 and chunk >= 64)
    for impl in SCAN_IMPLS:
        for delta in DELTAS:
            for chunk in CHUNKS:
                broken = impl == "legacy" and (delta >= 0.3 or (delta >= 0.1 and chunk >= 64))
                yield pytest.param(impl, delta, chunk, marks=_xfail_if(broken))
def sequential_selective_scan_fp64(x, dt, A, B, C, D):   # h = exp(dt*A) h + dt*B*x ; y = <C,h> + D*x ; A (dim, state)
def rel_max_err(got, want): return (got.double() - want).abs().max().item() / want.abs().max().item()
def _fixture(delta, seq_len=128, dim=8, state=16): torch.manual_seed(0); A = -arange(1..state).repeat(dim,1); x, dt=full(delta), B, C randn; D ones
```

Principles: CPU-collected and unconditionally run (never behind `cuda`); parametrised over implementation × delta/forget-bias × chunk; legacy failures are `xfail(strict=True)` so they fail loudly the moment they pass; document-boundary cases (mid-chunk, on a chunk edge, several docs) and bit-exact isolation (each document's output equals its standalone output); feature-off bit-identity (λ=1 trapezoid, zero-angle RoPE, zero-init bias); every parameter receives a gradient; `step()` equals the chunked forward; the arm table equals the state file's `arms`.

## 11. Cluster material (verbatim)

### 11.1 SLURM headers

Training wrapper (1 GPU), from `scripts/train_stage0_150m_h100.sh:19-31`:

```bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1   # ga03: ARM/Grace node; gx13v1: faulty GPU (cudaErrorContained)
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --job-name=h100_stage0_150m
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append   # pot-hpi-aisc-batch is preemptible: without this a requeue TRUNCATES the log
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
```

Array wrapper, from `scripts/screen_arms_h100.sh:266-278, 282-320`:

```bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --gpus=1
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --job-name=screen
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
ARMS="${ARMS:-S0 S2 S3 S4 S6}"
read -r -a ARM_LIST <<< "${ARMS}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -ge "${#ARM_LIST[@]}" ]; then echo "FATAL: array index ${IDX} but only ${#ARM_LIST[@]} arms"; exit 1; fi
ARM="${ARM_LIST[$IDX]}"
unset EXPERIMENT          # a stray EXPERIMENT in the submitting shell would funnel every arm into one output dir
export ARM STEPS WARMUP_STEPS VAL_EVERY
export SAVE_TOP_K="${SAVE_TOP_K_SCREEN:-0}"
bash scripts/slurm/train_pretrain_1gpu.sh
```

CPU job (env setup), from `scripts/setup_env_h100.sh:453-461`:

```bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --exclude=ga03   # ARM/Grace node; x86 .venv python -> "cannot execute binary file: Exec format error"
#SBATCH --account=aisc
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=setup_env
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
```

Preflight / watch (CPU, with qos), from `scripts/preflight_mamba3_h100.sh:344-353` and `mamba3_watch.sh:542-551`:

```bash
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --exclude=ga03,gx17v1,gx13v1
#SBATCH --mem=16G            # watch: 4G
#SBATCH --cpus-per-task=4    # watch: 2
#SBATCH --time=00:20:00      # watch: 00:10:00
#SBATCH --job-name=preflight
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
```

Profiler (1 GPU), from `scripts/profile_e1_confirm_h100.sh:668-680`: as the training header with `--qos=aisc`, `--mem=64G`, `--cpus-per-task=8`, `--time=03:00:00`.

Multi-GPU: **`#SBATCH --gpus=1` in a file is not overridden by an env var** (SLURM parses directives statically). A 4-GPU job needs its own wrapper with `--gpus=4` and the `sbatch --gpus=4` flag; never combine `--gpus` with a typed `--gres` ("Invalid GRES specification (with and without type identification)", hit live 2026-08-30). The wrapper checks `torch.cuda.device_count() >= NUM_GPUS` and exits 1 otherwise. Under DDP `trainer.max_steps` counts global optimizer steps, so the token budget per step scales with `NUM_GPUS`.

### 11.2 Environment block (every GPU wrapper), from `train_stage0_h100.sh:143-144, 179-188`

```bash
SCRATCH_ROOT="${SCRATCH_ROOT:-/sc/scratch/$USER/lexhybrid}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
export HF_HOME="${SCRATCH_ROOT}/.hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"          # training/eval jobs offline; fetch jobs set 0
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"
mkdir -p logs
source "${VENV_ACTIVATE}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')"
nvidia-smi
```

Replace the reference's `echo "branch: $(git rev-parse ...)"` with `cat .sync_stamp`.

### 11.3 `setup_env` body (reference `setup_env_h100.sh:465-508`, replaced by the uv lock — decision 18)

The reference installed torch from the cu128 index before `requirements.txt`, because the node driver is CUDA 12.8 and the default PyPI wheel (built for CUDA 13.0) fails at runtime; its venv had no editable install (job 2552094). The legal repo encodes both facts in `pyproject.toml` / `uv.lock` instead: torch comes from the explicit cu128 index on Linux, and `uv sync` installs the project editable.

```bash
if ! command -v uv &> /dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; fi
uv sync --locked                        # torch 2.11.0+cu128 from uv.lock; project installed editable
.venv/bin/python scripts/check_env.py   # Python 3.11, torch 2.11.0+cu128, CUDA 12.8, every package == uv.lock
```

### 11.4 Training command shape, from `train_stage0_h100.sh:199-234`

```bash
python scripts/train_pretrain.py \
  model=${MODEL_CONFIG} seed=${SEED} dataset=${DATASET_CONFIG} trainer=${TRAINER_CFG} distill=${DISTILL_CFG} \
  trainer.max_epochs=-1 trainer.max_steps=${MAX_STEPS} trainer.accumulate_grad_batches=${ACCUM} \
  trainer.val_every_opt_steps=${VAL_EVERY} trainer.log_every_n_steps=25 trainer.compile_model=${COMPILE} \
  dataset.batch_size=${BATCH_SIZE} dataset.row_len=${ROW_LEN} dataset.num_workers=4 \
  callbacks.checkpoint.every_n_train_steps=${CKPT_EVERY} callbacks.checkpoint.save_top_k=${SAVE_TOP_K} \
  experiment_name=${EXPERIMENT} output_dir=${SCRATCH_ROOT}/outputs/${EXPERIMENT} \
  model.learning_rate=${LR} model.warmup_steps=${WARMUP} model.gradient_clip_val=${GRAD_CLIP} model.use_gradient_checkpointing=${GRAD_CKPT} \
  +model.scheduler_name=wsd +model.beta2_schedule=true +model.beta2_start=0.999 +model.beta2_end=0.974 \
  ${RESUME_ARG} ${EXTRA_OVERRIDES}
  # ^ EXTRA_OVERRIDES deliberately unquoted: the arm's overrides are separate Hydra arguments.
  # RESUME_ARG="+resume_from_checkpoint=${OUT}/checkpoints/last.ckpt" when that file exists (defect 12: the reference never passed it).
```

Reference recipe values worth keeping: LR 4e-4 (6e-4 collapsed the 150M at step 666), warmup 2000, grad clip 0.5–1.0 (a single spike at step 24,749 collapsed a 150M run under clip 1.0; the reference then used 0.5), `GRAD_CKPT=true`, `every_n_train_steps 2000`, WSD with β2 0.999 → 0.974.

### 11.5 Arm pattern, from `scripts/mamba3_arms.py`

```python
class Arm(NamedTuple):
    config: str; overrides: Dict[str, Any]; seed: int; isolates: str; expect: List[str]; walltime: str = "12:00:00"
ARMS = {"S1": Arm("hybrid_legal_base", {}, 42, "control", ["mamba3x7","attentionx2","mlstmx3","tfla_impl=exact"]), ...}
def hydra_overrides(arm) -> List[str]: ["model.{k}={v}" for k, v in sorted(arm.overrides.items())]   # booleans lower-cased
# cmd_env: prints `export MODEL_CONFIG= SEED= MAX_STEPS= WARMUP= SAVE_TOP_K= EXPERIMENT=screen_<arm>_s<seed> EXTRA_OVERRIDES=` (shlex-quoted) + suggested walltime
# cmd_verify [--full]: builds every arm via HybridConfig.from_hydra(yaml + overrides), asserts every `expect` token is in model.architecture_fingerprint(); --full asserts the parameter band; every override key must be a HybridConfig field
```

`ARM` is resolved on the compute node: the wrapper runs `python scripts/screen_arms.py env "$ARM"` after activating the venv and `eval`s the exports; a caller-supplied `EXPERIMENT` is saved before and restored after (a probe once wrote into a screen run's directory, job 2513598).

### 11.6 Trainer / callback yamls (reference values)

`h100_single_gpu.yaml`: `accelerator gpu, devices 1, precision bf16-mixed, strategy auto, compile_model true (the distill script read the wrong key; legal reads compile_model), max_epochs 1, max_steps 10000, val_check_interval 500, check_val_every_n_epoch null, log_every_n_steps 25, accumulate_grad_batches 1, enable_checkpointing true, default_root_dir ${output_dir}, num_sanity_val_steps 2`. `h100_multi_ddp.yaml`: `devices -1, precision bf16-mixed, strategy ddp, sync_batchnorm true (drop), find_unused_parameters true (dead key; build DDPStrategy in code), max_steps 100000, val_check_interval 5000, log_every_n_steps 100`. `callbacks/default.yaml`: `checkpoint: monitor val/loss, mode min, save_top_k 3, save_last true, every_n_train_steps 500, auto_insert_metric_name false`; `lr_monitor step`. `config.yaml`: `seed 42` (expose it in every wrapper), `output_dir ./outputs/${experiment_name}` (legal: on scratch), `checkpoint_dir ${output_dir}/checkpoints`.

Reference checkpoint filename: `stage0_kd-step{step:06d}` (a `/` in a metric name once turned every save into a directory).

### 11.7 Watch script essentials (`mamba3_watch.sh`)

`set -uo pipefail`; `squeue --me --format="%.10i %.12P %.9N %.2t %.10M %.20j"`; `sacct --starttime=now-3days --user=$USER --format="JobID%-16,JobName%-18,State%-14,ExitCode%-8,Elapsed%-10,Start%-16,NodeList%-8" | grep -Ev "\.(batch|extern|[0-9]+) "`; per-run checkpoint listing (`last.ckpt`, step files); `grep -cE 'Traceback|CUDA out of memory|RuntimeError|DUE TO TIME LIMIT|\bnan\b|Segmentation fault'` per log; count run headers per log to detect a requeue that restarted from step 0; `du` only when running as a job. No python anywhere.

### 11.8 Profiler protocol (`profile_e1_confirm_h100.sh:717-731`)

```bash
point () {   # arm_name, model, seq_len, iters, extra flags...
  local arm="$1" mdl="$2" len="$3" it="$4"; shift 4
  export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/ind_${arm}_${len}"
  rm -rf "${TORCHINDUCTOR_CACHE_DIR}"; mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
  python scripts/performance_profile.py --sweep --models "${mdl}" --seq-lengths "${len}" --batch_size "${BATCH_SIZE}" \
    --num_iterations "${it}" --dtype "${DTYPE}" --output-dir "${OUTPUT_DIR}/${arm}_L${len}" "$@"
}
# one sequence length per PROCESS, one Inductor cache per point; the Transformer rows run FIRST; check effective_chunk_size in every row
```

## 12. `.gitignore` guards (as ported in P1-A)

The reference's guards, plus `Reference/`, the coverage and JUnit outputs, and the local cluster config. The file itself is grouped and commented; these are its patterns:

```
__pycache__/  *.py[cod]  *.egg-info/  build/  dist/
.venv/  .venv-vllm/  venv/  env/
Reference/
*.pth  *.pt  *.ckpt  *.safetensors  *.parquet  *.hdf5  *.h5
data/raw/  data/shards/  data/hf/  data/cache/
outputs/  checkpoints/  logs/  cluster/  wandb/  lightning_logs/  .hydra/  multirun/
.pytest_cache/  .ruff_cache/  .coverage  .coverage.*  coverage.xml  junit*.xml  htmlcov/
.DS_Store  Thumbs.db
.sync_stamp  scripts/slurm/cluster.env
*.md
!README.md  !CLAUDE.md  !LEGAL_BUILD_PLAN.md  !Docs/*.md  !analysis/*.md  !analysis/**/*.md
!tests/fixtures/**  !data/manifests/**
```

`legal_build_state.json` needs no exception: no pattern denies `*.json`.

The reference's `test_every_analysis_deliverable_can_actually_enter_the_repo` (an untracked `*.md` was once reported as committed) lives on in two places:
- `scripts/check_repo_hygiene.py` (validate gate 6) fails when a critical file (`README.md`, `CLAUDE.md`, `LEGAL_BUILD_PLAN.md`, `legal_build_state.json`, `uv.lock`, `pyproject.toml`, `.python-version`, `.gitignore`, `.github/workflows/ci.yml`) is missing or ignored, and when any existing `Docs/*.md`, `analysis/*.md`, `analysis/**/*.md` or `tests/fixtures/**` file is ignored.
- `tests/test_port_map.py` fails when a port-map destination is ignored.

## 13. `validate.sh` (as ported in P1-W)

The reference ran three gates: Hydra invariants, pytest, and a CPU smoke. The port keeps them and adds the plan, environment, static and hygiene gates. The interpreter is `$PYTHON`, else `./.venv/bin/python`, which must exist (`uv sync --locked` builds it). `set -uo pipefail` without `-e`: every gate runs and records its own status, the summary prints PASS, WARN or FAIL per gate, and the script exits 1 if any gate failed.

Modes and their pytest selections (plan §8.3; `tests/test_ci_workflow.py` asserts they match):

| Mode | Used for | Marker selection | Extras |
|---|---|---|---|
| default | every code box | `not slow and not cuda and not multigpu and not network` | shellcheck may WARN if not installed |
| `--full` | every `Pn-Z` gate | `not cuda and not multigpu and not network` | |
| `--ci` | the CI `test` job | as `--full` | coverage with the `fail_under` floor, `junit.xml`, shellcheck mandatory |
| `--coverage` | combined with a mode (P1-W5) | the mode's | coverage report |

Gates:
0. Plan and state consistent: `python3 scripts/plan_state.py check`.
1. Environment equals the lock: `uv lock --check`; `scripts/check_env.py` (interpreter, every installed version, editable install, CUDA build).
2. Static: `ruff check .`; `ruff format --check .`; `bash -n` and `shellcheck -x` on every `scripts/**/*.sh`.
3. Model configs: `scripts/check_configs.py`. For every `configs/model/*.yaml` it composes `config` with `overrides=[f"model={name}", "dataset=synthetic", "trainer=cpu_debug", "experiment_name=ci_check"]` (P3 switches the dataset to `legal_smoke`) inside `initialize_config_dir(config_dir=REPO/configs, version_base="1.3")` after `GlobalHydra.instance().clear()`. It builds `HybridConfig.from_hydra` and asserts a non-empty `layer_pattern`, `vocab_size == 151936` for legal yamls (50257 for the `ref_*` replicas), `tfla_impl` pinned when `mlstm` is in the pattern, `scan_impl` pinned when `mamba` is in the pattern, and no legacy operator outside `hybrid_legal_legacy` and `ref_*` (R8).
4. Tests: `pytest tests/ -m "<selection>" --tb=short -q`.
5. CPU smoke: `scripts/smoke_model.py` builds a four-layer model over `mamba3, mlstm, attention, mamba` with the shipping operators (`tfla_impl=exact`, `scan_impl=exact`, `norm_topology=hybrid`). It runs forward on `(2, 32)` with labels and `doc_ids`, asserts finite loss and logits, runs backward, and asserts that every `requires_grad` parameter received a gradient.
6. Repository hygiene: `scripts/check_repo_hygiene.py`, `.gitignore`-aware in every mode. The CI `hygiene` job runs it with `--ci` on the pristine checkout.

## 14. Measured numbers, job ids, incidents, stack (for sanity checks and the writeups)

Jobs: efficiency 2579631, 2579642, 2580198, 2582482, 2582775, 2583455; screen array 2552165; Stage-0 2553736; preflight 2552163; decoders 2553737/43/49; uncompiled sweep 2561023.

| Measurement | Value |
|---|---|
| Inference L=16,384, batch 4, bf16: Mamba-3 chunk 64 / chunk 128 / compiled 64 / compiled 128 / Transformer | 562.22 / 402.96 / 140.35 / **122.90** ms (7.169 GB) / 164.85 ms (7.152 GB) |
| Inference L=8,192 | 269.99 / 195.27 / 70.48 / **62.09** ms (3.824 GB) / 71.26 ms (3.776 GB) |
| Uncompiled default (2561023) L=256 / 2,048 / 16,384: Mamba-3 | 20.6 ms 0.51 GB / 86.2 ms 1.24 GB / 678.2 ms 7.09 GB (Transformer 4.9 / 16.1 / 163.9 ms) |
| Training step L=1,024 / 2,048: Transformer vs compiled chunk 128 | 29.65 vs 36.35 ms / 52.17 (7.55 GB) vs 141.14 ms (12.02 GB) |
| Training memory uncompiled L=2,048 / 8,192 | 13.9 GB / 54.4 GB |
| Chunk sweep uncompiled L=16,384: 64 / 128 / 256 / 512 | 557.8 / 398.6 / 411.8 / 512.8 ms (256 and 512 fail to compile: Inductor `SplitScan` cumsum `TypeError`) |
| Where the time went, L=16,384 uncompiled | 9 Mamba-3 layers 77.8% (scan 67.5%), 3 mLSTM 13.4%, rest 8.8%; Amdahl bound 4.50× |
| Decode, prompt 256, 64 new tokens, batch 1 | cached 0.00635 s/tok (157.6 tok/s), full recompute 0.03293 s/tok; TTFT 1.79 s vs 0.45 s |
| Equivalence | operator vs fp64 2.1e-07; logits across chunk 2.3e-05–3.4e-05; compile 3.0e-05; 0/400 decoded reports differ chunk 64 vs 128 |
| Screen 12K steps (PubMed, seed 42 / 1234) | A0 19.387 / 18.933; A1 16.294; A2 16.708 / 16.376; A2x 15.566 / 15.788; A3 16.719; A4-lo/mid/hi 16.431/16.534/16.199 (hi at 1234: 18.912); bar 0.642 |
| Stage-0 120K steps val PPL | m3 (A2x) 11.674; A2 12.550; v2 legacy 13.180; Transformer 11.222 |
| Wall per arm (1 H100, 512 rows, 24.6K tok/step, 2.6B teacher) | A2 1:20:55; A2x 5:45:19 (exact TFLA ≈ 20% slower than A2's 2.18 it/s); A1 4:11:13 (24 h wall needed); 120K backbone 57 h 23 m |
| Defect magnitudes | selective scan rel-max-err 0.92 at Δ=0.705; TFLA legacy 0.88 at forget bias 0; Δ at init 0.807 vs reference ~0.021; `_slow_forward` vs TFLA 0.42 max abs |
| Checkpoint | 2.4 GB per 150M Lightning checkpoint (GPT-2 vocab); quota incident: `save_top_k=3` + `last.ckpt` = 9.6 GB per arm, 3 of 4 seed arms died on `Disk quota exceeded` (job 2542399) |

Incidents to design against: (1) login node refuses `python` and `bash script.sh`; the refusal word-splits into "command not found" noise; a pre-submit `eval "$(python ... env A2)"` exported nothing and the job ran the wrapper's defaults — a 12K arm became a 120K A0 (job 2513581); (2) a shared Inductor cache timed two chunk sizes identically to the microsecond and produced a withdrawn 0.47 exponent; shapes compiled later in one process measured worse; (3) `strict=False` loading with the wrong yaml / `prefix_k` cost 0.0145 ROUGE-L, four times the whole architecture gap; (4) `norm_topology` silently dropped when building the config (Phase 9), `scan_impl`/`tfla_impl`/`dt_init_strategy` reached the yaml but not the model (job 2513007), `theta_max` missing from the dataclass collapsed every RoPE arm to 1166.7 PPL; (5) requeued arms restarted from step 0 because no wrapper passed `ckpt_path`; (6) a GPT-2 tokenizer on German legal text fragments compounds (unmeasured; P3-D measures).

Stack recorded on the cluster (2026-07-08 and later): Python 3.11 via uv, torch 2.11.0+cu128, transformers 5.13, lightning 2.6.5; the H100 node driver is CUDA 12.8; the Mac ran Python 3.14.

## 15. The 18 defects (fixed in P2; each with its reference line)

| # | Defect | Reference location | Fix (P2 box) |
|---|---|---|---|
| 1 | `cu_seqlens` is a `(B,L)` per-position document-id tensor, not cumulative offsets | `scripts/train.py:186-197`, every mixer | rename `doc_ids` (P1) |
| 2 | `mlstm_*` config keys never reach `mLSTMBlock` | `layers/hybrid_block.py:143-149` (no prefix strip; `_KNOWN["mlstm_"]` at 107 accepts them) | P2-A |
| 3 | mLSTM gate-bias init zeroed by `_init_weights`; no `post_model_init` | `layers/mlstm_block.py:123-124`, `models/hybrid_lm.py:186-191` | P2-B |
| 4 | Mamba-3 reads the shared `expand_factor` | `layers/hybrid_block.py:129-142` | P2-A |
| 5 | `AttentionBlock` has no `step()`/KV cache; `supports_cached_decode()` all-or-nothing | `layers/attention_block.py:210-214`, `models/hybrid_lm.py:388-395` | P2-I |
| 6 | Loss not masked at document boundaries | `models/hybrid_lm.py:261-272` | P2-L |
| 7 | Prefill token-by-token; scan pads with a fresh segment id so the returned state would be zero on partial chunks; `.item()` sync | `models/hybrid_lm.py:411-429`, `kernels/ssd/ssd_interface.py:117, 170-189` | P2-F, P2-G, P2-K |
| 8 | Attention RoPE absolute within packed rows; `_rope_tables` rebuilds with θ=10000; θ=10k too small for 8K+ | `layers/attention_block.py:192-199, 227-229` | P2-H |
| 9 | TFLA chunk size hard-wired by L; dead `m_state`; exact path on the sequential fallback at init with forget bias 0 and chunk 128 | `kernels/tfla/tfla_interface.py:152-170, 232-240, 290-298` | P2-C |
| 10 | `RMSNorm` no fp32 upcast | `layers/normalization.py:32-34` | P2-E |
| 11 | `beam_search_cached` no EOS; `length_penalty` inert; uncached reference lived in a dropped script | `models/hybrid_lm.py:510-572` | P2-N, P2-P |
| 12 | Distill script: no doc ids/EOS; `compile` vs `compile_model`; requeue restarts from 0; NaN guard returns a leaf; dataset id hard-coded; `val_check_interval` counts micro-batches; `find_unused_parameters` key dead | `scripts/train_stage0_distill.py:282-297, 229-240, 460, 535-555`; `configs/trainer/h100_multi_ddp.yaml:21` | P2-U, P6-A |
| 13 | `evaluate_lm.py` only warns on missing keys; `bits_per_byte` is bits per token | `scripts/evaluate_lm.py:377-379, 166` | P1-Q/P1-R |
| 14 | `pytest.ini` uses `[tool:pytest]` | `pytest.ini:3` | P1-A |
| 15 | `mLSTMBlock._forward_segmented` Python loop per (row, segment) | `layers/mlstm_block.py:367-395` | P2-D |
| 16 | mLSTM `step` casts `C`,`n` to the activation dtype; SSD `state` fp32 only by promotion | `layers/mlstm_block.py:354-361`, `kernels/ssd/ssd_interface.py:170` | P2-E |
| 17 | Tying before init draws the tied matrix twice; `tie_word_embeddings` default False | `models/hybrid_lm.py:136-140` | P2-M |
| 18 | Dense `(B,1,L,L)` fp32 attention mask; SDPA off the flash kernel | `layers/attention_block.py:104-127, 238-244` | P2-J |

## 16. Training utilities (verbatim formulas)

WSD (`training/schedulers.py:31-94`):

```python
def wsd_factor(step, warmup_steps, stable_steps, decay_steps, min_lr_ratio=0.0, warmup_start_factor=0.01):
    if warmup_steps > 0 and step < warmup_steps:
        return warmup_start_factor + (1.0 - warmup_start_factor) * (step / warmup_steps)
    decay_start = warmup_steps + stable_steps
    if step < decay_start: return 1.0
    if decay_steps <= 0: return min_lr_ratio
    p = min(1.0, (step - decay_start) / decay_steps)
    return max(min_lr_ratio, 1.0 - math.sqrt(p))
class WSDScheduler(LambdaLR):
    def __init__(self, optimizer, max_steps, warmup_ratio=0.01, stable_ratio=0.85, decay_ratio=0.14, min_lr_ratio=0.0, warmup_start_factor=0.01, warmup_steps=None, last_epoch=-1):
        # ratios must sum to 1; with an absolute warmup_steps: decay = round(max_steps*decay_ratio), stable = max_steps - warmup - decay
        # 120000 steps, warmup 2000 -> decay 16800, stable 101200, decay_start 103200 ; 12000 steps, warmup 500 -> decay 1680, decay_start 10320
def beta2_for_step(step, decay_start, decay_steps, beta2_start=0.999, beta2_end=0.974):   # constant before decay, linear across it
def apply_beta2_schedule(optimizer, step, decay_start, decay_steps, beta2_start=0.999, beta2_end=0.974) -> b2   # sets param_group["betas"] = (b1, b2)
```

Parameter groups (`training/optimizer.py:13-77`): no weight decay when the lower-cased parameter name contains any of `bias, norm, ln, layernorm, layer_norm, rmsnorm, embedding, embed, pos_embed, token_embedding`; `configure_optimizer(model, optimizer_name="adamw", learning_rate=3e-4, weight_decay=0.1, betas=(0.9,0.999), eps=1e-8, momentum=0.9, foreach=True)`; `foreach` only on CUDA.

Lightning base (`training/lightning_module.py:19-283`): defaults `learning_rate=3e-4, weight_decay=0.1, warmup_steps=2000, max_steps=100000, optimizer_name="adamw", scheduler_name="cosine", gradient_clip_val=1.0, compile_model=False, beta2_schedule=False, beta2_start=0.999, beta2_end=0.974`; `compile_model → self.model = torch.compile(self.model)`; `training_step` logs `train/loss`, `train/perplexity`, `train/lr`; `validation_step` logs `val/loss`, `val/perplexity`; `configure_optimizers`: `wsd` returns early with `_build_wsd_scheduler` (absolute `warmup_steps`), else cosine (`T_max = max_steps − warmup`, `eta_min = 0.1·lr`) / linear / constant wrapped in `LinearLR(start_factor=0.01)` warmup via `SequentialLR`, interval `step`; `on_before_optimizer_step` clips manually with `clip_grad_norm_` and logs `train/grad_norm` (trainer clipping off); `on_train_batch_start` applies the β2 anneal keyed on `global_step`.

`SignalCheckpointCallback(checkpoint_dir, filename="interrupt.ckpt")`: installs `SIGTERM`/`SIGUSR1` handlers on `LOCAL_RANK == 0` in `setup`; `on_exception` also saves; saves once via `trainer.save_checkpoint`; raises `SystemExit(0)` after a signal. No wrapper set `#SBATCH --signal=`; SLURM's default SIGTERM applies. Pair with `ckpt_path=last.ckpt` in the wrapper for real resume.

`write_run_metadata(cfg, output_dir, extra=None) -> Path` writes `run_metadata.json` with `timestamp_utc, python_version, argv, cwd, resolved_config` (+ `entrypoint` in `extra`); legal version replaces `git_sha, git_branch, git_dirty` with `sync_stamp` (contents of `.sync_stamp` or "") and `tree_hash` (sha256 over sorted file hashes of `lexhybrid/ configs/ scripts/`).

Handoff recipe (`verify_stage0_handoff.py/.sh`): extract `model.`-prefixed keys from `last.ckpt` (`{k[6:]: v for k, v in sd.items() if k.startswith("model.")}`) into `model_only.pt` if absent; build the model from the yaml via `from_hydra`; `load_state_dict(strict=False)`; PASS = 0 missing and 0 unexpected and > 0 norm keys; exit 0/1.

## 17. Training entry point facts (`scripts/train_stage0_distill.py`)

Startup order: `torch.set_float32_matmul_precision("high")` → `pl.seed_everything(cfg.seed, workers=True)` → dirs → `write_run_metadata` → `nvidia-smi` → tokenizer → student via `from_hydra` (optional `+lm_checkpoint=` warm start) → teacher frozen bf16 `.eval()` → data → module → callbacks (`ModelCheckpoint(dirpath, monitor="val/loss", mode="min", save_top_k, save_last=True, every_n_train_steps, filename="stage0_kd-step{step:06d}")`, `LearningRateMonitor("step")`, `SignalCheckpointCallback`) → `TensorBoardLogger(save_dir=log_dir, name=...)` → `pl.Trainer(max_steps, max_epochs, accelerator, devices, precision, accumulate_grad_batches, val_check_interval, log_every_n_steps, callbacks, logger)` → `trainer.fit(module, train_dl, val_dl, ckpt_path=cfg.get("resume_from_checkpoint"))`. Validation is pure CE. The NaN guard returned `torch.tensor(0.0, requires_grad=True)` and raised after 50 hits (P2-U replaces it with an optimizer-step skip). The reference's KD was mean-pooled cosine on hidden states (never logits); the legal KD is logit-KL and is new code (P2-T).

## 18. Evaluation script facts

`evaluate_lm.py`: `--checkpoint --model-config --layer-pattern --dataset --split --batch-size 4 --max-length 1024 --max-batches --output-dir --throughput --generate --device --norm-topology`; token-weighted perplexity; throughput at `[128,256,512,1024]` with 3 warmup / 10 trials on random ids; peak memory via `max_memory_allocated`; `results.json` keys `model_config, checkpoint, dataset, split, total_params, test_perplexity, test_loss, bits_per_byte(→bits_per_token), num_tokens_evaluated, peak_gpu_memory_gb, throughput, generation_samples, timestamp, tokens_per_second`.

`performance_profile.py`: `--model/--models --seq-lengths --batch-sizes/--batch_size 4 --num_iterations 10 --dtype {fp32,fp16,bf16} --backward --sweep --decode --per-layer --prompt-len --new-tokens --attn-backend {auto,math,flash,efficient} --chunk-size --compile --output-dir`; `measure_point`: 3 warmups, `reset_peak_memory_stats`, median of `perf_counter` between syncs, OOM → `{"oom": True}`; `fit_log_slope` least squares of log(y) on log(L) per (model, batch); outputs `efficiency_curves.csv/json` with `points` + `scaling_exponents`, each row carrying `attn_backend, compiled, chunk_size, effective_chunk_size`; `layer_split.json/csv` via forward hooks + CUDA events + a self-restoring patch on `ssd_chunked_scan`, Amdahl = `1/(1−share)`; decode: TTFT, s/token, second-half/first-half growth, cached vs full.

`check_operator_equivalence.py`: `R1_TOLERANCE = 1e-4`; `rel_max_err` in fp64; operator cases contiguous and boundaries inside a chunk (`seg[:,100:]=1; seg[1,173:]=2`) vs `ssd_sequential_reference`, chunk 64 baseline; pass rule `vs_base ≤ 1e-4 and vs_oracle ≤ max(2·base_err, 1e-4)`; model logits vs baseline per chunk size, optionally under compile; flags `--device --model --chunk-sizes --seq-length --compile --skip-model`; exit 0/1.

`bootstrap_compare.py`: paired index resample per draw for both systems, 1,000 draws, percentile CI, `significant = CI excludes 0`, `frac_sign_flipped`; Markdown table `| metric | A | B | diff | 95% CI | verdict |`. `analyze_generation_diversity.py`: exact-duplicate clusters, distinct-1..4, TTR, sampled self-BLEU-4 (300×40), mean tokens; pre-registered verdict thresholds.

## 19. Writeup skeleton

See `Docs/analysis_TEMPLATE.md`. The reference's `analysis/mamba3_results.md` order: header blockquote (branch, plan/state names, measurement window, where the baselines come from, precedence rule "the seed tables below are the record") → §1 Summary as numbered findings including nulls → §2 the defect table → §3 screen table with walls → §4 backbone gate → §5 downstream with the decision rule first, then `mean ± SD` tables with verdict columns, per-seed tables that show contradictions, paired-bootstrap tables → §6 efficiency → §7 "What this licenses" / "Not licensed" / "Open limitations" → §8 Reproduction with every job id.
