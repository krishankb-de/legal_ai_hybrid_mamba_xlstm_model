"""Operator-level numerical correctness against float64 sequential oracles (plan P1-U).

Ported from the reference's ``tests/test_mamba3_numerics.py`` (M1-M4) and
``tests/test_scan_correctness.py`` (Phase 14C). These tests PIN known defects as well as prove the
fixes: every ``legacy`` case that is broken is ``xfail(strict=True)``, so it fails loudly the moment
it starts passing, and every ``exact`` case must pass. Everything is CPU-collected and runs
unconditionally -- the reference's old kernel tests asserted only shape/NaN and were CUDA-gated,
which is how the defect survived.

The defect: both chunked recurrences computed ``A_cum * cumsum(Bx / A_cum.clamp(eps))``; wherever
the clamp fires, a token's own contribution to the state is annihilated rather than perturbed.
"""

import pytest
import torch

from lexhybrid.kernels.selective_scan.scan_interface import (
    selective_scan,
    selective_scan_exact,
    selective_scan_parallel,
    selective_scan_sequential_reference,
)
from lexhybrid.kernels.ssd import ssd_chunked_scan, ssd_sequential_reference, ssd_step
from lexhybrid.kernels.tfla import TFLAFallbackError, sequential_mlstm_reference, tfla_forward_parallel
from lexhybrid.layers.mamba_block import MambaBlock
from lexhybrid.layers.rotary import TWO_PI, apply_rotary, cumulative_angles

# Deltas span the reference Mamba init range logU[1e-3, 1e-1] and the range the reference
# actually operated in (Delta ~ 0.70 pre_rms / 0.82 hybrid).
DELTAS = [1e-3, 1e-2, 1e-1, 0.3, 0.705, 1.0]
CHUNKS = [8, 64]
TOL = 1e-6

_FLIPS = "strict=True, so it fails loudly the moment it starts passing."
DEFECT_SCAN = f"reference M1: divide-and-clamp in the legacy selective scan. {_FLIPS}"
DEFECT_TFLA = f"reference M1: divide-and-clamp in the legacy TFLA intra-chunk term. {_FLIPS}"
SCAN_IMPLS = ("legacy", "exact")


def _xfail_if(condition: bool, reason: str = DEFECT_SCAN):
    return [pytest.mark.xfail(strict=True, reason=reason)] if condition else []


def rel_max_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return (got.double() - want.double()).abs().max().item() / want.double().abs().max().item()


# ---------------------------------------------------------------------------------------------
# Selective scan (Mamba-1)
# ---------------------------------------------------------------------------------------------


def sequential_selective_scan_fp64(x, dt, A, B, C, D):
    """float64 ground truth for the Mamba-1 (S6) recurrence, written independently of the kernel.

    h_t = exp(dt_t * A) . h_{t-1} + (dt_t * B_t) x_t;  y_t = <C_t, h_t> + D * x_t
    """
    x64, dt64 = x.double(), dt.double()
    A64, B64, C64, D64 = A.double(), B.double(), C.double(), D.double()
    batch, seq_len, dim = x64.shape
    h = torch.zeros(batch, dim, A64.shape[1], dtype=torch.float64)
    ys = []
    for t in range(seq_len):
        decay = torch.exp(dt64[:, t].unsqueeze(-1) * A64)
        inp = (dt64[:, t].unsqueeze(-1) * B64[:, t].unsqueeze(1)) * x64[:, t].unsqueeze(-1)
        h = decay * h + inp
        ys.append(torch.einsum("bdn,bn->bd", h, C64[:, t]) + D64 * x64[:, t])
    return torch.stack(ys, dim=1)


def _scan_fixture(delta: float, seq_len: int = 128, dim: int = 8, state: int = 16):
    """Inputs matching the shipped init: A = -[1..N] repeated per channel."""
    torch.manual_seed(0)
    A = -torch.arange(1.0, state + 1).repeat(dim, 1)
    return (
        torch.randn(1, seq_len, dim),
        torch.full((1, seq_len, dim), delta),
        A,
        torch.randn(1, seq_len, state),
        torch.randn(1, seq_len, state),
        torch.ones(dim),
    )


def _scan_cases():
    """Measured: chunk=8 rescues delta=0.1 but nothing rescues delta>=0.3 on the legacy path."""
    for impl in SCAN_IMPLS:
        for delta in DELTAS:
            for chunk in CHUNKS:
                broken = impl == "legacy" and (delta >= 0.3 or (delta >= 0.1 and chunk >= 64))
                yield pytest.param(impl, delta, chunk, marks=_xfail_if(broken))


def _delta_cases():
    for impl in SCAN_IMPLS:
        for delta in DELTAS:
            yield pytest.param(impl, delta, marks=_xfail_if(impl == "legacy" and delta >= 0.1))


@pytest.mark.parametrize("scan_impl,delta,chunk_size", list(_scan_cases()))
def test_selective_scan_chunked_matches_sequential_reference(scan_impl, delta, chunk_size):
    x, dt, A, B, C, D = _scan_fixture(delta)
    want = sequential_selective_scan_fp64(x, dt, A, B, C, D)
    impl = selective_scan_parallel if scan_impl == "legacy" else selective_scan_exact
    err = rel_max_err(impl(x, dt, A, B, C, D, chunk_size=chunk_size), want)
    assert err <= TOL, f"scan_impl={scan_impl} delta={delta} chunk={chunk_size}: rel-max-err {err:.3e}"


@pytest.mark.parametrize("scan_impl,delta", list(_delta_cases()))
def test_selective_scan_public_api_matches_sequential_reference(scan_impl, delta):
    x, dt, A, B, C, D = _scan_fixture(delta)
    want = sequential_selective_scan_fp64(x, dt, A, B, C, D)
    err = rel_max_err(selective_scan(x, dt, A, B, C, D, scan_impl=scan_impl), want)
    assert err <= TOL, f"scan_impl={scan_impl} delta={delta}: rel-max-err {err:.3e}"


@pytest.mark.parametrize("scan_impl,delta", list(_delta_cases()))
def test_mamba_block_slow_forward_matches_sequential_reference(scan_impl, delta):
    """``use_fast_path=False`` shares the one implementation (the reference once had a second copy)."""
    dim, state = 8, 16
    x, dt, A, B, C, D = _scan_fixture(delta, dim=dim, state=state)
    block = MambaBlock(dim=dim, state_size=state, expand_factor=1, scan_impl=scan_impl).eval()
    with torch.no_grad():
        block.D.copy_(D)
        got = block._slow_forward(x, dt, A, B, C)
    assert rel_max_err(got, sequential_selective_scan_fp64(x, dt, A, B, C, D)) <= TOL


def test_legacy_scan_path_is_unchanged():
    """``scan_impl="legacy"`` must stay the original operator bit for bit (the ablation arm needs it)."""
    x, dt, A, B, C, D = _scan_fixture(0.705, seq_len=256)
    direct = selective_scan_parallel(
        x.float(), dt.float(), A.float(), B.float(), C.float(), D.float(), chunk_size=64
    )
    assert torch.equal(selective_scan(x, dt, A, B, C, D, scan_impl="legacy"), direct)


# Scan error bound (reference Phase 14C-1): one-sided regression guards on the known defect.
EXACT_TOL = 1e-5  # fp32 floor ~1e-7; the smallest defect signal on the grid is 5.6e-2
DEFECTIVE_CEILING = 2.0  # the defect must not exceed this; it is ~1.08 in the reference


def _audit_inputs(delta, batch=2, seq_len=128, dim=8, state_size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq_len, dim, generator=g)
    dt = torch.full((batch, seq_len, dim), float(delta))
    A = -torch.arange(1, state_size + 1, dtype=torch.float32).repeat(dim, 1)
    B = torch.randn(batch, seq_len, state_size, generator=g)
    C = torch.randn(batch, seq_len, state_size, generator=g)
    D = torch.randn(dim, generator=g)
    return x, dt, A, B, C, D


def _measure(delta, chunk_size):
    x, dt, A, B, C, D = _audit_inputs(delta)
    return rel_max_err(
        selective_scan_parallel(x, dt, A, B, C, D, chunk_size=chunk_size),
        sequential_selective_scan_fp64(x, dt, A, B, C, D),
    )


def _clamp_fire_fraction(delta, chunk_size, dim=8, state_size=16):
    A = -torch.arange(1, state_size + 1, dtype=torch.float32).repeat(dim, 1)
    dt_col = torch.full((chunk_size, dim), float(delta))
    log_a_cum = torch.cumsum(dt_col.unsqueeze(-1) * A.unsqueeze(0), dim=0)
    return (torch.exp(log_a_cum) < 1e-8).double().mean().item()


@pytest.mark.parametrize("delta", [1e-3, 1e-2])
@pytest.mark.parametrize("chunk_size", [4, 8, 16, 32, 64])
def test_legacy_scan_is_exact_where_the_clamp_never_fires(delta, chunk_size):
    assert _clamp_fire_fraction(delta, chunk_size) == 0.0, "precondition: clamp must not fire"
    assert _measure(delta, chunk_size) < EXACT_TOL


def test_shrinking_the_chunk_restores_exactness():
    assert _clamp_fire_fraction(0.1, 64) > 0.0 and _clamp_fire_fraction(0.1, 4) == 0.0
    assert _measure(0.1, 4) < EXACT_TOL < _measure(0.1, 64)


@pytest.mark.parametrize("delta", [0.1, 0.3, 0.705, 1.0])
def test_legacy_scan_error_does_not_exceed_documented_ceiling(delta):
    assert _measure(delta, 64) <= DEFECTIVE_CEILING


def test_legacy_scan_defect_is_monotone_in_delta():
    errs = [_measure(d, 64) for d in (0.1, 0.3, 1.0)]
    assert errs == sorted(errs), errs


def test_clamp_fires_at_the_reference_models_delta():
    assert _clamp_fire_fraction(0.705, 64) > 0.5


def test_runtime_reference_agrees_with_the_independent_oracle():
    x, dt, A, B, C, D = _audit_inputs(0.705)
    runtime = selective_scan_sequential_reference(x, dt, A, B, C, D)
    assert rel_max_err(runtime, sequential_selective_scan_fp64(x, dt, A, B, C, D)) < EXACT_TOL


def test_exact_scan_toggle_is_off_by_default_and_selects_the_oracle_when_on(monkeypatch):
    monkeypatch.delenv("HYBRID_EXACT_SCAN", raising=False)
    x, dt, A, B, C, D = _audit_inputs(0.705)
    default = selective_scan(x, dt, A, B, C, D)
    chunked = selective_scan_parallel(
        x.float(), dt.float(), A.float(), B.float(), C.float(), D.float(), chunk_size=32
    )
    assert torch.equal(default, chunked)
    monkeypatch.setenv("HYBRID_EXACT_SCAN", "1")
    exact = sequential_selective_scan_fp64(x, dt, A, B, C, D)
    assert rel_max_err(selective_scan(x, dt, A, B, C, D), exact) < EXACT_TOL
    assert rel_max_err(default, exact) > 0.1, "the toggle must be able to detect the defect"


# ---------------------------------------------------------------------------------------------
# TFLA (mLSTM)
# ---------------------------------------------------------------------------------------------

FORGET_BIASES = [0.0, 1.0, 2.0, 3.0]  # 0.0 is the reference's shipped forget_gate_bias_init


def _tfla_fixture(forget_bias: float, seq_len: int = 128, heads: int = 2, dim: int = 8):
    torch.manual_seed(0)
    shape = (1, heads, seq_len, dim)
    return (
        torch.randn(shape),
        torch.randn(shape) / dim**0.5,
        torch.randn(shape),
        torch.sigmoid(torch.randn(shape) - 10.0),  # i_gate at the shipped bias -10
        torch.sigmoid(torch.randn(shape) * 0.5 + forget_bias),
    )


@pytest.mark.parametrize(
    "tfla_impl,forget_bias,chunk_size",
    [
        pytest.param(impl, fb, cs, marks=_xfail_if(impl == "legacy" and fb <= 1.0 and cs >= 64, DEFECT_TFLA))
        for impl in ("legacy", "exact")
        for fb in FORGET_BIASES
        for cs in CHUNKS
    ],
)
def test_tfla_matches_sequential_reference(tfla_impl, forget_bias, chunk_size):
    """The legacy error is governed entirely by whether f_cum underflows the 1e-6 clamp."""
    q, k, v, i_gate, f_gate = _tfla_fixture(forget_bias)
    want = sequential_mlstm_reference(q, k, v, i_gate, f_gate)
    got = tfla_forward_parallel(q, k, v, i_gate, f_gate, chunk_size=chunk_size, tfla_impl=tfla_impl)
    err = rel_max_err(got, want)
    assert err <= TOL, f"tfla_impl={tfla_impl} forget_bias={forget_bias} chunk={chunk_size}: {err:.3e}"


@pytest.mark.parametrize("forget_bias", FORGET_BIASES)
def test_tfla_clamp_hit_rate_is_documented(forget_bias):
    torch.manual_seed(0)
    f = torch.sigmoid(torch.randn(200_000) * 0.5 + forget_bias)
    f_cum = torch.log(f.clamp(min=1e-6)).view(-1, 64).cumsum(-1).exp()
    hit_rate = (f_cum < 1e-6).double().mean().item()
    expected = {0.0: 0.70, 1.0: 0.36, 2.0: 0.0, 3.0: 0.0}[forget_bias]
    assert abs(hit_rate - expected) < 0.05, f"forget_bias={forget_bias}: clamp hit-rate {hit_rate:.3f}"


@pytest.mark.parametrize("forget_bias", [-4.0, -2.0])
@pytest.mark.parametrize("chunk_size", [64, 128])
def test_tfla_exact_survives_extreme_decay(forget_bias, chunk_size):
    """The overflow guard: re-centring alone produced NaN here; the sequential fallback does not."""
    q, k, v, i_gate, f_gate = _tfla_fixture(forget_bias)
    got = tfla_forward_parallel(
        q, k, v, i_gate, f_gate, chunk_size=chunk_size, tfla_impl="exact", fallback="sequential"
    )
    assert torch.isfinite(got).all()
    assert rel_max_err(got, sequential_mlstm_reference(q, k, v, i_gate, f_gate)) <= TOL


@pytest.mark.parametrize("chunk_size", [64, 128])
def test_tfla_fallback_is_an_error_unless_configured(chunk_size):
    """P2-C: the slow path is a configured choice; by default a too-wide chunk raises."""
    q, k, v, i_gate, f_gate = _tfla_fixture(-2.0)
    with pytest.raises(TFLAFallbackError, match="half-range"):
        tfla_forward_parallel(q, k, v, i_gate, f_gate, chunk_size=chunk_size, tfla_impl="exact")
    with pytest.raises(ValueError, match="fallback"):
        tfla_forward_parallel(q, k, v, i_gate, f_gate, chunk_size=chunk_size, fallback="silent")


def _packed_tfla_inputs(seq_len: int, boundaries: dict, forget_bias: float = 1.0, seed: int = 0):
    """Two rows with different boundary sets; doc_ids increase at each listed position."""
    torch.manual_seed(seed)
    shape = (2, 2, seq_len, 8)
    q, v = torch.randn(shape), torch.randn(shape)
    k = torch.randn(shape) / 8**0.5
    i_gate = torch.exp(torch.randn(shape) * 0.5 - 1.0)
    f_gate = torch.sigmoid(torch.randn(shape) * 0.5 + forget_bias)
    doc_ids = torch.zeros(2, seq_len, dtype=torch.long)
    for row, cuts in boundaries.items():
        for cut in cuts:
            doc_ids[row, cut:] += 1
    return q, k, v, i_gate, f_gate, doc_ids


def _tfla_per_document(q, k, v, i_gate, f_gate, doc_ids, **kw):
    out = torch.zeros_like(q)
    for b in range(q.shape[0]):
        ids = doc_ids[b]
        starts = [0] + [t for t in range(1, ids.numel()) if ids[t] != ids[t - 1]] + [ids.numel()]
        for s, e in zip(starts, starts[1:]):
            out[b : b + 1, :, s:e] = tfla_forward_parallel(
                *(t[b : b + 1, :, s:e] for t in (q, k, v, i_gate, f_gate)), **kw
            )
    return out


PACKED_CASES = [
    # (chunk, L, {row: boundary positions}) -- mid-chunk, on a chunk edge, several per chunk
    (8, 37, {0: [13], 1: [8, 16, 17]}),
    (16, 64, {0: [16], 1: [5, 40]}),
    (16, 100, {0: [31, 32, 33], 1: [48, 99]}),
    (64, 100, {0: [64], 1: [1, 50, 63]}),
]


@pytest.mark.parametrize(
    "chunk_size,seq_len,boundaries", PACKED_CASES, ids=[f"chunk{c}-L{n}" for c, n, _ in PACKED_CASES]
)
def test_tfla_packed_equals_per_document(chunk_size, seq_len, boundaries):
    """Defect 15 (P2-D): resets inside the kernel equal running every document alone, and equal
    the fp64 oracle with resets, at 1e-6."""
    q, k, v, i_gate, f_gate, doc_ids = _packed_tfla_inputs(seq_len, boundaries)
    kw = dict(chunk_size=chunk_size, tfla_impl="exact")
    packed = tfla_forward_parallel(q, k, v, i_gate, f_gate, doc_ids=doc_ids, **kw)
    alone = _tfla_per_document(q, k, v, i_gate, f_gate, doc_ids, **kw)
    oracle = sequential_mlstm_reference(q, k, v, i_gate, f_gate, doc_ids=doc_ids)
    assert rel_max_err(packed, alone) <= TOL, f"packed vs per-document {rel_max_err(packed, alone):.3e}"
    assert rel_max_err(packed, oracle) <= TOL, f"packed vs oracle {rel_max_err(packed, oracle):.3e}"


def test_tfla_documents_do_not_leak():
    """Changing document A cannot change a single bit of document B's output."""
    q, k, v, i_gate, f_gate, doc_ids = _packed_tfla_inputs(48, {0: [20], 1: [20]})
    base = tfla_forward_parallel(q, k, v, i_gate, f_gate, chunk_size=16, tfla_impl="exact", doc_ids=doc_ids)
    q2, v2 = q.clone(), v.clone()
    q2[:, :, :20] += 5.0
    v2[:, :, :20] -= 3.0
    moved = tfla_forward_parallel(
        q2, k, v2, i_gate, f_gate, chunk_size=16, tfla_impl="exact", doc_ids=doc_ids
    )
    assert torch.equal(base[:, :, 20:], moved[:, :, 20:])
    assert not torch.allclose(base[:, :, :20], moved[:, :, :20])


def test_tfla_sequential_fallback_resets_at_boundaries():
    """The fallback, when configured, honours document boundaries too."""
    q, k, v, i_gate, f_gate, doc_ids = _packed_tfla_inputs(64, {0: [10, 37], 1: [32]}, forget_bias=-3.0)
    got = tfla_forward_parallel(
        q, k, v, i_gate, f_gate, chunk_size=64, tfla_impl="exact", fallback="sequential", doc_ids=doc_ids
    )
    assert rel_max_err(got, sequential_mlstm_reference(q, k, v, i_gate, f_gate, doc_ids=doc_ids)) <= TOL


@pytest.mark.parametrize("docs", [False, True], ids=["contiguous", "packed"])
@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 50])
def test_tfla_final_state_matches_oracle(seq_len, docs):
    """`return_state` gives the (C, n) the step recurrence continues from (P2-G), on partial chunks."""
    q, k, v, i_gate, f_gate, doc_ids = _packed_tfla_inputs(
        seq_len, {0: [seq_len // 2] if seq_len > 2 else [], 1: []}
    )
    ids = doc_ids if docs else None
    _, C, n = tfla_forward_parallel(
        q, k, v, i_gate, f_gate, chunk_size=16, tfla_impl="exact", doc_ids=ids, return_state=True
    )
    _, C_ref, n_ref = sequential_mlstm_reference(q, k, v, i_gate, f_gate, doc_ids=ids, return_state=True)
    assert rel_max_err(C, C_ref) <= TOL and rel_max_err(n, n_ref) <= TOL


def _tfla_model(chunk_size: int, forget_bias: float):
    from lexhybrid import HybridConfig, HybridLanguageModel

    torch.manual_seed(0)
    cfg = HybridConfig(
        vocab_size=128,
        dim=64,
        num_layers=2,
        layer_pattern=["mlstm", "mamba3"],
        head_dim=16,
        mamba3_d_state=16,
        mamba3_head_dim=16,
        tfla_impl="exact",
        tfla_fallback="error",
        mlstm_chunk_size=chunk_size,
        mlstm_forget_gate_bias_init=forget_bias,
        max_position_embeddings=512,
        dropout=0.0,
    )
    return HybridLanguageModel(cfg).eval()


@pytest.mark.parametrize("chunk_size,forget_bias", [(128, 3.0)], ids=["chunk=128, forget_bias=3.0"])
def test_tfla_fast_path_at_init(chunk_size, forget_bias):
    """Defect 9: the shipped configuration runs exact TFLA on its factorised path at init. With
    tfla_fallback="error" the forward would raise if any chunk needed the fallback."""
    model = _tfla_model(chunk_size, forget_bias)
    with torch.no_grad():
        out = model(torch.randint(0, 128, (2, 512))).logits
    assert torch.isfinite(out).all()


def test_tfla_needed_the_fallback_at_init_with_the_reference_forget_bias():
    """The record behind defect 9: forget bias 0 gives f = 0.5 and a half-range of
    0.5 * 128 * ln 2 = 44.4 > 40 at chunk 128, so the reference ran the fallback from step 0."""
    model = _tfla_model(128, 0.0)
    with torch.no_grad(), pytest.raises(TFLAFallbackError, match=r"half-range is 4[45]\.[0-9]"):
        model(torch.randint(0, 128, (2, 512)))


def test_tfla_chunk_size_reaches_kernel(monkeypatch):
    """Defect 9: the block's chunk size, not the sequence length, sets the kernel's chunk."""
    import lexhybrid.kernels.tfla.tfla_interface as tfla
    from lexhybrid.layers.mlstm_block import mLSTMBlock

    seen = []
    real = tfla.tfla_forward_parallel

    def spy(*args, **kwargs):
        seen.append(kwargs["chunk_size"])
        return real(*args, **kwargs)

    monkeypatch.setattr(tfla, "tfla_forward_parallel", spy)
    for configured, length, expected in [(16, 96, 16), (64, 200, 64), (128, 40, 40)]:
        block = mLSTMBlock(dim=32, head_dim=16, tfla_impl="exact", chunk_size=configured).eval()
        with torch.no_grad():
            block(torch.randn(1, length, 32))
        assert seen[-1] == expected, f"chunk {configured}, L {length}: kernel saw {seen[-1]}"


# ---------------------------------------------------------------------------------------------
# SSD (Mamba-2/3)
# ---------------------------------------------------------------------------------------------

SSD_SHAPES = [
    # (batch, seqlen, nheads, headdim, ngroups, dstate)
    (2, 128, 4, 16, 1, 32),  # ngroups=1: B/C fully shared
    (2, 128, 8, 16, 2, 64),
    (1, 100, 4, 8, 4, 16),  # seqlen not a multiple of any chunk size
    (3, 64, 6, 32, 6, 16),  # ngroups == nheads: no sharing at all
]


def ssd_fixture(shape, seed: int = 0):
    batch, seqlen, nheads, headdim, ngroups, dstate = shape
    torch.manual_seed(seed)
    f64 = torch.float64
    return dict(
        x=torch.randn(batch, seqlen, nheads, headdim, dtype=f64),
        dt=torch.rand(batch, seqlen, nheads, dtype=f64) * 0.1 + 1e-3,
        A=-torch.rand(nheads, dtype=f64) * 8 - 0.1,
        B=torch.randn(batch, seqlen, ngroups, dstate, dtype=f64),
        C=torch.randn(batch, seqlen, ngroups, dstate, dtype=f64),
        D=torch.ones(nheads, dtype=f64),
    )


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("shape", SSD_SHAPES, ids=lambda s: "x".join(str(v) for v in s))
def test_ssd_chunked_matches_sequential_reference(shape, chunk_size):
    kw = ssd_fixture(shape)
    assert rel_max_err(ssd_chunked_scan(chunk_size=chunk_size, **kw), ssd_sequential_reference(**kw)) <= 1e-12


@pytest.mark.parametrize(
    "boundaries", [(37,), (32,), (20, 55, 80)], ids=["mid-chunk", "on-chunk-edge", "multi-doc"]
)
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_ssd_document_boundaries_match_reference(boundaries, chunk_size):
    """Resets are boolean masks inside the scan; the on-chunk-edge case is where an off-by-one in
    the carry logic would hide."""
    kw = ssd_fixture((2, 96, 4, 16, 2, 32))
    ids = torch.zeros(2, 96, dtype=torch.long)
    for k, start in enumerate(boundaries):
        ids[:, start:] = k + 1
    want = ssd_sequential_reference(doc_ids=ids, **kw)
    assert rel_max_err(ssd_chunked_scan(chunk_size=chunk_size, doc_ids=ids, **kw), want) <= 1e-12


@pytest.mark.parametrize("chunk_size", [16, 64])
def test_ssd_document_isolation_is_bit_exact(chunk_size):
    """Perturbing document A must leave document B bit-identical, not merely close."""
    kw = ssd_fixture((2, 96, 4, 16, 2, 32))
    boundary = 37
    ids = torch.zeros(2, 96, dtype=torch.long)
    ids[:, boundary:] = 1
    ref = ssd_chunked_scan(chunk_size=chunk_size, doc_ids=ids, **kw)
    perturbed = dict(kw)
    perturbed["x"] = kw["x"].clone()
    perturbed["x"][:, :boundary] += torch.randn_like(perturbed["x"][:, :boundary]) * 5.0
    out = ssd_chunked_scan(chunk_size=chunk_size, doc_ids=ids, **perturbed)
    assert torch.equal(ref[:, boundary:], out[:, boundary:]), "doc B leaked from doc A"
    assert not torch.allclose(ref[:, :boundary], out[:, :boundary]), "doc A should have changed"


@pytest.mark.parametrize("docs", [False, True], ids=["contiguous", "packed"])
@pytest.mark.parametrize("seqlen", [1, 63, 64, 65, 200], ids=lambda n: f"L={n}")
def test_ssd_final_state_matches_oracle(seqlen, docs):
    """Defect 7 (P2-F): the state after the last token, including a partial last chunk -- where the
    reference's fresh padding segment returned zeros -- and a document starting inside it."""
    kw = ssd_fixture((2, seqlen, 4, 8, 1, 16))
    ids = None
    if docs and seqlen > 1:
        ids = torch.zeros(2, seqlen, dtype=torch.long)
        ids[0, seqlen // 3 :] = 1
        ids[1, max(seqlen - 3, 1) :] = 1  # a document starting in the last chunk
    beta = kw["dt"] * 0.5
    extra = [(beta, torch.roll(kw["B"], 1, 1), torch.roll(kw["x"], 1, 1))]  # a trapezoid-like term
    y, state = ssd_chunked_scan(chunk_size=64, doc_ids=ids, extra_terms=extra, return_final_state=True, **kw)
    y_ref, state_ref = ssd_sequential_reference(doc_ids=ids, extra_terms=extra, return_final_state=True, **kw)
    assert rel_max_err(y, y_ref) <= 1e-12
    assert rel_max_err(state, state_ref) <= 1e-6, f"final state {rel_max_err(state, state_ref):.3e}"
    assert state.abs().max() > 0


@pytest.mark.parametrize("autocast", [False, True], ids=["bf16-inputs", "bf16-autocast"])
def test_ssd_state_is_fp32_under_autocast(autocast):
    """Defect 16 (P2-E): the carried state is fp32 by declaration, whatever the activations are."""
    kw = {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in ssd_fixture(SSD_SHAPES[0]).items()
    }
    kw["dt"], kw["A"], kw["D"] = (
        kw["dt"].float(),
        kw["A"].float(),
        kw["D"].float(),
    )  # as Mamba3Block passes them
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        y, state = ssd_chunked_scan(chunk_size=32, return_final_state=True, **kw)
    assert state.dtype is torch.float32
    assert y.dtype is torch.bfloat16


def test_tfla_state_is_fp32_for_bf16_inputs():
    """The TFLA carried state is fp32 too (P2-E), so the cache a bf16 forward writes stays fp32."""
    q, k, v, i_gate, f_gate, _ = _packed_tfla_inputs(40, {0: [], 1: []})
    h16, C, n = tfla_forward_parallel(
        *(t.to(torch.bfloat16) for t in (q, k, v, i_gate, f_gate)),
        chunk_size=16,
        tfla_impl="exact",
        return_state=True,
    )
    assert h16.dtype is torch.bfloat16 and C.dtype is torch.float32 and n.dtype is torch.float32
    _, C_ref, _ = sequential_mlstm_reference(q, k, v, i_gate, f_gate, return_state=True)
    assert rel_max_err(C, C_ref) < 0.05


def test_ssd_extra_terms_are_linear():
    """Splitting one state-input term into two halves reproduces the original (the trapezoid hook)."""
    kw = ssd_fixture((2, 96, 4, 16, 2, 32))
    half = kw["dt"] * 0.5
    got = ssd_chunked_scan(chunk_size=32, coeff=half, extra_terms=[(half, kw["B"], kw["x"])], **kw)
    assert rel_max_err(got, ssd_sequential_reference(**kw)) <= 1e-12


def test_ssd_step_matches_the_chunked_scan():
    """The decode step and the training scan are the same recurrence."""
    kw = ssd_fixture((2, 48, 4, 16, 2, 32))
    batch, seqlen, nheads, headdim = kw["x"].shape
    state = torch.zeros(batch, nheads, headdim, kw["B"].shape[-1], dtype=torch.float64)
    ys = []
    for t in range(seqlen):
        y_t, state = ssd_step(
            kw["x"][:, t], kw["dt"][:, t], kw["A"], kw["B"][:, t], kw["C"][:, t], state, kw["D"]
        )
        ys.append(y_t)
    assert rel_max_err(torch.stack(ys, dim=1), ssd_chunked_scan(chunk_size=16, **kw)) <= 1e-12


# Exponential-trapezoidal rule (paper Prop. 1): h_t = a h_{t-1} + b B_{t-1} x_{t-1} + g B_t x_t


def trapezoid_terms(x, dt, A, B, lam, doc_ids=None):
    """Reference construction of (gamma, [(beta, shift(B), shift(x))]), mirroring the block."""
    alpha = torch.exp(dt * A)
    gamma = lam * dt
    beta = (1.0 - lam) * dt * alpha
    if doc_ids is not None:
        starts = torch.zeros_like(doc_ids, dtype=torch.bool)
        starts[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
        starts[:, 0] = True
        beta = beta.masked_fill(starts.unsqueeze(-1), 0.0)
    else:
        beta = beta.clone()
        beta[:, 0] = 0.0

    def shift(v):
        return torch.cat([torch.zeros_like(v[:, :1]), v[:, :-1]], dim=1)

    return gamma, [(beta, shift(B), shift(x))]


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("lam_mode", ["half", "one", "zero", "random"])
def test_trapezoid_matches_three_term_reference(lam_mode, chunk_size):
    kw = ssd_fixture((2, 96, 4, 16, 2, 32))
    shape = kw["dt"].shape
    lam = {
        "half": torch.full(shape, 0.5, dtype=torch.float64),
        "one": torch.ones(shape, dtype=torch.float64),
        "zero": torch.zeros(shape, dtype=torch.float64),
        "random": torch.rand(shape, dtype=torch.float64),
    }[lam_mode]
    gamma, extra = trapezoid_terms(kw["x"], kw["dt"], kw["A"], kw["B"], lam)
    want = ssd_sequential_reference(coeff=gamma, extra_terms=extra, **kw)
    got = ssd_chunked_scan(chunk_size=chunk_size, coeff=gamma, extra_terms=extra, **kw)
    assert rel_max_err(got, want) <= 1e-12


def test_trapezoid_respects_document_boundaries():
    kw = ssd_fixture((2, 96, 4, 16, 2, 32))
    ids = torch.zeros(2, 96, dtype=torch.long)
    ids[:, 20:], ids[:, 55:] = 1, 2
    lam = torch.rand(kw["dt"].shape, dtype=torch.float64)
    gamma, extra = trapezoid_terms(kw["x"], kw["dt"], kw["A"], kw["B"], lam, doc_ids=ids)
    want = ssd_sequential_reference(doc_ids=ids, coeff=gamma, extra_terms=extra, **kw)
    got = ssd_chunked_scan(chunk_size=32, doc_ids=ids, coeff=gamma, extra_terms=extra, **kw)
    assert rel_max_err(got, want) <= 1e-12


# ---------------------------------------------------------------------------------------------
# Data-dependent rotary angles (paper Prop. 2-4)
# ---------------------------------------------------------------------------------------------


def test_rotate_then_shift_is_not_the_same_as_shift_then_rotate():
    """Prop. 4's ordering constraint: B_{t-1} must carry its own Theta_{t-1}."""
    dt = torch.rand(1, 8, 1) * 0.1 + 0.05
    theta = torch.randn(1, 8, 2)
    angles = cumulative_angles(dt, theta)
    B = torch.randn(1, 8, 1, 8)

    def shift(v):
        return torch.cat([torch.zeros_like(v[:, :1]), v[:, :-1]], dim=1)

    assert not torch.allclose(shift(apply_rotary(B, angles)), apply_rotary(shift(B), angles))


@pytest.mark.parametrize("seq_len", [512, 4096])
def test_angle_accumulation_stays_accurate_at_length(seq_len):
    dt = torch.rand(1, seq_len, 1) * 0.9 + 0.1
    theta = torch.randn(1, seq_len, 16)
    exact = torch.remainder((dt.double() * theta.double()).cumsum(1), TWO_PI)
    ours = cumulative_angles(dt, theta).double()
    naive = torch.remainder((dt * theta).cumsum(1), TWO_PI).double()
    assert (ours - exact).abs().max() < 1e-6
    assert (ours - exact).abs().max() < (naive - exact).abs().max()
    assert 0.0 <= ours.min() and ours.max() < TWO_PI


def test_rope_angles_reset_per_document():
    dt = torch.rand(2, 64, 1) * 0.1 + 1e-3
    theta = torch.randn(2, 64, 8)
    ids = torch.zeros(2, 64, dtype=torch.long)
    ids[:, 20:], ids[:, 40:] = 1, 2
    segmented = cumulative_angles(dt, theta, doc_ids=ids)
    standalone = cumulative_angles(dt[:, 20:40], theta[:, 20:40])
    assert torch.allclose(segmented[:, 20:40], standalone, atol=1e-9)
