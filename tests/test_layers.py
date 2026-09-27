"""Block-level behaviour of the four mixers and HybridBlock (plan P1-D..M).

Ported from the reference's ``tests/test_layers.py`` and the block-level parts of
``tests/test_mamba3_numerics.py`` (M2-M6): document resets, bit-identity controls for disabled
features, every parameter receiving a gradient, and ``step()`` reproducing the chunked forward.
"""

import inspect
import math
import typing

import pytest
import torch
import torch.nn as nn

from lexhybrid.config.hybrid_config import LayerType
from lexhybrid.layers.attention_block import AttentionBlock, build_doc_boundary_attn_mask
from lexhybrid.layers.hybrid_block import HybridBlock, _mamba3_params
from lexhybrid.layers.mamba3_block import Mamba3Block
from lexhybrid.layers.mamba_block import MambaBlock
from lexhybrid.layers.mlstm_block import mLSTMBlock
from lexhybrid.layers.normalization import RMSNorm
from lexhybrid.layers.rotary import cumulative_angles


def m3_block(**kw):
    torch.manual_seed(0)
    defaults = dict(dim=128, d_state=64, head_dim=32)
    defaults.update(kw)
    return Mamba3Block(**defaults)


def step_through(block, x, **cache_kw):
    """Run ``block`` one token at a time through its cache and stack the outputs."""
    cache = block.allocate_inference_cache(x.shape[0], **cache_kw)
    with torch.no_grad():
        return torch.stack([block.step(x[:, t], cache) for t in range(x.shape[1])], dim=1)


def no_grad_params(module):
    return [n for n, p in module.named_parameters() if p.grad is None]


# -- RMSNorm and rotary ---------------------------------------------------------------------------


def test_rmsnorm_shape_and_unit_rms():
    norm = RMSNorm(16)
    x = torch.randn(3, 5, 16) * 7.0
    y = norm(x)
    assert y.shape == x.shape
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3, 5), atol=1e-4)


def test_rmsnorm_computes_in_fp32():
    """Defect 10 (P2-E): the mean square is taken in fp32 and the result cast back. For fp32 inputs
    the formula is the reference's, bit for bit."""
    norm = RMSNorm(64)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 64))
    x = torch.randn(4, 64) * 30.0
    ref = norm.weight * (x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + norm.eps))
    assert torch.equal(norm(x), ref)

    x16 = x.to(torch.bfloat16)
    x64 = x16.double()
    want = norm.weight.double() * x64 / torch.sqrt((x64**2).mean(-1, keepdim=True) + norm.eps)
    got = norm(x16)
    assert got.dtype is torch.bfloat16
    # Only the final rounding to bf16 remains: one bf16 ulp (2^-8 relative).
    assert ((got.double() - want).abs() / want.abs().clamp(min=1e-3)).max() <= 2**-8


def test_cumulative_angles_restart_per_document():
    dt = torch.rand(1, 12, 1)
    theta = torch.randn(1, 12, 2)
    ids = torch.tensor([[0] * 5 + [1] * 7])
    seg = cumulative_angles(dt, theta, doc_ids=ids)
    assert torch.allclose(seg[:, 5:], cumulative_angles(dt[:, 5:], theta[:, 5:]), atol=1e-9)


# -- Mamba-3 --------------------------------------------------------------------------------------


def test_mamba3_reduces_to_mamba2_by_default():
    block = m3_block()
    assert (block.use_trapezoid, block.use_rope, block.bc_bias, block.a_mode, block.mimo_rank) == (
        False,
        False,
        "none",
        "static",
        1,
    )
    assert block.use_conv is True and block.out_norm is None


def test_mamba3_forward_backward_is_finite_and_fully_connected():
    block = m3_block()
    out = block(torch.randn(2, 48, 128))
    out.sum().backward()
    assert out.shape == (2, 48, 128) and torch.isfinite(out).all()
    assert not no_grad_params(block)


@pytest.mark.parametrize("use_conv", [True, False])
def test_mamba3_document_reset_matches_running_the_document_alone(use_conv):
    """Isolation plus correctness: document B equals running document B on its own."""
    block = m3_block(use_conv=use_conv).eval()
    batch, seqlen, boundary = 2, 64, 29
    ids = torch.zeros(batch, seqlen, dtype=torch.long)
    ids[:, boundary:] = 1
    x = torch.randn(batch, seqlen, 128)
    with torch.no_grad():
        ref = block(x, doc_ids=ids)
        perturbed = x.clone()
        perturbed[:, :boundary] += torch.randn(batch, boundary, 128) * 5.0
        out = block(perturbed, doc_ids=ids)
        standalone = block(x[:, boundary:])
    assert torch.equal(ref[:, boundary:], out[:, boundary:]), "doc B leaked from doc A"
    assert torch.allclose(ref[:, boundary:], standalone, atol=1e-5)


def test_mimo_is_plumbed_but_refuses_to_run():
    with pytest.raises(NotImplementedError, match="MIMO"):
        m3_block(mimo_rank=4)


def _drive_lambda_to_one(block):
    """Zero the trap slice of in_proj and raise trap_bias so lambda is exactly 1.0 in fp32."""
    s = block._split
    offset = sum(s[:6])
    with torch.no_grad():
        block.in_proj.weight[offset : offset + s[6]].zero_()
        if block.trap_bias is not None:
            block.trap_bias.fill_(20.0)


@pytest.mark.parametrize("with_documents", [False, True])
def test_lambda_one_is_bit_identical_to_trapezoid_off(with_documents):
    """lambda = 1 is Euler exactly: sigmoid(20) == 1.0 in fp32 and a + 0.0 == a."""
    on = m3_block(use_trapezoid=True).eval()
    off = m3_block(use_trapezoid=False).eval()
    off.load_state_dict({k: v for k, v in on.state_dict().items() if k != "trap_bias"})
    _drive_lambda_to_one(on)
    _drive_lambda_to_one(off)
    x = torch.randn(2, 96, 128)
    kw = {}
    if with_documents:
        ids = torch.zeros(2, 96, dtype=torch.long)
        ids[:, 41:] = 1
        kw["doc_ids"] = ids
    with torch.no_grad():
        assert torch.equal(on(x, **kw), off(x, **kw))


def test_trapezoid_actually_changes_the_output_at_its_default():
    trap = m3_block(use_trapezoid=True).eval()
    euler = m3_block(use_trapezoid=False).eval()
    euler.load_state_dict({k: v for k, v in trap.state_dict().items() if k != "trap_bias"})
    x = torch.randn(2, 96, 128)
    with torch.no_grad():
        assert (trap(x) - euler(x)).abs().max() > 1e-3


@pytest.mark.parametrize("use_trapezoid", [False, True])
@pytest.mark.parametrize("a_mode", ["static", "data_dependent"])
def test_no_parameter_is_left_dangling_by_a_disabled_flag(use_trapezoid, a_mode):
    block = m3_block(use_trapezoid=use_trapezoid, a_mode=a_mode)
    block(torch.randn(2, 32, 128)).sum().backward()
    assert not no_grad_params(block)


def _zero_rope_slice(block):
    with torch.no_grad():
        o, n = sum(block._split[:7]), block._split[7]
        block.in_proj.weight[o : o + n].zero_()


@pytest.mark.parametrize("with_documents", [False, True])
def test_zero_angle_rope_is_bit_identical_to_rope_off(with_documents):
    on = m3_block(use_rope=True).eval()
    off = m3_block(use_rope=False).eval()
    off.load_state_dict(on.state_dict())
    _zero_rope_slice(on)
    _zero_rope_slice(off)
    x = torch.randn(2, 96, 128)
    kw = {}
    if with_documents:
        ids = torch.zeros(2, 96, dtype=torch.long)
        ids[:, 41:] = 1
        kw["doc_ids"] = ids
    with torch.no_grad():
        assert torch.equal(on(x, **kw), off(x, **kw))


def test_rope_changes_the_output_when_the_angles_are_not_zero():
    on = m3_block(use_rope=True).eval()
    off = m3_block(use_rope=False).eval()
    off.load_state_dict(on.state_dict())
    _zero_rope_slice(off)
    with torch.no_grad():
        o, n = sum(on._split[:7]), on._split[7]
        on.in_proj.weight[o : o + n].normal_(0, 0.5)
        x = torch.randn(2, 96, 128)
        assert (on(x) - off(x)).abs().max() > 1e-5


def test_theta_max_bounds_the_total_rotation_over_a_sequence():
    """At the reference's screen setting the bound was 81 turns over 512 tokens -- a scrambler."""
    block = m3_block(use_rope=True)
    assert 512 * block.dt_limit * block.theta_max / (2 * math.pi) > 50
    calm = m3_block(use_rope=True, theta_max=0.02)
    assert 512 * calm.dt_limit * calm.theta_max / (2 * math.pi) < 2


def _bc_bias_pair(**extra):
    none = m3_block(bc_bias="none", **extra).eval()
    zero = m3_block(bc_bias="zero_init", **extra).eval()
    missing, unexpected = zero.load_state_dict(none.state_dict(), strict=False)
    assert set(missing) == {"B_bias", "C_bias"} and not unexpected
    return none, zero


@pytest.mark.parametrize("with_documents", [False, True])
@pytest.mark.parametrize("features", [{}, {"use_trapezoid": True, "use_rope": True}])
def test_zero_init_bc_bias_is_bit_identical_to_no_bias(with_documents, features):
    none, zero = _bc_bias_pair(**features)
    x = torch.randn(2, 96, 128)
    kw = {}
    if with_documents:
        ids = torch.zeros(2, 96, dtype=torch.long)
        ids[:, 41:] = 1
        kw["doc_ids"] = ids
    with torch.no_grad():
        assert torch.equal(none(x, **kw), zero(x, **kw))


def test_one_init_bc_bias_is_a_real_arm_not_a_relabelling():
    _, zero = _bc_bias_pair()
    one = m3_block(bc_bias="one_init").eval()
    one.load_state_dict(zero.state_dict())
    with torch.no_grad():
        one.B_bias.fill_(1.0)
        one.C_bias.fill_(1.0)
        x = torch.randn(2, 96, 128)
        assert not torch.allclose(one(x), zero(x), atol=1e-5)


@pytest.mark.parametrize("bc_bias", ["none", "zero_init", "one_init"])
@pytest.mark.parametrize("use_conv", [True, False])
def test_no_parameter_is_left_dangling_by_an_m5_flag(bc_bias, use_conv):
    block = m3_block(bc_bias=bc_bias, use_conv=use_conv)
    block(torch.randn(2, 32, 128)).sum().backward()
    assert not no_grad_params(block)


def test_mimo_rank_one_is_bit_identical_to_leaving_it_alone():
    default = m3_block().eval()
    explicit = m3_block(mimo_rank=1).eval()
    explicit.load_state_dict(default.state_dict())
    x = torch.randn(2, 96, 128)
    with torch.no_grad():
        assert torch.equal(default(x), explicit(x))


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"use_trapezoid": True},
        {"use_rope": True, "theta_max": 0.2},
        {"use_conv": False},
        {"bc_bias": "one_init"},
        {"a_mode": "data_dependent"},
        {"use_trapezoid": True, "use_rope": True, "bc_bias": "one_init", "theta_max": 0.2},
    ],
)
def test_mamba3_step_reproduces_the_chunked_forward(flags):
    """The cache is an equivalence, not an approximation, under every flag."""
    block = m3_block(**flags).eval()
    x = torch.randn(2, 40, 128)
    with torch.no_grad():
        full = block(x)
    stepped = step_through(block, x)
    assert torch.allclose(full, stepped, atol=1e-5), f"max abs {(full - stepped).abs().max():.3e}"


MAMBA3_FLAG_SETS = [
    {},
    {"use_trapezoid": True},
    {"use_rope": True, "theta_max": 0.2},
    {"use_conv": False},
    {"bc_bias": "one_init", "a_mode": "data_dependent"},
    {"use_trapezoid": True, "use_rope": True, "bc_bias": "one_init", "theta_max": 0.2},
]


def _continue_with_steps(block, x, prefix, doc_ids=None):
    """Prefill ``x[:, :prefix]`` in one forward that fills the cache, then step the rest."""
    cache = block.allocate_inference_cache(x.shape[0])
    with torch.no_grad():
        head = block(x[:, :prefix], cache=cache, doc_ids=None if doc_ids is None else doc_ids[:, :prefix])
        after_prefill = dict(cache)
        tail = [block.step(x[:, t], cache) for t in range(prefix, x.shape[1])]
    tail = torch.stack(tail, dim=1) if tail else x[:, :0]
    return head, tail, after_prefill


@pytest.mark.parametrize("prefix", [1, 3, 23, 40], ids=lambda p: f"prefix{p}")
@pytest.mark.parametrize("flags", MAMBA3_FLAG_SETS, ids=lambda f: "-".join(sorted(f)) or "default")
def test_mamba3_forward_then_step_equals_step_only(flags, prefix):
    """P2-F: a forward pass fills the cache (state, conv window, trapezoid B/x, fp64 angle, seen) so
    stepping on from it equals stepping from the first token -- including a prompt shorter than
    the conv window and one that ends exactly on a chunk edge (chunk 8)."""
    block = m3_block(chunk_size=8, **flags).eval()
    x = torch.randn(2, 48, 128)
    head, tail, cache = _continue_with_steps(block, x, prefix)
    only = step_through(block, x)
    assert cache["seen"] == prefix and cache["ssm_state"].dtype is torch.float32
    assert torch.allclose(head, only[:, :prefix], atol=1e-5)
    assert torch.allclose(tail, only[:, prefix:], atol=1e-5), (
        f"max abs {(tail - only[:, prefix:]).abs().max():.3e}"
    )


@pytest.mark.parametrize("flags", MAMBA3_FLAG_SETS, ids=lambda f: "-".join(sorted(f)) or "default")
def test_mamba3_prefill_with_documents_continues_the_last_one(flags):
    """With packed documents in the prompt, the cache holds only the last document: continuing it
    equals stepping that document alone (conv window and state both cut at the boundary)."""
    block = m3_block(chunk_size=8, **flags).eval()
    x = torch.randn(2, 36, 128)
    ids = torch.zeros(2, 36, dtype=torch.long)
    ids[:, 21:] = 1  # the prompt [0, 23) ends two tokens into document 1
    _, tail, _ = _continue_with_steps(block, x, 23, doc_ids=ids)
    alone = step_through(block, x[:, 21:])
    assert torch.allclose(tail, alone[:, 2:], atol=1e-5), f"max abs {(tail - alone[:, 2:]).abs().max():.3e}"


def test_mamba3_forward_refuses_a_filled_cache():
    block = m3_block().eval()
    cache = block.allocate_inference_cache(1)
    with torch.no_grad():
        block(torch.randn(1, 4, 128), cache=cache)
        with pytest.raises(ValueError, match="empty cache"):
            block(torch.randn(1, 4, 128), cache=cache)


def test_mamba3_cache_size_is_independent_of_context():
    block = m3_block(use_rope=True, use_trapezoid=True)
    short = block.allocate_inference_cache(1)
    long = block.allocate_inference_cache(1)
    x = torch.randn(1, 64, 128)
    with torch.no_grad():
        for t in range(64):
            block.step(x[:, t], long)

    def size(c):
        return sum(v.numel() for v in c.values() if torch.is_tensor(v))

    assert size(long) == size(short)


# -- mLSTM ----------------------------------------------------------------------------------------


def test_mlstm_forward_is_finite():
    block = mLSTMBlock(dim=256, head_dim=64, num_heads=4)
    out = block(torch.randn(2, 128, 256))
    assert out.shape == (2, 128, 256) and torch.isfinite(out).all()


def test_mlstm_soft_cap_keeps_large_inputs_finite():
    block = mLSTMBlock(dim=64, head_dim=32, num_heads=2, gate_soft_cap=15.0)
    nn.init.constant_(block.i_gate_proj.weight, 1.0)
    nn.init.constant_(block.i_gate_proj.bias, 0.0)
    out = block(torch.ones(1, 4, 64) * 10.0)
    assert torch.isfinite(out).all()


def test_mlstm_input_gate_bias_init_in_the_block():
    block = mLSTMBlock(
        dim=64, head_dim=32, num_heads=2, input_gate_bias_init=-10.0, forget_gate_bias_init=3.0
    )
    assert torch.all(block.i_gate_proj.bias == -10.0) and torch.all(block.f_gate_proj.bias == 3.0)


def test_mlstm_step_reproduces_the_shipping_exact_tfla_operator():
    """Only exact TFLA equals a recurrence; the legacy kernel computes none a step could reproduce.

    Both blocks use the reference's forget bias 0.0 (f = 0.5): that is where legacy's clamp bites.
    At the shipped 3.0 the clamp is never reached and legacy happens to agree."""
    block = mLSTMBlock(dim=64, head_dim=16, tfla_impl="exact", forget_gate_bias_init=0.0).eval()
    x = torch.randn(2, 96, 64)
    with torch.no_grad():
        full = block(x)
    assert torch.allclose(full, step_through(block, x), atol=1e-5)

    legacy = mLSTMBlock(dim=64, head_dim=16, tfla_impl="legacy", forget_gate_bias_init=0.0).eval()
    legacy.load_state_dict(block.state_dict())
    with torch.no_grad():
        gap = (legacy(x) - step_through(legacy, x)).abs().max().item()
    assert gap > 1e-5, "legacy TFLA suddenly agrees with an exact recurrence -- revisit the defect record"


def test_mlstm_step_state_stays_fp32():
    """Defect 16 (P2-E): a bf16 decode keeps C and n in fp32, as Mamba3Block.step keeps its state."""
    torch.manual_seed(0)
    block32 = mLSTMBlock(dim=64, head_dim=16, tfla_impl="exact").eval()
    block16 = mLSTMBlock(dim=64, head_dim=16, tfla_impl="exact").eval()
    block16.load_state_dict(block32.state_dict())
    block16 = block16.to(torch.bfloat16)
    x = torch.randn(2, 24, 64)
    cache = block16.allocate_inference_cache(2, dtype=torch.bfloat16)
    with torch.no_grad():
        out16 = torch.stack([block16.step(x[:, t].to(torch.bfloat16), cache) for t in range(24)], dim=1)
    assert cache["C"].dtype is torch.float32 and cache["n"].dtype is torch.float32
    assert out16.dtype is torch.bfloat16
    out32 = step_through(block32, x)
    assert (out16.float() - out32).abs().max() / out32.abs().max() < 0.05


@pytest.mark.parametrize("prefix", [1, 17, 40], ids=lambda p: f"prefix{p}")
@pytest.mark.parametrize("hybrid_norm", [False, True], ids=["plain", "hybridnorm"])
def test_mlstm_forward_then_step_equals_step_only(prefix, hybrid_norm):
    """P2-G: TFLA's final (C, n) fill the cache, so stepping on after a one-pass prefill equals
    stepping from the first token (chunk 8: prompts ending mid-chunk and on a chunk edge)."""
    block = mLSTMBlock(
        dim=64, head_dim=16, tfla_impl="exact", chunk_size=8, use_hybrid_norm=hybrid_norm
    ).eval()
    x = torch.randn(2, 48, 64)
    cache = block.allocate_inference_cache(2)
    with torch.no_grad():
        head = block(x[:, :prefix], cache=cache)
        assert cache["seen"] == prefix and cache["C"].dtype is torch.float32
        tail = torch.stack([block.step(x[:, t], cache) for t in range(prefix, 48)], dim=1)
    only = step_through(block, x)
    assert torch.allclose(head, only[:, :prefix], atol=1e-5)
    assert torch.allclose(tail, only[:, prefix:], atol=1e-5), (
        f"max abs {(tail - only[:, prefix:]).abs().max():.3e}"
    )


def test_mlstm_prefill_with_documents_continues_the_last_one():
    block = mLSTMBlock(dim=64, head_dim=16, tfla_impl="exact", chunk_size=8).eval()
    x = torch.randn(2, 36, 64)
    ids = torch.zeros(2, 36, dtype=torch.long)
    ids[:, 21:] = 1
    cache = block.allocate_inference_cache(2)
    with torch.no_grad():
        block(x[:, :23], cache=cache, doc_ids=ids[:, :23])
        tail = torch.stack([block.step(x[:, t], cache) for t in range(23, 36)], dim=1)
    alone = step_through(block, x[:, 21:])
    assert torch.allclose(tail, alone[:, 2:], atol=1e-5)
    with pytest.raises(ValueError, match="empty cache"), torch.no_grad():
        block(x[:, :4], cache=cache)


def test_mlstm_document_reset_matches_running_the_document_alone():
    block = mLSTMBlock(dim=64, head_dim=16, tfla_impl="exact").eval()
    ids = torch.zeros(2, 50, dtype=torch.long)
    ids[:, 21:] = 1
    x = torch.randn(2, 50, 64)
    with torch.no_grad():
        packed = block(x, doc_ids=ids)
        alone = block(x[:, 21:])
    assert torch.allclose(packed[:, 21:], alone, atol=1e-6)


# -- attention ------------------------------------------------------------------------------------


def test_attention_is_causal_and_blocks_cross_document():
    block = AttentionBlock(dim=64, num_heads=4).eval()
    x = torch.randn(1, 20, 64)
    ids = torch.tensor([[0] * 8 + [1] * 12])
    with torch.no_grad():
        base = block(x, doc_ids=ids)
        future = x.clone()
        future[:, 15:] += 3.0
        assert torch.allclose(block(future, doc_ids=ids)[:, :15], base[:, :15], atol=1e-6), "not causal"
        other_doc = x.clone()
        other_doc[:, :8] += 3.0
        assert torch.allclose(block(other_doc, doc_ids=ids)[:, 8:], base[:, 8:], atol=1e-6), (
            "attends across docs"
        )


def test_attention_positions_restart_per_document():
    """Defect 8 (P2-H): a document in a packed row sees the RoPE positions it would see alone, so its
    output equals running it alone -- for the second document too, at any offset."""
    from lexhybrid.layers.attention_block import positions_within_documents

    torch.manual_seed(0)
    block = AttentionBlock(dim=64, num_heads=4, rope_theta=500000.0, max_position_embeddings=64).eval()
    x = torch.randn(2, 30, 64)
    ids = torch.tensor([[0] * 11 + [1] * 19, [0] * 5 + [1] * 17 + [2] * 8])
    assert positions_within_documents(ids)[1].tolist() == list(range(5)) + list(range(17)) + list(range(8))
    with torch.no_grad():
        packed = block(x, doc_ids=ids)
        assert torch.allclose(packed[0, 11:], block(x[:1, 11:])[0], atol=1e-6)
        assert torch.allclose(packed[1, 5:22], block(x[1:, 5:22])[0], atol=1e-6)
        assert torch.allclose(packed[1, 22:], block(x[1:, 22:])[0], atol=1e-6)


def test_attention_rope_tables_extend_with_the_configured_theta():
    """Defect 8's second half: past `max_position_embeddings` the reference rebuilt its tables with
    theta 10,000 whatever rope_theta said. A short cache must now give the long cache's output."""
    torch.manual_seed(0)
    short = AttentionBlock(dim=64, num_heads=4, rope_theta=500000.0, max_position_embeddings=8).eval()
    long = AttentionBlock(dim=64, num_heads=4, rope_theta=500000.0, max_position_embeddings=64).eval()
    long.load_state_dict(short.state_dict())
    x = torch.randn(1, 40, 64)
    with torch.no_grad():
        assert torch.allclose(short(x), long(x), atol=1e-6)
        wrong = AttentionBlock(dim=64, num_heads=4, rope_theta=10000.0, max_position_embeddings=64).eval()
        wrong.load_state_dict(short.state_dict())
        assert not torch.allclose(short(x), wrong(x), atol=1e-4)


def _packed_doc_ids(batch: int, seq_len: int) -> torch.Tensor:
    ids = torch.zeros(batch, seq_len, dtype=torch.long)
    ids[0, 37:] = 1
    ids[0, 70:] = 2
    if batch > 1:
        ids[1, 1:] = 1  # a one-token document, then one crossing the 64/128 tile edges
    return ids


def test_doc_block_mask_matches_the_dense_mask():
    """Semantics of the P2-J block mask, checked with eager FlexAttention (runs everywhere): causal
    inside each document, nothing across one -- the dense mask's exact meaning."""
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import flex_attention

    from lexhybrid.layers.attention_block import build_doc_block_mask

    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 100, 16) for _ in range(3))
    ids = _packed_doc_ids(2, 100)
    dense = F.scaled_dot_product_attention(q, k, v, attn_mask=build_doc_boundary_attn_mask(ids))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "flex_attention called without torch.compile()"
        flex = flex_attention(q, k, v, block_mask=build_doc_block_mask(ids))
    assert torch.allclose(flex, dense, atol=1e-5)


def test_attn_impl_auto_keeps_the_dense_mask_on_cpu():
    block = AttentionBlock(dim=64, num_heads=4)
    q = torch.zeros(1, 4, 8, 16)
    assert block._packed_impl(q, 0.0) == "sdpa"
    assert AttentionBlock(dim=64, num_heads=4, attn_impl="flex")._packed_impl(q, 0.0) == "flex"
    with pytest.raises(ValueError, match="dropout"):
        AttentionBlock(dim=64, num_heads=4, attn_impl="flex")._packed_impl(q, 0.1)
    with pytest.raises(ValueError, match="attn_impl"):
        AttentionBlock(dim=64, num_heads=4, attn_impl="dense")


@pytest.mark.linux_only
def test_flex_block_mask_equals_dense():
    """Defect 18 (P2-J): the compiled FlexAttention path (Inductor CPU; the CUDA variant is in
    tests/test_gpu.py) equals the dense-mask path for packed rows, forward and backward."""
    torch.manual_seed(0)
    dense = AttentionBlock(dim=64, num_heads=4, attn_impl="sdpa")
    flex = AttentionBlock(dim=64, num_heads=4, attn_impl="flex")
    flex.load_state_dict(dense.state_dict())
    x = torch.randn(2, 100, 64, requires_grad=True)
    ids = _packed_doc_ids(2, 100)
    out_dense = dense(x, doc_ids=ids)
    out_flex = flex(x, doc_ids=ids)
    assert torch.allclose(out_flex, out_dense, atol=1e-5), f"max abs {(out_flex - out_dense).abs().max():.3e}"
    g_dense = torch.autograd.grad(out_dense.square().sum(), dense.qkv_proj.weight)[0]
    g_flex = torch.autograd.grad(out_flex.square().sum(), flex.qkv_proj.weight)[0]
    assert torch.allclose(g_flex, g_dense, atol=1e-4)


def test_doc_boundary_mask_shape_and_values():
    mask = build_doc_boundary_attn_mask(torch.tensor([[0, 0, 1, 1]]))
    assert mask.shape == (1, 1, 4, 4)
    allowed = mask[0, 0] == 0
    assert allowed.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, True, True],
    ]


# -- Mamba-1 --------------------------------------------------------------------------------------


@pytest.mark.parametrize("scan_impl", ["legacy", "exact"])
def test_mamba1_forward_backward(scan_impl):
    block = MambaBlock(dim=64, state_size=8, scan_impl=scan_impl)
    out = block(torch.randn(2, 40, 64))
    out.sum().backward()
    assert out.shape == (2, 40, 64) and torch.isfinite(out).all()
    assert not no_grad_params(block)


# -- HybridBlock ----------------------------------------------------------------------------------


@pytest.mark.parametrize("layer_type", ["mamba", "mamba3", "mlstm", "attention"])
def test_hybrid_block_all_layer_types(layer_type):
    block = HybridBlock(
        dim=128,
        layer_type=layer_type,
        state_size=16,
        mamba3_d_state=32,
        mamba3_head_dim=16,
        head_dim=32,
        num_heads=4,
    )
    out = block(torch.randn(2, 64, 128))
    assert out.shape == (2, 64, 128) and torch.isfinite(out).all()


@pytest.mark.parametrize("use_mlp", [True, False])
def test_hybrid_block_with_and_without_mlp(use_mlp):
    block = HybridBlock(dim=64, layer_type="mamba", use_mlp=use_mlp, state_size=8)
    assert (block.mlp is not None) == use_mlp
    assert block(torch.randn(2, 16, 64)).shape == (2, 16, 64)


def test_hybrid_norm_ffn_is_post_residual_from_block_one():
    """Under `hybrid`, block >= 1 computes norm2(x + mlp(x)); block 0 stays pre-norm."""

    def build(is_first):
        blk = HybridBlock(
            dim=64, layer_type="mamba", norm_topology="hybrid", is_first_block=is_first, state_size=8
        )
        with torch.no_grad():
            blk.mlp[-1].weight.zero_()  # mlp(x) = 0
        return blk

    x = torch.randn(2, 16, 64)
    nonfirst = build(False)
    first = build(True)
    first.load_state_dict(nonfirst.state_dict())
    assert torch.allclose(nonfirst(x), nonfirst.norm2(first(x)), atol=1e-5)


def test_mlstm_kwargs_reach_block():
    """Defect 2 (P2-A): the reference filtered on unprefixed names, so every `mlstm_*` option fell
    back to the block default. Gate biases are checked on the bare block, before any model init."""
    block = HybridBlock(
        dim=64,
        layer_type="mlstm",
        head_dim=32,
        mlstm_gate_soft_cap=7.0,
        mlstm_input_gate_bias_init=-4.0,
        mlstm_forget_gate_bias_init=2.5,
        mlstm_chunk_size=16,
    ).mixer
    assert (block.gate_soft_cap, block.chunk_size) == (7.0, 16)
    assert torch.all(block.i_gate_proj.bias == -4.0) and torch.all(block.f_gate_proj.bias == 2.5)


def test_mlstm_kwargs_reach_block_through_the_config():
    from lexhybrid import HybridConfig
    from lexhybrid.models.hybrid_lm import _RESERVED_BLOCK_ARGS

    cfg = HybridConfig(
        layer_pattern=["mlstm"],
        num_layers=1,
        dim=64,
        head_dim=32,
        mlstm_gate_soft_cap=9.0,
        mlstm_chunk_size=32,
    )
    kw = {k: v for k, v in cfg.to_dict().items() if k not in _RESERVED_BLOCK_ARGS}
    mixer = HybridBlock(dim=64, layer_type="mlstm", **kw).mixer
    assert (mixer.gate_soft_cap, mixer.chunk_size) == (9.0, 32)


def test_mamba3_expand_factor_independent():
    """Defect 4 (P2-A): Mamba-3 read the Mamba-1 `expand_factor`; each mixer now reads its own."""
    common = dict(dim=64, state_size=8, mamba3_d_state=16, mamba3_head_dim=32, head_dim=32)
    m3 = HybridBlock(layer_type="mamba3", expand_factor=4, mamba3_expand_factor=1, **common).mixer
    m1 = HybridBlock(layer_type="mamba", expand_factor=4, mamba3_expand_factor=1, **common).mixer
    assert m3.inner_dim == 64 and m1.inner_dim == 256
    # And the prefixed key wins whichever order the flat bag delivers them in.
    reordered = HybridBlock(layer_type="mamba3", mamba3_expand_factor=1, expand_factor=4, **common).mixer
    assert reordered.inner_dim == 64


def test_prefixed_key_beats_the_shared_head_dim():
    block = HybridBlock(dim=64, layer_type="mamba3", mamba3_head_dim=16, head_dim=32, mamba3_d_state=16).mixer
    assert block.head_dim == 16 and block.nheads == 8


@pytest.mark.parametrize("bad_kwarg", ["mamba3_use_ropee", "mamba3_dstate", "mlstm_gate_softcap"])
def test_typo_in_a_prefixed_mixer_option_raises(bad_kwarg):
    with pytest.raises(ValueError, match="unknown mixer option"):
        HybridBlock(dim=64, layer_type="mamba3", mamba3_d_state=16, mamba3_head_dim=32, **{bad_kwarg: True})


def test_supports_doc_ids_matches_the_forward_signature():
    """The declared capability and the real signature must not drift apart."""
    for cls in (MambaBlock, Mamba3Block, mLSTMBlock, AttentionBlock):
        declared = getattr(cls, "supports_doc_ids", None)
        assert declared is not None, f"{cls.__name__} must declare supports_doc_ids"
        assert declared == ("doc_ids" in inspect.signature(cls.forward).parameters)


def test_every_layer_type_declares_the_capability():
    for layer_type in typing.get_args(LayerType):
        block = HybridBlock(
            dim=64,
            layer_type=layer_type,
            state_size=8,
            head_dim=32,
            num_heads=2,
            mamba3_d_state=16,
            mamba3_head_dim=32,
        )
        assert block._mixer_takes_doc_ids is True


def test_the_dispatcher_whitelist_is_derived_not_listed():
    params = inspect.signature(Mamba3Block.__init__).parameters
    expected = {
        n
        for n, p in params.items()
        if n not in ("self", "dim") and p.kind is not inspect.Parameter.VAR_KEYWORD
    }
    assert set(_mamba3_params()) == expected and "theta_max" in expected


def test_block_step_refuses_a_mixer_without_step():
    block = HybridBlock(dim=64, layer_type="mamba", state_size=8)
    with pytest.raises(NotImplementedError, match="no step"):
        block.step(torch.randn(2, 64), None)
