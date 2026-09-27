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
    """Only exact TFLA equals a recurrence; the legacy kernel computes none a step could reproduce."""
    block = mLSTMBlock(dim=64, head_dim=16, tfla_impl="exact").eval()
    x = torch.randn(2, 96, 64)
    with torch.no_grad():
        full = block(x)
    assert torch.allclose(full, step_through(block, x), atol=1e-5)

    legacy = mLSTMBlock(dim=64, head_dim=16, tfla_impl="legacy").eval()
    legacy.load_state_dict(block.state_dict())
    with torch.no_grad():
        gap = (legacy(x) - step_through(legacy, x)).abs().max().item()
    assert gap > 1e-5, "legacy TFLA suddenly agrees with an exact recurrence -- revisit the defect record"


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
