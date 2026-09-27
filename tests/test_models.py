"""Model-level behaviour of HybridLanguageModel (plan P1-N)."""

import copy

import pytest
import torch

from lexhybrid import HybridConfig, HybridLanguageModel
from lexhybrid.config import load_model_config


def tiny(pattern, **kw):
    torch.manual_seed(0)
    defaults = dict(
        vocab_size=128,
        dim=64,
        num_layers=len(pattern),
        layer_pattern=pattern,
        state_size=8,
        mamba3_d_state=16,
        mamba3_head_dim=32,
        head_dim=32,
        num_heads=2,
        max_position_embeddings=64,
    )
    defaults.update(kw)
    return HybridLanguageModel(HybridConfig(**defaults))


def test_forward_backward_all_mixers_every_parameter_gets_a_gradient():
    model = tiny(
        ["mamba", "mamba3", "mlstm", "attention"],
        tfla_impl="exact",
        scan_impl="exact",
        norm_topology="hybrid",
    )
    ids = torch.randint(0, 128, (2, 32))
    doc_ids = torch.tensor([[0] * 12 + [1] * 20] * 2)
    out = model(ids, labels=ids, doc_ids=doc_ids)
    assert torch.isfinite(out.loss) and out.logits.shape == (2, 32, 128)
    out.loss.backward()
    assert [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None] == []


def test_return_tuple_and_hidden_states():
    model = tiny(["mamba3", "attention"])
    ids = torch.randint(0, 128, (1, 10))
    loss, logits, hidden = model(ids, labels=ids, output_hidden_states=True, return_dict=False)
    assert logits.shape == (1, 10, 128) and len(hidden) == 3 and loss.ndim == 0


def test_inputs_embeds_equals_input_ids_path():
    model = tiny(["mamba3", "mlstm"]).eval()
    ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        a = model(ids).logits
        b = model(inputs_embeds=model.embeddings(ids)).logits
    assert torch.equal(a, b)
    with pytest.raises(ValueError, match="Exactly one"):
        model(ids, inputs_embeds=model.embeddings(ids))


def test_attention_mask_zeroes_padded_positions_at_the_embedding():
    model = tiny(["mamba3"]).eval()
    ids = torch.randint(0, 128, (1, 6))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        a = model(ids, attention_mask=mask).logits
        b = model(inputs_embeds=model.embeddings(ids) * mask.unsqueeze(-1).float()).logits
    assert torch.equal(a, b)


def test_get_layer_types_cycles_the_pattern():
    model = tiny(["mamba", "mamba3", "mlstm", "attention"], num_layers=8)
    assert model.get_layer_types() == ["mamba", "mamba3", "mlstm", "attention"] * 2


def test_generation_lengths():
    model = tiny(["mamba3", "mlstm"], tfla_impl="exact")
    ids = torch.randint(0, 128, (1, 5))
    assert model.generate(ids, max_new_tokens=7).shape == (1, 12)
    assert model.generate_cached(ids, max_new_tokens=7).shape == (1, 12)


def test_arch_fingerprint_tokens():
    fp = HybridLanguageModel(
        load_model_config(
            "ref_hybrid_m3", dim=128, num_heads=4, head_dim=32, mamba3_head_dim=32, vocab_size=256
        )
    ).architecture_fingerprint()
    assert fp.startswith("ARCH ")
    for token in (
        "mamba3x9",
        "mlstmx3",
        "d_state=128",
        "trapezoid=False",
        "rope=False",
        "scan_impl=legacy",
        "params=",
    ):
        assert token in fp, f"fingerprint is missing {token!r}: {fp}"
    assert "mamba3(" not in tiny(["attention"]).architecture_fingerprint()


@pytest.mark.parametrize("scan_impl", ["legacy", "exact"])
def test_scan_and_tfla_impl_are_threaded_from_config_to_block(scan_impl):
    model = tiny(["mamba", "mlstm"], scan_impl=scan_impl, tfla_impl=scan_impl)
    mamba = next(la.mixer for la in model.layers if la.layer_type == "mamba")
    mlstm = next(la.mixer for la in model.layers if la.layer_type == "mlstm")
    assert mamba.scan_impl == scan_impl and mlstm.tfla_impl == scan_impl


def test_mamba3_config_flags_reach_the_block():
    model = tiny(
        ["mamba3", "mlstm"],
        dim=128,
        num_heads=4,
        mamba3_d_state=32,
        mamba3_head_dim=16,
        mamba3_chunk_size=8,
        mamba3_use_conv=False,
        mamba3_a_mode="data_dependent",
        mamba3_dt_limit=0.5,
        mamba3_bc_bias="one_init",
        mamba3_theta_max=0.02,
        mamba3_use_rope=True,
    )
    mixer = model.layers[0].mixer
    assert (mixer.d_state, mixer.head_dim, mixer.chunk_size) == (32, 16, 8)
    assert mixer.use_conv is False and mixer.conv1d is None
    assert mixer.a_mode == "data_dependent" and mixer.dt_limit == 0.5 and mixer.theta_max == 0.02
    assert mixer.bc_bias == "one_init" and torch.equal(mixer.B_bias, torch.ones_like(mixer.B_bias))
    with pytest.raises(NotImplementedError, match="MIMO"):
        tiny(["mamba3"], mamba3_mimo_rank=4)


# Reference M1-B/F: Delta at init under each norm topology and dt init.
def _delta_at_init(model, seq_len=128, batch=4):
    torch.manual_seed(0)
    block = next(la for la in model.layers if la.layer_type == "mamba")
    mixer = block.mixer
    ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
    with torch.no_grad():
        x = block.norm1(model.embeddings(ids))
        x_inner, _ = mixer.in_proj(x).chunk(2, dim=-1)
        x_conv = mixer.activation(mixer.conv1d(x_inner.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        dt = mixer.dt_proj(mixer.x_proj(x_conv)[..., : mixer.dt_rank])
        if mixer.dt_norm is not None:
            dt = mixer.dt_norm(dt)
        return torch.nn.functional.softplus(dt)


def _delta_model(norm_topology, dt_init_strategy="none"):
    return tiny(
        ["mamba", "mlstm"],
        vocab_size=512,
        dim=128,
        state_size=16,
        num_heads=4,
        max_position_embeddings=128,
        norm_topology=norm_topology,
        dt_init_strategy=dt_init_strategy,
    )


_DEFECT_DELTA = "reference M1: no Mamba dt init, and dt_norm would erase one. strict=True."


@pytest.mark.parametrize(
    "norm_topology,dt_init_strategy,broken",
    [
        ("pre_rms", "none", True),
        ("pre_rms", "mamba", False),
        ("hybrid", "none", True),
        ("hybrid", "mamba", True),  # dt_norm erases the fix -- pinned as broken on purpose
        ("hybrid_bc", "none", True),
        ("hybrid_bc", "mamba", False),
    ],
)
def test_delta_at_init_is_in_the_mamba_range(norm_topology, dt_init_strategy, broken, request):
    if broken:
        request.applymarker(pytest.mark.xfail(strict=True, reason=_DEFECT_DELTA))
    mean = _delta_at_init(_delta_model(norm_topology, dt_init_strategy)).mean().item()
    assert 1e-3 <= mean <= 1.5e-1, f"{norm_topology}/{dt_init_strategy}: Delta mean {mean:.4f}"


def test_dt_norm_erases_the_dt_init():
    with_norm = _delta_at_init(_delta_model("hybrid", "mamba")).mean().item()
    without_norm = _delta_at_init(_delta_model("hybrid_bc", "mamba")).mean().item()
    assert without_norm < 0.05 < with_norm and with_norm / without_norm > 5.0


def test_dt_proj_bias_is_zeroed_by_model_init():
    mixer = next(la for la in _delta_model("hybrid").layers if la.layer_type == "mamba").mixer
    assert torch.equal(mixer.dt_proj.bias, torch.zeros_like(mixer.dt_proj.bias))


def test_mlstm_gate_bias_is_zeroed_by_model_init_defect_3():
    """Pins recorded defect 3 until P2-B fixes it: the model's weight pass zeroes the gate biases."""
    mixer = next(la for la in tiny(["mlstm"]).layers).mixer
    assert torch.equal(mixer.i_gate_proj.bias, torch.zeros_like(mixer.i_gate_proj.bias))


def test_hybrid_bc_keeps_bc_norms_but_drops_dt_norm():
    hybrid = next(la.mixer for la in _delta_model("hybrid").layers if la.layer_type == "mamba")
    hybrid_bc = next(la.mixer for la in _delta_model("hybrid_bc").layers if la.layer_type == "mamba")
    assert hybrid.dt_norm is not None and hybrid.B_norm is not None
    assert hybrid_bc.dt_norm is None and hybrid_bc.B_norm is not None and hybrid_bc.C_norm is not None


@pytest.mark.parametrize("pattern", [["mamba3"], ["mamba3", "mlstm"]])
def test_ssd_runs_in_a_pure_bf16_model_without_autocast(pattern):
    """A `.to(bfloat16)` model without autocast once died in the SSD einsums (reference job 2560261)."""
    ref = tiny(pattern).eval()
    ids = torch.randint(0, 128, (2, 48))
    with torch.no_grad():
        out32 = ref(ids).logits
    bf = copy.deepcopy(ref).to(dtype=torch.bfloat16).eval()
    with torch.no_grad():
        out16 = bf(ids).logits
    assert out16.dtype is torch.bfloat16 and torch.isfinite(out16).all()
    assert (out16.float() - out32).abs().max() / out32.abs().max() < 0.05


def test_ssd_document_masking_survives_bf16():
    model = tiny(["mamba3"]).to(dtype=torch.bfloat16).eval()
    ids = torch.randint(0, 128, (1, 32))
    doc = torch.zeros(1, 32, dtype=torch.long)
    doc[:, 16:] = 1
    other = ids.clone()
    other[:, 16:] = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        a = model(ids, doc_ids=doc).logits[:, :16]
        b = model(other, doc_ids=doc).logits[:, :16]
    assert torch.equal(a, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_every_layer_type_runs_in_both_dtypes(dtype):
    model = tiny(["mamba", "mamba3", "mlstm", "attention"]).to(dtype=dtype).eval()
    ids = torch.randint(0, 128, (2, 32))
    doc = torch.zeros(2, 32, dtype=torch.long)
    doc[:, 16:] = 1
    with torch.no_grad():
        out = model(ids, doc_ids=doc).logits
    assert out.dtype is dtype and torch.isfinite(out).all()


def test_tied_embeddings_share_one_tensor():
    model = tiny(["mamba3"], tie_word_embeddings=True)
    assert model.lm_head.weight is model.embeddings.token_embedding.weight


def test_get_num_params_non_embedding_subtracts_the_token_embedding():
    model = tiny(["mamba3"])
    total = sum(p.numel() for p in model.parameters())
    assert model.get_num_params(non_embedding=False) == total
    assert model.get_num_params() == total - model.embeddings.token_embedding.weight.numel()


def test_gradient_checkpointing_matches_the_plain_forward():
    model = tiny(
        ["mamba3", "mlstm", "attention"], use_gradient_checkpointing=True, dropout=0.0, tfla_impl="exact"
    )
    model.train()
    ids = torch.randint(0, 128, (2, 24))
    loss_ckpt = model(ids, labels=ids).loss
    model.config.use_gradient_checkpointing = False
    loss_plain = model(ids, labels=ids).loss
    assert torch.allclose(loss_ckpt, loss_plain, atol=1e-6)
