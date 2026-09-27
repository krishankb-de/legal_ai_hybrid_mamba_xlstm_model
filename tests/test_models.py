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
        "vocab=256",
        "tied=False",  # the reference replica keeps the reference's untied head (exact counts)
        "expand=2",
        "mlstm(chunk_size=128, forget_bias=0.0, fallback=error)",
        "params=",
        "params_nonembed=",
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


@pytest.mark.parametrize("forget_bias", [0.0, 3.0])
def test_mlstm_gate_bias_survives_model_init(forget_bias):
    """Defect 3 (P2-B): the model's weight pass zeroed both gate biases; `post_model_init` restores
    the -10 input bias and the configured forget bias inside the built model."""
    model = tiny(["mlstm", "mamba3"], mlstm_forget_gate_bias_init=forget_bias)
    mixer = model.layers[0].mixer
    assert torch.all(mixer.i_gate_proj.bias == -10.0)
    assert torch.all(mixer.f_gate_proj.bias == forget_bias)
    # The weight pass still owns every other Linear bias: the output gate starts at zero.
    assert torch.equal(mixer.o_gate_proj.bias, torch.zeros_like(mixer.o_gate_proj.bias))


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


def test_loss_masks_boundary_keeps_eos():
    """Defect 6 (P2-L): a document's last token (its EOS) does not predict the next document's
    first; the EOS itself stays a target; a row that starts mid-document needs no mask at its start;
    -100 labels are ignored; `n_supervised_tokens` counts what entered the loss."""
    import torch.nn.functional as F

    model = tiny(["mamba3", "attention"], dropout=0.0).eval()
    eos = 3
    ids = torch.randint(4, 128, (2, 16))
    ids[0, 6] = eos  # document 0 of row 0 ends at position 6
    doc = torch.tensor([[0] * 7 + [1] * 9, [5] * 10 + [6] * 6])  # row 1 starts mid-document 5
    labels = ids.clone()
    labels[1, 12] = -100
    with torch.no_grad():
        out = model(ids, labels=labels, doc_ids=doc)
    targets = labels[:, 1:].clone()
    targets[0, 6] = -100  # EOS(6) -> first token of doc 1: masked
    targets[1, 9] = -100  # last of doc 5 -> first of doc 6: masked
    want = F.cross_entropy(out.logits[:, :-1].reshape(-1, 128), targets.reshape(-1), ignore_index=-100)
    assert torch.allclose(out.loss, want, atol=1e-6)
    assert out.n_supervised_tokens.item() == 2 * 15 - 3
    assert targets[0, 5] == eos, "the EOS stays a target, predicted from inside its document"
    with torch.no_grad():
        unmasked = model(ids, labels=labels)
    assert unmasked.n_supervised_tokens.item() == 2 * 15 - 1 and not torch.allclose(unmasked.loss, out.loss)


def test_loss_with_nothing_supervised_is_zero_not_nan():
    model = tiny(["mamba3"]).eval()
    ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        out = model(ids, labels=torch.full_like(ids, -100))
    assert out.loss.item() == 0.0 and out.n_supervised_tokens.item() == 0


def test_param_counts_tied_untied():
    """Defect 17 and decision 11 (P2-M): init then tie; `non_embedding = total - embedding - (lm_head if
    untied)`, so tying changes the total but never the non-embedding count; both are in the ARCH line."""
    untied = tiny(["mamba3", "mlstm", "attention"], tie_word_embeddings=False)
    tied = tiny(["mamba3", "mlstm", "attention"], tie_word_embeddings=True)
    emb = untied.embeddings.token_embedding.weight.numel()
    assert untied.get_num_params(False) == sum(p.numel() for p in untied.parameters())
    assert untied.get_num_params(False) - tied.get_num_params(False) == emb  # the head, counted once
    assert untied.get_num_params(True) == untied.get_num_params(False) - 2 * emb
    assert tied.get_num_params(True) == tied.get_num_params(False) - emb
    assert tied.get_num_params(True) == untied.get_num_params(True)
    # Init then tie: under the same seed the tied matrix is the embedding's own draw. Tying first
    # (the reference) overwrote it with the lm_head's draw.
    assert torch.equal(tied.embeddings.token_embedding.weight, untied.embeddings.token_embedding.weight)
    assert not torch.equal(tied.lm_head.weight, untied.lm_head.weight)
    fp = tied.architecture_fingerprint()
    assert f"params={tied.get_num_params(False):,}" in fp
    assert f"params_nonembed={tied.get_num_params(True):,}" in fp and "tied=True" in fp


def test_mtp_head_off_is_bit_identical():
    """P2-S: at mtp_n = 1 nothing is built -- the other MTP settings are inert -- and turning MTP on
    leaves every main-model weight and the main logits bit-identical (the head is built last)."""
    pattern = ["mamba3", "mlstm", "attention"]
    off = tiny(pattern, dropout=0.0).eval()
    off_other = tiny(pattern, dropout=0.0, mtp_loss_weight=0.9, mtp_layer_type="attention").eval()
    on = tiny(pattern, dropout=0.0, mtp_n=2).eval()
    assert off.mtp_head is None and not any("mtp" in k for k in off.state_dict())
    ids = torch.randint(0, 128, (2, 24))
    with torch.no_grad():
        a, b, c = (m(ids, labels=ids) for m in (off, off_other, on))
    assert torch.equal(a.logits, b.logits) and torch.equal(a.loss, b.loss) and a.mtp_loss is None
    for name, tensor in off.state_dict().items():
        assert torch.equal(tensor, on.state_dict()[name]), name
    assert torch.equal(a.logits, c.logits)
    assert torch.allclose(c.loss, a.loss + 0.3 * c.mtp_loss, atol=1e-6)
    assert on.get_num_params(False) > off.get_num_params(False) and "mtp(n=2" in on.architecture_fingerprint()


def test_mtp_loss_backward():
    """Two MTP depths train: every parameter (main and MTP) receives a gradient, packed documents
    included, and a zero weight gives back exactly the language-model loss."""
    model = tiny(["mamba3", "attention"], dropout=0.0, mtp_n=3, mtp_loss_weight=0.5, tfla_impl="exact")
    ids = torch.randint(0, 128, (2, 40))
    doc = torch.zeros(2, 40, dtype=torch.long)
    doc[:, 17:] = 1
    out = model(ids, labels=ids, doc_ids=doc)
    assert torch.isfinite(out.mtp_loss) and out.mtp_loss > 0
    out.loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient: {missing[:5]}"
    assert len(model.mtp_head.depths) == 2
    model.config.mtp_loss_weight = 0.0
    off = tiny(["mamba3", "attention"], dropout=0.0, tfla_impl="exact")
    off.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("mtp_head.")})
    with torch.no_grad():
        zero = model(ids, labels=ids, doc_ids=doc)
        base = off(ids, labels=ids, doc_ids=doc)
    assert torch.allclose(zero.loss, base.loss, atol=1e-6) and zero.mtp_loss > 0


def test_mtp_masks_targets_across_documents():
    """Depth k supervises position i only if i and i + k + 1 share a document."""
    from lexhybrid.kernels.segments import segment_ids

    doc = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    seg = segment_ids(doc, 1, 7, doc.device)
    k, length = 1, 7 - 2
    keep = seg[:, k + 1 : k + 1 + length] == seg[:, :length]
    assert keep.tolist() == [[True, False, False, True, True]]


def test_tie_word_embeddings_defaults_to_true():
    assert HybridConfig().tie_word_embeddings is True


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


@pytest.mark.linux_only
@pytest.mark.slow
def test_compile_forward_chunk128(monkeypatch):
    """P2-Y: hybrid_legal_base at reduced width -- chunk 128 for both Mamba-3 and mLSTM, packed rows
    -- compiled on the Inductor CPU backend (gcc; runs in CI, the CUDA variant runs in P4-J1) equals
    eager at 1e-4, and exact TFLA stays on its factorised path: the sequential fallback is spied
    and tfla_fallback is "error", so either path would fail the test."""
    import lexhybrid.kernels.tfla.tfla_interface as tfla

    def fallback_taken(*args, **kwargs):
        raise AssertionError("exact TFLA took the sequential fallback")

    monkeypatch.setattr(tfla, "_intra_chunk_sequential", fallback_taken)
    torch.manual_seed(0)
    cfg = load_model_config(
        "hybrid_legal_base",
        dim=64,
        num_heads=2,
        head_dim=32,
        mamba3_head_dim=32,
        vocab_size=512,
        max_position_embeddings=512,
    )
    assert (cfg.mlstm_chunk_size, cfg.mamba3_chunk_size, cfg.tfla_fallback) == (128, 128, "error")
    model = HybridLanguageModel(cfg).eval()
    ids = torch.randint(0, 512, (2, 256))
    doc = torch.zeros(2, 256, dtype=torch.long)
    doc[0, 100:] = 1
    doc[1, 128:] = 1  # a boundary exactly on the chunk edge
    with torch.no_grad():
        eager = model(ids, doc_ids=doc).logits
        torch._dynamo.reset()
        compiled = torch.compile(model, dynamic=False)
        got = compiled(ids, doc_ids=doc).logits
    assert torch.allclose(got, eager, atol=1e-4), f"compile drift {(got - eager).abs().max():.3e}"
