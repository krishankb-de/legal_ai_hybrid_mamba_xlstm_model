"""Cached decode must be an equivalence with full recomputation (reference M6-D; plan P1-N)."""

import pytest
import torch

from lexhybrid import HybridConfig, HybridLanguageModel
from lexhybrid.decoding import beam_search_uncached


def cached_lm(vocab=97, **cfg_kw):
    torch.manual_seed(0)
    kw = dict(
        vocab_size=vocab,
        dim=64,
        num_layers=4,
        layer_pattern=["mamba3", "mlstm"],
        head_dim=16,
        num_heads=4,
        max_position_embeddings=64,
        tfla_impl="exact",
        mamba3_d_state=32,
        mamba3_head_dim=16,
        mamba3_chunk_size=8,
    )
    kw.update(cfg_kw)
    return HybridLanguageModel(HybridConfig(**kw)).eval()


def test_cached_decode_matches_full_recompute():
    """Compared on logits, not sampled tokens, so the assertion is deterministic and stronger."""
    model = cached_lm()
    assert model.supports_cached_decode()
    ids = torch.randint(0, 97, (2, 7))
    with torch.no_grad():
        hidden = model.embeddings(ids)
        caches = model.allocate_inference_cache(2)
        cached = model.prefill(hidden, caches)
        full = model(inputs_embeds=hidden).logits[:, -1]
        assert torch.allclose(full, cached, atol=1e-5)
        for _ in range(6):
            nxt = full.argmax(-1, keepdim=True)
            hidden = torch.cat([hidden, model.embeddings(nxt)], dim=1)
            cached = model.step_logits(model.embeddings(nxt)[:, 0], caches)
            full = model(inputs_embeds=hidden).logits[:, -1]
            assert torch.allclose(full, cached, atol=1e-5), f"drift {(full - cached).abs().max():.3e}"


@pytest.mark.parametrize(
    "flags", [{}, {"mamba3_use_trapezoid": True, "mamba3_use_rope": True, "mamba3_theta_max": 0.2}]
)
def test_cached_beam_equals_uncached(flags):
    """Beams reorder and duplicate every step; a beam inheriting another's state writes fluent, wrong
    text, and token equality against the uncached search is the check that catches it."""
    model = cached_lm(**flags)
    ids = torch.randint(0, 97, (1, 6))
    uncached = beam_search_uncached(model, ids, beam_size=3, max_new_tokens=8)
    cached = model.beam_search_cached(ids, beam_size=3, max_new_tokens=8)
    assert torch.equal(uncached, cached), f"uncached {uncached.tolist()} vs cached {cached.tolist()}"


def test_cached_decode_refuses_a_stack_it_cannot_serve():
    """All-or-nothing: Mamba-1 (and, until P2-I, attention) has no step()."""
    for pattern in (["mamba3", "mamba"], ["mamba3", "attention"]):
        model = cached_lm(layer_pattern=pattern, state_size=8)
        assert not model.supports_cached_decode()
        with pytest.raises(NotImplementedError, match="step"):
            model.generate_cached(torch.randint(0, 97, (1, 4)), max_new_tokens=2)


def test_reorder_cache_gathers_tensors_and_copies_scalars():
    model = cached_lm()
    caches = model.allocate_inference_cache(3)
    caches[0]["ssm_state"][1].fill_(7.0)
    moved = model.reorder_cache(caches, torch.tensor([1, 1, 0]))
    assert torch.all(moved[0]["ssm_state"][0] == 7.0) and torch.all(moved[0]["ssm_state"][2] == 0.0)
    assert moved[0]["seen"] == caches[0]["seen"]


def test_beam_search_refuses_a_batch():
    model = cached_lm()
    with pytest.raises(ValueError, match="one sample"):
        model.beam_search_cached(torch.randint(0, 97, (2, 4)))
    with pytest.raises(ValueError, match="one sample"):
        beam_search_uncached(model, torch.randint(0, 97, (2, 4)))
