"""Decoding (plans P1-N, P2-I..P2-R).

Cached decode must be an equivalence with full recomputation (reference M6-D). The uncached
references (P2-N) are tested against a bigram "model" whose next-token scores depend only on the
last token, so every expected output is computable by hand."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lexhybrid import HybridConfig, HybridLanguageModel
from lexhybrid.decoding import (
    apply_repetition_penalty,
    beam_search,
    beam_search_uncached,
    filter_logits,
    greedy,
    sample,
)


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


# The three families P2-K and the P2 gate name: the recurrent hybrid, the Transformer, and the legal
# base pattern (M3 M3 M3 A L L L M3 M3 A M3 M3, legacy-free), all at test width.
FAMILIES = {
    "hybrid": ["mamba3", "mlstm"],
    "transformer": ["attention"],
    "base_pattern": ["mamba3"] * 3
    + ["attention"]
    + ["mlstm"] * 3
    + ["mamba3"] * 2
    + ["attention"]
    + ["mamba3"] * 2,
}


def family_lm(family):
    pattern = FAMILIES[family]
    return cached_lm(layer_pattern=pattern, num_layers=len(pattern), mlstm_chunk_size=8)


def _prefill_then_greedy(model, ids, steps):
    """(cached, uncached) logits at the prompt end and for ``steps`` greedy tokens."""
    with torch.no_grad():
        caches = model.allocate_inference_cache(
            ids.shape[0], dtype=model.lm_head.weight.dtype, max_seq_len=ids.shape[1] + steps
        )
        cached = [model.prefill(model.embeddings(ids), caches)]
        full = [model(ids).logits[:, -1]]
        for _ in range(steps):
            nxt = full[-1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            cached.append(model.step_logits(model.embeddings(nxt)[:, 0], caches))
            full.append(model(ids).logits[:, -1])
    return torch.stack(cached), torch.stack(full)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_prefill_equals_uncached(family):
    """P2-K (defect 7): the one-pass prefill leaves every cache exactly where token-by-token decoding
    would, in fp32 to 1e-5 over the prompt end and 6 continued tokens."""
    model = family_lm(family)
    assert model.supports_cached_decode()
    cached, full = _prefill_then_greedy(model, torch.randint(0, 97, (2, 21)), steps=6)
    assert torch.allclose(cached, full, atol=1e-5), f"{family}: max abs {(cached - full).abs().max():.3e}"


@pytest.mark.parametrize("family", list(FAMILIES))
def test_prefill_equals_uncached_bf16_argmax(family):
    """In bf16 the two paths round differently, so the claim is argmax identity wherever the
    uncached top-2 margin exceeds bf16 noise -- and the margin is made large enough that most
    positions are tested (a vacuous pass is refused)."""
    model = family_lm(family)
    with torch.no_grad():
        model.lm_head.weight.mul_(40.0)  # widen logit margins beyond bf16 rounding
    model = model.to(torch.bfloat16)
    cached, full = _prefill_then_greedy(model, torch.randint(0, 97, (4, 21)), steps=6)
    top2 = full.float().topk(2, dim=-1).values
    decisive = (top2[..., 0] - top2[..., 1]) > 0.25
    assert decisive.float().mean() >= 0.5, f"{family}: only {decisive.float().mean():.0%} decisive positions"
    agree = cached.argmax(-1) == full.argmax(-1)
    assert bool(agree[decisive].all()), (
        f"{family}: argmax differs at {int((~agree & decisive).sum())} positions"
    )


def test_prefill_with_documents_continues_the_last_document():
    """A packed prompt: decoding continues each row's last document as if it were alone."""
    model = family_lm("base_pattern")
    ids = torch.randint(0, 97, (2, 20))
    doc = torch.zeros(2, 20, dtype=torch.long)
    doc[0, 7:] = 1
    doc[1, 13:] = 1
    with torch.no_grad():
        caches = model.allocate_inference_cache(2, max_seq_len=24)
        packed = model.prefill(model.embeddings(ids), caches, doc_ids=doc)
        alone = torch.stack([model(ids[:1, 7:]).logits[0, -1], model(ids[1:, 13:]).logits[0, -1]])
    assert torch.allclose(packed, alone, atol=1e-5)


@pytest.mark.parametrize("penalty", [1.0, 1.3], ids=["no-penalty", "penalty1.3"])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_cached_greedy_equals_uncached(family, penalty):
    """P2-O: cached greedy equals the uncached reference token for token, batch 3 with rows that
    finish at different steps (EOS is a token row 0 emits at its third step, so it must fire)."""
    from lexhybrid.decoding import greedy, greedy_cached

    model = family_lm(family)
    ids = torch.randint(0, 97, (3, 9))
    free = greedy(model, ids, max_new_tokens=6, repetition_penalty=penalty)
    eos = int(free[0, ids.shape[1] + 2])
    want = greedy(model, ids, max_new_tokens=12, eos_token_id=eos, pad_token_id=0, repetition_penalty=penalty)
    got = greedy_cached(
        model, ids, max_new_tokens=12, eos_token_id=eos, pad_token_id=0, repetition_penalty=penalty
    )
    assert torch.equal(got, want), f"{family}: cached {got.tolist()} vs uncached {want.tolist()}"
    assert (want[0, ids.shape[1] :] == eos).any()


def test_cached_sample_equals_uncached_under_one_generator():
    from lexhybrid.decoding import sample, sample_cached

    model = family_lm("base_pattern")
    ids = torch.randint(0, 97, (2, 7))
    kw = dict(max_new_tokens=10, temperature=0.8, top_k=20, top_p=0.9, repetition_penalty=1.2)
    want = sample(model, ids, generator=torch.Generator().manual_seed(3), **kw)
    got = sample_cached(model, ids, generator=torch.Generator().manual_seed(3), **kw)
    assert torch.equal(got, want)


def test_model_generate_methods_delegate_to_the_decoders():
    model = family_lm("hybrid")
    ids = torch.randint(0, 97, (1, 5))
    torch.manual_seed(1)
    a = model.generate_cached(ids, max_new_tokens=4, top_k=1)
    torch.manual_seed(1)
    b = model.generate(ids, max_new_tokens=4, top_k=1)
    assert torch.equal(a, b) and a.shape == (1, 9)


@pytest.mark.parametrize("length_penalty", [0.0, 1.0], ids=["lp0", "lp1"])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_cached_beam_equals_uncached_with_eos(family, length_penalty):
    """P2-P / defect 11: with EOS, length normalisation and a repetition penalty, cached beam search
    returns the uncached reference's tokens. EOS is a token the free best beam emits, so
    hypotheses do finish and leave the beam; the caches (KV included) follow every reorder."""
    from lexhybrid.decoding import beam_search, beam_search_cached

    model = family_lm(family)
    ids = torch.randint(0, 97, (1, 8))
    free = beam_search(model, ids, beam_size=3, max_new_tokens=6)
    eos = int(free[0, ids.shape[1] + 1])
    kw = dict(
        beam_size=3,
        max_new_tokens=14,
        eos_token_id=eos,
        length_penalty=length_penalty,
        repetition_penalty=1.2,
    )
    want = beam_search(model, ids, **kw)
    got = beam_search_cached(model, ids, **kw)
    assert torch.equal(got, want), f"{family}: cached {got.tolist()} vs uncached {want.tolist()}"
    assert torch.equal(
        model.beam_search_cached(ids, **{k: v for k, v in kw.items() if k != "beam_size"}, beam_size=3), want
    )


def test_cached_decode_refuses_a_stack_it_cannot_serve():
    """All-or-nothing: Mamba-1 has no step(), so the legacy stack cannot decode with a cache."""
    model = cached_lm(layer_pattern=["mamba3", "mamba"], state_size=8)
    assert not model.supports_cached_decode()
    with pytest.raises(NotImplementedError, match="step"):
        model.generate_cached(torch.randint(0, 97, (1, 4)), max_new_tokens=2)
    assert cached_lm(layer_pattern=["mamba3", "attention", "mlstm"]).supports_cached_decode()


def test_attention_step_matches_forward():
    """Defect 5 (P2-I): the KV-cache step reproduces the full forward, from an empty cache and after
    a one-pass prefill that fills it, with and without QK-norm."""
    from lexhybrid.layers.attention_block import AttentionBlock

    for qk_norm in (False, True):
        torch.manual_seed(0)
        block = AttentionBlock(
            dim=64, num_heads=4, use_hybrid_norm=qk_norm, max_position_embeddings=32
        ).eval()
        x = torch.randn(2, 20, 64)
        with torch.no_grad():
            full = block(x)
            cache = block.allocate_inference_cache(2, max_seq_len=20)
            stepped = torch.stack([block.step(x[:, t], cache) for t in range(20)], dim=1)
            assert torch.allclose(full, stepped, atol=1e-5), f"step drift {(full - stepped).abs().max():.3e}"
            cache = block.allocate_inference_cache(2, max_seq_len=20)
            head = block(x[:, :7], cache=cache)
            tail = torch.stack([block.step(x[:, t], cache) for t in range(7, 20)], dim=1)
            assert torch.allclose(torch.cat([head, tail], 1), full, atol=1e-5)
            with pytest.raises(RuntimeError, match="KV cache full"):
                block.step(x[:, 0], cache)


def test_attention_prefill_with_documents_continues_the_last_one():
    """After a packed prompt the cache masks earlier documents and RoPE continues the last one."""
    from lexhybrid.layers.attention_block import AttentionBlock

    torch.manual_seed(0)
    block = AttentionBlock(dim=64, num_heads=4).eval()
    x = torch.randn(2, 24, 64)
    ids = torch.tensor([[0] * 9 + [1] * 15, [0] * 16 + [1] * 8])
    with torch.no_grad():
        cache = block.allocate_inference_cache(2, max_seq_len=24)
        block(x[:, :18], cache=cache, doc_ids=ids[:, :18])
        assert cache["doc_start"].tolist() == [9, 16] and cache["masked"]
        tail = torch.stack([block.step(x[:, t], cache) for t in range(18, 24)], dim=1)
        assert torch.allclose(tail[0], block(x[:1, 9:])[0, 9:], atol=1e-5)
        assert torch.allclose(tail[1], block(x[1:, 16:])[0, 2:], atol=1e-5)


def test_reorder_cache_with_attention():
    """Beams reorder the KV cache row-wise, and cached beam search over a stack with attention
    returns the uncached search's tokens."""
    model = cached_lm(layer_pattern=["mamba3", "attention", "mlstm"])
    caches = model.allocate_inference_cache(3, max_seq_len=8)
    attn = caches[1]
    attn["k"][1].fill_(5.0)
    attn["doc_start"][1] = 2
    moved = model.reorder_cache(caches, torch.tensor([1, 1, 0]))[1]
    assert torch.all(moved["k"][0] == 5.0) and torch.all(moved["k"][2] == 0.0)
    assert moved["doc_start"].tolist() == [2, 2, 0] and moved["seen"] == attn["seen"]

    ids = torch.randint(0, 97, (1, 6))
    uncached = beam_search_uncached(model, ids, beam_size=3, max_new_tokens=8)
    cached = model.beam_search_cached(ids, beam_size=3, max_new_tokens=8)
    assert torch.equal(uncached, cached), f"uncached {uncached.tolist()} vs cached {cached.tolist()}"


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


# -- uncached references (P2-N) ----------------------------------------------------------------

EOS = 3


class Bigram(nn.Module):
    """``logits[b, t] = table[ids[b, t]]``: the next-token scores given the last token."""

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", torch.as_tensor(table, dtype=torch.float32))

    def forward(self, ids, return_dict=True):
        return SimpleNamespace(logits=self.table[ids])


def log_table(rows):
    """Rows of probabilities -> log-probabilities (so a beam's score is a sum of known logs)."""
    return torch.tensor(rows, dtype=torch.float64).log().float()


# From token 0: EOS directly (p .30) or 1 (p .50); from 1: 2 (p .90); from 2: EOS (p .95).
# Hypotheses: [0, EOS] with log .30 = -1.204 (1 token); [0, 1, 2, EOS] with log(.5*.9*.95) = -0.849
# (3 tokens). Raw sums prefer the long one; so does every length penalty >= 0.
TABLE = log_table(
    [
        [0.05, 0.50, 0.15, 0.30],
        [0.02, 0.03, 0.90, 0.05],
        [0.02, 0.02, 0.01, 0.95],
        [0.25, 0.25, 0.25, 0.25],
    ]
)


def test_uncached_beam_stops_on_eos():
    """P2-N / defect 11: finished hypotheses leave the beam with their EOS and are ranked by
    `sum_logprob / generated ** length_penalty`; nothing follows an EOS."""
    model = Bigram(TABLE)
    out = beam_search(model, torch.tensor([[0]]), beam_size=2, max_new_tokens=10, eos_token_id=EOS)
    assert out.tolist() == [[0, 1, 2, EOS]]


def test_length_penalty_is_not_inert():
    """The reference's `length_penalty` divided every live beam by the same length, so it never
    changed a ranking. Here a short and a long hypothesis swap places when it changes."""
    # From 0: EOS (p .40) or 1 (p .60); from 1: 2 (p .50); from 2: EOS (p .50).
    # short [0, EOS]: log .4 = -0.916 (1 token); long [0,1,2,EOS]: log .15 = -1.897 (3 tokens).
    table = log_table(
        [
            [0.00001, 0.59999, 0.00000001, 0.4],
            [0.25, 0.0, 0.5, 0.25],
            [0.25, 0.25, 0.0, 0.5],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    model = Bigram(table)
    kw = dict(beam_size=2, max_new_tokens=6, eos_token_id=EOS)
    short = beam_search(model, torch.tensor([[0]]), length_penalty=0.0, **kw)
    long_ = beam_search(model, torch.tensor([[0]]), length_penalty=1.0, **kw)
    assert short.tolist() == [[0, EOS]]  # -0.916 > -1.897
    assert long_.tolist() == [[0, 1, 2, EOS]]  # -1.897 / 3 = -0.632 > -0.916 / 1
    assert math.log(0.15) / 3 > math.log(0.4)


def test_uncached_greedy_stops_on_eos_and_pads_finished_rows():
    model = Bigram(TABLE)
    ids = torch.tensor([[0], [2]])  # row 1 emits EOS at once; row 0 after three tokens
    out = greedy(model, ids, max_new_tokens=8, eos_token_id=EOS, pad_token_id=0)
    assert out.tolist() == [[0, 1, 2, EOS], [2, EOS, 0, 0]]  # stops when every row has finished
    assert greedy(model, ids, max_new_tokens=8, eos_token_id=EOS)[1].tolist() == [2, EOS, EOS, EOS]
    no_eos = greedy(model, ids, max_new_tokens=5)
    assert no_eos.shape == (2, 6)


def test_repetition_penalty_breaks_a_loop():
    """CTRL penalty: a token already in the history has its positive logit divided (negative:
    multiplied), so a self-loop the unpenalised greedy decoder would repeat forever is left."""
    table = torch.tensor([[2.0, 1.0, 0.5, -1.0]] * 4)  # token 0 always wins unpenalised
    model = Bigram(table)
    plain = greedy(model, torch.tensor([[1]]), max_new_tokens=3)
    assert plain.tolist() == [[1, 0, 0, 0]]
    # p = 5: after [1, 0] token 0 scores 2/5 = 0.4 < token 2's 0.5, so 2 is chosen; after [1, 0, 2]
    # token 0 (0.4) beats 1 (0.2) and 2 (0.1) again.
    penalised = greedy(model, torch.tensor([[1]]), max_new_tokens=3, repetition_penalty=5.0)
    assert penalised.tolist() == [[1, 0, 2, 0]]
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    got = apply_repetition_penalty(logits, torch.tensor([[0, 1]]), 2.0)
    assert got.tolist() == [[1.0, -4.0, 1.0]]


def test_sample_is_reproducible_and_top_k_one_is_greedy():
    torch.manual_seed(0)
    model = Bigram(torch.randn(6, 6))
    ids = torch.tensor([[0], [1]])
    a = sample(model, ids, max_new_tokens=12, generator=torch.Generator().manual_seed(7))
    b = sample(model, ids, max_new_tokens=12, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert torch.equal(sample(model, ids, max_new_tokens=6, top_k=1), greedy(model, ids, max_new_tokens=6))


def test_filter_logits():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
    assert torch.isinf(filter_logits(logits, top_k=2)[0, [0, 3]]).all()
    kept = torch.isfinite(filter_logits(logits, top_p=0.7))[0]
    assert kept.tolist() == [False, True, True, False]


def _tiny_lm():
    torch.manual_seed(0)
    cfg = HybridConfig(
        vocab_size=61,
        dim=32,
        num_layers=2,
        layer_pattern=["mamba3", "attention"],
        mamba3_d_state=16,
        mamba3_head_dim=16,
        head_dim=16,
        max_position_embeddings=64,
        dropout=0.0,
    )
    return HybridLanguageModel(cfg).eval()


def test_real_model_beam_ends_at_its_eos():
    """On a real model: pick as EOS a token the unconstrained best beam emits, so a hypothesis must
    finish; the returned one has nothing after its EOS. With length_penalty 0 (raw sums) a longer
    live beam cannot outscore it; with 1.0 a repetitive long beam legitimately can."""
    model = _tiny_lm()
    ids = torch.randint(0, 61, (1, 5))
    free = beam_search(model, ids, beam_size=3, max_new_tokens=6)
    eos = int(free[0, ids.shape[1] + 2])
    out = beam_search(model, ids, beam_size=3, max_new_tokens=40, eos_token_id=eos, length_penalty=0.0)
    gen = out[0, ids.shape[1] :].tolist()
    assert eos in gen and gen.index(eos) == len(gen) - 1


def test_uncached_beam_search_refuses_a_batch():
    with pytest.raises(ValueError, match="one sample"):
        beam_search(Bigram(TABLE), torch.tensor([[0], [1]]))


# -- pointer constraints (P2-Q) ----------------------------------------------------------------

POINTER_VOCAB = 300
RETRIEVED = {2: [1, 3], 5: [2], 9: [4, 5, 6]}


def _specials():
    from lexhybrid.decoding.pointer_constraints import SpecialTokens

    return SpecialTokens.from_offset(200)


def test_special_token_registry_matches_decision_7():
    from lexhybrid.decoding.pointer_constraints import SpecialTokens

    names = SpecialTokens.names()
    assert len(names) == 87 and len(set(names)) == 87
    assert names[:3] == ["<|q|>", "<|cite|>", "<|c1|>"] and "<|c16|>" in names and "<|s64|>" in names
    assert names[-5:] == ["<|unanswerable|>", "<|passage|>", "<|/passage|>", "<|question|>", "<|answer|>"]
    sp = _specials()
    assert sp.all_ids() == list(range(200, 287))
    assert sp.chunk_number(sp.chunks[4]) == 5 and sp.sentence_number(sp.sentences[63]) == 64
    assert sp.chunk_number(sp.q) is None


def test_pointer_fsm_transitions():
    from lexhybrid.decoding.pointer_constraints import FREE, PointerFSM

    sp = _specials()
    fsm = PointerFSM(sp, RETRIEVED, POINTER_VOCAB, eos_token_id=0)
    free = fsm.allowed((FREE, None))
    assert free[7] and free[sp.q] and free[sp.cite] and free[sp.unanswerable] and free[0]
    assert not free[sp.chunks[1]], "a chunk pointer needs a marker first"
    assert not any(free[t] for t in sp.prompt_only()), "prompt structure is never generated"
    after_q = fsm.replay([sp.q])
    assert sorted(t for t in range(POINTER_VOCAB) if fsm.allowed(after_q)[t]) == [
        sp.chunks[k - 1] for k in (2, 5, 9)
    ]
    after_c9 = fsm.replay([sp.q, sp.chunks[8]])
    assert sorted(t for t in range(POINTER_VOCAB) if fsm.allowed(after_c9)[t]) == [
        sp.sentences[j - 1] for j in (4, 5, 6)
    ]
    span = fsm.replay([sp.cite, sp.chunks[8], sp.sentences[3], sp.sentences[4], 11])
    assert span == (FREE, None)
    with pytest.raises(ValueError):
        fsm.replay([sp.q, sp.chunks[0]])  # chunk 1 was not retrieved
    with pytest.raises(ValueError):
        fsm.replay([sp.q, sp.chunks[1], sp.sentences[1]])  # s2 is not in chunk 2
    nothing = PointerFSM(sp, {}, POINTER_VOCAB)
    assert not nothing.allowed((FREE, None))[sp.q], "with nothing retrieved there is nothing to point at"


def _trim_open_pointer(tokens, sp):
    """Drop a pointer left open by the step budget (the decoder's max_new_tokens, not the FSM)."""
    tokens = list(tokens)
    while tokens and (tokens[-1] in (sp.q, sp.cite) or sp.chunk_number(tokens[-1]) is not None):
        tokens.pop()
    return tokens


def test_pointer_fsm_never_emits_unretrieved_pointer():
    """P2-Q: 1,000 steps of random logits (pointer tokens boosted so pointers are frequent) through
    the machine; an independent validator finds no invalid pointer, and pointers were exercised."""
    from lexhybrid.decoding.pointer_constraints import PointerFSM, validate_pointers

    sp = _specials()
    fsm = PointerFSM(sp, RETRIEVED, POINTER_VOCAB, eos_token_id=0)
    g = torch.Generator().manual_seed(0)
    generated = torch.zeros(1, 0, dtype=torch.long)
    boost = torch.zeros(POINTER_VOCAB)
    boost[sp.all_ids()] = 4.0
    for _ in range(1000):
        logits = torch.randn(1, POINTER_VOCAB, generator=g) * 3.0 + boost
        masked = fsm.mask_logits(logits, generated)
        token = torch.multinomial(torch.softmax(masked, -1), 1, generator=g)
        assert torch.isfinite(masked[0, token]).all()
        generated = torch.cat([generated, token], dim=1)
    tokens = _trim_open_pointer(generated[0].tolist(), sp)
    assert validate_pointers(tokens, sp, RETRIEVED) == []
    n_pointers = sum(sp.chunk_number(t) is not None for t in tokens)
    assert n_pointers >= 50, f"only {n_pointers} pointers exercised"
    assert {sp.chunk_number(t) for t in tokens if sp.chunk_number(t)} == set(RETRIEVED)


def test_pointer_constraint_inside_the_decoders():
    """The FSM as a logits processor: greedy (cached = uncached) and beam search only produce valid
    pointers on a real model whose pointer logits are boosted."""
    from lexhybrid.decoding import beam_search, greedy, greedy_cached
    from lexhybrid.decoding.pointer_constraints import PointerFSM, validate_pointers

    sp = _specials()
    model = cached_lm(vocab=POINTER_VOCAB, layer_pattern=["mamba3", "attention", "mlstm"], num_layers=3)
    with torch.no_grad():
        model.lm_head.weight[sp.all_ids()] += 0.5 * model.lm_head.weight.abs().max()
    ids = torch.randint(0, 200, (2, 6))
    fsm = PointerFSM(sp, RETRIEVED, POINTER_VOCAB)
    kw = dict(max_new_tokens=24, logits_processor=fsm.processor(ids.shape[1]))
    uncached, cached = greedy(model, ids, **kw), greedy_cached(model, ids, **kw)
    assert torch.equal(uncached, cached)
    beams = beam_search(model, ids[:1], beam_size=3, **kw)
    for row in [*uncached, *beams]:
        gen = _trim_open_pointer(row[ids.shape[1] :].tolist(), sp)
        assert validate_pointers(gen, sp, RETRIEVED) == []


# -- best-of-n (P2-R) --------------------------------------------------------------------------


def test_best_of_n_picks_scorer_max():
    """P2-R: n cached samples in one batch; the scorer sees each answer (prompt removed, cut after
    EOS) and the highest-scoring sample is returned; ties go to the earliest sample."""
    from lexhybrid.decoding import best_of_n, sample_cached

    model = family_lm("base_pattern")
    ids = torch.randint(0, 97, (1, 6))
    kw = dict(max_new_tokens=8, temperature=1.2)
    target = 5

    def count_target(tokens):
        return sum(t == target for t in tokens)

    best, scores = best_of_n(
        model, ids, 6, count_target, generator=torch.Generator().manual_seed(4), return_scores=True, **kw
    )
    samples = sample_cached(
        model, ids.expand(6, -1).contiguous(), generator=torch.Generator().manual_seed(4), **kw
    )
    want = max(range(6), key=lambda i: (scores[i], -i))
    assert scores == [float(count_target(r[6:].tolist())) for r in samples]
    assert torch.equal(best[0], samples[want])
    constant = best_of_n(model, ids, 4, lambda tokens: 1.0, generator=torch.Generator().manual_seed(4), **kw)
    first = sample_cached(
        model, ids.expand(4, -1).contiguous(), generator=torch.Generator().manual_seed(4), **kw
    )
    assert torch.equal(constant[0], first[0])


def test_generated_tokens_cuts_after_eos():
    from lexhybrid.decoding import generated_tokens

    row = torch.tensor([9, 9, 1, 2, 3, 7, 7])
    assert generated_tokens(row, 2) == [1, 2, 3, 7, 7]
    assert generated_tokens(row, 2, eos_token_id=3) == [1, 2, 3]
