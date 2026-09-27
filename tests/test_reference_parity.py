"""The port computes exactly what the reference computed (plan P1-Y). Never skipped.

Each fixture in ``tests/fixtures/reference_parity/`` was produced by the REFERENCE code
(``scripts/dump_reference_parity.py``): a state dict, a packed input with document ids, and the
outputs. Here the state dict is loaded strictly into the NEW block or model and every output must
agree to 1e-6 in fp32 (whole-model logits to 5e-6, see ``MODEL_TOL``); cached beam search must
produce the identical tokens. P2 fixes defects on
paths these fixtures pin; a fix that moves one of them updates the fixture in the same box, with a
note saying why.
"""

import dataclasses
import inspect

import pytest
import torch

from lexhybrid import HybridConfig, HybridLanguageModel
from lexhybrid.layers.attention_block import AttentionBlock
from lexhybrid.layers.mamba3_block import Mamba3Block
from lexhybrid.layers.mamba_block import MambaBlock
from lexhybrid.layers.mlstm_block import mLSTMBlock
from tests.conftest import REPO_ROOT

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "reference_parity"
TOL = 1e-6
# Whole-model outputs (4 blocks, MLPs, LM head) carry cross-platform fp32 BLAS differences: the
# fixtures were generated on macOS arm64, and on Linux x86_64 the logits differ by up to 1.4e-6
# (CI run 36322991795) while every block still agrees to 1e-6. A port error moves logits by 1e-3+.
MODEL_TOL = 5e-6
BLOCKS = {"mamba3": Mamba3Block, "mlstm": mLSTMBlock, "attention": AttentionBlock, "mamba": MambaBlock}

# ---- what the fixtures were computed with --------------------------------------------------------
# P2 changes defaults on purpose. The fixtures hold the reference's computation, so the parity test
# pins the reference's value wherever a default moved (PINS), and says why a new option cannot move
# a fixture (NEUTRAL). `test_every_changed_default_is_pinned_or_neutral` fails when a default changes
# or an option appears without an entry here: that is the moment to decide, not a surprise later.

# Defaults as the P1 port (== the reference) had them.
P1_CONFIG_DEFAULTS = {
    "vocab_size": 50257, "dim": 768, "num_layers": 12, "layer_pattern": ["mamba", "mamba", "mlstm"],
    "state_size": 16, "conv_size": 4, "expand_factor": 2, "dt_rank": None, "use_fast_path": True,
    "scan_impl": "legacy", "dt_init_strategy": "none", "dt_min": 0.001, "dt_max": 0.1,
    "tfla_impl": "legacy", "mamba3_d_state": 128, "mamba3_head_dim": 64, "mamba3_ngroups": 1,
    "mamba3_chunk_size": 64, "mamba3_use_conv": True, "mamba3_conv_size": 4,
    "mamba3_use_trapezoid": False, "mamba3_use_rope": False, "mamba3_rope_fraction": 0.5,
    "mamba3_theta_max": 1.0, "mamba3_dt_min": 0.001, "mamba3_dt_max": 0.1,
    "mamba3_dt_init_floor": 0.0001, "mamba3_bc_bias": "none", "mamba3_mimo_rank": 1,
    "mamba3_a_mode": "static", "mamba3_a_floor": 0.0001, "mamba3_dt_limit": 1.0,
    "mamba3_use_outproj_norm": False, "head_dim": 64, "num_heads": None, "proj_factor": 2,
    "mlstm_gate_soft_cap": 15.0, "mlstm_input_gate_bias_init": -10.0,
    "mlstm_forget_gate_bias_init": 0.0, "attn_dropout": 0.0, "rope_theta": 10000.0,
    "norm_type": "rms", "norm_topology": "pre_rms", "use_mlp": True, "mlp_ratio": 4.0,
    "max_position_embeddings": 2048, "dropout": 0.1, "initializer_range": 0.02,
    "tie_word_embeddings": False, "use_gradient_checkpointing": False, "model_type": "lexhybrid",
}  # fmt: skip
P1_BLOCK_DEFAULTS = {
    "mamba": {
        "state_size": 16, "conv_size": 4, "expand_factor": 2, "dt_rank": None, "use_fast_path": True,
        "use_hybrid_norm": False, "scan_impl": "legacy", "dt_init_strategy": "none",
        "dt_min": 0.001, "dt_max": 0.1, "use_dt_norm": None,
    },
    "mamba3": {
        "d_state": 128, "head_dim": 64, "expand_factor": 2, "ngroups": 1, "chunk_size": 64,
        "use_conv": True, "conv_size": 4, "use_trapezoid": False, "use_rope": False,
        "rope_fraction": 0.5, "bc_bias": "none", "mimo_rank": 1, "a_mode": "static",
        "a_floor": 0.0001, "dt_min": 0.001, "dt_max": 0.1, "dt_init_floor": 0.0001,
        "dt_limit": 1.0, "theta_max": 1.0, "use_outproj_norm": False, "use_hybrid_norm": False,
    },
    "mlstm": {
        "head_dim": 64, "num_heads": None, "tfla_impl": "legacy", "proj_factor": 2,
        "gate_soft_cap": 15.0, "input_gate_bias_init": -10.0, "forget_gate_bias_init": 0.0,
        "use_hybrid_norm": False,
    },
    "attention": {
        "num_heads": None, "head_dim": 64, "attn_dropout": 0.0, "rope_theta": 10000.0,
        "max_position_embeddings": 1024, "use_hybrid_norm": False,
    },
}  # fmt: skip

# Reference values passed explicitly wherever P2 changed a default.
CONFIG_PINS = {
    # The reference chose the TFLA chunk from the length: 32 for L <= 128 (the fixtures' L is 96).
    # Exact TFLA is chunk-invariant up to rounding; legacy is not (its clamp restarts per chunk).
    "mlstm_chunk_size": 32,
    # Overwritten by every fixture's state dict; pinned so the default change (P2-C) is visible.
    "mlstm_forget_gate_bias_init": 0.0,
    # RoPE's base frequency changes every attention output; the reference used 10,000 (P2-H).
    "rope_theta": 10000.0,
    # Tying changes the state dict (no separate lm_head.weight); the reference was untied (P2-M).
    "tie_word_embeddings": False,
}
BLOCK_PINS = {
    "mamba": {},
    "mamba3": {},
    "mlstm": {"chunk_size": 32, "forget_gate_bias_init": 0.0},
    "attention": {"rope_theta": 10000.0},
}
# New options that cannot change what a fixture computes.
NEUTRAL = {
    # Only decides whether exact TFLA may take its slow exact path; it raises or computes the same
    # function, and no fixture needs the path (half-range at chunk 32 is 11 < 40).
    "tfla_fallback": "raises or computes the same function; never needed by a fixture",
    # Equals the shared expand_factor (2) in every fixture, which is what the reference read.
    "mamba3_expand_factor": "equals the reference's shared expand_factor 2 in every fixture",
    # "auto" is the dense-mask SDPA path on CPU, where the fixtures run: the reference's kernel.
    "attn_impl": "auto means the reference's dense-mask SDPA on CPU",
    # Multi-token prediction is off at its default (mtp_n = 1): no module is built, no RNG drawn.
    "mtp_n": "1 = off: nothing is built",
    "mtp_loss_weight": "unused while mtp_n = 1",
    "mtp_layer_type": "unused while mtp_n = 1",
}


def _config_defaults() -> dict:
    return {
        f.name: f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        for f in dataclasses.fields(HybridConfig)
    }


def _block_defaults(cls) -> dict:
    return {
        name: p.default
        for name, p in inspect.signature(cls.__init__).parameters.items()
        if name not in ("self", "dim") and p.kind is not inspect.Parameter.VAR_KEYWORD
    }


def _unpinned(current: dict, p1: dict, pins: dict) -> list[str]:
    problems = []
    for name, value in current.items():
        if name in pins:
            if name in p1 and pins[name] != p1[name]:
                problems.append(
                    f"{name}: pinned to {pins[name]!r} but the reference default was {p1[name]!r}"
                )
        elif name not in p1:
            if name not in NEUTRAL:
                problems.append(f"{name}: new option, neither pinned nor declared NEUTRAL")
        elif value != p1[name]:
            problems.append(f"{name}: default moved {p1[name]!r} -> {value!r} and is not pinned")
    return problems


def test_every_changed_default_is_pinned_or_neutral():
    problems = _unpinned(_config_defaults(), P1_CONFIG_DEFAULTS, CONFIG_PINS)
    for name, cls in BLOCKS.items():
        problems += [
            f"{name}.{p}" for p in _unpinned(_block_defaults(cls), P1_BLOCK_DEFAULTS[name], BLOCK_PINS[name])
        ]
    assert not problems, "decide how each fixture sees these:\n  " + "\n  ".join(problems)


def load(name):
    return torch.load(FIXTURES / f"{name}.pt", weights_only=True)


def block_params():
    for name in BLOCKS:
        for i, case in enumerate(load(name)):
            yield pytest.param(name, i, id=f"{name}-{case['label']}")


def model_params():
    for i, case in enumerate(load("model")):
        yield pytest.param(i, id=case["label"])


def test_all_five_fixtures_exist():
    assert sorted(p.name for p in FIXTURES.glob("*.pt")) == [
        "attention.pt",
        "mamba.pt",
        "mamba3.pt",
        "mlstm.pt",
        "model.pt",
    ]


def _per_document(block, x, doc_ids):
    """Run every (row, document) span on its own, as the reference's mLSTM did."""
    rows = []
    for b in range(x.shape[0]):
        ids = doc_ids[b]
        starts = [0] + [t for t in range(1, ids.numel()) if ids[t] != ids[t - 1]] + [ids.numel()]
        rows.append(torch.cat([block(x[b : b + 1, s:e]) for s, e in zip(starts, starts[1:])], dim=1))
    return torch.cat(rows, dim=0)


@pytest.mark.parametrize("name,index", list(block_params()))
def test_block_matches_the_reference(name, index):
    case = load(name)[index]
    kwargs = {**BLOCK_PINS[name], **case["kwargs"]}
    block = BLOCKS[name](case["input"].shape[-1], **kwargs).eval()
    block.load_state_dict(case["state_dict"], strict=True)
    with torch.no_grad():
        if name == "mlstm" and kwargs.get("tfla_impl", "legacy") == "legacy":
            # P2-D moved document resets into the TFLA kernel. The reference ran each document on
            # its own; legacy TFLA's clamp restarts per chunk, so its packed output depends on
            # where a document sits in the chunk grid (3.6e-4 on this fixture's second document).
            # The reference's computation is the per-document one, and that is what is pinned.
            got_doc = _per_document(block, case["input"], case["doc_ids"])
        else:
            got_doc = block(case["input"], doc_ids=case["doc_ids"])
        got_nodoc = block(case["input"])
    assert (got_doc - case["output_doc"]).abs().max().item() <= TOL
    assert (got_nodoc - case["output_nodoc"]).abs().max().item() <= TOL
    if case["step_output"] is not None:
        cache = block.allocate_inference_cache(case["input"].shape[0])
        with torch.no_grad():
            stepped = torch.stack(
                [block.step(case["input"][:, t], cache) for t in range(case["input"].shape[1])], dim=1
            )
        assert (stepped - case["step_output"]).abs().max().item() <= TOL


@pytest.mark.parametrize("index", list(model_params()))
def test_model_matches_the_reference(index):
    case = load("model")[index]
    model = HybridLanguageModel(HybridConfig(**{**CONFIG_PINS, **case["kwargs"]})).eval()
    model.load_state_dict(case["state_dict"], strict=True)
    with torch.no_grad():
        with_doc = model(case["input_ids"], labels=case["input_ids"], doc_ids=case["doc_ids"])
        without_doc = model(case["input_ids"], labels=case["input_ids"])
    assert (with_doc.logits - case["logits_doc"]).abs().max().item() <= MODEL_TOL
    assert (without_doc.logits - case["logits_nodoc"]).abs().max().item() <= MODEL_TOL
    assert abs(without_doc.loss.item() - case["loss_nodoc"].item()) <= MODEL_TOL
    # P2-L fixed defect 6: the packed loss no longer supervises the cross-document prediction. The
    # expected value is the boundary-masked cross-entropy of the REFERENCE's own logits; the
    # reference's unmasked loss must now differ from ours.
    ids, doc = case["input_ids"], case["doc_ids"]
    targets = ids[:, 1:].masked_fill(doc[:, 1:] != doc[:, :-1], -100)
    want = torch.nn.functional.cross_entropy(
        case["logits_doc"][:, :-1].reshape(-1, case["logits_doc"].shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )
    assert abs(with_doc.loss.item() - want.item()) <= MODEL_TOL
    assert abs(with_doc.loss.item() - case["loss_doc"].item()) > MODEL_TOL, (
        "the boundary mask changed nothing"
    )
    assert with_doc.n_supervised_tokens.item() == targets.ne(-100).sum().item()
    if case["beam_tokens"] is not None:
        tokens = model.beam_search_cached(case["beam_prompt"], beam_size=3, max_new_tokens=10)
        assert torch.equal(tokens, case["beam_tokens"])
