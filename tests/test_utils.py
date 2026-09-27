"""Checkpoint utilities and run metadata (plan P1-Q)."""

import json

import pytest
import torch
from omegaconf import OmegaConf

from lexhybrid import HybridConfig, HybridLanguageModel
from lexhybrid.utils import run_metadata
from lexhybrid.utils.checkpoint import (
    detect_prefix,
    infer_architecture,
    infer_layer_types,
    load_state_dict_guarded,
    strip_prefixes,
)


def tiny(pattern, **kw):
    kw = dict(
        vocab_size=64,
        dim=64,
        num_layers=len(pattern),
        layer_pattern=pattern,
        state_size=8,
        mamba3_d_state=16,
        mamba3_head_dim=32,
        head_dim=32,
        num_heads=2,
        max_position_embeddings=32,
        **kw,
    )
    return HybridLanguageModel(HybridConfig(**kw))


@pytest.mark.parametrize("wrapper", ["", "model.", "_orig_mod.", "model._orig_mod.", "_orig_mod.model."])
@pytest.mark.parametrize("inner", ["", "lm."])
def test_strip_prefixes(wrapper, inner):
    """Wrapper prefixes are removed, then the retrieval encoder's inner `lm.`; its projection head
    (a sibling of `lm.`) and logit scale are dropped."""
    state = {
        wrapper + inner + "layers.0.mixer.dt_bias": 1,
        wrapper + "projection_head.w": 2,
        "logit_scale": 3,
    }
    assert strip_prefixes(state) == {"layers.0.mixer.dt_bias": 1}


def test_guarded_load_refuses_mismatch_and_accepts_match():
    a = tiny(["mamba3", "mlstm"])
    # Most keys absent -> more than half the model is missing -> hard failure.
    with pytest.raises(RuntimeError, match="missing"):
        load_state_dict_guarded(
            tiny(["mamba3", "mlstm"]),
            {"embeddings.token_embedding.weight": a.embeddings.token_embedding.weight},
        )
    # A shape mismatch on a shared key raises inside torch, before the fraction check.
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_state_dict_guarded(tiny(["attention", "attention"]), a.state_dict())
    # A few missing keys (under 50%, over 5%) warn but load.
    partial = {k: v for i, (k, v) in enumerate(a.state_dict().items()) if i % 5}
    with pytest.warns(RuntimeWarning, match="missing"):
        load_state_dict_guarded(tiny(["mamba3", "mlstm"]), partial)
    missing, unexpected = load_state_dict_guarded(tiny(["mamba3", "mlstm"]), a.state_dict())
    assert missing == [] and unexpected == []


def test_infer_layer_types_and_architecture():
    model = tiny(["mamba", "mamba3", "mlstm", "attention"], norm_topology="hybrid")
    state = {"model." + k: v for k, v in model.state_dict().items()}
    assert detect_prefix(state) == "model.layers."
    assert infer_layer_types(state) == ["mamba", "mamba3", "mlstm", "attention"]
    arch = infer_architecture(state)
    assert arch.norm_topology == "hybrid" and arch.mamba3_d_state == 16 and arch.mamba3_head_dim == 32
    assert arch.state_size == 8 and arch.config_kwargs()["num_layers"] == 4


def test_infer_layer_types_refuses_a_layer_without_fingerprint():
    with pytest.raises(ValueError):
        infer_layer_types({"layers.0.mixer.unknown_param": torch.zeros(1)})
    with pytest.raises(ValueError):
        detect_prefix({"embeddings.token_embedding.weight": torch.zeros(1)})


def test_run_metadata_records_provenance_without_git(tmp_path, monkeypatch):
    cfg = OmegaConf.create({"seed": 42, "model": {"dim": 64}})
    path = run_metadata.write_run_metadata(cfg, str(tmp_path), extra={"entrypoint": "test"})
    meta = json.loads(path.read_text())
    assert meta["resolved_config"]["seed"] == 42 and meta["entrypoint"] == "test"
    assert len(meta["tree_hash"]) == 64 and "git_sha" not in meta


def test_read_git_head_reads_files_only(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("abc123\n")
    assert run_metadata.read_git_head(tmp_path) == "abc123"
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text("# pack\ndef456 refs/heads/main\n")
    assert run_metadata.read_git_head(tmp_path) == "def456"
    assert run_metadata.read_git_head(tmp_path / "nowhere") == ""


def test_tree_hash_changes_with_content(tmp_path):
    (tmp_path / "lexhybrid").mkdir()
    f = tmp_path / "lexhybrid" / "a.py"
    f.write_text("x = 1\n")
    h1 = run_metadata.tree_hash(tmp_path)
    f.write_text("x = 2\n")
    assert run_metadata.tree_hash(tmp_path) != h1
