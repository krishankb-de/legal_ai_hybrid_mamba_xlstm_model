"""HybridConfig unit tests (plan P1-C)."""

import dataclasses

import pytest

from lexhybrid.config import HybridConfig, load_model_config, resolve_model_yaml


def test_default_config():
    c = HybridConfig()
    assert (c.dim, c.num_layers, c.num_heads, c.dt_rank) == (768, 12, 12, 48)
    assert c.model_type == "lexhybrid"


def test_auto_compute_heads():
    assert HybridConfig(dim=1024, head_dim=64, num_heads=None).num_heads == 16


def test_to_dict_from_dict_roundtrip():
    c1 = HybridConfig(dim=1024, num_layers=24, layer_pattern=["mamba3", "attention"])
    c2 = HybridConfig.from_dict(c1.to_dict())
    assert c2 == c1


@pytest.mark.parametrize(
    "field,value",
    [
        ("layer_pattern", ["mamba", "slstm"]),
        ("norm_topology", "post"),
        ("scan_impl", "fast"),
        ("tfla_impl", "fast"),
        ("mamba3_bc_bias", "two"),
        ("mamba3_a_mode", "dynamic"),
        ("dt_init_strategy", "xavier"),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValueError):
        HybridConfig(**{field: value})


def test_from_hydra_filters_unknown_keys_and_keeps_every_field():
    raw = {
        "dim": 64,
        "learning_rate": 4e-4,
        "warmup_steps": 2000,
        "model_type": "hybrid_lm",
        "layer_pattern": ("mamba3",),
    }
    c = HybridConfig.from_hydra(raw, num_layers=3, not_a_field=1)
    assert (c.dim, c.num_layers, c.layer_pattern, c.model_type) == (64, 3, ["mamba3"], "lexhybrid")


def test_every_mamba3_block_parameter_is_reachable_from_the_config():
    """Every lever Mamba3Block accepts exists as a mamba3_* field (theta_max was once missing)."""
    import inspect

    from lexhybrid.layers.mamba3_block import Mamba3Block

    params = inspect.signature(Mamba3Block.__init__).parameters
    levers = {
        n
        for n, p in params.items()
        if n not in ("self", "dim") and p.kind is not inspect.Parameter.VAR_KEYWORD
    }
    fields = {f.name for f in dataclasses.fields(HybridConfig)}
    shared = {"use_hybrid_norm"}  # injected by HybridBlock; expand_factor has its own field since P2-A
    assert sorted(lv for lv in levers - shared if f"mamba3_{lv}" not in fields) == []


def test_every_mlstm_block_parameter_is_reachable_from_the_config():
    """Defect 2's twin: every mLSTMBlock lever is a shared field or an `mlstm_*` field."""
    import inspect

    from lexhybrid.layers.hybrid_block import _MLSTM_SHARED
    from lexhybrid.layers.mlstm_block import mLSTMBlock

    params = inspect.signature(mLSTMBlock.__init__).parameters
    levers = {n for n in params if n not in ("self", "dim")}
    fields = {f.name for f in dataclasses.fields(HybridConfig)}
    unreachable = sorted(
        lv
        for lv in levers
        if not (
            (lv in _MLSTM_SHARED and (lv in fields or lv == "use_hybrid_norm")) or f"mlstm_{lv}" in fields
        )
    )
    assert unreachable == []


def test_get_layer_config_covers_every_mixer():
    c = HybridConfig(layer_pattern=["mamba", "mamba3", "mlstm", "attention"], num_layers=4)
    assert c.get_layer_config(0)["scan_impl"] == "legacy"
    assert c.get_layer_config(1)["d_state"] == 128
    assert c.get_layer_config(2)["tfla_impl"] == "legacy"
    assert c.get_layer_config(3)["rope_theta"] == 500000.0  # P2-H


def test_save_pretrained_writes_config_json(tmp_path):
    path = HybridConfig(dim=64).save_pretrained(str(tmp_path))
    assert path.endswith("config.json") and (tmp_path / "config.json").exists()


def test_load_model_config_by_name_and_by_path():
    by_name = load_model_config("ref_hybrid_m3")
    by_path = load_model_config(str(resolve_model_yaml("ref_hybrid_m3")))
    assert by_name == by_path and by_name.norm_topology == "hybrid"
    with pytest.raises(FileNotFoundError):
        load_model_config("no_such_model")
