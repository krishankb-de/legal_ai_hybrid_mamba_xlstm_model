"""Configuration invariants (plan P1-V): the living spec of what every yaml guarantees."""

import pytest
import torch
import yaml

from lexhybrid import HybridLanguageModel
from lexhybrid.config import load_model_config
from tests.conftest import REPO_ROOT, load_script

CHECK = load_script("check_configs")
MODEL_NAMES = CHECK.model_names()


def test_there_are_model_configs():
    assert "ref_hybrid_m3" in MODEL_NAMES and "ref_transformer" in MODEL_NAMES


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_yaml_composes_and_pins_its_operators(name):
    assert CHECK.check_model(name) == []


def test_the_gate_catches_an_unpinned_and_a_legacy_yaml(tmp_path, monkeypatch):
    model_dir = tmp_path / "configs" / "model"
    model_dir.mkdir(parents=True)
    for sub in ("dataset", "trainer", "callbacks"):
        (tmp_path / "configs" / sub).mkdir()
    for f in ("config.yaml", "dataset/synthetic.yaml", "trainer/cpu_debug.yaml", "callbacks/default.yaml"):
        (tmp_path / "configs" / f).write_text((REPO_ROOT / "configs" / f).read_text())
    (model_dir / "hybrid_legal_bad.yaml").write_text(
        yaml.safe_dump(
            {
                "vocab_size": 151936,
                "dim": 64,
                "num_layers": 2,
                "layer_pattern": ["mlstm", "mamba"],
                "scan_impl": "legacy",
                "head_dim": 32,
            }
        )
    )
    (tmp_path / "configs" / "config.yaml").write_text(
        (tmp_path / "configs" / "config.yaml")
        .read_text()
        .replace("model: hybrid_legal_base", "model: hybrid_legal_bad")
    )
    monkeypatch.setattr(CHECK, "CONFIG_DIR", tmp_path / "configs")
    problems = CHECK.check_model("hybrid_legal_bad")
    assert any("does not pin tfla_impl" in p for p in problems)
    assert any("scan_impl legacy outside" in p for p in problems)


@pytest.mark.slow
@pytest.mark.parametrize("name,expected", [("ref_hybrid_m3", 184_192_200), ("ref_transformer", 183_386_880)])
def test_reference_replicas_reproduce_the_reference_parameter_counts_exactly(name, expected):
    """Structural parity with the reference: same modules, same shapes, to the parameter."""
    torch.manual_seed(0)
    model = HybridLanguageModel(load_model_config(name))
    assert sum(p.numel() for p in model.parameters()) == expected


@pytest.mark.slow
def test_dropping_the_conv_moves_parameters_by_exactly_the_conv():
    """1536 + 2*128 channels x (4 weights + 1 bias) = 8,960 per layer, nine mamba3 layers."""
    with_conv = HybridLanguageModel(load_model_config("ref_hybrid_m3"))
    without = HybridLanguageModel(load_model_config("ref_hybrid_m3", mamba3_use_conv=False))

    def n(m):
        return sum(p.numel() for p in m.parameters())

    assert n(with_conv) - n(without) == 9 * 8_960
    assert not any("conv1d" in k for k in without.state_dict())


def test_trainer_and_dataset_configs_compose():
    cfg = CHECK.compose_model("ref_hybrid_m3")
    assert cfg.trainer.max_steps == 2 and cfg.dataset.name == "synthetic"
    assert cfg.callbacks.checkpoint.save_top_k == 0 and cfg.callbacks.checkpoint.save_last is True
    for trainer in ("h100_single_gpu", "h100_multi_ddp"):
        t = yaml.safe_load((REPO_ROOT / "configs" / "trainer" / f"{trainer}.yaml").read_text())
        assert t["precision"] == "bf16-mixed" and t["compile_model"] is False and "val_every_opt_steps" in t


def test_smoke_model_gate_passes():
    assert load_script("smoke_model").main() == 0


LEGAL = ("hybrid_legal_base", "transformer_legal_base", "hybrid_legal_legacy")


@pytest.mark.parametrize("name", LEGAL)
def test_legal_yamls_compose_and_pin_operators(name):
    """P2-V: the three legal model configs compose, pin their operators (R8, decision 17), use the
    Qwen3 vocabulary and a tied head, and declare every HybridConfig field so an arm can override
    any of them as `model.<field>=...` without a `+`."""
    import dataclasses

    from lexhybrid import HybridConfig

    assert CHECK.check_model(name) == []
    raw = yaml.safe_load((REPO_ROOT / "configs" / "model" / f"{name}.yaml").read_text())
    missing = sorted({f.name for f in dataclasses.fields(HybridConfig)} - set(raw))
    assert not missing, f"{name} does not declare {missing}"
    cfg = load_model_config(name)
    assert (cfg.vocab_size, cfg.tie_word_embeddings, cfg.dim) == (151936, True, 768)
    base = yaml.safe_load((REPO_ROOT / "configs" / "model" / "hybrid_legal_base.yaml").read_text())
    for key in ("learning_rate", "warmup_steps", "weight_decay", "gradient_clip_val", "scheduler"):
        assert raw[key] == base[key], f"{name}: {key} differs from the base"
    if name == "hybrid_legal_base":  # decision 10, value by value
        assert (
            cfg.layer_pattern
            == ["mamba3"] * 3
            + ["attention"]
            + ["mlstm"] * 3
            + ["mamba3"] * 2
            + ["attention"]
            + ["mamba3"] * 2
        )
        assert (cfg.num_layers, cfg.num_heads, cfg.norm_topology) == (12, 12, "hybrid")
        assert (cfg.tfla_impl, cfg.tfla_fallback, cfg.mamba3_d_state) == ("exact", "error", 128)
        assert (cfg.mamba3_chunk_size, cfg.mlstm_chunk_size, cfg.mlstm_forget_gate_bias_init) == (
            128,
            128,
            3.0,
        )
        assert (cfg.rope_theta, cfg.dropout, cfg.max_position_embeddings) == (500000.0, 0.0, 8192)
    elif name == "transformer_legal_base":
        assert (cfg.num_layers, cfg.layer_pattern, cfg.norm_topology) == (15, ["attention"], "pre_rms")
    else:
        assert cfg.layer_pattern.count("mamba") == 9 and cfg.layer_pattern[4:7] == ["mlstm"] * 3
        assert (cfg.scan_impl, cfg.tfla_impl) == ("legacy", "legacy")


def test_an_arm_can_override_any_base_field_without_a_plus():
    cfg = CHECK.compose_model("hybrid_legal_base")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        arm = compose(
            config_name="config",
            overrides=["model.mamba3_use_rope=true", "model.mtp_n=2", "model.mamba3_d_state=64"],
        )
    assert (arm.model.mamba3_use_rope, arm.model.mtp_n, arm.model.mamba3_d_state) == (True, 2, 64)
    assert cfg.model.mamba3_use_rope is False


# -- P2-W: screen variants and parameter bands ------------------------------------------------

PARAMS = load_script("param_counts")
SCREEN_VARIANTS = {
    "hybrid_legal_attn0": {"layer_pattern"},
    "hybrid_legal_attn4": {"layer_pattern"},
    "hybrid_legal_ds64": {"mamba3_d_state"},
    "hybrid_legal_mtp": {"mtp_n"},
}


@pytest.fixture(scope="module")
def counts():
    return {r["name"]: r for r in PARAMS.all_counts()}


@pytest.mark.parametrize("name,lever", list(SCREEN_VARIANTS.items()))
def test_screen_variants_differ_from_the_base_in_one_lever(name, lever):
    """OFAT (R3): each screen yaml equals the base in every field but its own lever."""
    load = lambda n: yaml.safe_load((REPO_ROOT / "configs" / "model" / f"{n}.yaml").read_text())  # noqa: E731
    base, arm = load("hybrid_legal_base"), load(name)
    assert set(arm) == set(base)
    assert {k for k in base if base[k] != arm[k]} == lever
    assert CHECK.check_model(name) == []


def test_param_bands(counts):
    """P2-W: the plan's predicted counts and bands, and analysis/param_counts.md is current."""
    base = counts["hybrid_legal_base"]
    assert round(base["non_embedding"] / 1e6, 1) == 104.1 and round(base["total"] / 1e6, 1) == 220.8
    predicted = {
        "transformer_legal_base": 2.0,
        "hybrid_legal_attn0": 2.8,
        "hybrid_legal_attn4": -2.8,
        "hybrid_legal_ds64": -0.75,
    }
    for name, pct in predicted.items():
        assert abs(counts[name]["delta_pct"] - pct) < 0.05, (
            f"{name}: {counts[name]['delta_pct']:+.2f}% vs predicted {pct:+}%"
        )
    assert abs(counts["transformer_legal_base"]["delta_pct"]) <= 2.5
    for name in SCREEN_VARIANTS:
        assert abs(counts[name]["delta_pct"]) <= 3.0, f"{name}: {counts[name]['delta_pct']:+.2f}%"
    assert (
        counts["hybrid_legal_mtp"]["mtp_head"] > 0
        and counts["hybrid_legal_mtp"]["non_embedding"] == base["non_embedding"]
    )
    assert (REPO_ROOT / "analysis" / "param_counts.md").read_text() == PARAMS.render(PARAMS.all_counts()), (
        "analysis/param_counts.md is stale: run scripts/param_counts.py --write"
    )


@pytest.mark.xfail(
    strict=True,
    reason="plan P2-W bound 'legacy vs base <= 1%' does not hold for the 9 x mamba + 3 x mlstm shape of "
    "P2-V: measured +2.36% (it holds against an all-recurrent 9 x mamba3 + 3 x mlstm hybrid). Recorded "
    "negative awaiting a user decision; strict, so it fails loudly if the configs change.",
)
def test_legacy_is_within_one_percent_of_the_base(counts):
    assert abs(counts["hybrid_legal_legacy"]["delta_pct"]) <= 1.0
