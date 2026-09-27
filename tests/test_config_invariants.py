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
        .replace("model: ref_hybrid_m3", "model: hybrid_legal_bad")
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
