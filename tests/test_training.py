"""Training utilities: WSD, beta2 anneal, optimizer groups, Lightning module, SLURM callback (P1-O/P)."""

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from lexhybrid import HybridConfig, HybridLanguageModel
from lexhybrid.data.synthetic import SyntheticPackedDataset
from lexhybrid.training.callbacks import SignalCheckpointCallback
from lexhybrid.training.lightning_module import HybridLightningModule
from lexhybrid.training.metrics import compute_bits_per_token, compute_mqar_accuracy, compute_perplexity
from lexhybrid.training.optimizer import configure_optimizer, get_parameter_groups
from lexhybrid.training.schedulers import WSDScheduler, apply_beta2_schedule, beta2_for_step, wsd_factor


def _opt():
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)


@pytest.mark.parametrize(
    "max_steps,warmup,decay,stable,decay_start",
    [(120_000, 2000, 16_800, 101_200, 103_200), (12_000, 500, 1_680, 9_820, 10_320)],
)
def test_wsd_shape_matches_the_reference_record(max_steps, warmup, decay, stable, decay_start):
    s = WSDScheduler(_opt(), max_steps=max_steps, warmup_steps=warmup)
    assert (s.decay_steps, s.stable_steps, s.decay_start) == (decay, stable, decay_start)


def test_wsd_factor_phases():
    assert wsd_factor(0, 100, 800, 100) == pytest.approx(0.01)
    assert wsd_factor(50, 100, 800, 100) == pytest.approx(0.01 + 0.99 * 0.5)
    assert wsd_factor(500, 100, 800, 100) == 1.0
    assert wsd_factor(900 + 25, 100, 800, 100) == pytest.approx(1 - 0.25**0.5)
    assert wsd_factor(10_000, 100, 800, 100, min_lr_ratio=0.1) == pytest.approx(0.1)


def test_wsd_ratio_validation():
    with pytest.raises(ValueError):
        WSDScheduler(_opt(), max_steps=100, warmup_ratio=0.5, stable_ratio=0.5, decay_ratio=0.5)


def test_beta2_anneal():
    assert beta2_for_step(10, decay_start=100, decay_steps=100) == 0.999
    assert beta2_for_step(150, decay_start=100, decay_steps=100) == pytest.approx(0.9865)
    assert beta2_for_step(500, decay_start=100, decay_steps=100) == pytest.approx(0.974)
    opt = _opt()
    assert apply_beta2_schedule(opt, 200, 100, 100) == pytest.approx(0.974)
    assert opt.param_groups[0]["betas"][1] == pytest.approx(0.974)


def test_parameter_groups_exclude_norms_biases_and_embeddings_from_decay():
    model = HybridLanguageModel(
        HybridConfig(
            vocab_size=64,
            dim=32,
            num_layers=1,
            layer_pattern=["mlstm"],
            head_dim=16,
            num_heads=2,
            max_position_embeddings=32,
        )
    )
    decay, no_decay = get_parameter_groups(model, weight_decay=0.1)
    names = {id(p): n for n, p in model.named_parameters()}
    decayed = {names[id(p)] for p in decay["params"]}
    assert not any(("bias" in n) or ("norm" in n) or ("embed" in n) for n in decayed)
    assert "lm_head.weight" in decayed and no_decay["weight_decay"] == 0.0
    assert isinstance(configure_optimizer(model), torch.optim.AdamW)
    with pytest.raises(ValueError):
        configure_optimizer(model, optimizer_name="lion")


def test_metrics():
    loss = torch.tensor(2.0)
    assert compute_perplexity(loss) == pytest.approx(torch.exp(loss).item())
    assert compute_bits_per_token(loss).item() == pytest.approx(2.0 / 0.6931471805599453)
    logits = torch.tensor([[[0.0, 5.0], [5.0, 0.0]]])
    assert compute_mqar_accuracy(logits, torch.tensor([[1, 1]])).item() == pytest.approx(0.5)


@pytest.mark.parametrize("scheduler", ["wsd", "cosine", "linear", "constant"])
def test_lightning_module_trains_two_steps_on_cpu(tmp_path, scheduler):
    torch.manual_seed(0)
    model = HybridLanguageModel(
        HybridConfig(
            vocab_size=64,
            dim=32,
            num_layers=2,
            layer_pattern=["mamba3", "mlstm"],
            mamba3_d_state=16,
            mamba3_head_dim=16,
            head_dim=16,
            num_heads=2,
            tfla_impl="exact",
            max_position_embeddings=64,
        )
    )
    module = HybridLightningModule(
        model, learning_rate=1e-3, warmup_steps=1, max_steps=4, scheduler_name=scheduler, beta2_schedule=True
    )
    data = DataLoader(SyntheticPackedDataset(num_rows=8, row_len=24, vocab_size=64), batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=1,
        val_check_interval=1,
        default_root_dir=str(tmp_path),
    )
    before = [p.detach().clone() for p in model.parameters()]
    trainer.fit(module, data, data)
    assert trainer.global_step == 2
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), "no parameter moved"
    assert "val/loss" in trainer.callback_metrics


def test_signal_callback_saves_once(tmp_path):
    saved = []

    class FakeTrainer:
        def save_checkpoint(self, path):
            saved.append(path)

    cb = SignalCheckpointCallback(str(tmp_path / "ckpt"))
    cb._trainer = FakeTrainer()
    cb._save("test")
    cb._save("again")
    assert saved == [str(tmp_path / "ckpt" / "interrupt.ckpt")]
