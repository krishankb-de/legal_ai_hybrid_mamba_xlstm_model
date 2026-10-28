"""Training utilities: WSD, beta2 anneal, optimizer groups, Lightning module, SLURM callback (P1-O/P)."""

import math

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


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_parameter_groups_exclude_norms_biases_and_embeddings_from_decay(tied):
    model = HybridLanguageModel(
        HybridConfig(
            vocab_size=64,
            dim=32,
            num_layers=1,
            layer_pattern=["mlstm"],
            head_dim=16,
            num_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=tied,
        )
    )
    decay, no_decay = get_parameter_groups(model, weight_decay=0.1)
    names = {id(p): n for n, p in model.named_parameters()}
    decayed = {names[id(p)] for p in decay["params"]}
    assert not any(("bias" in n) or ("norm" in n) or ("embed" in n) for n in decayed)
    assert no_decay["weight_decay"] == 0.0
    # Untied, the head is an ordinary decayed matrix. Tied (the default since P2-M), the shared
    # tensor is listed once, under its embedding name, and is not decayed.
    assert ("lm_head.weight" in decayed) is (not tied)
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))
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


# -- distillation (P2-T) -----------------------------------------------------------------------


def _kd_inputs(batch=2, length=37, dim=16, teacher_dim=24, vocab=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, length, dim, generator=g, requires_grad=True)
    teacher_hidden = torch.randn(batch, length, teacher_dim, generator=g)
    targets = torch.randint(0, vocab, (batch, length), generator=g)
    targets[0, 5] = targets[1, 20] = targets[1, -1] = -100
    torch.manual_seed(seed)
    head = torch.nn.Linear(dim, vocab, bias=False)
    teacher_head = torch.nn.Linear(teacher_dim, vocab, bias=False)
    return hidden, teacher_hidden, targets, head, teacher_head


def _full_kd(hidden, teacher_hidden, targets, head, teacher_head, alpha, T):
    import torch.nn.functional as F

    s, t = head(hidden), teacher_head(teacher_hidden)
    mask = targets != -100
    ce = F.cross_entropy(s.reshape(-1, s.shape[-1]), targets.reshape(-1), ignore_index=-100)
    kl = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True, reduction="none").sum(
        -1
    )
    kl = (kl * mask).sum() / mask.sum()
    return (1 - alpha) * ce + alpha * T**2 * kl, ce, kl


@pytest.mark.parametrize("slab", [8, 37, 512], ids=lambda s: f"slab{s}")
def test_slab_kl_equals_full_kl_small(slab):
    """P2-T: the slab-wise, checkpointed CE + KL equals the materialised computation -- value and
    gradients -- with -100 targets inside and at the end of a slab."""
    from lexhybrid.training.distill import slab_ce_kl

    hidden, teacher_hidden, targets, head, teacher_head = _kd_inputs()
    got = slab_ce_kl(hidden, targets, teacher_hidden, head, teacher_head, alpha=0.5, T=2.0, slab=slab)
    want, ce, kl = _full_kd(hidden, teacher_hidden, targets, head, teacher_head, 0.5, 2.0)
    assert torch.allclose(got["loss"], want, atol=1e-6)
    assert torch.allclose(got["ce"], ce, atol=1e-6) and torch.allclose(got["kl"], kl, atol=1e-6)
    assert got["n_supervised"].item() == 2 * 37 - 3
    g_slab = torch.autograd.grad(got["loss"], [hidden, head.weight])
    g_full = torch.autograd.grad(want, [hidden, head.weight])
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g_slab, g_full))
    assert teacher_head.weight.grad is None, "the teacher is never trained"


def test_slab_loss_without_a_teacher_is_plain_ce():
    import torch.nn.functional as F

    from lexhybrid.training.distill import slab_ce_kl

    hidden, _, targets, head, _ = _kd_inputs()
    got = slab_ce_kl(hidden, targets, None, head, slab=10)
    want = F.cross_entropy(head(hidden).reshape(-1, 50), targets.reshape(-1), ignore_index=-100)
    assert torch.allclose(got["loss"], want, atol=1e-6) and got["kl"].item() == 0.0


def _tiny_hf_teacher(arch):
    import transformers

    cfg_cls, model_cls = {
        "qwen2": (transformers.Qwen2Config, transformers.Qwen2ForCausalLM),
        "qwen3": (transformers.Qwen3Config, transformers.Qwen3ForCausalLM),
    }[arch]
    cfg = cfg_cls(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    return model_cls(cfg).eval()


@pytest.mark.parametrize("arch", ["qwen2", "qwen3"])
def test_teacher_position_ids_isolation(arch):
    """Decision 9: a packed row reaches the teacher with per-document position ids and a block-
    diagonal mask, so every document's teacher logits equal the standalone run (1e-3; exact in
    fp32 here). Offline: a tiny random model built from a transformers config."""
    from lexhybrid.training.distill import teacher_forward_packed, teacher_inputs_packed

    teacher = _tiny_hf_teacher(arch)
    ids = torch.randint(0, 128, (2, 40))
    doc = torch.zeros(2, 40, dtype=torch.long)
    doc[0, 11:] = 1
    doc[0, 30:] = 2
    doc[1, 25:] = 1
    head = teacher.get_output_embeddings()
    with torch.no_grad():
        packed = head(teacher_forward_packed(teacher, ids, doc))
        for row, cuts in ((0, [0, 11, 30, 40]), (1, [0, 25, 40])):
            for s, e in zip(cuts, cuts[1:]):
                alone = teacher(ids[row : row + 1, s:e]).logits[0]
                assert (packed[row, s:e] - alone).abs().max() < 1e-3, f"row {row} doc [{s},{e})"
        leak = head(
            teacher.get_decoder()(
                input_ids=ids, position_ids=teacher_inputs_packed(ids, doc)[0]
            ).last_hidden_state
        )
        assert (leak[0, 11:30] - teacher(ids[:1, 11:30]).logits[0]).abs().max() > 1e-3, "positions alone leak"


class _LargestTensor(torch.utils._python_dispatch.TorchDispatchMode):
    """Records the numel of the largest tensor any op produces (forward and backward)."""

    def __init__(self):
        super().__init__()
        self.largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in torch.utils._pytree.tree_leaves(out):
            if isinstance(t, torch.Tensor):
                self.largest = max(self.largest, t.numel())
        return out


def test_peak_memory_scales_with_slab_not_row():
    """P2-T: with the slab loss the largest tensor ever allocated -- forward and backward -- is
    one slab's logits, whatever the row length; materialised logits grow with the row."""
    from lexhybrid.training.distill import slab_ce_kl

    batch, vocab, slab = 2, 3000, 64
    sizes = {}
    for length in (256, 1024):
        hidden, teacher_hidden, targets, head, teacher_head = _kd_inputs(length=length, vocab=vocab)
        with _LargestTensor() as mode:
            out = slab_ce_kl(hidden, targets, teacher_hidden, head, teacher_head, slab=slab)
            out["loss"].backward()
        sizes[length] = mode.largest
    assert sizes[256] == sizes[1024] <= batch * slab * vocab, sizes
    with _LargestTensor() as mode:
        hidden, teacher_hidden, targets, head, teacher_head = _kd_inputs(length=1024, vocab=vocab)
        _full_kd(hidden, teacher_hidden, targets, head, teacher_head, 0.5, 2.0)[0].backward()
    assert mode.largest >= batch * 1024 * vocab


# -- pretraining module and entry point (P2-U) -------------------------------------------------


def _compose(*overrides):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from tests.conftest import REPO_ROOT

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


@pytest.mark.parametrize("trainer_name", ["cpu_debug", "h100_single_gpu", "h100_multi_ddp"])
def test_val_check_interval_product(trainer_name, script):
    """Defect 12: validation cadence is set in optimizer steps and converted to Lightning's
    micro-batch count as val_every_opt_steps * accumulate_grad_batches."""
    tp = script("train_pretrain")
    cfg = _compose(f"trainer={trainer_name}")
    kwargs = tp.build_trainer_kwargs(cfg.trainer)
    want = cfg.trainer.val_every_opt_steps * cfg.trainer.accumulate_grad_batches
    assert kwargs["val_check_interval"] == want and "val_every_opt_steps" not in kwargs
    if trainer_name == "cpu_debug":
        trainer = pl.Trainer(logger=False, **kwargs)
        assert trainer.val_check_interval == want
    with pytest.raises(ValueError, match="val_every_opt_steps"):
        tp.build_trainer_kwargs({**kwargs, "val_every_opt_steps": 1, "val_check_interval": 5})


def test_ddp_strategy_is_built_explicitly(script):
    """...and it launches its own ranks inside the one SLURM task (P6-A): under SLURM, Lightning
    would otherwise take SLURM_NTASKS=1 as the world and expect srun to start the other ranks."""
    from lightning_fabric.plugins.environments import LightningEnvironment
    from pytorch_lightning.strategies import DDPStrategy

    tp = script("train_pretrain")
    strategy = tp.build_trainer_kwargs(_compose("trainer=h100_multi_ddp").trainer)["strategy"]
    assert isinstance(strategy, DDPStrategy)
    assert strategy._ddp_kwargs == {"find_unused_parameters": False, "gradient_as_bucket_view": True}
    assert isinstance(strategy.cluster_environment, LightningEnvironment)
    assert tp.build_strategy("auto") == "auto"


def test_ddp_ignores_slurm_inside_a_batch_job(script, monkeypatch):
    """With SLURM_NTASKS set (every batch job) the Trainer keeps the strategy's own environment."""
    import pytorch_lightning as pl
    from lightning_fabric.plugins.environments import LightningEnvironment

    tp = script("train_pretrain")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_JOB_NAME", "train_4gpu")
    trainer = pl.Trainer(accelerator="cpu", devices=2, strategy=tp.build_strategy("ddp"), logger=False)
    assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)


def test_only_the_launcher_rank_writes_run_metadata(script, monkeypatch):
    tp = script("train_pretrain")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert tp.is_launcher_rank()
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert tp.is_launcher_rank()
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert not tp.is_launcher_rank()


@pytest.mark.parametrize("name", ["qwen3_1p7b", "qwen3_8b"])
def test_build_teacher_loads_the_pinned_revision_offline(name, script, monkeypatch):
    """Job 2589362: offline, a teacher loaded without its revision resolves refs/main, which the
    pinned 1.7B snapshot never wrote. build_teacher passes the config's revision through."""
    import transformers

    seen = {}

    def fake_from_pretrained(repo, **kwargs):
        seen.update(repo=repo, **kwargs)
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = _compose(f"distill={name}")
    teacher = script("train_pretrain").build_teacher(cfg)
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    assert seen["repo"] == cfg.distill.teacher and seen["local_files_only"] is True
    assert seen["revision"] == cfg.distill.revision
    assert _compose().get("distill") is None and script("train_pretrain").build_teacher(_compose()) is None


def test_arch_only_prints_the_fingerprint_and_trains_nothing(tmp_path, script, capsys):
    """The wrappers' step-0 check (FL1): `+arch_only=true` prints the ARCH line of exactly the model
    the overrides describe, on meta, and writes nothing."""
    tp = script("train_pretrain")
    out = tmp_path / "out"
    assert tp.run(_compose("model=hybrid_legal_ds64", f"output_dir={out}", "+arch_only=true")) is None
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("ARCH ") and "mamba3(d_state=64" in line and "vocab=151936" in line
    assert not out.exists()


def test_step_stats_callback_times_optimizer_steps(tmp_path, capsys):
    """STEPSTATS (P4-T, P5-F): one timing per optimizer step, not per micro-batch; a line every
    `every_n_steps` and a final one; peak memory is n/a without CUDA."""
    from lexhybrid.training.callbacks import StepStatsCallback
    from lexhybrid.training.lightning_module import HybridLightningModule

    stats = StepStatsCallback(every_n_steps=1, skip_steps=0)
    module = HybridLightningModule(_tiny_student(), warmup_steps=1, max_steps=2)
    data = DataLoader(SyntheticPackedDataset(num_rows=8, row_len=24, vocab_size=128), batch_size=2)
    _cpu_trainer(tmp_path, callbacks=[stats], accumulate_grad_batches=2, limit_val_batches=0).fit(
        module, data
    )
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("STEPSTATS")]
    assert len(stats.durations) == 2 and all(d > 0 for d in stats.durations)
    assert [ln.split()[2] for ln in lines] == ["step=1", "step=2", "step=2"]
    final = dict(kv.split("=", 1) for kv in lines[-1].split()[1:])
    assert final["rank"] == "0" and final["final"] == "1" and final["n"] == "2"
    assert float(final["s_per_step"]) > 0 and final["peak_alloc_gb"] == "n/a"
    validated = StepStatsCallback(every_n_steps=0, skip_steps=0)  # validation after every step
    _cpu_trainer(tmp_path, callbacks=[validated], max_steps=3).fit(
        HybridLightningModule(_tiny_student(), warmup_steps=1, max_steps=3), data, data
    )
    assert len(validated.durations) == 2, "steps after a validation run are timed from its end"
    skipping = StepStatsCallback(every_n_steps=0, skip_steps=5)
    _cpu_trainer(tmp_path, callbacks=[skipping], limit_val_batches=0).fit(
        HybridLightningModule(_tiny_student(), warmup_steps=1, max_steps=2), data
    )
    assert skipping.durations == [] and "s_per_step=n/a" in capsys.readouterr().out


def _tiny_student(**kw):
    torch.manual_seed(0)
    defaults = dict(
        vocab_size=128,
        dim=32,
        num_layers=2,
        layer_pattern=["mamba3", "mlstm"],
        mamba3_d_state=16,
        mamba3_head_dim=16,
        head_dim=16,
        tfla_impl="exact",
        mlstm_chunk_size=8,
        max_position_embeddings=64,
        dropout=0.0,
    )
    defaults.update(kw)
    return HybridLanguageModel(HybridConfig(**defaults))


def _cpu_trainer(tmp_path, **kw):
    defaults = dict(
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
    defaults.update(kw)
    return pl.Trainer(**defaults)


def test_nan_guard_skips_step(tmp_path):
    """Defect 12: a non-finite loss returns None, so Lightning skips backward AND the optimizer step.
    With weight decay 0.1 any step would move the weights even on zero gradients (the reference's
    leaf-tensor guard did exactly that); here they stay bit-identical, and the run stops after
    `max_nan_skips` consecutive skips."""
    from lexhybrid.training.pretrain_module import PretrainLightningModule

    module = PretrainLightningModule(
        _tiny_student(), learning_rate=1e-2, weight_decay=0.1, warmup_steps=0, max_steps=10, max_nan_skips=3
    )
    real = module.compute_losses

    def poisoned(batch, **kw):
        out = real(batch, **kw)
        out["loss"] = out["loss"] * float("nan")
        return out

    module.compute_losses = poisoned
    before = {k: v.clone() for k, v in module.model.state_dict().items()}
    data = DataLoader(SyntheticPackedDataset(num_rows=16, row_len=24, vocab_size=128), batch_size=2)
    with pytest.raises(RuntimeError, match="consecutive non-finite"):
        _cpu_trainer(tmp_path, max_steps=5).fit(module, data)
    assert module.nan_skips == 4
    for k, v in module.model.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} moved although every step was skipped"


def test_pretrain_module_trains_with_kd_and_mtp(tmp_path):
    """Two CPU steps with a tiny random Qwen2 teacher and MTP on: finite CE, KL and MTP terms; the
    student moves; the teacher neither trains nor enters the checkpointed state."""
    from lexhybrid.training.pretrain_module import PretrainLightningModule

    teacher = _tiny_hf_teacher("qwen2")
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    student = _tiny_student(mtp_n=2)
    module = PretrainLightningModule(
        student,
        teacher=teacher,
        learning_rate=1e-3,
        warmup_steps=1,
        max_steps=4,
        scheduler_name="wsd",
        loss_slab=8,
    )
    before = {k: v.clone() for k, v in student.state_dict().items()}
    batch = next(
        iter(DataLoader(SyntheticPackedDataset(num_rows=2, row_len=24, vocab_size=128), batch_size=2))
    )
    out = module.compute_losses(batch, use_teacher=True, use_mtp=True)
    assert all(torch.isfinite(out[k]) for k in ("loss", "ce", "kl", "mtp")) and out["kl"] > 0
    data = DataLoader(SyntheticPackedDataset(num_rows=8, row_len=24, vocab_size=128), batch_size=2)
    _cpu_trainer(tmp_path).fit(module, data, data)
    assert not all(torch.equal(v, before[k]) for k, v in student.state_dict().items())
    assert all(torch.equal(v, teacher_before[k]) for k, v in teacher.state_dict().items())
    assert not any(k.startswith("teacher") or "_teacher" in k for k in module.state_dict())


@pytest.mark.slow
def test_train_pretrain_script_runs_and_resumes(tmp_path, script):
    """The entry point end to end on CPU: 2 steps with last.ckpt written every step, then a resumed
    run to step 3 from that checkpoint -- the requeue path of the SLURM wrappers."""
    tp = script("train_pretrain")
    overrides = [
        "model.vocab_size=64",
        "model.dim=32",
        "model.num_layers=2",
        "model.layer_pattern=[mamba3,mlstm]",
        "model.num_heads=2",
        "model.head_dim=16",
        "model.mamba3_head_dim=16",
        "model.mamba3_d_state=16",
        "model.max_position_embeddings=64",
        "model.tfla_impl=exact",
        "model.mlstm_chunk_size=8",
        "trainer=cpu_debug",
        "trainer.enable_checkpointing=true",
        "callbacks.checkpoint.every_n_train_steps=1",
        "dataset.row_len=24",
        "dataset.num_rows=8",
        "dataset.batch_size=2",
        f"output_dir={tmp_path}",
        "+logger=false",
    ]
    first = tp.run(_compose(*overrides))
    last = tmp_path / "checkpoints" / "last.ckpt"
    assert first.global_step == 2 and last.exists()
    assert (tmp_path / "run_metadata.json").exists()
    calls = []
    real_step = tp.PretrainLightningModule.training_step

    def counted(self, batch, batch_idx):
        calls.append(int(self.global_step))
        return real_step(self, batch, batch_idx)

    tp.PretrainLightningModule.training_step = counted
    try:
        resumed = tp.run(_compose(*overrides, "trainer.max_steps=3", f"+resume_from_checkpoint={last}"))
    finally:
        tp.PretrainLightningModule.training_step = real_step
    assert resumed.global_step == 3 and calls == [2], (
        f"resumed run stepped from {calls} (a restart would be [0, 1, 2])"
    )
    with pytest.raises(FileNotFoundError):
        tp.run(_compose(*overrides, f"+resume_from_checkpoint={tmp_path / 'missing.ckpt'}"))


def test_train_pretrain_runs_on_packed_shards(tmp_path, script):
    """P3-V: the entry point trains on the packed-shard mixture (dataset=legal_smoke retargeted at
    shards built here with a toy one-id-per-character tokenizer)."""
    import json

    from lexhybrid.data.corpus.collectors.fineweb2_de import parse_fineweb_row
    from lexhybrid.data.corpus.collectors.oldp import parse_oldp_case
    from lexhybrid.data.datasets import build_source_shards
    from tests.conftest import REPO_ROOT

    fixtures = REPO_ROOT / "tests" / "fixtures" / "collectors"
    oldp = [
        parse_oldp_case(json.loads(p.read_text())) for p in sorted((fixtures / "oldp").glob("case_*.json"))
    ]
    web = [parse_fineweb_row(r) for r in json.loads((fixtures / "fineweb2_de" / "rows.json").read_text())]
    encode = lambda text: [min(ord(c), 1022) + 1 for c in text]  # noqa: E731 -- ids < 1024, EOS 0
    for docs in (oldp, web):
        build_source_shards(docs, encode, 0, 32, tmp_path / "shards", fraction=0.001, minimum=1)
    tp = script("train_pretrain")
    trainer = tp.run(
        _compose(
            "dataset=legal_smoke",
            f"dataset.shards_root={tmp_path / 'shards'}",
            "dataset.row_len=32",
            "dataset.train_samples=8",
            "dataset.batch_size=2",
            "~dataset.sources",
            "+dataset.sources={oldp:{group:legal},fineweb2_de:{group:general}}",
            "model.vocab_size=1024",
            "model.dim=32",
            "model.num_layers=2",
            "model.layer_pattern=[mamba3,mlstm]",
            "model.num_heads=2",
            "model.head_dim=16",
            "model.mamba3_head_dim=16",
            "model.mamba3_d_state=16",
            "model.max_position_embeddings=64",
            "model.tfla_impl=exact",
            "model.mlstm_chunk_size=8",
            "trainer=cpu_debug",
            f"output_dir={tmp_path / 'out'}",
            "+logger=false",
        )  # fmt: skip
    )
    assert trainer.global_step == 2


class _CaptureLogger(pl.loggers.Logger):
    """Keeps every ``log_metrics`` call in memory: ``rows = [(step, {tag: value})]``."""

    def __init__(self):
        super().__init__()
        self.rows = []

    @property
    def name(self):
        return "capture"

    @property
    def version(self):
        return 0

    def log_metrics(self, metrics, step=None):
        self.rows.append((step, dict(metrics)))

    def log_hyperparams(self, params, *args, **kwargs):
        pass

    def by_step(self, prefix):
        out = {}
        for step, metrics in self.rows:
            out.setdefault(step, {}).update({k: v for k, v in metrics.items() if k.startswith(prefix)})
        return {s: m for s, m in out.items() if m}


def _diag_fit(tmp_path, callbacks, max_steps=4, **trainer_kw):
    from lexhybrid.training.pretrain_module import PretrainLightningModule

    student = _tiny_student(
        layer_pattern=["mamba3", "attention", "mlstm"], num_layers=3, norm_topology="hybrid"
    )
    module = PretrainLightningModule(student, learning_rate=1e-3, warmup_steps=1, max_steps=max_steps)
    data = DataLoader(SyntheticPackedDataset(num_rows=64, row_len=24, vocab_size=128), batch_size=2)
    logger = _CaptureLogger()
    trainer = _cpu_trainer(
        tmp_path, callbacks=callbacks, logger=logger, max_steps=max_steps, log_every_n_steps=1,
        accumulate_grad_batches=2, limit_val_batches=0, **trainer_kw,
    )  # fmt: skip
    trainer.fit(module, data)
    return trainer, student, logger


def test_spike_diagnostics_log_every_part_without_changing_training(tmp_path):
    """decisions.p5_divergence: per-part gradient and update norms every optimizer step, forward
    statistics (activations, mLSTM gates and half-range, attention logit bound, Mamba dt, head
    logits) every n steps; the weights end bit-identical to a run without the callback, and the
    TFLA patch is removed at teardown."""
    from lexhybrid.kernels.tfla import tfla_interface
    from lexhybrid.training.diagnostics import SpikeDiagnosticsCallback

    forward, row_block = tfla_interface.tfla_forward_parallel, tfla_interface._row_block
    _, plain, _ = _diag_fit(tmp_path, [])
    diag = SpikeDiagnosticsCallback(every_n_steps=2)
    _, traced, logger = _diag_fit(tmp_path, [diag])
    for k, v in plain.state_dict().items():
        assert torch.equal(v, traced.state_dict()[k]), f"{k} differs with the diagnostics on"
    assert (tfla_interface.tfla_forward_parallel, tfla_interface._row_block) == (forward, row_block)

    parts = {"embed", "L0_mamba3", "L1_attention", "L2_mlstm", "final_norm"}
    grads, updates = logger.by_step("diag/grad/"), logger.by_step("diag/update/")
    assert sorted(grads) == sorted(updates) == [0, 1, 2, 3]
    assert all({k.split("/")[-1] for k in m} == parts for m in grads.values())
    assert all(v > 0 for m in updates.values() for v in m.values())
    clip = {s: m["train/grad_norm"] for s, m in logger.by_step("train/grad_norm").items()}
    for step, m in grads.items():  # the parts add up to the norm the module clips
        assert math.isclose(math.sqrt(sum(v * v for v in m.values())), clip[step], rel_tol=1e-4)

    fwd = {s: m for s, m in logger.by_step("diag/").items() if any("/act/" in k for k in m)}
    assert sorted(fwd) == [0, 2]
    want = {
        "diag/act/L0_mamba3_rms", "diag/act/L2_mlstm_max", "diag/mlp/L1_attention_max",
        "diag/mlstm/L2_ilogit_max", "diag/mlstm/L2_flogit_min", "diag/mlstm/L2_half_range",
        "diag/mlstm/L2_row_block_min", "diag/attn/L1_logit_bound", "diag/mamba/L0_dt_max",
        "diag/head/logit_max",
    }  # fmt: skip
    assert all(want <= set(m) for m in fwd.values()), want - set(fwd[0])
    assert all(math.isfinite(v) for m in fwd.values() for v in m.values())
    assert set(logger.by_step("diag/param/")) == {0, 2}


def test_spike_diagnostics_flag_spikes_and_stop_at_a_step(tmp_path, capsys):
    from lexhybrid.training.diagnostics import SpikeDiagnosticsCallback

    diag = SpikeDiagnosticsCallback(every_n_steps=100, spike_factor=0.0, stop_at_step=22)
    trainer, _, _ = _diag_fit(tmp_path, [diag], max_steps=30)
    out = capsys.readouterr().out
    assert trainer.global_step == 22 and "DIAG STOP step=22" in out
    assert diag.spikes == [20, 21]  # the median needs 20 steps of history first
    line = next(ln for ln in out.splitlines() if ln.startswith("DIAG SPIKE step=20 "))
    assert "top=" in line and "median=" in line


def test_spike_diagnostics_record_the_tfla_row_block_regime():
    """The S1 failure regime: a strong forget gate (log f = -8 per step, half-range 64 over a
    16-step chunk) takes the per-block path; the trace records the half-range and the block."""
    from lexhybrid.kernels.tfla import tfla_interface
    from lexhybrid.training.diagnostics import SpikeDiagnosticsCallback

    diag = SpikeDiagnosticsCallback()
    diag._patch_tfla()
    try:
        diag._active_step, diag._current_mlstm = 0, "L9"
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 1, 16, 4) for _ in range(3))
        f = torch.full((1, 1, 16, 4), math.exp(-8.0))
        out = tfla_interface.tfla_forward_parallel(
            q, k, v, torch.ones_like(f), f, chunk_size=16, tfla_impl="exact"
        )
        assert torch.isfinite(out).all()
        assert math.isclose(diag._stats["mlstm/L9_half_range"], 64.0, rel_tol=1e-5)
        assert 1 <= diag._stats["mlstm/L9_row_block_min"] < 16
    finally:
        diag.teardown(None, None, "fit")
    assert tfla_interface._row_block.__name__ == "_row_block"


def test_spike_diagnostics_flag_a_non_finite_gradient_first(capsys):
    """The step the NaN guard skips is a spike too, and the part that went non-finite leads."""
    from types import SimpleNamespace

    from lexhybrid.training.diagnostics import SpikeDiagnosticsCallback

    good, bad = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(2))
    good.grad, bad.grad = torch.full((3,), 2.0), torch.tensor([1.0, float("nan")])
    diag = SpikeDiagnosticsCallback(every_n_steps=100)
    diag.parts = {"L0_mamba3": [good], "L1_attention": [bad]}
    diag.on_before_optimizer_step(
        SimpleNamespace(global_step=5, logger=None, is_global_zero=True), None, None
    )
    line = capsys.readouterr().out.strip()
    assert diag.spikes == [5] and not diag.history
    assert (
        line.startswith("DIAG SPIKE step=5 grad_norm=nan") and "top=L1_attention:nan,L0_mamba3:3.464" in line
    )


def test_inspect_checkpoint_maps_adamw_state_to_parameter_names(tmp_path, script, monkeypatch, capsys):
    """decisions.p5_divergence: the AdamW state of a Lightning checkpoint is indexed by position;
    the inspector names it through get_parameter_groups. Each named second moment must equal the
    live optimizer's for that parameter, and a checkpoint of another model is refused."""
    from lexhybrid.training.pretrain_module import PretrainLightningModule

    student = _tiny_student(
        layer_pattern=["mamba3", "attention", "mlstm"], num_layers=3, norm_topology="hybrid"
    )
    module = PretrainLightningModule(student, learning_rate=1e-3, warmup_steps=1, max_steps=2)
    data = DataLoader(SyntheticPackedDataset(num_rows=8, row_len=24, vocab_size=128), batch_size=2)
    trainer = _cpu_trainer(tmp_path, limit_val_batches=0)
    trainer.fit(module, data)
    path = tmp_path / "last.ckpt"
    trainer.save_checkpoint(str(path))

    ic = script("inspect_checkpoint")
    monkeypatch.setattr(ic, "load_model_config", lambda name: student.config)
    assert ic.main(["--checkpoint", str(path), "--model-config", "tiny", "--label", "t"]) == 0
    lines = capsys.readouterr().out.splitlines()
    params = {
        ln.split()[2]: dict(kv.split("=") for kv in ln.split()[3:]) for ln in lines if ln.startswith("PARAM")
    }
    assert set(params) == {n for n, _ in student.named_parameters()}
    state = trainer.optimizers[0].state
    for name, p in student.named_parameters():
        want = state[p]["exp_avg_sq"].sqrt().max().item()
        assert math.isclose(float(params[name]["v_sqrt_max"]), want, rel_tol=1e-3), name
    parts = {ln.split()[2] for ln in lines if ln.startswith("PART")}
    assert parts == {"embed", "L0_mamba3", "L1_attention", "L2_mlstm", "final_norm"}

    other = _tiny_student(layer_pattern=["mamba3", "mlstm"], num_layers=2)
    monkeypatch.setattr(ic, "load_model_config", lambda name: other.config)
    with pytest.raises((RuntimeError, KeyError)):
        ic.main(["--checkpoint", str(path), "--model-config", "tiny"])


def test_val_perplexity_is_exp_of_the_token_weighted_mean_ce(tmp_path):
    """Screen attempt 1 logged val/perplexity as the mean of per-batch exp(CE): 31.69 where
    exp(mean CE) was 8.19 (job 2636358). Two batches at CE 1 and 3 over 30 and 10 tokens must give
    exp(1.5) = 4.48, equal to exp(val/loss); the old rule gave (30e + 10e^3) / 40 = 7.06."""
    from lexhybrid.training.pretrain_module import PretrainLightningModule

    module = PretrainLightningModule(_tiny_student(), warmup_steps=1, max_steps=2)
    script = iter([(1.0, 30), (3.0, 10)])

    def fixed(batch, **kw):
        ce, n = next(script)
        ce = torch.tensor(ce, requires_grad=True)
        return {"loss": ce, "ce": ce, "n_supervised": torch.tensor(n)}

    module.compute_losses = fixed
    data = DataLoader(SyntheticPackedDataset(num_rows=4, row_len=24, vocab_size=128), batch_size=2)
    (metrics,) = _cpu_trainer(tmp_path, limit_val_batches=2).validate(module, data, verbose=False)
    assert math.isclose(metrics["val/loss"], 1.5, rel_tol=1e-6)
    assert math.isclose(metrics["val/perplexity"], math.exp(1.5), rel_tol=1e-6)


@pytest.mark.parametrize(
    "model_cfg",
    ["hybrid_legal_base", "hybrid_legal_attn0", "hybrid_legal_attn4", "hybrid_legal_ds64",
     "hybrid_legal_legacy", "hybrid_legal_mtp", "transformer_legal_base"],
)  # fmt: skip
def test_legal_configs_train_with_adamw_beta2_095(model_cfg, script, capsys):
    """decisions.p5_fix: every legal config's AdamW runs at beta2 0.95, and the yaml value reaches the
    optimizer through train_pretrain.build_module (the OPTIM line the job log shows)."""
    tp = script("train_pretrain")
    module = tp.build_module(_compose(f"model={model_cfg}"), _tiny_student(), None)
    optimizer = module.configure_optimizers()["optimizer"]
    assert all(g["betas"] == (0.9, 0.95) for g in optimizer.param_groups)
    assert "OPTIM adamw lr=0.0004 betas=(0.9, 0.95)" in capsys.readouterr().out


def test_adam_beta2_defaults_and_the_schedule_guard():
    from lexhybrid.training.lightning_module import HybridLightningModule

    optimizer = HybridLightningModule(_tiny_student(), max_steps=10).configure_optimizers()
    optimizer = optimizer["optimizer"] if isinstance(optimizer, dict) else optimizer
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    with pytest.raises(ValueError, match="anneals from beta2_start"):
        HybridLightningModule(_tiny_student(), beta2_schedule=True, adam_beta2=0.95)


def test_no_teacher_is_loaded_when_kd_alpha_is_0(script, capsys):
    """S6 (distill.alpha=0) ran the 8B teacher forward every step in screen attempt 1 although the
    loss was plain CE; build_teacher now returns None before loading anything."""
    tp = script("train_pretrain")
    assert tp.build_teacher(_compose("distill=qwen3_8b", "distill.alpha=0.0")) is None
    assert "teacher: none (distill.alpha=0)" in capsys.readouterr().out
    assert tp.build_teacher(_compose()) is None  # no distill config at all


def test_late_context_ppl_counts_document_context_and_matches_training_val_ce(script):
    """P5-F's late-25% PPL (user's choice, 2026-10-13): a predicted token counts as late when at
    least late_from tokens of ITS OWN document precede it in the row; late_row is the row-position
    variant. The overall CE equals what training's validation_step logs as val/loss."""
    from lexhybrid.training.pretrain_module import PretrainLightningModule

    ev = script("evaluate_val_ppl")
    doc = torch.tensor([[0] * 10 + [1] * 14, [5] * 24])
    assert ev.document_offsets(doc).tolist() == [list(range(10)) + list(range(14)), list(range(24))]

    student = _tiny_student().eval()
    torch.manual_seed(1)
    ids = torch.randint(0, 128, (2, 24))
    sums = ev.late_context_sums(
        student, [{"input_ids": ids, "labels": ids, "doc_ids": doc}], late_from=8, slab=5
    )
    # row 0: 23 predictions minus the one into doc 1's first token; late_doc = doc 0 at 8, 9 and
    # doc 1 at offsets 8..13 (positions 18..23); late_row = positions 8..23 minus position 10
    # row 1: one document, so both late sets are positions 8..23
    assert sums["rows"] == 2 and sums["all"]["tokens"] == 22 + 23
    assert sums["late_doc"]["tokens"] == (2 + 6) + 16 and sums["late_row"]["tokens"] == 15 + 16
    module = PretrainLightningModule(student)
    want = module.compute_losses(
        {"input_ids": ids, "labels": ids, "doc_ids": doc}, use_teacher=False, use_mtp=False
    )
    assert math.isclose(sums["all"]["ce_sum"] / sums["all"]["tokens"], want["ce"].item(), rel_tol=1e-5)
    line, out = ev.summarise(sums, "t")
    assert (
        line.startswith("VALPPL t tokens=45 ppl=")
        and "late_doc_tokens=24" in line
        and "late_row_tokens=31" in line
    )
    assert math.isclose(out["ppl"], math.exp(want["ce"].item()), rel_tol=1e-5)


class _IndexRows(torch.utils.data.Dataset):
    """Row i carries its index, so a run's data order can be read back."""

    def __init__(self, n, offset=0):
        self.n, self.offset = n, offset

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.offset + i)
        return {"input_ids": torch.randint(1, 128, (24,), generator=g), "idx": torch.tensor(self.offset + i)}


class _SeenRows(pl.Callback):
    def __init__(self):
        self.seen = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.seen += batch["idx"].tolist()


def _schedule_fit(tmp_path, name, rows, stop=None, resume=None, accumulate=2, batch_size=2):
    from pytorch_lightning.callbacks import ModelCheckpoint

    from lexhybrid.data.datasets import ScheduleSampler
    from lexhybrid.training.callbacks import StopAtStepCallback
    from lexhybrid.training.lightning_module import HybridLightningModule

    torch.manual_seed(0)
    module = HybridLightningModule(_tiny_student(), warmup_steps=1, max_steps=8, scheduler_name="wsd")
    seen = _SeenRows()
    callbacks = [
        seen,
        ModelCheckpoint(dirpath=tmp_path / name, every_n_train_steps=2, save_last=True, save_top_k=0),
    ]
    if stop:
        callbacks.append(StopAtStepCallback(stop))
    trainer = _cpu_trainer(tmp_path, max_steps=8, accumulate_grad_batches=accumulate, limit_val_batches=0,
                           callbacks=callbacks, enable_checkpointing=True, use_distributed_sampler=False)  # fmt: skip
    sampler = ScheduleSampler(len(rows), batch_size).bind(trainer)
    data = DataLoader(rows, batch_size=batch_size, sampler=sampler, drop_last=True)
    trainer.fit(module, data, ckpt_path=resume)
    return trainer, module, seen.seen


def test_a_resumed_run_reads_on_through_the_schedule_and_matches_an_unbroken_one(tmp_path):
    """decisions.resume_data: Lightning restarts a plain loader at row 0 on resume (screen run S5's
    continuation re-read its first rows). Stopped at step 4 (StopAtStepCallback, the LR horizon still
    8) and resumed, the run reads rows 16..31 and ends with the weights of an unbroken 8-step run."""
    rows = _IndexRows(100)
    _, whole, seen_whole = _schedule_fit(tmp_path, "whole", rows)
    first, _, seen_a = _schedule_fit(tmp_path, "a", rows, stop=4)
    assert first.global_step == 4 and seen_a == list(range(16))
    second, resumed, seen_b = _schedule_fit(tmp_path, "b", rows, resume=str(tmp_path / "a" / "last.ckpt"))
    assert second.global_step == 8 and seen_b == list(range(16, 32)) and seen_whole == list(range(32))
    for (k, v), w in zip(resumed.state_dict().items(), whole.state_dict().values(), strict=True):
        assert torch.equal(v, w), k


def test_a_stage_on_a_new_schedule_starts_it_at_row_0_and_a_requeue_reads_on(tmp_path):
    """P6 stage B: resumed from stage A's last step onto a shorter schedule of its own (8,192-token
    rows), with another micro-batch and accumulation, it reads that schedule from its first row; a
    requeue inside stage B reads on. A schedule longer than stage A's count still never repeats."""
    _schedule_fit(tmp_path, "a", _IndexRows(100), stop=4)  # 8 micro-batches of stage A
    a_ckpt = str(tmp_path / "a" / "last.ckpt")
    kw = dict(accumulate=1, batch_size=4)
    trainer, _, seen = _schedule_fit(tmp_path, "b", _IndexRows(16, offset=1000), resume=a_ckpt, **kw)
    assert trainer.global_step == 8 and seen == list(range(1000, 1016))
    _, _, b1 = _schedule_fit(tmp_path, "b1", _IndexRows(16, offset=1000), resume=a_ckpt, stop=6, **kw)
    b1_ckpt = str(tmp_path / "b1" / "last.ckpt")
    _, _, b2 = _schedule_fit(tmp_path, "b2", _IndexRows(16, offset=1000), resume=b1_ckpt, **kw)
    assert b1 + b2 == list(range(1000, 1016))
    _, _, l1 = _schedule_fit(tmp_path, "l1", _IndexRows(400, offset=1000), resume=a_ckpt, stop=6, **kw)
    l1_ckpt = str(tmp_path / "l1" / "last.ckpt")
    _, _, l2 = _schedule_fit(tmp_path, "l2", _IndexRows(400, offset=1000), resume=l1_ckpt, **kw)
    assert len(set(l1 + l2)) == 16 and l2[0] == l1[-1] + 1  # fresh rows, read on across the requeue


def test_schedule_sampler_splits_ranks_evenly_and_skips_what_was_trained(monkeypatch):
    from types import SimpleNamespace

    from lexhybrid.data import datasets

    sampler = datasets.ScheduleSampler(10, batch_size=2)
    for rank, want in (
        (0, [0, 3, 6]),
        (1, [1, 4, 7]),
        (2, [2, 5, 8]),
    ):  # 10 // 3 = 3 each: row 9 is never split
        monkeypatch.setattr(datasets, "_rank_world", lambda r=rank: (r, 3))
        assert len(sampler) == 3 and list(sampler) == want
    monkeypatch.setattr(datasets, "_rank_world", lambda: (1, 2))
    progress = SimpleNamespace(current=SimpleNamespace(ready=1))  # one micro-batch of 2 done this epoch
    sampler.bind(
        SimpleNamespace(fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(batch_progress=progress)))
    )
    assert list(sampler) == [5, 7, 9]
    shard = datasets.ShardSampler(5)
    seen = []
    for rank in range(3):
        monkeypatch.setattr(datasets, "_rank_world", lambda r=rank: (r, 3))
        seen.append(list(shard))
    assert [len(s) for s in seen] == [2, 2, 2] and sorted(i for s in seen for i in s) == [0, 0, 1, 2, 3, 4]


class _RankRows(pl.Callback):
    """Writes the row indices each DDP rank trains on to <dir>/rank<r>.txt."""

    def __init__(self, directory):
        self.directory = str(directory)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        with open(f"{self.directory}/rank{trainer.global_rank}.txt", "a") as f:
            f.write(" ".join(map(str, batch["idx"].tolist())) + "\n")


def _ddp_fit(tmp_path, name, stop=None, resume=None):
    from pytorch_lightning.callbacks import ModelCheckpoint

    from lexhybrid.data.datasets import ScheduleSampler
    from lexhybrid.training.callbacks import StopAtStepCallback
    from lexhybrid.training.lightning_module import HybridLightningModule

    out = tmp_path / name
    out.mkdir()
    torch.manual_seed(0)
    module = HybridLightningModule(_tiny_student(), warmup_steps=1, max_steps=4, scheduler_name="wsd")
    callbacks = [
        _RankRows(out),
        ModelCheckpoint(dirpath=out, every_n_train_steps=2, save_last=True, save_top_k=0),
    ]
    if stop:
        callbacks.append(StopAtStepCallback(stop))
    trainer = _cpu_trainer(tmp_path, max_steps=4, devices=2, strategy="ddp_spawn", limit_val_batches=0,
                           callbacks=callbacks, enable_checkpointing=True, use_distributed_sampler=False)  # fmt: skip
    rows = _IndexRows(64)
    data = DataLoader(rows, batch_size=2, sampler=ScheduleSampler(len(rows), 2).bind(trainer), drop_last=True)
    trainer.fit(module, data, ckpt_path=resume)
    return [
        [int(i) for ln in (out / f"rank{r}.txt").read_text().split("\n") for i in ln.split()] for r in (0, 1)
    ]


def test_schedule_sampler_splits_ddp_ranks_and_reads_on_after_a_resume(tmp_path):
    """Under DDP (two CPU ranks, gloo) each rank reads its own rows, r, r + 2, ...; stopped at step 2
    and resumed, every rank reads on where it stopped (P6's four ranks and their requeues)."""
    whole = _ddp_fit(tmp_path, "whole")
    assert whole == [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
    first = _ddp_fit(tmp_path, "a", stop=2)
    second = _ddp_fit(tmp_path, "b", resume=str(tmp_path / "a" / "last.ckpt"))
    assert [f + s for f, s in zip(first, second, strict=True)] == whole


def test_last_ckpt_holds_the_final_step_when_it_is_not_a_checkpoint_step(tmp_path):
    """P6 stage B ends at step 18,037 with checkpoints every 1,000: without FinalCheckpointCallback
    its last.ckpt would stop at 18,000. Here 5 steps with checkpoints every 2."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    from lexhybrid.training.callbacks import FinalCheckpointCallback
    from lexhybrid.training.lightning_module import HybridLightningModule

    ckpts = tmp_path / "ckpt"
    module = HybridLightningModule(_tiny_student(), warmup_steps=1, max_steps=5)
    callbacks = [
        FinalCheckpointCallback(ckpts),
        ModelCheckpoint(dirpath=ckpts, every_n_train_steps=2, save_last=True, save_top_k=0),
    ]
    data = DataLoader(SyntheticPackedDataset(num_rows=16, row_len=24, vocab_size=128), batch_size=2)
    _cpu_trainer(
        tmp_path, max_steps=5, limit_val_batches=0, callbacks=callbacks, enable_checkpointing=True
    ).fit(module, data)
    assert torch.load(ckpts / "last.ckpt", map_location="cpu", weights_only=False)["global_step"] == 5
