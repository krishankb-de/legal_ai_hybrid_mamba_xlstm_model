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
