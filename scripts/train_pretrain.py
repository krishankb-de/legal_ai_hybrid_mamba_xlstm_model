#!/usr/bin/env python3
"""Pretraining entry point (the SLURM wrappers call it).

    python scripts/train_pretrain.py model=hybrid_legal_base dataset=synthetic trainer=cpu_debug
    python scripts/train_pretrain.py ... distill=qwen3_1p7b +resume_from_checkpoint=/path/last.ckpt

Differences from the upstream reference implementation's training script:

* ``val_check_interval`` counts micro-batches in Lightning; the configs say how often to validate
  in OPTIMIZER steps (``val_every_opt_steps``), so the script sets
  ``val_check_interval = val_every_opt_steps * accumulate_grad_batches`` and asserts the Trainer
  holds that product. A trainer yaml that sets ``val_check_interval`` itself is refused.
* DDP is built explicitly as ``DDPStrategy(find_unused_parameters=False,
  gradient_as_bucket_view=True)``; the upstream implementation's ``find_unused_parameters`` yaml key was never
  read, because its entry point passed the string ``"ddp"``.
* ``compile_model`` is honoured (the upstream implementation read ``compile``).
* ``resume_from_checkpoint`` (the wrapper passes ``last.ckpt`` when it exists) resumes the run; a
  requeued upstream job restarted from step 0.

For the SLURM wrappers:

* DDP runs as ONE SLURM task that starts its own rank processes (``LightningEnvironment``), so a
  4-GPU job needs ``--gpus=4`` and no ``srun``/``--ntasks-per-node``; left to itself, Lightning
  would detect SLURM from ``SLURM_NTASKS`` and expect one srun task per GPU.
* ``+arch_only=true`` builds the model this exact config describes, prints its ``ARCH`` line and
  exits: the wrappers check the arm's expected tokens against it before step 0.
* ``StepStatsCallback`` prints ``STEPSTATS`` lines (s/step, peak GB, per rank).
* ``SpikeDiagnosticsCallback`` (``callbacks.diagnostics.enabled``, off by default) logs per-part
  gradient, update, activation and gate statistics for a loss-spike diagnosis.
"""

import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from lexhybrid.config import HybridConfig
from lexhybrid.data.synthetic import SyntheticPackedDataset
from lexhybrid.models.hybrid_lm import HybridLanguageModel
from lexhybrid.training.callbacks import (
    FinalCheckpointCallback,
    SignalCheckpointCallback,
    StepStatsCallback,
    StopAtStepCallback,
)
from lexhybrid.training.diagnostics import SpikeDiagnosticsCallback
from lexhybrid.training.pretrain_module import PretrainLightningModule
from lexhybrid.utils.run_metadata import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent
# Trainer yaml keys this script consumes itself instead of passing to pl.Trainer.
_SCRIPT_KEYS = ("val_every_opt_steps", "compile_model", "strategy", "stop_at_step")


def build_trainer_kwargs(trainer_cfg) -> dict:
    """``pl.Trainer`` kwargs from a trainer yaml, with ``val_check_interval`` derived."""
    if isinstance(trainer_cfg, DictConfig):
        t = dict(OmegaConf.to_container(trainer_cfg, resolve=True))
    else:
        t = dict(trainer_cfg)
    if "val_check_interval" in t:
        raise ValueError(
            "trainer yamls set val_every_opt_steps (optimizer steps); val_check_interval counts "
            "micro-batches and is derived by the entry point"
        )
    every = int(t["val_every_opt_steps"])
    accumulate = int(t.get("accumulate_grad_batches", 1))
    kwargs = {k: v for k, v in t.items() if k not in _SCRIPT_KEYS}
    kwargs["val_check_interval"] = every * accumulate
    kwargs["strategy"] = build_strategy(t.get("strategy", "auto"))
    return kwargs


def build_strategy(name):
    """The strategy object: DDP is always built explicitly with fixed settings, and
    launches its own rank processes inside the one SLURM task (``LightningEnvironment``)."""
    if name == "ddp":
        from lightning_fabric.plugins.environments import LightningEnvironment

        return DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            cluster_environment=LightningEnvironment(),
        )
    return name


def build_model(cfg: DictConfig) -> HybridLanguageModel:
    model = HybridLanguageModel(HybridConfig.from_hydra(cfg.model))
    warm = cfg.get("lm_checkpoint")
    if warm:
        from lexhybrid.utils.checkpoint import load_state_dict_guarded, strip_prefixes

        state = torch.load(warm, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        load_state_dict_guarded(model, strip_prefixes(state))
    return model


def build_teacher(cfg: DictConfig):
    """The frozen Hugging Face teacher named by ``cfg.distill.teacher``, or None without distill."""
    if cfg.get("distill") is None:
        return None
    if float(cfg.distill.get("alpha", 0.5)) == 0.0:
        # S6 (no KD): the loss is plain CE either way; without this the 8B forward still ran every
        # step and S6's s/step and peak GB carried it (screen attempt 1).
        print("teacher: none (distill.alpha=0)", flush=True)
        return None
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, cfg.distill.get("dtype", "bfloat16"))
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.distill.teacher,
        revision=cfg.distill.get("revision", "main"),  # the snapshot fetch_hf.sh put in the cache
        dtype=dtype,
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    )
    return teacher.eval().requires_grad_(False)


def build_module(cfg: DictConfig, model: HybridLanguageModel, teacher) -> PretrainLightningModule:
    """The Lightning module with the model yaml's optimisation keys and the distill config's KD terms."""
    distill = cfg.get("distill") or {}
    m = cfg.model
    return PretrainLightningModule(
        model,
        teacher=teacher,
        kd_alpha=distill.get("alpha", 0.5),
        kd_temperature=distill.get("temperature", 2.0),
        loss_slab=distill.get("slab", 512),
        learning_rate=m.get("learning_rate", 3e-4),
        weight_decay=m.get("weight_decay", 0.1),
        warmup_steps=m.get("warmup_steps", 2000),
        max_steps=cfg.trainer.max_steps,
        scheduler_name=m.get("scheduler", "wsd"),
        gradient_clip_val=m.get("gradient_clip_val", 1.0),
        compile_model=bool(cfg.trainer.get("compile_model", False)),
        beta2_schedule=m.get("beta2_schedule", False),
        adam_beta2=m.get("adam_beta2", 0.999),
    )


def build_dataloaders(cfg: DictConfig, vocab_size: int):
    """Train/val loaders: seeded synthetic rows, or the packed-shard mixture."""
    d = cfg.dataset
    if d.name == "packed":
        from lexhybrid.data.datasets import ScheduleSampler, ShardSampler, mixture_from_config

        train, val = mixture_from_config(d, "train"), mixture_from_config(d, "val")
        # The training order is the mixture's seeded schedule, read on from where a resumed run
        # stands (ScheduleSampler, bound to the trainer in run(); Lightning's own DDP sampler is off).
        return (
            DataLoader(
                train, batch_size=d.batch_size, sampler=ScheduleSampler(len(train), d.batch_size),
                drop_last=True, num_workers=d.num_workers,
            ),
            DataLoader(val, batch_size=d.eval_batch_size, sampler=ShardSampler(len(val)), num_workers=d.num_workers),
        )  # fmt: skip
    if d.name != "synthetic":
        raise ValueError(f"unknown dataset {d.name!r} (synthetic or packed)")
    common = dict(
        row_len=d.row_len, vocab_size=vocab_size, min_doc_len=d.min_doc_len, max_doc_len=d.max_doc_len
    )
    train = SyntheticPackedDataset(num_rows=d.num_rows, seed=d.seed, **common)
    val = SyntheticPackedDataset(num_rows=max(d.eval_batch_size, 2), seed=d.seed + 1_000_000, **common)
    return (
        DataLoader(train, batch_size=d.batch_size, shuffle=True, num_workers=d.num_workers),
        DataLoader(val, batch_size=d.eval_batch_size, num_workers=d.num_workers),
    )


def build_callbacks(cfg: DictConfig, enable_checkpointing: bool) -> list:
    callbacks = [SignalCheckpointCallback(cfg.checkpoint_dir)]
    stats = cfg.callbacks.get("step_stats")
    if stats is not None and stats.get("enabled", True):
        callbacks.append(StepStatsCallback(stats.get("every_n_steps", 500), stats.get("skip_steps", 5)))
    diag = cfg.callbacks.get("diagnostics")
    if diag is not None and diag.get("enabled", False):
        callbacks.append(
            SpikeDiagnosticsCallback(
                diag.get("every_n_steps", 10), diag.get("spike_factor", 5.0), diag.get("stop_at_step", 0)
            )
        )
    ck = cfg.callbacks.checkpoint
    if enable_checkpointing:
        callbacks.append(FinalCheckpointCallback(cfg.checkpoint_dir))  # the final step, not only every n
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                monitor=ck.monitor,
                mode=ck.mode,
                save_top_k=ck.save_top_k,
                save_last=ck.save_last,
                filename=ck.filename,
                auto_insert_metric_name=ck.auto_insert_metric_name,
                every_n_train_steps=ck.every_n_train_steps,
            )
        )
    if cfg.trainer.get("stop_at_step"):
        callbacks.append(StopAtStepCallback(cfg.trainer.stop_at_step))
    if cfg.callbacks.lr_monitor.enabled and cfg.get("logger", True):
        callbacks.append(LearningRateMonitor(logging_interval=cfg.callbacks.lr_monitor.logging_interval))
    return callbacks


def is_launcher_rank() -> bool:
    """True in the process the wrapper started: Lightning's DDP launcher starts ranks 1..N-1 by
    re-running this script with ``LOCAL_RANK`` set, so only rank 0 lacks it (or has 0)."""
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def print_arch(cfg: DictConfig) -> str:
    """The ``ARCH`` line of the model ``cfg.model`` builds (on ``meta``: no weights, no memory)."""
    with torch.device("meta"):
        line = HybridLanguageModel(HybridConfig.from_hydra(cfg.model)).architecture_fingerprint()
    print(line, flush=True)
    return line


def check_stage_keys(cfg: DictConfig) -> None:
    """Stage A: ``trainer.stop_at_step`` ends training at a checkpoint step inside the LR
    schedule's horizon ``trainer.max_steps`` (WSD over the combined A+B horizon), so ``last.ckpt`` is
    that step and stage B resumes from it."""
    stop, horizon = cfg.trainer.get("stop_at_step"), int(cfg.trainer.max_steps)
    every = int(cfg.callbacks.checkpoint.every_n_train_steps)
    if stop is None:
        return
    if not 0 < int(stop) <= horizon:
        raise ValueError(f"trainer.stop_at_step={stop} must lie in (0, max_steps={horizon}]")
    if int(stop) % every:
        raise ValueError(
            f"trainer.stop_at_step={stop} is not a checkpoint step (every {every}): last.ckpt would miss it"
        )


def run(cfg: DictConfig) -> pl.Trainer | None:
    """The whole pipeline, callable from tests with a composed config."""
    if cfg.get("arch_only"):
        print_arch(cfg)
        return None
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)
    for key in ("output_dir", "checkpoint_dir", "log_dir"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)
    if is_launcher_rank():  # the DDP ranks re-run this script; one run_metadata.json, from rank 0
        write_run_metadata(cfg, cfg.output_dir, extra={"entrypoint": "scripts/train_pretrain.py"})

    model = build_model(cfg)
    print(model.architecture_fingerprint(), flush=True)  # wrappers grep this at step 0
    module = build_module(cfg, model, build_teacher(cfg))
    check_stage_keys(cfg)
    train_dl, val_dl = build_dataloaders(cfg, model.config.vocab_size)

    kwargs = build_trainer_kwargs(cfg.trainer)
    if cfg.dataset.name == "packed":
        kwargs["use_distributed_sampler"] = False  # ScheduleSampler / ShardSampler split the ranks
    use_logger = cfg.get("logger", True)
    trainer = pl.Trainer(
        callbacks=build_callbacks(cfg, kwargs.get("enable_checkpointing", True)),
        logger=TensorBoardLogger(save_dir=cfg.log_dir, name=cfg.experiment_name) if use_logger else False,
        **kwargs,
    )
    expected = int(cfg.trainer.val_every_opt_steps) * int(cfg.trainer.get("accumulate_grad_batches", 1))
    assert trainer.val_check_interval == expected, (trainer.val_check_interval, expected)
    if hasattr(train_dl.sampler, "bind"):  # reads on from Lightning's restored batch count
        train_dl.sampler.bind(trainer)

    ckpt_path = cfg.get("resume_from_checkpoint")
    if ckpt_path and not Path(ckpt_path).exists():
        raise FileNotFoundError(f"resume_from_checkpoint={ckpt_path} does not exist")
    trainer.fit(module, train_dl, val_dl, ckpt_path=ckpt_path or None)
    return trainer


@hydra.main(version_base="1.3", config_path=str(REPO_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    sys.exit(main())
