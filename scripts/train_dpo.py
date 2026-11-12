#!/usr/bin/env python3
"""Preference optimisation (DPO or KTO) of an SFT model on verifier-labelled pairs.

    python scripts/train_dpo.py model=hybrid_legal_base trainer=h100_multi_ddp +dpo=default \\
        dpo.sft_checkpoint=<sft run>/checkpoints/last.ckpt dpo.pairs=<pairs>/train.jsonl \\
        dpo.val_pairs=<pairs>/dev.jsonl [dpo.loss=kto dpo.beta=0.1 dpo.learning_rate=5e-7] \\
        [dpo.hf_model=Qwen/Qwen3-0.6B-Base] [+resume_from_checkpoint=<run>/checkpoints/last.ckpt]

The pairs come from ``scripts/build_dpo_pairs.py``. Policy and reference both start from
``dpo.sft_checkpoint``, every key loaded and none missing (``load_sft_model``): the hybrid of
``cfg.model`` or, with ``dpo.hf_model``, the Hugging Face baseline behind ``CausalLMAdapter``. The
reference is a frozen copy held outside the module (``DPOLightningModule``).

``resume_from_checkpoint`` restores the policy, the optimizer and the step from a run's own
``last.ckpt`` (the wrapper passes it on a requeue), and the reference is rebuilt from the SFT
checkpoint. A checkpoint saved with another loss, beta, learning rate or SFT checkpoint is refused:
its ``last.ckpt`` belongs to another run of a sweep.

The training pairs are read on a seeded schedule of ``max_steps x accumulate x world size x
dpo.batch_size`` pairs (``PairSchedule``, ``ScheduleSampler``), so a requeue continues exactly, and
``trainer.max_steps`` counts global optimizer steps. Pairs that cannot fill one optimizer step stop the
run, and so do validation pairs that leave a rank without a batch: KTO drops a short last batch, whose
mismatched rows could be a single pair matched with itself.

The trainer yaml, the DDP strategy (one SLURM task, ranks started by Lightning) and
``val_every_opt_steps`` are ``train_pretrain.py``'s. With ``dpo.val_pairs``, as for SFT,
``best_val_checkpoint`` keeps the ``callbacks.checkpoint.save_top_k`` best-``val/loss`` checkpoints and
``last.ckpt`` follows it, both saved at validation end; without, the step-based ``ModelCheckpoint``
writes ``last.ckpt``.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from lexhybrid.config import HybridConfig
from lexhybrid.data.datasets import ScheduleSampler, ShardSampler
from lexhybrid.training.callbacks import ValidationEndCheckpoint
from lexhybrid.training.dpo_module import DPOLightningModule, PairSchedule, collate_pairs, load_sft_model
from lexhybrid.training.sft_module import best_val_checkpoint
from lexhybrid.utils.checkpoint import check_run_settings
from lexhybrid.utils.run_metadata import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent
RESUME_MUST_MATCH = ("loss", "beta", "learning_rate", "sft_checkpoint")


def _script(name: str):
    """A sibling script as a module: ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tp = _script("train_pretrain")


def build_policy(cfg: DictConfig) -> nn.Module:
    d = cfg.dpo
    config = None if d.hf_model else HybridConfig.from_hydra(cfg.model)
    return load_sft_model(d.sft_checkpoint, config, d.hf_model, d.hf_revision)


def build_module(cfg: DictConfig, policy: nn.Module) -> DPOLightningModule:
    d, m = cfg.dpo, cfg.model
    return DPOLightningModule(
        policy, copy.deepcopy(policy), loss=d.loss, beta=d.beta, desirable_weight=d.desirable_weight,
        undesirable_weight=d.undesirable_weight, sft_checkpoint=str(d.sft_checkpoint),
        learning_rate=d.learning_rate, weight_decay=d.weight_decay, warmup_steps=d.warmup_steps,
        max_steps=cfg.trainer.max_steps, scheduler_name=d.scheduler,
        gradient_clip_val=m.get("gradient_clip_val", 1.0), adam_beta2=m.get("adam_beta2", 0.999),
    )  # fmt: skip


def build_callbacks(cfg: DictConfig, enable_checkpointing: bool) -> list:
    """Pretraining's callbacks. With dev pairs its step-based checkpoint is replaced by the
    best-``val/loss`` checkpoints and then ``last.ckpt``, both saved at validation end: a ``last.ckpt``
    saved at the step, before its validation, would resume past that validation with the ranking of
    the one before. Without dev pairs nothing validates and the step-based checkpoint writes
    ``last.ckpt`` only."""
    top_k = int(cfg.callbacks.checkpoint.save_top_k)
    validates = bool(cfg.dpo.get("val_pairs"))
    if top_k and not validates:
        raise ValueError("callbacks.checkpoint.save_top_k ranks by val/loss: give dpo.val_pairs")
    # no monitor: Lightning defers a monitored step save until the metric exists, so a kill would lose it
    last_only = OmegaConf.merge(cfg, {"callbacks": {"checkpoint": {"save_top_k": 0, "monitor": None}}})
    callbacks = tp.build_callbacks(last_only, enable_checkpointing)
    if not validates:
        return callbacks
    callbacks = [c for c in callbacks if not isinstance(c, ModelCheckpoint)]
    if enable_checkpointing:
        if top_k:
            callbacks.append(best_val_checkpoint(cfg.checkpoint_dir, top_k))
        callbacks.append(ValidationEndCheckpoint(cfg.checkpoint_dir))  # after the ranking above
    return callbacks


def read_pairs(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def build_dataloaders(cfg: DictConfig, trainer):
    """The training schedule's loader and the dev loader, or None without ``dpo.val_pairs``."""
    d = cfg.dpo
    kto = d.loss == "kto"
    if kto and d.batch_size < 2:
        raise ValueError("KTO's reference point mismatches the pairs of a batch: dpo.batch_size must be >= 2")
    pairs = read_pairs(d.pairs)
    world, accumulate = trainer.world_size, trainer.accumulate_grad_batches
    if len(pairs) < world * accumulate * d.batch_size:
        raise ValueError(
            f"{d.pairs}: {len(pairs)} pairs cannot fill one optimizer step ({world} ranks x {accumulate} "
            f"micro-batches x dpo.batch_size {d.batch_size})"
        )
    n = trainer.max_steps * accumulate * world * d.batch_size
    train = PairSchedule(pairs, num_samples=n, seed=cfg.seed)
    common = dict(batch_size=d.batch_size, collate_fn=collate_pairs, num_workers=d.num_workers)
    train_dl = DataLoader(
        train, sampler=ScheduleSampler(len(train), d.batch_size).bind(trainer), drop_last=True, **common
    )
    if not d.get("val_pairs"):
        return train_dl, None
    val = PairSchedule(read_pairs(d.val_pairs))
    if len(val) < (world * d.batch_size if kto else 1):
        raise ValueError(f"{d.val_pairs}: {len(val)} pairs leave a rank without a validation batch")
    return train_dl, DataLoader(val, sampler=ShardSampler(len(val)), drop_last=kto, **common)


def check_resume(ckpt_path, module: DPOLightningModule) -> None:
    """Refuse a checkpoint trained with other settings; the header alone is read (``mmap``)."""
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True).get(
        "hyper_parameters", {}
    )
    differ = {k: (saved.get(k), module.hparams[k]) for k in RESUME_MUST_MATCH if saved.get(k) != module.hparams[k]}  # fmt: skip
    if differ:
        raise ValueError(
            f"{ckpt_path} was trained with other settings (saved, now): {differ}; give this run its own EXPERIMENT"
        )


def run(cfg: DictConfig) -> pl.Trainer:
    """The whole run, callable from tests with a composed config."""
    if cfg.get("dpo") is None:
        raise ValueError("no dpo config: pass +dpo=default")
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)
    for key in ("output_dir", "checkpoint_dir", "log_dir"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)
    if tp.is_launcher_rank():
        write_run_metadata(cfg, cfg.output_dir, extra={"entrypoint": "scripts/train_dpo.py"})

    policy = build_policy(cfg)
    print(f"HF {cfg.dpo.hf_model}@{cfg.dpo.hf_revision}" if cfg.dpo.hf_model else policy.architecture_fingerprint(), flush=True)  # fmt: skip
    module = build_module(cfg, policy)
    ckpt_path = cfg.get("resume_from_checkpoint")
    if ckpt_path:
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"resume_from_checkpoint={ckpt_path} does not exist")
        check_resume(ckpt_path, module)
    kwargs = tp.build_trainer_kwargs(cfg.trainer)
    kwargs["use_distributed_sampler"] = False  # ScheduleSampler / ShardSampler split the ranks
    trainer = pl.Trainer(
        callbacks=build_callbacks(cfg, kwargs.get("enable_checkpointing", True)),
        logger=TensorBoardLogger(save_dir=cfg.log_dir, name=cfg.experiment_name) if cfg.get("logger", True) else False,
        **kwargs,
    )  # fmt: skip
    settings = {k: v for k, v in OmegaConf.to_container(cfg.dpo, resolve=True).items() if k != "num_workers"}
    settings |= {"seed": cfg.seed, "max_steps": trainer.max_steps,
                 "accumulate_grad_batches": trainer.accumulate_grad_batches}  # fmt: skip
    check_run_settings(cfg.output_dir, settings, bool(ckpt_path), tp.is_launcher_rank())
    train_dl, val_dl = build_dataloaders(cfg, trainer)
    trainer.fit(module, train_dl, val_dl, ckpt_path=ckpt_path or None)
    return trainer


@hydra.main(version_base="1.3", config_path=str(REPO_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    sys.exit(main())
