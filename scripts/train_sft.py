#!/usr/bin/env python3
"""Supervised fine-tuning entry point: answer-token CE on ``scripts/build_sft_data.py``'s splits.

    python scripts/train_sft.py sft.data_dir=data/sft lm_checkpoint=/path/to/pretrain/last.ckpt
    python scripts/train_sft.py ... trainer=h100_multi_ddp sft.learning_rate=8e-5 +resume_from_checkpoint=...

The config is ``configs/sft/default.yaml``: the pretraining model, trainer and callbacks groups with
the SFT keys. The pretraining entry point's machinery is reused: the trainer kwargs
(``val_every_opt_steps`` and the explicit DDP strategy), the callbacks (``last.ckpt``, the signal
checkpoint, ``STEPSTATS``), and ``run_metadata.json`` from the launcher rank. Three things differ:

* ``lm_checkpoint`` loads strictly: weights only (the SFT schedule starts at step 0), and any missing
  or unexpected key stops the run, because fine-tuning a partly loaded model fails without a word.
* The step-based ``ModelCheckpoint`` writes ``last.ckpt`` only; ``best_val_checkpoint`` keeps the
  ``callbacks.checkpoint.save_top_k`` best-dev-loss checkpoints, saved at validation end.
* ``resume_from_checkpoint`` resumes an SFT run, the requeue path.

``trainer.max_steps`` counts global optimizer steps: the training schedule holds
``max_steps x accumulate x world size x sft.batch_size`` items, read once (``ScheduleSampler``).
``scripts/train_sft_hf.py`` runs this same ``run`` on a Hugging Face model.
"""

import importlib.util
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from lexhybrid.config import HybridConfig
from lexhybrid.data.datasets import ScheduleSampler, ShardSampler
from lexhybrid.models.hybrid_lm import HybridLanguageModel
from lexhybrid.training.sft_module import SFTLightningModule, SFTSplit, best_val_checkpoint, collate
from lexhybrid.utils.checkpoint import check_run_settings, strip_prefixes
from lexhybrid.utils.run_metadata import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    """A sibling script as a module: ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tp = _script("train_pretrain")


def build_model(cfg: DictConfig) -> HybridLanguageModel:
    """The hybrid of ``cfg.model`` with the weights of ``cfg.lm_checkpoint``; every key must match."""
    if not cfg.get("lm_checkpoint"):
        raise ValueError("lm_checkpoint: SFT starts from pretrained weights")
    model = HybridLanguageModel(HybridConfig.from_hydra(cfg.model))
    state = torch.load(cfg.lm_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_prefixes(state.get("state_dict", state)), strict=True)
    return model


def build_callbacks(cfg: DictConfig, enable_checkpointing: bool) -> list:
    """Pretraining's callbacks with its step-based checkpoint writing ``last.ckpt`` only, plus the
    best-dev-loss checkpoint saved at validation end."""
    top_k = int(cfg.callbacks.checkpoint.save_top_k)
    # no monitor: Lightning defers a monitored step save until the metric exists, so a kill would lose it
    last_only = OmegaConf.merge(cfg, {"callbacks": {"checkpoint": {"save_top_k": 0, "monitor": None}}})
    callbacks = tp.build_callbacks(last_only, enable_checkpointing)
    if enable_checkpointing and top_k:
        callbacks.append(best_val_checkpoint(cfg.checkpoint_dir, top_k))
    return callbacks


def build_module(cfg: DictConfig, model) -> SFTLightningModule:
    s = cfg.sft
    return SFTLightningModule(
        model,
        loss_slab=s.loss_slab,
        learning_rate=s.learning_rate,
        weight_decay=s.weight_decay,
        warmup_steps=s.warmup_steps,
        max_steps=cfg.trainer.max_steps,
        scheduler_name=s.scheduler,
        gradient_clip_val=s.gradient_clip_val,
        adam_beta2=s.adam_beta2,
    )


def run(cfg: DictConfig, model=None, entrypoint: str = "scripts/train_sft.py") -> pl.Trainer:
    """The whole run; ``model`` None builds the hybrid of ``cfg.model`` from ``cfg.lm_checkpoint``."""
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)
    for key in ("output_dir", "checkpoint_dir", "log_dir"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)
    if tp.is_launcher_rank():  # the DDP ranks re-run this script; one run_metadata.json, from rank 0
        write_run_metadata(cfg, cfg.output_dir, extra={"entrypoint": entrypoint})
    if model is None:
        model = build_model(cfg)
        print(model.architecture_fingerprint(), flush=True)
    module = build_module(cfg, model)

    kwargs = tp.build_trainer_kwargs(cfg.trainer)
    kwargs["use_distributed_sampler"] = False  # ScheduleSampler / ShardSampler split the ranks
    logger = (
        TensorBoardLogger(save_dir=cfg.log_dir, name=cfg.experiment_name)
        if cfg.get("logger", True)
        else False
    )
    trainer = pl.Trainer(
        callbacks=build_callbacks(cfg, kwargs.get("enable_checkpointing", True)), logger=logger, **kwargs
    )
    s, data = cfg.sft, Path(cfg.sft.data_dir)
    n = trainer.max_steps * trainer.accumulate_grad_batches * trainer.world_size * s.batch_size
    train = SFTSplit(data / "train.parquet", num_samples=n, seed=cfg.seed)
    val = SFTSplit(data / "dev.parquet")
    train_dl = DataLoader(
        train, batch_size=s.batch_size, sampler=ScheduleSampler(len(train), s.batch_size).bind(trainer),
        drop_last=True, collate_fn=collate, num_workers=s.num_workers,
    )  # fmt: skip
    val_dl = DataLoader(
        val,
        batch_size=s.eval_batch_size,
        sampler=ShardSampler(len(val)),
        collate_fn=collate,
        num_workers=s.num_workers,
    )

    ckpt_path = cfg.get("resume_from_checkpoint")
    if ckpt_path and not Path(ckpt_path).exists():
        raise FileNotFoundError(f"resume_from_checkpoint={ckpt_path} does not exist")
    settings = {k: v for k, v in OmegaConf.to_container(cfg.sft, resolve=True).items() if k != "num_workers"}
    settings |= {"lm_checkpoint": cfg.get("lm_checkpoint"), "seed": cfg.seed, "max_steps": trainer.max_steps,
                 "accumulate_grad_batches": trainer.accumulate_grad_batches}  # fmt: skip
    check_run_settings(cfg.output_dir, settings, bool(ckpt_path), tp.is_launcher_rank())
    trainer.fit(module, train_dl, val_dl, ckpt_path=ckpt_path or None)
    return trainer


@hydra.main(version_base="1.3", config_path=str(REPO_ROOT / "configs"), config_name="sft/default")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    sys.exit(main())
