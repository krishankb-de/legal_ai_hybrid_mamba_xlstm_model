"""PyTorch Lightning wrapper for causal-LM training (ported base class of the reference).

P1 keeps the reference behaviour: manual gradient clipping in ``on_before_optimizer_step``
(Lightning's own clipping is not used), WSD with an absolute warmup, the beta2 anneal keyed on
``global_step``, and cosine/linear/constant alternatives wrapped in a linear warmup. P2-U adds the
pretraining module (KD, NaN guard that skips the step, the ``val_check_interval`` product rule).
"""

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR

from lexhybrid.models.hybrid_lm import HybridLanguageModel
from lexhybrid.training.metrics import compute_perplexity
from lexhybrid.training.optimizer import configure_optimizer
from lexhybrid.training.schedulers import WSDScheduler, apply_beta2_schedule


class HybridLightningModule(pl.LightningModule):
    """Training loop around a ``HybridLanguageModel``.

    Args:
        model: the model to train.
        learning_rate, weight_decay: AdamW settings.
        warmup_steps: absolute warmup (WSD keeps its decay ratio and absorbs the rest).
        max_steps: schedule horizon (optimizer steps).
        optimizer_name: ``"adamw"`` | ``"adam"`` | ``"sgd"``.
        scheduler_name: ``"wsd"`` | ``"cosine"`` | ``"linear"`` | ``"constant"``.
        gradient_clip_val: max global grad norm, clipped manually before each optimizer step.
        compile_model: wrap the model in ``torch.compile``.
        beta2_schedule, beta2_start, beta2_end: anneal AdamW beta2 across the WSD decay phase.
        adam_beta2: AdamW's beta2 (beta1 stays 0.9). The legal configs set 0.95 (decisions.p5_fix):
            at 0.999 the second moment lags ~700 steps, and one KD-amplified gradient spike gave
            the update that collapsed screen runs S1-s42 and S3-s43.
    """

    def __init__(
        self,
        model: HybridLanguageModel,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        optimizer_name: str = "adamw",
        scheduler_name: str = "cosine",
        gradient_clip_val: float = 1.0,
        compile_model: bool = False,
        beta2_schedule: bool = False,
        beta2_start: float = 0.999,
        beta2_end: float = 0.974,
        adam_beta2: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = torch.compile(model) if compile_model else model

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.gradient_clip_val = gradient_clip_val
        self.beta2_schedule = bool(beta2_schedule)
        self.beta2_start = float(beta2_start)
        self.beta2_end = float(beta2_end)
        self.adam_beta2 = float(adam_beta2)
        if self.beta2_schedule and self.adam_beta2 != self.beta2_start:
            raise ValueError(
                f"beta2_schedule anneals from beta2_start={self.beta2_start}, but adam_beta2={self.adam_beta2}"
            )
        # Populated by _build_wsd_scheduler when scheduler_name == "wsd".
        self._wsd_decay_start: int | None = None
        self._wsd_decay_steps: int | None = None

    def forward(self, input_ids: torch.Tensor, **kwargs):
        return self.model(input_ids, **kwargs)

    def _shared_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        outputs = self.model(input_ids, labels=labels, doc_ids=batch.get("doc_ids"), return_dict=True)
        return outputs.loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/perplexity", compute_perplexity(loss), prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        labels = batch.get("labels", batch["input_ids"])
        self._accumulate_val(loss, int((labels[..., 1:] != -100).sum()))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_sums = torch.zeros(2, dtype=torch.float64, device=self.device)  # CE x tokens, tokens

    def _accumulate_val(self, ce: torch.Tensor, n: int) -> None:
        """Token-weighted sums for the epoch's ``val/perplexity``. Logged per batch with
        ``on_epoch=True`` it was the mean of per-batch exp(CE), which Jensen's inequality inflates
        (screen attempt 1: 31.69 logged against exp(mean CE) = 8.19, job 2636358)."""
        self._val_sums += torch.stack(
            [
                ce.detach().double() * n,
                torch.tensor(float(n), dtype=torch.float64, device=self._val_sums.device),
            ]
        )

    def on_validation_epoch_end(self) -> None:
        sums = self._val_sums.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums)
        if sums[1] > 0:
            self.log("val/perplexity", compute_perplexity(sums[0] / sums[1]).float(), prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/perplexity", compute_perplexity(loss), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        betas = (0.9, self.adam_beta2)
        optimizer = configure_optimizer(
            self.model,
            optimizer_name=self.optimizer_name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=betas,
        )
        print(
            f"OPTIM {self.optimizer_name} lr={self.learning_rate} betas={betas} weight_decay={self.weight_decay}",
            flush=True,
        )
        if self.scheduler_name == "wsd":
            # WSD owns its warmup; it is not wrapped in the warmup SequentialLR below.
            scheduler = self._build_wsd_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        if self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.max_steps - self.warmup_steps, eta_min=self.learning_rate * 0.1
            )
        elif self.scheduler_name == "linear":
            scheduler = LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.max_steps - self.warmup_steps
            )
        else:
            scheduler = ConstantLR(optimizer, factor=1.0)
        if self.warmup_steps > 0:
            warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_steps)
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, scheduler], milestones=[self.warmup_steps]
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_before_optimizer_step(self, optimizer):
        """Clip gradients and log the pre-clip norm (manual, so foreach AdamW and AMP coexist)."""
        if self.gradient_clip_val > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gradient_clip_val)
            self.log("train/grad_norm", grad_norm, on_step=True)

    def _build_wsd_scheduler(self, optimizer: Optimizer) -> WSDScheduler:
        """A WSD scheduler with an absolute warmup; records the decay window for the beta2 anneal."""
        abs_warmup = int(self.warmup_steps) if self.warmup_steps else None
        sched = WSDScheduler(optimizer, max_steps=self.max_steps, warmup_steps=abs_warmup)
        self._wsd_decay_start = sched.decay_start
        self._wsd_decay_steps = sched.decay_steps
        return sched

    def on_train_batch_start(self, batch, batch_idx):
        """Apply the beta2 schedule when WSD and ``beta2_schedule`` are both active."""
        if not (self.beta2_schedule and self.scheduler_name == "wsd"):
            return
        if self._wsd_decay_start is None or self._wsd_decay_steps is None:
            return
        b2 = apply_beta2_schedule(
            self.trainer.optimizers[0],
            step=int(self.global_step),
            decay_start=self._wsd_decay_start,
            decay_steps=self._wsd_decay_steps,
            beta2_start=self.beta2_start,
            beta2_end=self.beta2_end,
        )
        self.log("train/adam_beta2", b2, on_step=True)
