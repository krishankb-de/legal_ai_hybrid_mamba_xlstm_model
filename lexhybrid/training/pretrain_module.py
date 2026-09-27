"""Pretraining Lightning module: slab-wise CE, optional logit KD, MTP, and a real NaN guard (P2-U).

Differences from the reference's stage-0 distillation module (defect 12):

* The loss is ``lexhybrid.training.distill.slab_ce_kl`` over ``model.backbone`` and ``model.head``:
  (B, L, V) logits never exist, the document-boundary mask applies to CE and KL alike, and a teacher
  (optional) sees packed rows as separate documents. MTP adds ``mtp_loss_weight * mtp_loss``.
* A non-finite loss **skips the step**: ``training_step`` returns None, which makes Lightning skip
  the backward pass and the optimizer step. The reference returned
  ``torch.tensor(0.0, requires_grad=True)`` -- a leaf with no graph -- so AdamW still stepped (and
  applied weight decay) on stale gradients. Under DDP every rank takes the same decision (an
  all-reduce of the flag), or the ranks would deadlock. A non-finite gradient norm clears the
  gradients before the step, which AdamW then skips parameter by parameter.
* ``compile_model`` is honoured (the reference's script read ``compile``): the backbone is compiled.
* The teacher is held outside the module's registered children, so it is neither checkpointed
  (a 1.7B teacher would add 3.4 GB to every ``last.ckpt``) nor wrapped by DDP.

Validation is plain CE (no teacher, no MTP), as in the reference.
"""

import torch
import torch.distributed as dist

from lexhybrid.models.hybrid_lm import HybridLanguageModel, boundary_masked_labels
from lexhybrid.training.distill import slab_ce_kl, teacher_forward_packed
from lexhybrid.training.lightning_module import HybridLightningModule
from lexhybrid.training.metrics import compute_perplexity


class PretrainLightningModule(HybridLightningModule):
    """Args: those of ``HybridLightningModule`` plus

    teacher: optional frozen Hugging Face causal LM (decision 8); None trains on CE alone.
    kd_alpha, kd_temperature: decision 9 (0.5, 2.0).
    loss_slab: positions per slab of the CE/KL computation.
    max_nan_skips: consecutive non-finite steps tolerated before the run is stopped.
    """

    def __init__(
        self,
        model: HybridLanguageModel,
        teacher=None,
        kd_alpha: float = 0.5,
        kd_temperature: float = 2.0,
        loss_slab: int = 512,
        max_nan_skips: int = 50,
        compile_model: bool = False,
        **kwargs,
    ):
        super().__init__(model, compile_model=False, **kwargs)
        self.save_hyperparameters(ignore=["model", "teacher"])
        self._backbone = torch.compile(model.backbone) if compile_model else model.backbone
        # Not an nn.Module attribute on purpose: not checkpointed, not moved by DDP, never trained.
        self.__dict__["_teacher"] = teacher
        if teacher is not None:
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
        self.kd_alpha = float(kd_alpha)
        self.kd_temperature = float(kd_temperature)
        self.loss_slab = int(loss_slab)
        self.max_nan_skips = int(max_nan_skips)
        self.nan_skips = 0
        self._consecutive_nan = 0

    @property
    def teacher(self):
        return self.__dict__["_teacher"]

    def on_fit_start(self):
        if self.teacher is not None:
            self.teacher.to(self.device)

    # -- losses -----------------------------------------------------------------------------------

    def compute_losses(self, batch: dict[str, torch.Tensor], use_teacher: bool, use_mtp: bool) -> dict:
        input_ids = batch["input_ids"]
        doc_ids = batch.get("doc_ids")
        labels = batch.get("labels", input_ids)
        residual, _ = self._backbone(input_ids, doc_ids=doc_ids)
        targets = boundary_masked_labels(labels, doc_ids)
        teacher_hidden = teacher_head = None
        if use_teacher and self.teacher is not None:
            teacher_hidden = teacher_forward_packed(self.teacher, input_ids, doc_ids)[:, :-1]
            teacher_head = self.teacher.get_output_embeddings()
        out = slab_ce_kl(
            residual[:, :-1],
            targets,
            teacher_hidden,
            self.model.head,
            teacher_head,
            alpha=self.kd_alpha,
            T=self.kd_temperature,
            slab=self.loss_slab,
        )
        if use_mtp and self.model.mtp_head is not None:
            out["mtp"] = self.model.mtp_head.loss(
                residual,
                input_ids,
                self.model.embeddings.token_embedding,
                self.model.head,
                doc_ids=doc_ids,
                labels=labels,
                slab=self.loss_slab,
            )
            out["loss"] = out["loss"] + self.model.config.mtp_loss_weight * out["mtp"]
        return out

    def _all_finite(self, loss: torch.Tensor) -> bool:
        finite = torch.isfinite(loss.detach()).to(torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        return bool(finite.item())

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        out = self.compute_losses(batch, use_teacher=True, use_mtp=True)
        loss = out["loss"]
        if not self._all_finite(loss):
            self.nan_skips += 1
            self._consecutive_nan += 1
            self.log("train/nan_skips", float(self.nan_skips), on_step=True)
            if self._consecutive_nan > self.max_nan_skips:
                raise RuntimeError(f"{self._consecutive_nan} consecutive non-finite losses; stopping the run")
            return None  # Lightning skips backward and the optimizer step for this batch
        self._consecutive_nan = 0
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/ce", out["ce"], on_step=True)
        self.log("train/perplexity", compute_perplexity(out["ce"]), on_step=True)
        if self.teacher is not None:
            self.log("train/kl", out["kl"], on_step=True)
        if "mtp" in out:
            self.log("train/mtp_loss", out["mtp"], on_step=True)
        self.log("train/n_supervised_tokens", out["n_supervised"].float(), on_step=True, reduce_fx="sum")
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        out = self.compute_losses(batch, use_teacher=False, use_mtp=False)
        n = int(out["n_supervised"])
        self.log("val/loss", out["ce"], prog_bar=True, on_epoch=True, batch_size=max(n, 1), sync_dist=True)
        self.log(
            "val/perplexity",
            compute_perplexity(out["ce"]),
            on_epoch=True,
            batch_size=max(n, 1),
            sync_dist=True,
        )
        return out["ce"]

    def on_before_optimizer_step(self, optimizer):
        """Clip and log the gradient norm; a non-finite norm clears the gradients (step skipped)."""
        if self.gradient_clip_val <= 0:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gradient_clip_val)
        self.log("train/grad_norm", grad_norm, on_step=True)
        if not torch.isfinite(grad_norm):
            for p in self.parameters():
                p.grad = None
            self.nan_skips += 1
            self.log("train/nan_skips", float(self.nan_skips), on_step=True)
