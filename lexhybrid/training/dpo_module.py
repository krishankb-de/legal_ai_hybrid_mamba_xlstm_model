"""Preference optimisation of an SFT model: DPO and KTO against a frozen reference copy.

Both losses read one number per sequence, the summed log-probability of its answer tokens under the
policy and under the reference (``answer_logps``: prompt and padding positions excluded):

* **DPO** (Rafailov et al. 2023), on (chosen, rejected) pairs of one prompt::

      -logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))

  log 2 while the policy equals the reference; it falls as the policy raises the chosen answer over
  the rejected one relative to the reference.
* **KTO** (Ethayarajh et al. 2024), unpaired: with ``r = pi - ref``, a desirable answer costs
  ``w_d * (1 - sigmoid(beta * (r - z0)))`` and an undesirable one ``w_u * (1 - sigmoid(beta * (z0 - r)))``.
  The reference point ``z0`` estimates KL(policy || reference) as the batch mean of ``r`` over
  *mismatched* rows (prompt i with the chosen answer of pair i + 1), gathered over the ranks, clamped
  at 0 and detached. A pair file feeds KTO with every chosen answer desirable and every rejected one
  undesirable; the mismatch needs at least two pairs per batch.

``answer_logps`` applies the head only at the positions that predict an answer token, so the
(B, L, V) logits at the 151,936-token vocabulary never exist: an answer is a few hundred of the
prompt's thousands of positions. The head and the log-softmax run in fp32 with autocast off: the
losses read differences of sums over hundreds of tokens, which bf16's few significant digits would
swamp. Rows are right-padded, which a causal model never sees.

The reference is held outside the module's registered children (as the KD teacher is): it is not
checkpointed, not wrapped by DDP, never put in train mode and never trained. A resumed run restores
the policy from its ``last.ckpt`` and rebuilds the reference from the SFT checkpoint.

``load_sft_model`` builds the model both the pair builder and the training start from: the hybrid or,
for the Hugging Face baseline, a ``CausalLMAdapter``, with every key of the checkpoint loaded and none
missing. ``PairSchedule`` serves the pairs on a seeded schedule, so sample ``i`` is always the same
pair and ``ScheduleSampler`` resumes a requeued run where it stopped.
"""

import os
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from lexhybrid.config import HybridConfig
from lexhybrid.data.tokenizer import BASE_LEN, EOS_ID, SPECIALS
from lexhybrid.models.hybrid_lm import HybridLanguageModel
from lexhybrid.training.lightning_module import HybridLightningModule
from lexhybrid.training.sft_module import CausalLMAdapter, wrap_lora
from lexhybrid.utils.checkpoint import load_every_key, strip_prefixes

LOSSES = ("dpo", "kto")


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Mean DPO loss over pairs; every argument is (B,) summed answer log-probabilities."""
    margin = (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)
    return -F.logsigmoid(beta * margin).mean()


def kto_loss(
    policy: torch.Tensor,
    reference: torch.Tensor,
    desirable: torch.Tensor,
    kl: torch.Tensor,
    beta: float,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> torch.Tensor:
    """Mean KTO loss over (N,) answers; ``desirable`` is (N,) bool, ``kl`` the reference point's
    raw estimate (clamped at 0 and detached here)."""
    r = policy - reference
    z0 = kl.detach().clamp(min=0)
    good = desirable_weight * (1 - torch.sigmoid(beta * (r - z0)))
    bad = undesirable_weight * (1 - torch.sigmoid(beta * (z0 - r)))
    return torch.where(desirable, good, bad).mean()


def answer_logps(head, residual: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """(B,) summed log-probability of each row's answer tokens.

    ``residual`` (B, L, D) is the stream before ``head`` (``model.backbone``); ``labels`` (B, L) holds
    the answer token ids and -100 on prompt and padding (``sft_format.encode``'s layout). Position t
    predicts ``labels[t + 1]``.
    """
    targets = labels[:, 1:]
    mask = targets != -100
    with torch.autocast(device_type=residual.device.type, enabled=False):
        logits = head(residual[:, :-1][mask].float())  # (N, V), answer-predicting positions only
        logp = -F.cross_entropy(logits, targets[mask], reduction="none")
    return logp.new_zeros(labels.shape[0]).index_add(0, mask.nonzero()[:, 0], logp)


def load_sft_model(
    checkpoint, config: HybridConfig | None = None, hf_model: str | None = None, hf_revision: str = "main"
) -> nn.Module:
    """The SFT model of ``checkpoint``: the hybrid of ``config``, or a ``CausalLMAdapter`` over the
    architecture of ``hf_model``; a key missing on either side raises."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = strip_prefixes(state.get("state_dict", state))
    if hf_model is None:
        model = HybridLanguageModel(config)
        load_every_key(model, state, checkpoint)
        return model
    from transformers import AutoConfig, AutoModelForCausalLM

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    hf_config = AutoConfig.from_pretrained(hf_model, revision=hf_revision, local_files_only=offline)
    lm = AutoModelForCausalLM.from_config(
        hf_config, dtype=torch.float32
    )  # the weights come from the checkpoint
    embedding = lm.get_input_embeddings()
    rank = next((v.shape[0] for k, v in state.items() if ".lora_A." in k), 0)
    if rank:  # the keys name peft's wrappers: resized as the SFT run did, then wrapped the same way
        if embedding.num_embeddings < BASE_LEN + len(SPECIALS):
            lm.resize_token_embeddings(BASE_LEN + len(SPECIALS))
        lm = wrap_lora(lm, rank)
    else:
        weights = state.get(next(n for n, p in lm.named_parameters() if p is embedding.weight))
        if weights is not None and embedding.num_embeddings != weights.shape[0]:
            lm.resize_token_embeddings(
                weights.shape[0]
            )  # the SFT run resized it to hold the pointer specials
    load_every_key(lm, state, checkpoint)
    return CausalLMAdapter(lm)


class PairSchedule(Dataset):
    """``num_samples`` pairs on a seeded schedule (one permutation per pass), or every pair once, in order."""

    def __init__(self, pairs: Sequence[dict], num_samples: int | None = None, seed: int = 0):
        self.pairs = pairs
        if num_samples is None:
            self.index = np.arange(len(pairs))
        else:
            rng = np.random.default_rng(seed)
            passes = -(-num_samples // len(pairs))
            self.index = np.concatenate([rng.permutation(len(pairs)) for _ in range(passes)])[:num_samples]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        return self.pairs[int(self.index[i])]


def _pad(rows: Sequence[tuple[Sequence[int], Sequence[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    """(prompt, answer) rows as right-padded ``input_ids`` and ``labels`` (-100 off the answer)."""
    length = max(len(p) + len(a) for p, a in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for i, (p, a) in enumerate(rows):
        ids[i, : len(p) + len(a)] = torch.tensor([*p, *a])
        labels[i, len(p) : len(p) + len(a)] = torch.tensor(list(a))
    return {"input_ids": ids, "labels": labels}


def collate_pairs(pairs: Sequence[dict], pad_id: int = EOS_ID) -> dict[str, torch.Tensor]:
    """A batch of B pair rows (``prompt_ids``, ``chosen_ids``, ``rejected_ids``): ``input_ids`` and
    ``labels`` hold the B chosen rows, then the B rejected rows; ``kl_input_ids`` / ``kl_labels`` the
    B mismatched rows of KTO's reference point (prompt i, chosen answer of pair (i + 1) mod B)."""
    prompts = [p["prompt_ids"] for p in pairs]
    answers = [p["chosen_ids"] for p in pairs] + [p["rejected_ids"] for p in pairs]
    kl_rows = [(prompts[i], pairs[(i + 1) % len(pairs)]["chosen_ids"]) for i in range(len(pairs))]
    kl = {f"kl_{k}": v for k, v in _pad(kl_rows, pad_id).items()}
    return {**_pad(list(zip(prompts * 2, answers)), pad_id), **kl}


class DPOLightningModule(HybridLightningModule):
    """Args: those of ``HybridLightningModule`` plus

    reference: the frozen reference model (a copy of the SFT model the policy starts from).
    loss: ``"dpo"`` | ``"kto"``.
    beta: the losses' inverse temperature.
    desirable_weight, undesirable_weight: KTO's class weights.
    sft_checkpoint: where policy and reference came from; a saved hyperparameter, so a resume can
        refuse a checkpoint of another run.

    ``model`` and ``reference`` are ``HybridLanguageModel``s or ``CausalLMAdapter``s.
    """

    def __init__(
        self,
        model: nn.Module,
        reference: nn.Module,
        loss: str = "dpo",
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
        sft_checkpoint: str | None = None,
        **kwargs,
    ):
        if loss not in LOSSES:
            raise ValueError(f"loss {loss!r} is not one of {LOSSES}")
        super().__init__(model, **kwargs)
        self.save_hyperparameters(ignore=["model", "reference"])
        if getattr(model, "mtp_head", None) is not None:  # outside the loss: DDP would wait for its gradients
            model.mtp_head.requires_grad_(False)
        # Not an nn.Module attribute on purpose: not checkpointed, not moved by DDP, never trained.
        self.__dict__["_reference"] = reference.eval().requires_grad_(False)
        self.loss_name = loss
        self.beta = float(beta)
        self.desirable_weight = float(desirable_weight)
        self.undesirable_weight = float(undesirable_weight)

    @property
    def reference(self) -> nn.Module:
        return self.__dict__["_reference"]

    def on_fit_start(self):
        self.reference.to(self.device)

    @staticmethod
    def _logps(model, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return answer_logps(model.head, model.backbone(input_ids)[0], labels)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The loss and the logged rewards (``beta * (pi - ref)``) of a ``collate_pairs`` batch."""
        policy = self._logps(self.model, batch["input_ids"], batch["labels"])
        with torch.no_grad():
            reference = self._logps(self.reference, batch["input_ids"], batch["labels"])
        b = policy.shape[0] // 2
        rewards = self.beta * (policy - reference).detach()
        out = {
            "reward_margin": (rewards[:b] - rewards[b:]).mean(),
            "reward_accuracy": (rewards[:b] > rewards[b:]).float().mean(),
        }
        if self.loss_name == "dpo":
            out["loss"] = dpo_loss(policy[:b], policy[b:], reference[:b], reference[b:], self.beta)
            return out
        with torch.no_grad():
            kl_policy = self._logps(self.model, batch["kl_input_ids"], batch["kl_labels"])
            kl_reference = self._logps(self.reference, batch["kl_input_ids"], batch["kl_labels"])
        out["kl"] = self.all_gather((kl_policy - kl_reference).mean()).mean()
        desirable = torch.arange(2 * b, device=policy.device) < b
        out["loss"] = kto_loss(
            policy, reference, desirable, out["kl"], self.beta, self.desirable_weight, self.undesirable_weight
        )
        return out

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out, pairs = self.compute_loss(batch), batch["input_ids"].shape[0] // 2
        for name, value in out.items():
            self.log(f"train/{name}", value, prog_bar=name == "loss", on_step=True, batch_size=pairs)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return out["loss"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out, pairs = self.compute_loss(batch), batch["input_ids"].shape[0] // 2
        for name, value in out.items():
            self.log(
                f"val/{name}", value, prog_bar=name == "loss", on_epoch=True, sync_dist=True, batch_size=pairs
            )
        return out["loss"]
