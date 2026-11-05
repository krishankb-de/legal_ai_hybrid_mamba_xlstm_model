"""Supervised fine-tuning: cross-entropy on the answer tokens only, for the hybrid and a Hugging Face baseline.

An SFT item is ``sft_format.encode``'s output: the prompt (question, passages, ``<|answer|>``) has
label -100 and the target -- text, pointer tokens and the final EOS -- is supervised. ``collate``
right-pads a batch of such items (labels -100). One example per row needs no ``doc_ids``: they
would be all 0, which is plain causal attention, and with ``doc_ids=None`` the attention layers keep
SDPA's causal kernel (the document-mask flex path is compiled for fixed shapes and would recompile at
every new padded length). Right padding never reaches a real position of a causal model.

The loss is ``slab_cross_entropy`` over ``model.backbone`` and ``model.head``, so (B, L, vocab)
logits never exist; there is no teacher and no multi-token-prediction term. Optimizer, schedule,
gradient clipping and the ``OPTIM`` line are ``HybridLightningModule``'s. ``CausalLMAdapter`` gives a
Hugging Face causal LM the same ``backbone``/``head`` split, so both models share this loop.

The data are ``scripts/build_sft_data.py``'s parquet splits. ``SFTSplit`` with ``num_samples``
serves a seeded schedule (one permutation per pass over the items), so sample ``i`` is always the
same item and ``ScheduleSampler`` resumes a requeued run where it stopped.
"""

from types import SimpleNamespace

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import Dataset

from lexhybrid.data.tokenizer import BASE_LEN, SPECIALS
from lexhybrid.models.slab_loss import slab_cross_entropy
from lexhybrid.training.lightning_module import HybridLightningModule


def best_val_checkpoint(dirpath, save_top_k: int = 1) -> ModelCheckpoint:
    """The ``save_top_k`` lowest-``val/loss`` checkpoints, ``best-step<N>.ckpt``, saved at the end of
    each validation. A step-interval ``ModelCheckpoint`` saves before that step's validation, so it
    pairs each score with the weights one validation later and its best path is off by one."""
    return ModelCheckpoint(
        dirpath=dirpath,
        monitor="val/loss",
        mode="min",
        save_top_k=save_top_k,
        filename="best-step{step:06d}",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )


class SFTSplit(Dataset):
    """One split (``train.parquet`` / ``dev.parquet``) as ``{input_ids, labels}`` items, held as flat
    int32 arrays. ``num_samples`` items on a seeded schedule, or every item once, in order."""

    def __init__(self, path, num_samples: int | None = None, seed: int = 0):
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["input_ids", "labels"])
        self.columns = {}
        for name in ("input_ids", "labels"):
            col = table.column(name).combine_chunks()
            self.columns[name] = (col.offsets.to_numpy(), col.values.to_numpy())
        n = table.num_rows
        if num_samples is None:
            self.index = np.arange(n)
        else:
            if not n:
                raise ValueError(f"{path}: no items to train on")
            rng = np.random.default_rng(seed)
            passes = -(-num_samples // n)
            self.index = np.concatenate([rng.permutation(n) for _ in range(passes)])[:num_samples]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        k = int(self.index[i])
        return {
            name: torch.from_numpy(values[offsets[k] : offsets[k + 1]].astype(np.int64))
            for name, (offsets, values) in self.columns.items()
        }


def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad ``input_ids`` (with 0: no real position sees a pad) and ``labels`` (with -100)."""
    width = max(len(it["input_ids"]) for it in items)
    input_ids = torch.zeros(len(items), width, dtype=torch.long)
    labels = torch.full((len(items), width), -100, dtype=torch.long)
    for row, it in enumerate(items):
        input_ids[row, : len(it["input_ids"])] = it["input_ids"]
        labels[row, : len(it["labels"])] = it["labels"]
    return {"input_ids": input_ids, "labels": labels}


# Every attention and MLP projection of a Qwen3 layer.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def wrap_lora(lm, rank: int):
    """``lm`` frozen but for rank-``rank`` LoRA adapters on ``LORA_TARGETS`` and the pointer specials'
    rows of the embedding and the output layer. ``lora_alpha = rank``: the adapters are not rescaled.
    Both layers are named: an untied output layer would otherwise never learn to emit a pointer."""
    from peft import LoraConfig, get_peft_model

    pointers = list(range(BASE_LEN, BASE_LEN + len(SPECIALS)))
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=list(LORA_TARGETS),
        trainable_token_indices={"embed_tokens": pointers, "lm_head": pointers},
    )
    return get_peft_model(lm, config)


class CausalLMAdapter(nn.Module):
    """A Hugging Face causal LM behind the hybrid's interfaces: the ``backbone``/``head`` split (the
    decoder's last hidden state, already through its final norm, and the output layer) and the cached
    decode of ``lexhybrid.decoding.generate`` (``allocate_inference_cache``, ``prefill``,
    ``step_logits``), advancing the model's KV cache with the head on the last position only."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.config = SimpleNamespace(vocab_size=lm.config.vocab_size)

    def backbone(self, input_ids: torch.Tensor):
        return self.lm.get_decoder()(input_ids=input_ids, use_cache=False).last_hidden_state, None

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.get_output_embeddings()(hidden)

    def supports_cached_decode(self) -> bool:
        return True

    @property
    def lm_head(self) -> nn.Module:
        return self.lm.get_output_embeddings()

    def embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(input_ids)

    def allocate_inference_cache(
        self, batch_size: int, device=None, dtype=None, max_seq_len: int = 0
    ) -> dict:
        return {"kv": None}  # the model's dynamic cache, from prefill on

    def _advance(self, inputs_embeds: torch.Tensor, caches: dict) -> torch.Tensor:
        out = self.lm.get_decoder()(inputs_embeds=inputs_embeds, past_key_values=caches["kv"], use_cache=True)
        caches["kv"] = out.past_key_values
        return self.head(out.last_hidden_state[:, -1])

    def prefill(
        self, inputs_embeds: torch.Tensor, caches: dict, doc_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if doc_ids is not None:
            raise ValueError("doc_ids: a Hugging Face model has no document boundaries")
        return self._advance(inputs_embeds, caches)

    def step_logits(self, embedding: torch.Tensor, caches: dict) -> torch.Tensor:
        return self._advance(embedding.unsqueeze(1), caches)


class SFTLightningModule(HybridLightningModule):
    """Args: those of ``HybridLightningModule`` (``learning_rate`` is the plain value the LR sweep
    scales) plus ``loss_slab``, positions per slab of the CE. ``model`` is a ``HybridLanguageModel``
    or a ``CausalLMAdapter``."""

    def __init__(self, model: nn.Module, loss_slab: int = 512, **kwargs):
        super().__init__(model, **kwargs)
        self.save_hyperparameters(ignore=["model"])
        if getattr(model, "mtp_head", None) is not None:  # outside the loss: DDP would wait for its gradients
            model.mtp_head.requires_grad_(False)
        self.loss_slab = int(loss_slab)

    def answer_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mean CE over the supervised targets, their count)``; position t predicts label t + 1."""
        residual, _ = self.model.backbone(batch["input_ids"])
        ce_sum, n = slab_cross_entropy(
            residual[:, :-1], batch["labels"][:, 1:], self.model.head, self.loss_slab
        )
        return ce_sum / n.clamp(min=1), n

    def _shared_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.answer_loss(batch)[0]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        ce, n = self.answer_loss(batch)
        n = int(n)
        self.log("val/loss", ce, prog_bar=True, on_epoch=True, batch_size=max(n, 1), sync_dist=True)
        self._accumulate_val(ce, n)  # val/perplexity = exp(token-weighted mean CE), at epoch end
        return ce
