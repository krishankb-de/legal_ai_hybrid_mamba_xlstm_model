#!/usr/bin/env python3
"""The Hugging Face baseline of supervised fine-tuning: ``sft.hf_model`` (``Qwen/Qwen3-0.6B-Base``)
trained by ``scripts/train_sft.py``'s ``run`` -- the same splits, schedule, steps and answer-token
loss -- through ``CausalLMAdapter``.

    python scripts/train_sft_hf.py sft.data_dir=data/sft trainer=h100_multi_ddp

The splits are token ids of the student's tokenizer (Qwen3's, plus the 87 pointer specials at ids
151,669..151,755), which the Qwen3 family shares; the model's embedding is resized when it has
fewer rows than that vocabulary. Weights load in fp32 (bf16 autocast trains them, as for the
hybrid), offline under ``HF_HUB_OFFLINE=1``.

``sft.lora_rank`` > 0 trains LoRA adapters instead (``wrap_lora``): every base weight frozen and
loaded in bf16, the adapters in fp32, with gradient checkpointing.

    python scripts/train_sft_hf.py sft.hf_model=Qwen/Qwen3-8B-Base sft.lora_rank=16 sft.learning_rate=1e-4
"""

import importlib.util
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from lexhybrid.data.tokenizer import BASE_LEN, SPECIALS
from lexhybrid.training.sft_module import CausalLMAdapter, wrap_lora

REPO_ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    """A sibling script as a module: ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_lm(cfg: DictConfig):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        cfg.sft.hf_model,
        revision=cfg.sft.hf_revision,
        dtype=torch.bfloat16 if cfg.sft.lora_rank else torch.float32,  # a frozen base at half the memory
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    )


def adapt(lm, lora_rank: int = 0) -> CausalLMAdapter:
    """The model with room for every pointer special, behind the hybrid's ``backbone``/``head``."""
    vocab = BASE_LEN + len(SPECIALS)
    if lm.get_input_embeddings().num_embeddings < vocab:
        lm.resize_token_embeddings(vocab)
    if lora_rank:
        # 36 layers of activations over 4 x 8,192 tokens do not fit one H100 without it
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lm = wrap_lora(lm, lora_rank)
        params = list(lm.parameters())
        trainable = sum(p.numel() for p in params if p.requires_grad)
        print(
            f"LORA rank={lora_rank} trainable={trainable:,} of {sum(p.numel() for p in params):,}", flush=True
        )
    return CausalLMAdapter(lm)


def run(cfg: DictConfig):
    print(f"HF {cfg.sft.hf_model}@{cfg.sft.hf_revision}", flush=True)
    return _script("train_sft").run(
        cfg, model=adapt(load_lm(cfg), cfg.sft.lora_rank), entrypoint="scripts/train_sft_hf.py"
    )


@hydra.main(version_base="1.3", config_path=str(REPO_ROOT / "configs"), config_name="sft/default")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    sys.exit(main())
