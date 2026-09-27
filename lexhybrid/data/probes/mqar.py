"""Multi-query associative recall, MQAR (plan P3-W), with the reference's ``configs/dataset/mqar.yaml``
semantics (Zoology, Arora et al. 2023): a vocabulary of 8,192 ids split into keys (lower half) and
values (upper half), 4 key-value pairs, 4 queries, query gaps drawn from a power law with
``a = 0.01``, a 2,048-token context, data seed 42.

A sequence opens with the pairs ``k1 v1 .. kN vN``; the rest is filler, into which each query is
placed as the key followed by its value at a power-law-distributed gap. The model is right on a
query when its prediction at the key's position is that key's value. The ids are shifted by
``id_offset`` into the student's ordinary vocabulary (no special token), and the filler comes from
the ids above the key and value ranges, so a filler token can never look like a query.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MQARConfig:
    vocab_size: int = 8192
    num_kv_pairs: int = 4
    num_queries: int = 4
    power_a: float = 0.01
    context_length: int = 2048
    seed: int = 42
    id_offset: int = 1000


@dataclass
class MQARItem:
    input_ids: list[int]
    query_positions: list[int]  # positions of the query keys: the value is the next token
    values: list[int]


REFERENCE = MQARConfig()  # configs/dataset/mqar.yaml of the reference


def generate(num_items: int, cfg: MQARConfig = REFERENCE) -> list[MQARItem]:
    if cfg.num_queries > cfg.num_kv_pairs:
        raise ValueError("each query asks for a different key: num_queries <= num_kv_pairs")
    rng = np.random.default_rng(cfg.seed)
    half, n = cfg.vocab_size // 2, cfg.num_kv_pairs
    space = (cfg.context_length - 2 * n) // 2
    p = cfg.power_a * np.arange(1, space + 1, dtype=float) ** (cfg.power_a - 1)
    p /= p.sum()
    items = []
    for _ in range(num_items):
        keys = cfg.id_offset + rng.choice(np.arange(1, half), n, replace=False)
        values = cfg.id_offset + rng.choice(np.arange(half, cfg.vocab_size), n, replace=False)
        context = np.empty(2 * n, dtype=np.int64)
        context[0::2], context[1::2] = keys, values
        filler = cfg.id_offset + cfg.vocab_size + rng.integers(0, cfg.vocab_size, cfg.context_length - 2 * n)
        gaps = rng.choice(space, cfg.num_queries, replace=False, p=p)
        asked = rng.choice(n, cfg.num_queries, replace=False)
        for g, j in zip(gaps, asked, strict=True):
            filler[2 * g], filler[2 * g + 1] = keys[j], values[j]
        seq = np.concatenate([context, filler])
        items.append(
            MQARItem(
                input_ids=seq.tolist(),
                query_positions=sorted(int(2 * n + 2 * g) for g in gaps),
                values=[int(seq[2 * n + 2 * g + 1]) for g in sorted(gaps)],
            )
        )
    return items


@torch.no_grad()
def score(model, items: list[MQARItem], batch_size: int = 8, device: str = "cpu") -> dict:
    """Accuracy over all queries; the logits are computed at the query positions only."""
    model.eval()
    correct = total = 0
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        ids = torch.tensor([it.input_ids for it in batch], device=device)
        hidden, _ = model.backbone(ids)
        for b, it in enumerate(batch):
            pred = model.head(hidden[b, it.query_positions]).argmax(-1).tolist()
            correct += sum(int(p == v) for p, v in zip(pred, it.values, strict=True))
            total += len(it.values)
    return {"accuracy": correct / total if total else float("nan"), "queries": total, "items": len(items)}
