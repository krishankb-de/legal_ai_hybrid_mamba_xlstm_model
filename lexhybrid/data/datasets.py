"""Pretraining data over packed shards (plan P3-V): the split, the mixture and the arms.

**Split by document.** ``split_by_document`` puts a document in validation by a seeded hash of its
id: 0.1% of the documents, at least 2,000 (asserted; a corpus too small for that is an error, not a
smaller split). It runs before packing, so no document ever straddles train and validation.
``build_source_shards`` then packs each side into ``<root>/<row_len>/<source>/{train,val}/`` and
writes ``meta.json`` (documents, rows, tokens, licences, and whether every row is commercial-safe or
any is research-only).

**Mixture.** A source belongs to a group (``legal``, ``general``) with a weight; the groups take
their token shares (70 / 30, decision 13), each split among its sources by weight. Rows are fixed
length, so drawing rows in those proportions draws tokens in them. The training order is a seeded,
precomputed schedule: sample ``i`` is always the same row, so a run is reproducible and can resume
by starting at a sample offset. Validation reads every validation row of every admitted source once,
in order.

**Arms** (decision 2). ``commercial_safe`` admits only sources whose shards are commercial-safe and
free of research-only rows, and checks every row it serves; ``research`` admits everything. A
research-only source listed for the commercial-safe arm is an error, never a silent drop.
"""

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from lexhybrid.data.packing import pack, write_shards
from lexhybrid.data.schema import Document

VAL_FRACTION = 0.001
MIN_VAL_DOCS = 2000
ARMS = ("commercial_safe", "research")


def _unit(seed: int, key: str) -> float:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64


def split_by_document(
    ids: Sequence[str], fraction: float = VAL_FRACTION, minimum: int = MIN_VAL_DOCS, seed: int = 0
) -> set[str]:
    """The validation document ids: ``max(ceil(fraction * n), minimum)`` of them, by seeded hash."""
    n = len(ids)
    n_val = max(math.ceil(fraction * n), minimum)
    if n_val >= n:
        raise ValueError(
            f"{n} documents cannot give a validation split of {n_val} (fraction {fraction}, minimum {minimum})"
        )
    val = set(sorted(ids, key=lambda i: _unit(seed, i))[:n_val])
    assert minimum <= len(val) < n, (len(val), minimum, n)
    return val


def build_source_shards(
    docs: Sequence[Document],
    encode: Callable[[str], list[int]],
    eos_id: int,
    row_len: int,
    root: Path,
    fraction: float = VAL_FRACTION,
    minimum: int = MIN_VAL_DOCS,
    seed: int = 0,
    extra: dict | None = None,
) -> dict:
    """Split one source's documents, pack both sides, write the shards and ``meta.json`` (with
    ``extra``, e.g. whether the documents were scrubbed)."""
    sources = {d.source for d in docs}
    if len(sources) != 1:
        raise ValueError(f"one source per call, got {sorted(sources)}")
    (source,) = sources
    val_ids = split_by_document([d.id for d in docs], fraction, minimum, seed)
    out = root / str(row_len) / source
    meta = {
        "source": source, "row_len": row_len, "licences": sorted({d.licence for d in docs}),
        "commercial_safe": all(d.commercial_safe for d in docs), "research_only": any(d.research_only for d in docs),
        "val_fraction": fraction, "min_val_docs": minimum, "seed": seed, **(extra or {}),
    }  # fmt: skip
    for split, part in (
        ("train", [d for d in docs if d.id not in val_ids]),
        ("val", [d for d in docs if d.id in val_ids]),
    ):
        rows = list(pack(part, encode, eos_id, row_len))
        write_shards(rows, out / split)
        meta[split] = {"documents": len(part), "rows": len(rows), "tokens": sum(r.length for r in rows),
                       "document_ids": sorted(d.id for d in part)}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    return meta


@dataclass(frozen=True)
class SourceSpec:
    name: str
    group: str
    weight: float = 1.0
    arm: str = "commercial_safe"  # the least permissive arm the source may enter


def effective_weights(groups: dict[str, float], sources: Sequence[SourceSpec], arm: str) -> dict[str, float]:
    """Each admitted source's share of the rows: its group's share, split by weight within it
    (groups without an admitted source drop out and the rest are renormalised)."""
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} not in {ARMS}")
    admitted = [s for s in sources if arm == "research" or s.arm == "commercial_safe"]
    weights: dict[str, float] = {}
    for group, share in groups.items():
        members = [s for s in admitted if s.group == group and s.weight > 0]
        total = sum(s.weight for s in members)
        for s in members:
            weights[s.name] = share * s.weight / total
    z = sum(weights.values())
    if not z:
        raise ValueError("no source is admitted to the mixture")
    return {k: v / z for k, v in weights.items()}


class SourceShards:
    """The rows of one source's split, read shard by shard on first use."""

    def __init__(self, directory: Path, split: str):
        import pyarrow.parquet as pq

        self.meta = json.loads((directory / "meta.json").read_text())
        self.paths = sorted((directory / split).glob("shard-*.parquet"))
        if not self.paths:
            raise FileNotFoundError(f"no shards in {directory / split}")
        self.counts = [pq.ParquetFile(p).metadata.num_rows for p in self.paths]
        self._cache: dict[int, list[dict]] = {}

    def __len__(self) -> int:
        return sum(self.counts)

    def row(self, k: int) -> dict:
        for i, n in enumerate(self.counts):
            if k < n:
                if i not in self._cache:
                    import pyarrow.parquet as pq

                    self._cache = {i: pq.read_table(self.paths[i]).to_pylist()}  # one shard in memory
                return self._cache[i][k]
            k -= n
        raise IndexError(k)


class PackedMixture(Dataset):
    """Rows of the admitted sources as training items ``{input_ids, labels, doc_ids}``.

    ``split="train"``: ``num_samples`` rows drawn by ``weights`` on a seeded schedule (a source's
    rows are visited in a seeded order, reshuffled each time it wraps), starting at ``start``.
    ``split="val"``: every row of every source once. Labels past a row's ``length`` are -100."""

    def __init__(
        self,
        root: Path,
        row_len: int,
        split: str,
        weights: dict[str, float],
        arm: str,
        num_samples: int | None = None,
        seed: int = 0,
        start: int = 0,
    ):
        import numpy as np

        self.arm, self.row_len = arm, row_len
        self.sources = {
            name: SourceShards(Path(root) / str(row_len) / name, split) for name in sorted(weights)
        }
        for name, s in self.sources.items():
            if arm == "commercial_safe" and (s.meta["research_only"] or not s.meta["commercial_safe"]):
                raise ValueError(
                    f"{name}: shards are not commercial-safe and cannot enter the commercial_safe arm"
                )
        names = list(self.sources)
        if split == "val":
            self.index = [(n, k) for n in names for k in range(len(self.sources[n]))]
            return
        if num_samples is None:
            raise ValueError("a training mixture needs num_samples")
        rng = np.random.default_rng(seed)
        p = np.array([weights[n] for n in names], dtype=float)
        draws = rng.choice(len(names), size=start + num_samples, p=p / p.sum())
        seen = dict.fromkeys(names, 0)
        orders: dict[tuple[str, int], np.ndarray] = {}
        index = []
        for d in draws:
            name = names[d]
            n, k = len(self.sources[name]), seen[name]
            wrap = k // n
            if (name, wrap) not in orders:
                # seeded by the source's name, so adding a source leaves the others' orders alone
                key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
                orders[(name, wrap)] = np.random.default_rng([seed, key, wrap]).permutation(n)
            index.append((name, int(orders[(name, wrap)][k % n])))
            seen[name] = k + 1
        self.index = index[start:]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        name, k = self.index[i]
        row = self.sources[name].row(k)
        if self.arm == "commercial_safe" and (row["research_only"] or not row["commercial_safe"]):
            raise RuntimeError(
                f"{name} row {k} is not commercial-safe"
            )  # the meta check makes this unreachable
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        labels = input_ids.clone()
        labels[row["length"] :] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "doc_ids": torch.tensor(row["doc_ids"], dtype=torch.long),
        }


def source_specs(cfg_sources: dict) -> list[SourceSpec]:
    """``SourceSpec``s from a dataset yaml's ``sources`` block."""
    return [
        SourceSpec(name, s["group"], float(s.get("weight", 1.0)), s.get("arm", "commercial_safe"))
        for name, s in cfg_sources.items()
    ]


def mixture_from_config(d, split: str) -> PackedMixture:
    """The dataset a ``name: packed`` dataset yaml describes, for ``split`` train or val."""
    from omegaconf import OmegaConf

    groups = OmegaConf.to_container(d.groups, resolve=True) if not isinstance(d.groups, dict) else d.groups
    sources = (
        OmegaConf.to_container(d.sources, resolve=True) if not isinstance(d.sources, dict) else d.sources
    )
    weights = effective_weights(groups, source_specs(sources), d.arm)
    return PackedMixture(
        Path(d.shards_root), int(d.row_len), split, weights, d.arm,
        num_samples=int(d.train_samples) if split == "train" else None, seed=int(d.seed),
    )  # fmt: skip
