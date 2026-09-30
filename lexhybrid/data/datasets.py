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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from lexhybrid.data.packing import Packer, ShardWriter
from lexhybrid.data.schema import Document

VAL_FRACTION = 0.001
MIN_VAL_DOCS = 2000
ARMS = ("commercial_safe", "research")
SPLITS = ("train", "val")


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
    val_ids = split_by_document([d.id for d in docs], fraction, minimum, seed)
    metas = build_source_shards_stream(
        docs, val_ids, encode, eos_id, (row_len,), root, fraction, minimum, seed, extra
    )
    return metas[row_len]


def build_source_shards_stream(
    docs: Iterable[Document],
    val_ids: set[str],
    encode: Callable[[str], list[int]],
    eos_id: int,
    row_lens: Sequence[int],
    root: Path,
    fraction: float = VAL_FRACTION,
    minimum: int = MIN_VAL_DOCS,
    seed: int = 0,
    extra: dict | None = None,
) -> dict[int, dict]:
    """``build_source_shards`` for a source that does not fit in memory (P4-L): ONE pass over its
    documents, each tokenised once and packed at every row length, rows written to their shards as
    they complete. ``val_ids`` is ``split_by_document`` over the source's ids (a first pass, ids
    only). The shards and ``meta.json`` of each row length are exactly ``build_source_shards``'."""
    packers = {(n, s): Packer(n, eos_id) for n in row_lens for s in SPLITS}
    writers: dict[tuple[int, str], ShardWriter] = {}
    counts = {(n, s): {"rows": 0, "tokens": 0} for n in row_lens for s in SPLITS}
    ids: dict[str, list[str]] = {s: [] for s in SPLITS}
    source, licences, commercial_safe, research_only = None, set(), True, False

    def emit(key, rows):
        for row in rows:
            writers[key].write(row)
            counts[key]["rows"] += 1
            counts[key]["tokens"] += row.length

    for doc in docs:
        if source is None:
            source = doc.source
            writers.update(
                {(n, s): ShardWriter(root / str(n) / source / s) for n in row_lens for s in SPLITS}
            )
        elif doc.source != source:
            raise ValueError(f"one source per call, got {source!r} and {doc.source!r}")
        split = "val" if doc.id in val_ids else "train"
        ids[split].append(doc.id)
        licences.add(doc.licence)
        commercial_safe &= doc.commercial_safe
        research_only |= doc.research_only
        tokens = [*encode(doc.text), eos_id]
        for n in row_lens:
            emit((n, split), packers[(n, split)].add(doc, tokens))
    if source is None:
        raise ValueError("no documents")
    metas = {}
    for n in row_lens:
        meta = {
            "source": source, "row_len": n, "licences": sorted(licences), "commercial_safe": commercial_safe,
            "research_only": research_only, "val_fraction": fraction, "min_val_docs": minimum, "seed": seed,
            **(extra or {}),
        }  # fmt: skip
        for s in SPLITS:
            emit((n, s), packers[(n, s)].finish())
            writers[(n, s)].close()
            meta[s] = {"documents": len(ids[s]), **counts[(n, s)], "document_ids": sorted(ids[s])}
        out = root / str(n) / source
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
        metas[n] = meta
    return metas


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


def shard_directory(root: Path, row_len: int, source: str) -> tuple[Path, int]:
    """Where ``source``'s ``row_len``-token rows come from, and how many sub-rows a stored row gives.

    Shards packed at exactly ``row_len`` are read as they are (factor 1). Otherwise the smallest
    packed length that is a multiple of ``row_len`` serves each stored row as ``factor`` consecutive
    sub-rows: the P5 screen trains on 2,048-token rows cut from the 4,096-token shards (P4-L packs
    4,096 and 8,192 only). A cut is a stretch of the same packed stream; a sub-row that starts inside
    a document is an ordinary packed row (no position predicts into position 0).
    """
    root = Path(root)
    exact = root / str(row_len) / source
    if exact.is_dir():
        return exact, 1
    lengths = sorted(
        int(p.name)
        for p in (root.iterdir() if root.is_dir() else ())
        if p.name.isdigit() and int(p.name) > row_len and int(p.name) % row_len == 0 and (p / source).is_dir()
    )
    if not lengths:
        raise FileNotFoundError(
            f"no shards for {source} at {row_len} tokens or a multiple of it under {root}"
        )
    return root / str(lengths[0]) / source, lengths[0] // row_len


class SourceShards:
    """The rows of one source's split, read shard by shard on first use. With ``factor`` > 1 every
    stored row is served as ``factor`` consecutive sub-rows of ``stored_len / factor`` tokens."""

    def __init__(self, directory: Path, split: str, factor: int = 1):
        import pyarrow.parquet as pq

        self.meta = json.loads((directory / "meta.json").read_text())
        self.paths = sorted((directory / split).glob("shard-*.parquet"))
        if not self.paths:
            raise FileNotFoundError(f"no shards in {directory / split}")
        self.counts = [pq.ParquetFile(p).metadata.num_rows for p in self.paths]
        self.factor = int(factor)
        if self.factor < 1 or int(self.meta.get("row_len", self.factor)) % self.factor:
            raise ValueError(
                f"{directory}: stored rows of {self.meta.get('row_len')} tokens cannot be cut {factor} ways"
            )
        self._cache: dict[int, list[dict]] = {}

    def __len__(self) -> int:
        return sum(self.counts) * self.factor

    def _stored(self, k: int) -> dict:
        for i, n in enumerate(self.counts):
            if k < n:
                if i not in self._cache:
                    import pyarrow.parquet as pq

                    self._cache = {i: pq.read_table(self.paths[i]).to_pylist()}  # one shard in memory
                return self._cache[i][k]
            k -= n
        raise IndexError(k)

    def row(self, k: int) -> dict:
        if not 0 <= k < len(self):
            raise IndexError(k)
        stored, part = divmod(k, self.factor)
        row = self._stored(stored)
        if self.factor == 1:
            return row
        width = len(row["input_ids"]) // self.factor
        lo = part * width
        return {
            **row,
            "input_ids": row["input_ids"][lo : lo + width],
            "doc_ids": row["doc_ids"][lo : lo + width],
            "length": max(0, min(row["length"] - lo, width)),  # real tokens of this cut
        }


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
        self.sources = {}
        for name in sorted(weights):
            directory, factor = shard_directory(root, row_len, name)
            self.sources[name] = SourceShards(directory, split, factor)
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
