"""Dense retrieval: ``BAAI/bge-m3`` vectors (the L2-normalised [CLS] state) and exact inner-product search.

An index directory holds ``vectors.npy`` (float16, one row per chunk, in ``ids.json`` order) and
``config.json``. ``DenseIndex.build`` streams the texts through the encoder into a memory-mapped
matrix, so millions of chunks never sit in memory at once, and ``load`` maps it back read-only.
Search is an exact matrix product, in blocks of corpus rows and of queries, keeping the top ``k``:
no approximate index, so a ranking is reproducible.

The encoder is any callable ``(texts) -> (n, dim)`` float array with unit rows, so tests use a stand-in;
``BgeM3Encoder`` loads the model on first use from the Hugging Face cache (offline on the cluster).
"""

import hashlib
import itertools
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from lexhybrid.retrieval.bm25 import Hit
from lexhybrid.retrieval.index import write_atomic

DENSE_MODEL = "BAAI/bge-m3"
VECTORS_NAME, IDS_NAME, CONFIG_NAME = "vectors.npy", "ids.json", "config.json"
PROGRESS_NAME = "progress.json"

Encoder = Callable[[Sequence[str]], np.ndarray]


class BgeM3Encoder:
    """bge-m3's dense vectors: the [CLS] state of the last layer, L2-normalised, in batches."""

    def __init__(self, model_name: str = DENSE_MODEL, device: str | None = None, batch_size: int = 64,
                 max_length: int = 512):  # fmt: skip
        self.model_name, self.device, self.batch_size, self.max_length = (
            model_name,
            device,
            batch_size,
            max_length,
        )
        self._model = self._tok = None

    def _load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device).eval()

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        if self._model is None:
            self._load()
        out = []
        for start in range(0, len(texts), self.batch_size):
            enc = self._tok(
                list(texts[start : start + self.batch_size]), padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )  # fmt: skip
            with torch.inference_mode():
                cls = self._model(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self._model.config.hidden_size), np.float32)


class DenseIndex:
    def __init__(self, vectors: np.ndarray, ids: list[str], config: dict):
        self.vectors, self.ids, self.config = vectors, ids, config
        self.resumed = 0  # rows an earlier, interrupted build had committed

    @classmethod
    def build(cls, directory: Path, ids: Sequence[str], texts: Iterable[str], encode: Encoder,
              batch_size: int = 1024, **config) -> "DenseIndex":  # fmt: skip
        """Embed ``texts`` (one per id, same order) into ``directory`` and load the result.

        Every batch is committed, so a build killed at its time limit continues from the last one
        when called again with the same ids and config; anything else starts over."""
        ids = list(ids)
        if not ids:
            raise ValueError("no chunks to index")
        if len(set(ids)) != len(ids):
            raise ValueError("chunk ids must be unique")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in (CONFIG_NAME, IDS_NAME):  # an interrupted rebuild must not look complete
            (directory / name).unlink(missing_ok=True)
        key = hashlib.sha256(json.dumps([ids, config], sort_keys=True).encode()).hexdigest()
        progress = directory / PROGRESS_NAME
        matrix, row, batch = None, 0, []
        if progress.exists() and (state := json.loads(progress.read_text()))["key"] == key:
            matrix, row = np.lib.format.open_memmap(directory / VECTORS_NAME, mode="r+"), state["row"]
        resumed = row

        def flush():
            nonlocal matrix, row
            vecs = np.asarray(encode(batch), dtype=np.float32)
            if matrix is None:
                matrix = np.lib.format.open_memmap(
                    directory / VECTORS_NAME, mode="w+", dtype=np.float16, shape=(len(ids), vecs.shape[1])
                )
            matrix[row : row + len(vecs)] = vecs
            row += len(vecs)
            batch.clear()
            matrix.flush()
            write_atomic(progress, json.dumps({"key": key, "row": row}))

        for text in itertools.islice(texts, row, None):
            batch.append(text)
            if len(batch) == batch_size:
                flush()
        if batch:
            flush()
        if row != len(ids):
            raise ValueError(f"{row} texts for {len(ids)} ids")
        matrix.flush()
        (directory / IDS_NAME).write_text(json.dumps(ids, ensure_ascii=False))
        config = {"kind": "dense", "n_chunks": len(ids), "dim": int(matrix.shape[1]), **config}
        (directory / CONFIG_NAME).write_text(json.dumps(config, indent=1) + "\n")
        progress.unlink()
        index = cls.load(directory)
        index.resumed = resumed
        return index

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        directory = Path(directory)
        vectors = np.load(directory / VECTORS_NAME, mmap_mode="r")
        ids = json.loads((directory / IDS_NAME).read_text())
        if len(ids) != len(vectors):
            raise ValueError(f"{directory}: {len(ids)} ids for {len(vectors)} vectors")
        return cls(vectors, ids, json.loads((directory / CONFIG_NAME).read_text()))

    def query_vectors(self, queries: np.ndarray, k: int = 10, block: int = 65536, query_block: int = 512):
        """The best ``k`` chunks per query vector, best first (ties: the earlier chunk)."""
        k = min(k, len(self.ids))
        out: list[list[Hit]] = []
        for q0 in range(0, len(queries), query_block):
            q = np.asarray(queries[q0 : q0 + query_block], dtype=np.float32)
            best_s = np.full((len(q), 0), -np.inf, np.float32)
            best_i = np.zeros((len(q), 0), np.int64)
            for start in range(0, len(self.ids), block):
                rows = np.asarray(self.vectors[start : start + block], dtype=np.float32)
                s = np.concatenate([best_s, q @ rows.T], axis=1)
                i = np.concatenate(
                    [best_i, np.broadcast_to(np.arange(start, start + len(rows)), (len(q), len(rows)))], 1
                )
                # Columns hold the kept best (already ordered) and then this block in chunk order, so a
                # stable sort ranks by (-score, chunk index): ties go to the earlier chunk, whatever the
                # block size (copies of a passage embed identically).
                keep = np.argsort(-s, axis=1, kind="stable")[:, :k]
                best_s, best_i = np.take_along_axis(s, keep, 1), np.take_along_axis(i, keep, 1)
            for scores, idx in zip(best_s, best_i, strict=True):
                out.append(
                    [Hit(self.ids[j], float(v), r) for r, (v, j) in enumerate(zip(scores, idx), start=1)]
                )
        return out

    def query(self, text: str, encode: Encoder, k: int = 10) -> list[Hit]:
        return self.query_vectors(encode([text]), k)[0]
