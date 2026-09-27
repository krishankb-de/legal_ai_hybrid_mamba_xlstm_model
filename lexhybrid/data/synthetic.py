"""Seeded random-token rows with packed-document ids, for smoke tests and profiling.

Rows look like real packed pretraining rows: documents of random length separated by an EOS id,
with ``doc_ids`` incremented after each EOS (the convention of ``lexhybrid/data/packing.py`` in
P3), so every code path that consumes ``doc_ids`` is exercised without any data on disk.
"""

import torch
from torch.utils.data import Dataset


class SyntheticPackedDataset(Dataset):
    """``num_rows`` rows of ``row_len`` random tokens, each packed from several documents.

    Args:
        num_rows: dataset length.
        row_len: tokens per row.
        vocab_size: token ids are drawn from ``[0, vocab_size - 1)``; ``vocab_size - 1`` is EOS
            unless ``eos_id`` is given.
        min_doc_len, max_doc_len: document length range (EOS included).
        seed: base seed; row ``i`` is generated from ``seed + i``, so any index is reproducible.
        eos_id: end-of-document token id.
    """

    def __init__(
        self,
        num_rows: int = 64,
        row_len: int = 128,
        vocab_size: int = 1024,
        min_doc_len: int = 8,
        max_doc_len: int = 96,
        seed: int = 0,
        eos_id: int | None = None,
    ):
        if min_doc_len < 2 or max_doc_len < min_doc_len:
            raise ValueError("need 2 <= min_doc_len <= max_doc_len")
        self.num_rows = num_rows
        self.row_len = row_len
        self.vocab_size = vocab_size
        self.min_doc_len = min_doc_len
        self.max_doc_len = max_doc_len
        self.seed = seed
        self.eos_id = vocab_size - 1 if eos_id is None else eos_id

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.num_rows:
            raise IndexError(index)
        g = torch.Generator().manual_seed(self.seed + index)
        input_ids = torch.empty(self.row_len, dtype=torch.long)
        doc_ids = torch.empty(self.row_len, dtype=torch.long)
        pos, doc = 0, 0
        while pos < self.row_len:
            length = int(torch.randint(self.min_doc_len, self.max_doc_len + 1, (1,), generator=g))
            end = min(pos + length, self.row_len)
            body = torch.randint(0, self.vocab_size - 1, (end - pos,), generator=g)
            if end - pos == length:
                body[-1] = self.eos_id  # a completed document ends on EOS
            input_ids[pos:end] = body
            doc_ids[pos:end] = doc
            pos, doc = end, doc + 1
        return {"input_ids": input_ids, "labels": input_ids.clone(), "doc_ids": doc_ids}
