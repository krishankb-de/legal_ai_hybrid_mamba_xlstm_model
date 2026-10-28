"""EOS-delimited packing into fixed-length rows and parquet shards (plan P3-U).

Every document is tokenized (``encode_document``: no special token can come from its text) and
followed by Qwen3's ``<|endoftext|>`` (151,643). Documents are laid end to end into rows of
``row_len`` tokens; ``doc_ids`` numbers the documents inside a row from 0 and steps **after** the EOS,
so the EOS belongs to the document it ends (the model predicts it) and the loss masks only the
prediction across the boundary (P2). A document longer than the rest of a row carries on into the
next row (as that row's document 0): nothing is dropped. The last row of a source is filled with EOS
tokens that join its last document; its ``length`` column says how many tokens are real, and
training masks the rest.

One source per shard, so a row has one ``source`` and one licence; the mixture (P3-V) weights
sources by drawing their rows. Columns: ``input_ids`` and ``doc_ids`` (``row_len`` int32 each),
``length``, ``source``, ``licence`` (the documents' licences joined by ``+`` if they differ),
``commercial_safe`` (every document in the row is), ``research_only`` (any is), ``documents`` (the
``Document.id`` of each local document id).

    python -m lexhybrid.data.packing --in data/scrubbed/bger.jsonl --row-len 4096
"""

import argparse
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from lexhybrid.data.schema import Document

REPO_ROOT = Path(__file__).resolve().parents[2]
ROW_LENS = (4096, 8192)
ROWS_PER_SHARD = 1000


@dataclass
class Row:
    input_ids: list[int]
    doc_ids: list[int]
    length: int
    source: str
    licences: list[str] = field(default_factory=list)
    commercial_safe: bool = True
    research_only: bool = False
    documents: list[str] = field(default_factory=list)

    @property
    def licence(self) -> str:
        return "+".join(sorted(set(self.licences)))


class Packer:
    """Rows of ``row_len`` tokens from one source's documents, fed one tokenised document at a time
    (``pack`` is this over an iterable). One token stream can feed several packers -- one per row
    length and split -- so the at-scale build (P4-L) tokenises every document once."""

    def __init__(self, row_len: int, eos_id: int):
        self.row_len, self.eos_id = row_len, eos_id
        self.source: str | None = None
        self.row: Row | None = None

    def add(self, doc: Document, tokens: list[int]) -> Iterator[Row]:
        """The rows ``doc`` completes; ``tokens`` is its encoding followed by the EOS."""
        if self.source is None:
            self.source = doc.source
        elif doc.source != self.source:
            raise ValueError(f"one source per pack: {self.source!r} then {doc.source!r}")
        pos = 0
        while pos < len(tokens):
            if self.row is None:
                self.row = Row([], [], 0, self.source)
            row = self.row
            local = row.doc_ids[-1] + 1 if row.doc_ids else 0
            take = tokens[pos : pos + self.row_len - len(row.input_ids)]
            row.input_ids += take
            row.doc_ids += [local] * len(take)
            row.documents.append(doc.id)
            row.licences.append(doc.licence)
            row.commercial_safe &= doc.commercial_safe
            row.research_only |= doc.research_only
            pos += len(take)
            if len(row.input_ids) == self.row_len:
                row.length = self.row_len
                self.row = None
                yield row

    def finish(self) -> Iterator[Row]:
        """The last row, padded with EOS tokens that join its last document."""
        row, self.row = self.row, None
        if row is not None and row.input_ids:
            row.length = len(row.input_ids)
            pad = self.row_len - row.length
            row.input_ids += [self.eos_id] * pad
            row.doc_ids += [row.doc_ids[-1]] * pad
            yield row


def pack(
    documents: Iterable[Document], encode: Callable[[str], list[int]], eos_id: int, row_len: int
) -> Iterator[Row]:
    """Rows of ``row_len`` tokens from the documents of one source, in order."""
    packer = Packer(row_len, eos_id)
    for doc in documents:
        yield from packer.add(doc, [*encode(doc.text), eos_id])
    yield from packer.finish()


def _schema():
    import pyarrow as pa

    return pa.schema([
        ("input_ids", pa.list_(pa.int32())), ("doc_ids", pa.list_(pa.int32())), ("length", pa.int32()),
        ("source", pa.string()), ("licence", pa.string()), ("commercial_safe", pa.bool_()),
        ("research_only", pa.bool_()), ("documents", pa.list_(pa.string())),
    ])  # fmt: skip


class ShardWriter:
    """Parquet shards ``shard-00000.parquet``... of at most ``rows_per_shard`` rows, written as rows
    arrive (``write_shards`` is this over an iterable); ``close`` writes the last, partial shard."""

    def __init__(self, out_dir: Path, rows_per_shard: int = ROWS_PER_SHARD):
        self.out_dir, self.rows_per_shard = Path(out_dir), rows_per_shard
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.paths: list[Path] = []
        self._batch: list[Row] = []
        self._schema = _schema()

    def write(self, row: Row) -> None:
        self._batch.append(row)
        if len(self._batch) == self.rows_per_shard:
            self._flush()

    def _flush(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.out_dir / f"shard-{len(self.paths):05d}.parquet"
        cols = {name: [getattr(r, name) for r in self._batch] for name in self._schema.names}
        pq.write_table(pa.table(cols, schema=self._schema), path)
        self.paths.append(path)
        self._batch.clear()

    def close(self) -> list[Path]:
        if self._batch:
            self._flush()
        return self.paths


def write_shards(rows: Iterable[Row], out_dir: Path, rows_per_shard: int = ROWS_PER_SHARD) -> list[Path]:
    """Parquet shards ``shard-00000.parquet``... of at most ``rows_per_shard`` rows."""
    writer = ShardWriter(out_dir, rows_per_shard)
    for row in rows:
        writer.write(row)
    return writer.close()


def read_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def main(argv=None) -> int:
    from lexhybrid.data.tokenizer import EOS_ID, encode_document, load_tokenizer

    parser = argparse.ArgumentParser(description="Pack one source's documents into parquet shards.")
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="documents, one JSON per line")
    parser.add_argument("--row-len", type=int, choices=ROW_LENS, default=4096)
    parser.add_argument("--out", type=Path, help="default data/shards/<row-len>/<source>")
    parser.add_argument("--max-rows", type=int, help="stop after this many rows")
    args = parser.parse_args(argv)
    docs = [
        Document.from_json(line) for line in args.in_path.read_text(encoding="utf-8").splitlines() if line
    ]
    tok = load_tokenizer(with_specials=False)
    rows = pack(docs, lambda t: encode_document(tok, t), EOS_ID, args.row_len)
    if args.max_rows:
        rows = (r for _, r in zip(range(args.max_rows), rows))
    out = args.out or REPO_ROOT / "data" / "shards" / str(args.row_len) / docs[0].source
    paths = write_shards(rows, out)
    print(f"{docs[0].source}: {len(docs)} documents -> {len(paths)} shard(s) in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
