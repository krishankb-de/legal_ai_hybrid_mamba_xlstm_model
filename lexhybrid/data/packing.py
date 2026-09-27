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


def pack(
    documents: Iterable[Document], encode: Callable[[str], list[int]], eos_id: int, row_len: int
) -> Iterator[Row]:
    """Rows of ``row_len`` tokens from the documents of one source, in order."""
    row: Row | None = None
    source = None
    for doc in documents:
        if source is None:
            source = doc.source
        elif doc.source != source:
            raise ValueError(f"one source per pack: {source!r} then {doc.source!r}")
        tokens = [*encode(doc.text), eos_id]
        pos = 0
        while pos < len(tokens):
            if row is None:
                row = Row([], [], 0, source)
            local = row.doc_ids[-1] + 1 if row.doc_ids else 0
            take = tokens[pos : pos + row_len - len(row.input_ids)]
            row.input_ids += take
            row.doc_ids += [local] * len(take)
            row.documents.append(doc.id)
            row.licences.append(doc.licence)
            row.commercial_safe &= doc.commercial_safe
            row.research_only |= doc.research_only
            pos += len(take)
            if len(row.input_ids) == row_len:
                row.length = row_len
                yield row
                row = None
    if row is not None and row.input_ids:  # the last row: EOS padding joins the last document
        row.length = len(row.input_ids)
        pad = row_len - row.length
        row.input_ids += [eos_id] * pad
        row.doc_ids += [row.doc_ids[-1]] * pad
        yield row


def write_shards(rows: Iterable[Row], out_dir: Path, rows_per_shard: int = ROWS_PER_SHARD) -> list[Path]:
    """Parquet shards ``shard-00000.parquet``... of at most ``rows_per_shard`` rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())), ("doc_ids", pa.list_(pa.int32())), ("length", pa.int32()),
        ("source", pa.string()), ("licence", pa.string()), ("commercial_safe", pa.bool_()),
        ("research_only", pa.bool_()), ("documents", pa.list_(pa.string())),
    ])  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, batch = [], []

    def flush():
        path = out_dir / f"shard-{len(paths):05d}.parquet"
        cols = {name: [getattr(r, name) for r in batch] for name in schema.names}
        pq.write_table(pa.table(cols, schema=schema), path)
        paths.append(path)
        batch.clear()

    for row in rows:
        batch.append(row)
        if len(batch) == rows_per_shard:
            flush()
    if batch:
        flush()
    return paths


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
