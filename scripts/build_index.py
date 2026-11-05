#!/usr/bin/env python3
"""Build a retrieval index from collected documents.

    .venv/bin/python scripts/build_index.py --kind bm25 --out data/index/smoke
    .venv/bin/python scripts/build_index.py --kind bm25 --docs data/scrubbed/gii.jsonl data/scrubbed/rii.jsonl \
        --out data/index/smoke
    .venv/bin/python scripts/build_index.py --kind dense --out data/index/smoke   # after the bm25 build

Documents come from scrubbed JSONL files (default: every ``data/scrubbed/*.jsonl`` of an indexed
type; the LER scrub runs before any index build, so a file under a ``raw`` directory is refused
unless ``--allow-unscrubbed``); scrub parts (``<source>.part-<k>-of-<n>.jsonl``) are read like whole
files. Statutes and regulations are chunked by provision, decisions by window; other
document types are not indexed. The index directory gets ``chunks.jsonl`` (every chunk, for the
prompt builder and the renderer), ``bm25/`` (``lexhybrid.retrieval.bm25.BM25Index``),
``index.json`` (what was indexed: sources, documents, chunks by type, settings) and
``documents.jsonl`` (``lexhybrid.retrieval.index.DocumentDigest`` of every input document of
an indexed type, which ``scripts/resync_index.py`` compares with the current documents).
``--kind dense`` then embeds every chunk of ``chunks.jsonl`` with bge-m3 into ``dense/``
(``lexhybrid.retrieval.dense.DenseIndex``) and adds its settings, size and speed to ``index.json``.

The retrieval index keeps every copy of a text: it is built from the collected documents, not from
the pretraining deduplication, so a passage held by two sources is found
in both.
"""

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from lexhybrid.data.schema import Document
from lexhybrid.retrieval.bm25 import BM25Index
from lexhybrid.retrieval.chunking import Chunk, chunk_document
from lexhybrid.retrieval.copies import copy_groups, read_dedup_logs
from lexhybrid.retrieval.dense import DENSE_MODEL, VECTORS_NAME, BgeM3Encoder, DenseIndex
from lexhybrid.retrieval.index import (
    CHUNKS_NAME,
    DOCUMENTS_NAME,
    INDEXED_TYPES,
    MANIFEST_NAME,
    document_digest,
    write_atomic,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
KINDS = ("bm25", "dense")


def read_documents(paths: list[Path]):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield Document.from_json(line)


def build(
    kind: str, paths: list[Path], out: Path, allow_unscrubbed: bool = False, dedup: list[Path] = ()
) -> dict:
    if kind != "bm25":
        raise SystemExit(f"--kind {kind}: build() makes the bm25 index; --kind dense uses build_dense()")
    raw = [p for p in paths if "raw" in p.parts]
    if raw and not allow_unscrubbed:
        raise SystemExit(f"not scrubbed (the LER scrub runs before any index build): {[str(p) for p in raw]}")
    if not paths:
        raise SystemExit("no document files")
    out.mkdir(parents=True, exist_ok=True)
    # index.json is written last: a rebuild that fails leaves no stale manifest beside the new
    # chunks.jsonl / documents.jsonl, so register and the re-sync refuse the directory
    (out / MANIFEST_NAME).unlink(missing_ok=True)
    shutil.rmtree(out / "dense", ignore_errors=True)  # vectors of the previous chunks would misalign
    # A chunk of a document whose text another source also holds names
    # those documents, so its citation can name every source (lexhybrid/retrieval/copies.py)
    groups = copy_groups(read_dedup_logs(dedup)) if dedup else {}
    chunks, docs, by_type, sources = [], 0, Counter(), Counter()
    with (
        open(out / CHUNKS_NAME, "w", encoding="utf-8") as f,
        open(out / DOCUMENTS_NAME, "w", encoding="utf-8") as listed,
    ):
        for doc in read_documents(paths):
            docs += 1
            if doc.doc_type in INDEXED_TYPES:
                listed.write(document_digest(doc).to_json() + "\n")
            for chunk in chunk_document(doc):
                if groups.get(chunk.doc_id):
                    chunk = replace(chunk, copies=groups[chunk.doc_id])
                f.write(chunk.to_json() + "\n")
                chunks.append(chunk)
                by_type[chunk.doc_type] += 1
                sources[chunk.source] += 1
    if not chunks:
        raise SystemExit(f"{docs} documents, none of an indexed type (statute, regulation, decision)")
    index = BM25Index.build(chunks)
    index.save(out / "bm25")
    manifest = {
        "kind": kind, "built_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "inputs": [str(p) for p in paths], "documents": docs, "chunks": len(chunks),
        "chunks_by_type": dict(by_type), "chunks_by_source": dict(sources), "bm25": index.config,
        "dedup": [str(p) for p in dedup], "chunks_with_copies": sum(1 for c in chunks if c.copies),
    }  # fmt: skip
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def build_dense(out: Path, encode, model: str) -> dict:
    """Every chunk of ``out/chunks.jsonl`` embedded into ``out/dense/``; its entry in ``index.json``."""
    manifest_path = out / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(f"{out}: no {MANIFEST_NAME}; build the bm25 index there first")
    manifest = json.loads(manifest_path.read_text())

    def chunks():
        with open(out / CHUNKS_NAME, encoding="utf-8") as f:
            yield from (Chunk.from_json(line) for line in f if line.strip())

    ids = [c.id for c in chunks()]
    start = time.time()
    index = DenseIndex.build(out / "dense", ids, (c.text for c in chunks()), encode, model=model)
    seconds = max(time.time() - start, 1e-9)
    manifest["dense"] = {
        **index.config, "seconds": round(seconds, 1), "resumed_rows": index.resumed,
        "chunks_per_s": round((len(ids) - index.resumed) / seconds, 1),
        "bytes": (out / "dense" / VECTORS_NAME).stat().st_size,
    }  # fmt: skip
    write_atomic(manifest_path, json.dumps(manifest, indent=1) + "\n")
    return manifest["dense"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", default="bm25", choices=KINDS)
    parser.add_argument(
        "--docs", type=Path, nargs="*", help="scrubbed JSONL files (default: data/scrubbed/*.jsonl)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-unscrubbed", action="store_true")
    parser.add_argument(
        "--dedup", type=Path, nargs="*", default=[],
        help="the pack's dedup.jsonl: cross-source copies are linked into Chunk.copies",
    )  # fmt: skip
    parser.add_argument("--dense-model", default=DENSE_MODEL)
    parser.add_argument("--device", default=None, help="for --kind dense (default: cuda if available)")
    args = parser.parse_args(argv)
    if args.kind == "dense":
        d = build_dense(args.out, BgeM3Encoder(args.dense_model, device=args.device), args.dense_model)
        print(
            f"DENSE {d['n_chunks']} chunks x {d['dim']} -> {d['bytes']:,} bytes, {d['chunks_per_s']} chunks/s"
        )
        return 0
    paths = args.docs if args.docs else sorted((REPO_ROOT / "data" / "scrubbed").glob("*.jsonl"))
    manifest = build(args.kind, paths, args.out, args.allow_unscrubbed, args.dedup)
    print(
        f"{manifest['documents']} documents -> {manifest['chunks']} chunks {manifest['chunks_by_type']} "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
