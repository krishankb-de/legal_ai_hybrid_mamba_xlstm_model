#!/usr/bin/env python3
"""Build a retrieval index from collected documents (plan P7-C; P7-D adds ``--kind dense``).

    .venv/bin/python scripts/build_index.py --kind bm25 --out data/index/smoke
    .venv/bin/python scripts/build_index.py --kind bm25 --docs data/scrubbed/gii.jsonl data/scrubbed/rii.jsonl \
        --out data/index/smoke

Documents come from scrubbed JSONL files (default: every ``data/scrubbed/*.jsonl`` of an indexed
type; R11 runs the LER scrub before any index build, so a file under a ``raw`` directory is refused
unless ``--allow-unscrubbed``); scrub parts (``<source>.part-<k>-of-<n>.jsonl``) are read like whole
files. Statutes and regulations are chunked by provision, decisions by window (P7-A/B); other
document types are not indexed. The index directory gets ``chunks.jsonl`` (every chunk, for the
prompt builder and the renderer), ``bm25/`` (``lexhybrid.retrieval.bm25.BM25Index``) and
``index.json`` (what was indexed: sources, documents, chunks by type, settings).

The retrieval index keeps every copy of a text: it is built from the collected documents, not from
the pretraining deduplication (decision 13 as amended), so a passage held by two sources is found
in both.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from lexhybrid.data.schema import Document
from lexhybrid.retrieval.bm25 import BM25Index
from lexhybrid.retrieval.chunking import chunk_document

REPO_ROOT = Path(__file__).resolve().parent.parent
KINDS = ("bm25",)


def read_documents(paths: list[Path]):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield Document.from_json(line)


def build(kind: str, paths: list[Path], out: Path, allow_unscrubbed: bool = False) -> dict:
    if kind not in KINDS:
        raise SystemExit(f"--kind {kind}: only {KINDS} so far (P7-D adds dense)")
    raw = [p for p in paths if "raw" in p.parts]
    if raw and not allow_unscrubbed:
        raise SystemExit(
            f"not scrubbed (R11: the LER scrub runs before any index build): {[str(p) for p in raw]}"
        )
    if not paths:
        raise SystemExit("no document files")
    out.mkdir(parents=True, exist_ok=True)
    chunks, docs, by_type, sources = [], 0, Counter(), Counter()
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for doc in read_documents(paths):
            docs += 1
            for chunk in chunk_document(doc):
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
    }  # fmt: skip
    (out / "index.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", default="bm25", choices=KINDS)
    parser.add_argument(
        "--docs", type=Path, nargs="*", help="scrubbed JSONL files (default: data/scrubbed/*.jsonl)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-unscrubbed", action="store_true")
    args = parser.parse_args(argv)
    paths = args.docs if args.docs else sorted((REPO_ROOT / "data" / "scrubbed").glob("*.jsonl"))
    manifest = build(args.kind, paths, args.out, args.allow_unscrubbed)
    print(
        f"{manifest['documents']} documents -> {manifest['chunks']} chunks {manifest['chunks_by_type']} "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
