#!/usr/bin/env python3
"""Packed shards from the collected documents (plan P3-V; the P3-Z smoke shards; P4-L at scale).

    .venv/bin/python scripts/build_shards.py --root data/shards/smoke --row-len 4096 \
        --val-fraction 0.1 --min-val-docs 2
    .venv/bin/python scripts/build_shards.py --root $SCRATCH_ROOT/data/shards --row-len 4096 8192 \
        --scrubbed $SCRATCH_ROOT/data/scrubbed --raw $SCRATCH_ROOT/data/raw --workers 32   # P4-L

Per source the documents come from ``<scrubbed>/<source>.jsonl`` (the LER scrub, P3-R), or from the
parts ``<source>.part-<k>-of-<n>.jsonl`` a scrub array wrote (every part must be there). R11 runs
the scrub before pretraining, so a source with only ``<raw>/<source>/<source>.jsonl`` is refused
unless ``--allow-unscrubbed`` is given, and ``meta.json`` records ``scrubbed`` either way.

Documents are packed in collection order (the order of the collected file, which keeps a law's
provisions and a court's decisions together), not in the deduplication's visiting order.

Streaming, so the corpus never has to fit in memory. Pass 1 reads every document once for its
deduplication signature, all sources together (P3-S; decision 13 as amended: a duplicate inside one
source is dropped, a copy in another source is kept and linked; the log of both goes to
``<root>/<row_len>/dedup.jsonl``, and the links let retrieval cite every source of a passage). Pass 2 packs each source -- ``--workers`` sources at a time --
at every ``--row-len`` from one tokenisation, after its split by document. One source per shard
directory, so a research-only source (``multilegalpile``) is always a shard set of its own and the
commercial-safe arm never reads it (P3-V). ``<root>/build_summary.json`` holds the per-source
counts the corpus manifest (P4-M) reads.
"""

import argparse
import itertools
import json
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict
from functools import partial
from pathlib import Path

from lexhybrid.data.corpus.dedup import deduplicate_stream
from lexhybrid.data.datasets import MIN_VAL_DOCS, VAL_FRACTION, build_source_shards_stream, split_by_document
from lexhybrid.data.packing import ROW_LENS
from lexhybrid.data.schema import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("gii", "rii", "oldp", "eurlex", "ris", "fedlex", "bger", "dip", "fineweb2_de", "multilegalpile")
_PART = re.compile(r"\.part-(\d+)-of-(\d+)\.jsonl$")


def source_files(
    source: str, scrubbed_dir: Path, raw_dir: Path, allow_unscrubbed: bool
) -> tuple[list[Path], bool] | None:
    """The files that hold ``source``'s documents and whether they are scrubbed; None if absent."""
    whole = scrubbed_dir / f"{source}.jsonl"
    if whole.exists():
        return [whole], True
    parts = sorted(scrubbed_dir.glob(f"{source}.part-*-of-*.jsonl"))
    if parts:
        found = {tuple(map(int, _PART.search(p.name).groups())): p for p in parts}  # (k, n) -> path
        totals = {n for _, n in found}
        n = totals.pop() if len(totals) == 1 else None
        if n is None or sorted(found) != [(k, n) for k in range(n)]:
            raise SystemExit(f"{source}: incomplete or mixed scrub parts {sorted(p.name for p in parts)}")
        return [found[(k, n)] for k in range(n)], True
    raw = raw_dir / source / f"{source}.jsonl"
    if raw.exists():
        if not allow_unscrubbed:
            raise SystemExit(
                f"{source}: not scrubbed (R11); run lexhybrid.data.corpus.scrub_ler or pass --allow-unscrubbed"
            )
        return [raw], False
    return None


def iter_documents(paths: list[Path]) -> Iterator[Document]:
    """The documents in collection order. Scrub parts hold every n-th line of the collected file
    (part k: lines k, k+n, ...), so they are read round-robin, one line from each in turn, which
    puts the lines back in their original order: a law's provisions stay next to each other."""
    files = [open(path, encoding="utf-8") for path in paths]  # closed in the finally below
    try:
        for lines in itertools.zip_longest(*files):
            for line in lines:
                if line is not None and line.strip():
                    yield Document.from_json(line)
    finally:
        for f in files:
            f.close()


def qwen3_encoder() -> Callable[[str], list[int]]:
    from lexhybrid.data.tokenizer import encode_document, load_tokenizer

    return partial(encode_document, load_tokenizer(with_specials=False))


def pack_source(job: dict, encoder_factory: Callable = qwen3_encoder) -> dict:
    """Pass 2 for one source (runs in a worker process): its kept documents, packed at every row
    length. Returns the counts of every row length and split (no document ids)."""
    from lexhybrid.data.tokenizer import EOS_ID

    dropped = job["dropped"]
    docs = (d for d in iter_documents(job["paths"]) if d.id not in dropped)
    metas = build_source_shards_stream(
        docs, job["val_ids"], encoder_factory(), job.get("eos_id", EOS_ID), job["row_lens"], job["root"],
        job["fraction"], job["minimum"], job["seed"], extra={"scrubbed": job["scrubbed"]},
    )  # fmt: skip
    return {
        str(n): {k: v for k, v in meta.items() if k not in ("train", "val")}
        | {s: {k: v for k, v in meta[s].items() if k != "document_ids"} for s in ("train", "val")}
        for n, meta in metas.items()
    }


def build(args, encoder_factory: Callable = qwen3_encoder, eos_id: int | None = None) -> dict:
    files = {}
    for source in args.sources:
        found = source_files(source, args.scrubbed, args.raw, args.allow_unscrubbed)
        if found is not None:
            files[source] = found
    if not files:
        raise SystemExit("no documents found")
    everything = (d for paths, _ in files.values() for d in iter_documents(paths))
    kept, log = deduplicate_stream(everything, workers=args.workers)
    for n in args.row_len:
        out = args.root / str(n)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dedup.jsonl").write_text("".join(json.dumps(asdict(e)) + "\n" for e in log))
    dropped = {e.id for e in log if e.dropped}
    linked = sum(1 for e in log if not e.dropped)
    print(
        f"{len(kept) + len(dropped)} documents, {len(dropped)} duplicates within a source dropped, "
        f"{linked} copies across sources kept and linked",
        flush=True,
    )
    jobs = []
    for source, (paths, scrubbed) in files.items():
        ids = [i for i in kept if i.startswith(source + ":")]  # Document ids are "<source>:..."
        try:
            val_ids = split_by_document(ids, args.val_fraction, args.min_val_docs, args.seed)
        except ValueError as e:
            raise SystemExit(f"{source}: {e}") from e
        job = dict(
            source=source, paths=paths, scrubbed=scrubbed, row_lens=tuple(args.row_len), root=args.root,
            dropped={i for i in dropped if i.startswith(source + ":")}, fraction=args.val_fraction,
            minimum=args.min_val_docs, seed=args.seed, val_ids=val_ids,
        )  # fmt: skip
        if eos_id is not None:
            job["eos_id"] = eos_id
        jobs.append(job)
    work = partial(pack_source, encoder_factory=encoder_factory)
    if args.workers > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            results = pool.map(work, jobs, chunksize=1)
    else:
        results = [work(job) for job in jobs]
    summary = {
        "documents": len(kept) + len(dropped),
        "duplicates_dropped": len(dropped),
        "copies_linked": linked,
        "sources": {},
    }
    for job, result in zip(jobs, results):
        summary["sources"][job["source"]] = result
        for n, meta in result.items():
            t, v = meta["train"], meta["val"]
            print(f"{job['source']:15s} L={n} train {t['documents']:7d} docs {t['rows']:7d} rows {t['tokens']:13,d} "
                  f"tokens | val {v['documents']:5d} docs {v['rows']:5d} rows | scrubbed={job['scrubbed']}")  # fmt: skip
    (args.root / "build_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "shards")
    parser.add_argument("--row-len", type=int, nargs="+", default=[4096], choices=ROW_LENS)
    parser.add_argument("--sources", nargs="*", default=list(SOURCES))
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--min-val-docs", type=int, default=MIN_VAL_DOCS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-unscrubbed", action="store_true")
    parser.add_argument("--scrubbed", type=Path, default=REPO_ROOT / "data" / "scrubbed")
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--workers", type=int, default=1, help="processes for the signatures and the packing")
    args = parser.parse_args(argv)
    args.row_len = sorted(set(args.row_len))
    build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
