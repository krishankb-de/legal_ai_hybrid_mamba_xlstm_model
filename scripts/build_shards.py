#!/usr/bin/env python3
"""Packed shards from the collected documents (plan P3-V; the P3-Z smoke shards).

    .venv/bin/python scripts/build_shards.py --root data/shards/smoke --row-len 4096 \
        --val-fraction 0.1 --min-val-docs 2

Per source the documents come from ``data/scrubbed/<source>.jsonl`` (the LER scrub, P3-R). R11 runs
the scrub before pretraining, so a source without a scrubbed file is refused unless
``--allow-unscrubbed`` is given, and ``meta.json`` records ``scrubbed`` either way. All sources are
deduplicated together first (P3-S; the commercial-safe copy survives, the log goes to
``<root>/<row_len>/dedup.jsonl``), then each is split by document and packed (``build_source_shards``).
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from lexhybrid.data.corpus.dedup import deduplicate
from lexhybrid.data.datasets import MIN_VAL_DOCS, VAL_FRACTION, build_source_shards
from lexhybrid.data.schema import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("gii", "rii", "oldp", "eurlex", "ris", "fedlex", "bger", "dip", "fineweb2_de", "multilegalpile")


def _read(path: Path) -> list[Document]:
    return [
        Document.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_sources(
    sources, scrubbed_dir: Path, raw_dir: Path, allow_unscrubbed: bool
) -> dict[str, tuple[list, bool]]:
    out = {}
    for source in sources:
        scrubbed = scrubbed_dir / f"{source}.jsonl"
        raw = raw_dir / source / f"{source}.jsonl"
        if scrubbed.exists():
            out[source] = (_read(scrubbed), True)
        elif raw.exists() and allow_unscrubbed:
            out[source] = (_read(raw), False)
        elif raw.exists():
            raise SystemExit(
                f"{source}: not scrubbed (R11); run lexhybrid.data.corpus.scrub_ler or pass --allow-unscrubbed"
            )
    return out


def main(argv=None) -> int:
    from lexhybrid.data.tokenizer import EOS_ID, encode_document, load_tokenizer

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "shards")
    parser.add_argument("--row-len", type=int, default=4096, choices=(4096, 8192))
    parser.add_argument("--sources", nargs="*", default=list(SOURCES))
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--min-val-docs", type=int, default=MIN_VAL_DOCS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-unscrubbed", action="store_true")
    parser.add_argument("--scrubbed", type=Path, default=REPO_ROOT / "data" / "scrubbed")
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "raw")
    args = parser.parse_args(argv)

    loaded = load_sources(args.sources, args.scrubbed, args.raw, args.allow_unscrubbed)
    if not loaded:
        print("no documents found", file=sys.stderr)
        return 1
    kept, log = deduplicate(d for docs, _ in loaded.values() for d in docs)
    out = args.root / str(args.row_len)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dedup.jsonl").write_text("".join(json.dumps(asdict(e)) + "\n" for e in log))
    tok = load_tokenizer(with_specials=False)
    encode = lambda text: encode_document(tok, text)  # noqa: E731
    print(f"{sum(len(d) for d, _ in loaded.values())} documents, {len(log)} duplicates dropped")
    for source, (_, scrubbed) in loaded.items():
        docs = [d for d in kept if d.source == source]
        meta = build_source_shards(
            docs, encode, EOS_ID, args.row_len, args.root, args.val_fraction, args.min_val_docs, args.seed,
            extra={"scrubbed": scrubbed},
        )  # fmt: skip
        t, v = meta["train"], meta["val"]
        print(f"{source:15s} train {t['documents']:4d} docs {t['rows']:4d} rows {t['tokens']:9,d} tokens | "
              f"val {v['documents']:3d} docs {v['rows']:3d} rows | scrubbed={scrubbed}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
