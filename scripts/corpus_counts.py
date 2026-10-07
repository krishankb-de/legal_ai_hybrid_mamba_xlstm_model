#!/usr/bin/env python3
"""Per-source corpus counts from the collection manifests (plan P4-L verdict).

    .venv/bin/python scripts/corpus_counts.py --manifests $SCRATCH_ROOT/data/manifests \
        --raw $SCRATCH_ROOT/data/raw --sources gii rii oldp eurlex dip fineweb2_de multilegalpile

Documents and characters per source, jurisdiction and licence come from the manifests (one line per
document with ``n_chars``, streamed). Tokens are an ESTIMATE: characters times the Qwen3 tokens per
character of a stride sample of the source's raw documents (``--sample`` per source: every k-th
line, so the sample spans the whole file), taken per jurisdiction. The exact counts come from the
pack (``build_summary.json``, P4-M). Rows outside ``--jurisdictions`` are reported and marked
excluded (decisions.market). Prints one table row per (source, jurisdiction, licence), then the
JSON summary as its last line.
"""

import argparse
import json
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

from lexhybrid.data.corpus.licences import LICENCES
from lexhybrid.data.schema import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
GERMAN_MARKET = ("gii", "rii", "oldp", "eurlex", "dip", "fineweb2_de", "multilegalpile")


def manifest_counts(path: Path) -> dict[tuple[str, str], dict]:
    """(jurisdiction, licence) -> documents and characters over one manifest."""
    out: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                c = out.setdefault((r["jurisdiction"], r["licence"]), {"documents": 0, "chars": 0})
                c["documents"] += 1
                c["chars"] += r["n_chars"]
    return out


def tokens_per_char(
    path: Path, n_lines: int, sample: int, encode: Callable[[str], list[int]]
) -> dict[str, dict]:
    """jurisdiction -> sampled documents, characters, tokens and tokens per character over every
    k-th line of ``path`` (k = n_lines // sample, at least 1)."""
    step = max(1, n_lines // max(1, sample))
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % step or not line.strip():
                continue
            doc = Document.from_json(line)
            s = out.setdefault(doc.jurisdiction, {"documents": 0, "chars": 0, "tokens": 0})
            s["documents"] += 1
            s["chars"] += len(doc.text)
            s["tokens"] += len(encode(doc.text))
    for s in out.values():
        s["tokens_per_char"] = s["tokens"] / s["chars"] if s["chars"] else None
    return out


def count(
    sources, manifests: Path, raw: Path, sample: int, jurisdictions, encode: Callable[[str], list[int]]
) -> dict:
    rows, missing = [], []
    for source in sources:
        man = manifests / f"{source}.jsonl"
        if not man.exists():
            missing.append(source)
            continue
        groups = manifest_counts(man)
        n_docs = sum(c["documents"] for c in groups.values())
        ratios = tokens_per_char(raw / source / f"{source}.jsonl", n_docs, sample, encode)
        pooled = sum(s["tokens"] for s in ratios.values()) / max(1, sum(s["chars"] for s in ratios.values()))
        for (jur, lic), c in sorted(groups.items()):
            s = ratios.get(jur)
            ratio = s["tokens_per_char"] if s and s["tokens_per_char"] else pooled
            rows.append({
                "source": source, "jurisdiction": jur, "licence": lic,
                "commercial_safe": LICENCES[lic].commercial_safe,
                "included": jurisdictions is None or jur in jurisdictions,
                "documents": c["documents"], "chars": c["chars"],
                "sample_documents": s["documents"] if s else 0, "tokens_per_char": round(ratio, 4),
                "est_tokens": round(c["chars"] * ratio),
            })  # fmt: skip
    kept = [r for r in rows if r["included"]]
    totals = {
        arm: {
            "documents": sum(r["documents"] for r in kept if pick(r)),
            "chars": sum(r["chars"] for r in kept if pick(r)),
            "est_tokens": sum(r["est_tokens"] for r in kept if pick(r)),
        }
        for arm, pick in (("commercial_safe", lambda r: r["commercial_safe"]), ("research", lambda r: True))
    }
    return {"rows": rows, "totals": totals, "missing": missing, "sample_per_source": sample,
            "jurisdictions": sorted(jurisdictions) if jurisdictions is not None else None}  # fmt: skip


def qwen3_encoder() -> Callable[[str], list[int]]:
    from lexhybrid.data.tokenizer import encode_document, load_tokenizer

    return partial(encode_document, load_tokenizer(with_specials=False))


def main(argv=None, encoder_factory: Callable = qwen3_encoder) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifests", type=Path, default=REPO_ROOT / "data" / "manifests")
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--sources", nargs="+", default=list(GERMAN_MARKET))
    parser.add_argument(
        "--jurisdictions", nargs="+", default=["DE", "EU"], help="the rows that enter training"
    )
    parser.add_argument(
        "--sample", type=int, default=2000, help="documents per source for tokens per character"
    )
    args = parser.parse_args(argv)
    summary = count(
        args.sources, args.manifests, args.raw, args.sample, set(args.jurisdictions), encoder_factory()
    )
    print(f"{'source':15s} {'jur':3s} {'licence':16s} {'in':3s} {'documents':>10s} {'chars':>15s} "
          f"{'tok/char':>8s} {'est. tokens':>15s}")  # fmt: skip
    for r in summary["rows"]:
        print(f"{r['source']:15s} {r['jurisdiction']:3s} {r['licence']:16s} {'yes' if r['included'] else 'no':3s} "
              f"{r['documents']:10,d} {r['chars']:15,d} {r['tokens_per_char']:8.4f} {r['est_tokens']:15,d}")  # fmt: skip
    for arm, t in summary["totals"].items():
        print(
            f"TOTAL {arm}: {t['documents']:,d} documents, {t['chars']:,d} chars, ~{t['est_tokens']:,d} tokens"
        )
    if summary["missing"]:
        print(f"no manifest for: {' '.join(summary['missing'])}")
    print(json.dumps(summary))
    return 1 if summary["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
