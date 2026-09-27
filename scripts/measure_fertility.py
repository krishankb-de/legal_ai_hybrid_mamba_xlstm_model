#!/usr/bin/env python3
"""Tokenizer fertility on DACH legal text (plan P3-D; prediction in analysis/tokenizer_fertility.md).

    .venv/bin/python scripts/measure_fertility.py --collect      # live sample -> data/cache/fertility/
    .venv/bin/python scripts/measure_fertility.py                # measure the sample and the fixtures

``--collect`` runs the eight legal collectors (and FineWeb-2 German as a general reference), each
until it has ``--words`` words in whole documents, skipping documents over ``--max-doc-words``,
and writes ``data/cache/fertility/<source>.jsonl`` (full documents, git-ignored) plus
``analysis/tokenizer_fertility_sample.jsonl`` (id, sha256 and word count of every sampled document).
The measurement prints one table row per source and tokenizer: words, tokens, tokens/word; words are
whitespace-separated tokens of ``Document.text``.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "cache" / "fertility"
SAMPLE_MANIFEST = REPO_ROOT / "analysis" / "tokenizer_fertility_sample.jsonl"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "collectors"
LEGAL = ("gii", "rii", "oldp", "eurlex", "ris", "fedlex", "bger", "dip")
REFERENCE = ("fineweb2_de",)
GPT2 = "openai-community/gpt2"


def n_words(text: str) -> int:
    return len(text.split())


def _collectors():
    from lexhybrid.data.corpus.collectors import bger, dip, eurlex, fedlex, fineweb2_de, gii, oldp, rii, ris

    return {
        "gii": gii.GIICollector(),
        "rii": rii.RIICollector(),
        "oldp": oldp.OLDPCollector(),
        "eurlex": eurlex.EURLexCollector(),
        "ris": ris.RISCollector(),
        "fedlex": fedlex.FedlexCollector(),
        "bger": bger.BGerCollector(),
        "dip": dip.DIPCollector(),
        "fineweb2_de": fineweb2_de.FineWeb2DECollector(),
    }


def take(documents, words: int, max_doc_words: int) -> list:
    """Whole documents in order until ``words`` words, skipping any longer than ``max_doc_words``."""
    out, total = [], 0
    for doc in documents:
        w = n_words(doc.text)
        if w > max_doc_words:
            continue
        out.append(doc)
        total += w
        if total >= words:
            break
    return out


def collect(words: int, max_doc_words: int, sources, recollect: bool = False) -> None:
    """Sample each source (keeping a sample already on disk unless ``recollect``), then rewrite the
    sample manifest from the files."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    collectors = _collectors()
    for source in sources:
        path = SAMPLE_DIR / f"{source}.jsonl"
        if path.exists() and not recollect:
            print(f"{source}: kept the sample in {path.name}", flush=True)
            continue
        c = collectors[source]
        if source == "ris":  # norms and decisions half and half
            docs = take(c.norms(), words // 2, max_doc_words) + take(
                c.decisions(), words - words // 2, max_doc_words
            )
        else:
            docs = take(c.iter_documents(), words, max_doc_words)
        with open(path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(doc.to_json() + "\n")
        print(f"{source}: {len(docs)} documents, {sum(n_words(d.text) for d in docs):,} words", flush=True)
    manifest = []
    for source in sources:
        for line in (SAMPLE_DIR / f"{source}.jsonl").read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            manifest.append(
                {"source": source, "id": d["id"], "sha256": d["sha256"], "words": n_words(d["text"])}
            )
    SAMPLE_MANIFEST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest))


def _texts(path: Path) -> list[str]:
    return [
        json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def fixture_texts() -> dict[str, list[str]]:
    """Document texts of the P3-F..M fixtures, parsed by their collectors."""
    from lexhybrid.data.corpus.collectors import bger, dip, eurlex, fedlex, gii, oldp, rii, ris

    out: dict[str, list[str]] = {}
    out["gii"] = [
        d.text for d in gii.parse_gii_xml((FIXTURES / "gii" / "beurkg_excerpt.xml").read_bytes(), "beurkg")
    ]
    out["rii"] = [
        rii.parse_rii_xml(p.read_bytes(), "https://www.rechtsprechung-im-internet.de").text
        for p in sorted((FIXTURES / "rii").glob("jb-*.xml"))
    ]
    out["oldp"] = [
        oldp.parse_oldp_case(json.loads(p.read_text())).text
        for p in sorted((FIXTURES / "oldp").glob("case_*.json"))
    ]
    out["eurlex"] = [
        d.text
        for d in eurlex.parse_eurlex_xhtml(
            (FIXTURES / "eurlex" / "32016R0679_excerpt.xhtml").read_bytes(), "32016R0679"
        )
    ]
    norms = [
        ris.parse_ris_norm(
            json.loads((FIXTURES / "ris" / f"{n}.json").read_text()),
            (FIXTURES / "ris" / f"{n}.xml").read_bytes(),
        )
        for n in ("NOR12018413", "NOR40172917", "NOR12019037")
    ]
    out["ris"] = [d.text for d in norms if d is not None]
    out["fedlex"] = [
        d.text
        for p in sorted((FIXTURES / "fedlex").glob("*.xml"))
        for d in fedlex.parse_fedlex_xml(p.read_bytes())
    ]
    out["bger"] = [
        bger.parse_bger(p.read_text(encoding="utf-8"), json.loads(p.with_suffix(".json").read_text())).text
        for p in sorted((FIXTURES / "bger").glob("CH_BGer_*.html"))
    ]
    out["dip"] = [
        dip.parse_dip_drucksache(json.loads(p.read_text())).text
        for p in sorted((FIXTURES / "dip").glob("*.json"))
    ]
    return out


def measure(texts_by_source: dict[str, list[str]], tokenizers: dict) -> list[dict]:
    rows = []
    for source, texts in texts_by_source.items():
        words = sum(n_words(t) for t in texts)
        for name, encode in tokenizers.items():
            tokens = sum(len(encode(t)) for t in texts)
            rows.append({"source": source, "tokenizer": name, "docs": len(texts), "words": words, "tokens": tokens,
                         "fertility": tokens / words if words else float("nan")})  # fmt: skip
    return rows


def pooled(rows: list[dict], sources, tokenizer: str) -> dict:
    sel = [r for r in rows if r["source"] in sources and r["tokenizer"] == tokenizer]
    words, tokens = sum(r["words"] for r in sel), sum(r["tokens"] for r in sel)
    return {"source": "pooled legal", "tokenizer": tokenizer, "docs": sum(r["docs"] for r in sel), "words": words,
            "tokens": tokens, "fertility": tokens / words if words else float("nan")}  # fmt: skip


def load_tokenizers() -> dict:
    from transformers import AutoTokenizer

    from lexhybrid.data.tokenizer import cache_dir, encode_document, load_tokenizer

    qwen3 = load_tokenizer(with_specials=False)
    gpt2 = AutoTokenizer.from_pretrained(GPT2, cache_dir=cache_dir())
    return {
        "Qwen3": lambda t: encode_document(qwen3, t),
        "GPT-2": lambda t: gpt2(t, add_special_tokens=False)["input_ids"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--collect", action="store_true", help="collect the live sample first")
    parser.add_argument("--words", type=int, default=125_000, help="words per source (default 125,000)")
    parser.add_argument("--max-doc-words", type=int, default=50_000, help="skip longer documents")
    parser.add_argument("--recollect", action="store_true", help="collect again even where a sample exists")
    args = parser.parse_args(argv)
    if args.collect:
        collect(args.words, args.max_doc_words, LEGAL + REFERENCE, args.recollect)
    tokenizers = load_tokenizers()
    sample = {
        s: _texts(SAMPLE_DIR / f"{s}.jsonl")
        for s in LEGAL + REFERENCE
        if (SAMPLE_DIR / f"{s}.jsonl").exists()
    }
    missing = [s for s in LEGAL if s not in sample]
    if missing:
        print(f"no sample for {missing}: run with --collect", file=sys.stderr)
        return 1
    rows = measure(sample, tokenizers)
    rows += [pooled(rows, LEGAL, name) for name in tokenizers]
    fixture_rows = measure(fixture_texts(), tokenizers)
    fixture_rows += [pooled(fixture_rows, LEGAL, name) for name in tokenizers]
    print("| sample | source | tokenizer | documents | words | tokens | tokens/word |")
    print("|---|---|---|---:|---:|---:|---:|")
    for label, table in (("live 2026-09-27", rows), ("P3 fixtures", fixture_rows)):
        for r in table:
            print(
                f"| {label} | {r['source']} | {r['tokenizer']} | {r['docs']:,} | {r['words']:,} | {r['tokens']:,} | {r['fertility']:.3f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
