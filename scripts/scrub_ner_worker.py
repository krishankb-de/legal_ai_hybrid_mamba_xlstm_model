#!/usr/bin/env python3
"""German legal-entity recognition for the LER scrub (plan P3-R1, P3-R; decision 22).

Runs in the scrub environment (``envs/scrub``: flair with transformers < 5) and imports nothing from
``lexhybrid``; ``lexhybrid.data.corpus.scrub_ler`` starts it as a subprocess:

    envs/scrub/.venv/bin/python scripts/scrub_ner_worker.py < docs.jsonl > entities.jsonl

Input lines ``{"id": ..., "text": ...}``; output lines ``{"id": ..., "entities": [[start, end,
label, score], ...]}`` with character offsets into ``text``. Each line of a text is tagged on its own,
long lines in chunks of at most ``--max-tokens`` whitespace tokens split after sentence ends;
``Sentence(use_tokenizer=False)`` as the model card advises for legal text. Throughput goes to
stderr at the end. The model is ``flair/ner-german-legal`` at a pinned revision (LER German F1 96.35;
19 tags), fetched into ``--cache-dir`` (default ``data/hf/hub``) on first use.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

MODEL = "flair/ner-german-legal"
REVISION = "69c73dd705665388224cff919a529dd53750fca6"
REPO_ROOT = Path(__file__).resolve().parent.parent
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+(?=\S)")


def chunks(text: str, max_tokens: int) -> list[tuple[int, str]]:
    """``(offset, piece)`` for every line of ``text``, long lines cut after sentence ends into pieces
    of at most ``max_tokens`` whitespace tokens (a single longer sentence is cut at the limit)."""
    out, pos = [], 0
    for line in text.split("\n"):
        start = pos
        pos += len(line) + 1
        if not line.strip():
            continue
        pieces, cut = [], 0
        for m in _SENTENCE_END.finditer(line):
            pieces.append((cut, m.start()))
            cut = m.end()
        pieces.append((cut, len(line)))
        cur_start, cur_end, n = None, None, 0
        for a, b in pieces:
            words = len(line[a:b].split())
            if cur_start is not None and n + words > max_tokens:
                out.append((start + cur_start, line[cur_start:cur_end]))
                cur_start, n = None, 0
            while words > max_tokens:  # one very long sentence: cut it at the token limit
                spans = [m.span() for m in re.finditer(r"\S+", line[a:b])]
                cut_at = a + spans[max_tokens - 1][1]
                out.append((start + a, line[a:cut_at]))
                a = cut_at + len(line[cut_at:b]) - len(line[cut_at:b].lstrip())
                words = len(line[a:b].split())
            if words == 0:
                continue
            cur_start = a if cur_start is None else cur_start
            cur_end, n = b, n + words
        if cur_start is not None:
            out.append((start + cur_start, line[cur_start:cur_end]))
    return out


def load_tagger(cache_dir: str):
    from flair.models import SequenceTagger
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL, "pytorch_model.bin", revision=REVISION, cache_dir=cache_dir)
    return SequenceTagger.load(path)


def tag(tagger, docs: list[dict], max_tokens: int, batch_size: int) -> list[dict]:
    from flair.data import Sentence

    out = []
    for doc in docs:
        pieces = chunks(doc["text"], max_tokens)
        sentences = [Sentence(piece, use_tokenizer=False) for _, piece in pieces]
        if sentences:
            tagger.predict(sentences, mini_batch_size=batch_size)
        entities = []
        for (offset, piece), sentence in zip(pieces, sentences, strict=True):
            for span in sentence.get_spans("ner"):
                label = span.get_label("ner")
                start, end = offset + span.start_position, offset + span.end_position
                assert doc["text"][start:end] == piece[span.start_position : span.end_position]
                entities.append([start, end, label.value, round(float(label.score), 4)])
        out.append({"id": doc["id"], "entities": entities})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="German legal NER over JSONL documents (stdin -> stdout).")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "data" / "hf" / "hub"))
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    # flair logs to sys.stdout; keep stdout for the JSONL results and send everything else to stderr.
    out, sys.stdout = sys.stdout, sys.stderr
    tagger = load_tagger(args.cache_dir)
    t0, n_docs, n_chars = time.perf_counter(), 0, 0
    for line in sys.stdin:
        if not line.strip():
            continue
        doc = json.loads(line)
        (result,) = tag(tagger, [doc], args.max_tokens, args.batch_size)
        out.write(json.dumps(result, ensure_ascii=False) + "\n")
        out.flush()
        n_docs, n_chars = n_docs + 1, n_chars + len(doc["text"])
    dt = time.perf_counter() - t0
    print(json.dumps({"docs": n_docs, "chars": n_chars, "seconds": round(dt, 2),
                      "docs_per_s": round(n_docs / dt, 3) if dt else None}), file=sys.stderr)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
