"""Lexical retrieval with BM25 over the retrieval chunks (plan P7-C; decision 14: ``bm25s``).

Every chunk's ``text`` (citation header, hierarchy, sentences; P7-A/B) is lower-cased, split into
word tokens, stripped of NLTK's German stopwords (``bm25s``' ``"de"`` list) and reduced by the
Snowball German stemmer (``PyStemmer``), which also folds umlauts and ß: ``Kündigungen``,
``kündigen`` -> ``kundig``, ``Mietverhältnisses`` -> ``mietverhaltnis``. One-character tokens are
kept (``bm25s``' default pattern drops them), because legal queries hinge on them: ``§ 1 BeurkG``,
``Art. 6 DSGVO``, ``lit. f``. Scoring is Lucene's BM25 (k1 1.5, b 0.75), the ``bm25s`` default.

An index is a directory: the ``bm25s`` arrays, the chunk ids in index order, and
``lexhybrid_bm25.json`` with the tokenizer settings a query must reuse (a query tokenised
differently from the corpus matches nothing and says nothing).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

LANGUAGE = "german"  # the Snowball stemmer
STOPWORDS = "de"  # bm25s: NLTK's German list
TOKEN_PATTERN = r"(?u)\b\w+\b"  # bm25s' default r"(?u)\b\w\w+\b" drops "1" of "§ 1"
CONFIG_NAME = "lexhybrid_bm25.json"
IDS_NAME = "chunk_ids.json"


def _stem():
    import Stemmer

    return Stemmer.Stemmer(LANGUAGE).stemWords


def tokenize(texts: Sequence[str]) -> list[list[str]]:
    """The index's tokens of each text (lower-cased, German stopwords out, stemmed)."""
    import bm25s

    return bm25s.tokenize(
        list(texts), lower=True, token_pattern=TOKEN_PATTERN, stopwords=STOPWORDS, stemmer=_stem(),
        return_ids=False, show_progress=False,
    )  # fmt: skip


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    rank: int  # 1-based


class BM25Index:
    """A ``bm25s`` index over chunks, queried by text."""

    def __init__(self, retriever, ids: list[str], config: dict):
        self.retriever, self.ids, self.config = retriever, ids, config

    @classmethod
    def build(cls, chunks: Sequence, k1: float = 1.5, b: float = 0.75, method: str = "lucene") -> "BM25Index":
        """Index ``chunks`` (anything with ``id`` and ``text``: ``lexhybrid.retrieval.chunking.Chunk``)."""
        import bm25s

        if not chunks:
            raise ValueError("no chunks to index")
        ids = [c.id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk ids must be unique")
        retriever = bm25s.BM25(k1=k1, b=b, method=method)
        retriever.index(tokenize([c.text for c in chunks]), show_progress=False)
        config = {
            "kind": "bm25", "library": "bm25s", "k1": k1, "b": b, "method": method, "language": LANGUAGE,
            "stopwords": STOPWORDS, "token_pattern": TOKEN_PATTERN, "n_chunks": len(ids),
        }  # fmt: skip
        return cls(retriever, ids, config)

    def query(self, text: str, k: int = 10) -> list[Hit]:
        """The best ``k`` chunks for ``text``, best first; chunks sharing no term score 0 and are
        left out, so a query of stopwords only returns nothing."""
        return self.query_many([text], k)[0]

    def query_many(self, texts: Sequence[str], k: int = 10) -> list[list[Hit]]:
        vocab = self.retriever.vocab_dict
        tokens = [[t for t in q if t in vocab] for q in tokenize(texts)]
        out: list[list[Hit]] = [[] for _ in texts]
        todo = [i for i, q in enumerate(tokens) if q]
        if not todo:
            return out
        k = min(k, len(self.ids))
        docs, scores = self.retriever.retrieve(
            [tokens[i] for i in todo], k=k, show_progress=False, n_threads=1
        )  # fmt: skip
        for row, i in enumerate(todo):
            ranked = [(int(d), float(s)) for d, s in zip(docs[row], scores[row]) if s > 0]
            out[i] = [Hit(self.ids[d], s, rank) for rank, (d, s) in enumerate(ranked, start=1)]
        return out

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(directory), show_progress=False)
        (directory / IDS_NAME).write_text(json.dumps(self.ids, ensure_ascii=False))
        (directory / CONFIG_NAME).write_text(json.dumps(self.config, indent=1) + "\n")
        return directory

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        import bm25s

        directory = Path(directory)
        config = json.loads((directory / CONFIG_NAME).read_text())
        mismatch = {
            name: (config.get(name), want)
            for name, want in (
                ("language", LANGUAGE),
                ("stopwords", STOPWORDS),
                ("token_pattern", TOKEN_PATTERN),
            )
            if config.get(name) != want
        }
        if mismatch:
            raise ValueError(
                f"{directory}: built with other tokenizer settings {mismatch}; rebuild the index"
            )
        retriever = bm25s.BM25.load(str(directory), show_progress=False)
        ids = json.loads((directory / IDS_NAME).read_text())
        if len(ids) != config["n_chunks"]:
            raise ValueError(f"{directory}: {len(ids)} chunk ids for an index of {config['n_chunks']}")
        return cls(retriever, ids, config)
