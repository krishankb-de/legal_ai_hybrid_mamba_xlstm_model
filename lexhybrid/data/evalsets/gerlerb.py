"""GerLeRB, the German Legislative Retrieval Benchmark (plan P3-P): questions over federal statutes.

Source: Zenodo record 15745124 (2025-06-26, CC BY 4.0): ``topics.txt`` (``qid<TAB>query``, 367
questions), ``qrels.txt`` (``qid<TAB>iteration<TAB>docno<TAB>label``) and ``corpus.trec.gz``
(90,798 paragraphs as TREC ``<DOC><DOCNO>..</DOCNO><TEXT>..</TEXT></DOC>``). A ``docno`` is
``<paragraph>_<GII slug>`` (``812_bgb``), so ``gii_id`` maps it onto the GII collector's ids
(``gii:bgb:812``) and P7 can score GerLeRB against the project's own statute index.

The questions were generated with GPT-4.1 from court-decision contexts and checked by legal experts:
evaluation only, never training data (R10).
"""

import gzip
import re
from collections.abc import Iterable, Iterator

from lexhybrid.data.corpus.collectors.base import http_get
from lexhybrid.data.evalsets.base import EvalSetInfo, Passage, Query, RetrievalSet, cli

RECORD = "15745124"
FILE_URL = "https://zenodo.org/api/records/" + RECORD + "/files/{name}/content"
INFO = EvalSetInfo(
    name="gerlerb",
    task="retrieval",
    licence="CC-BY-4.0",
    source=f"https://zenodo.org/records/{RECORD}",
    note="queries generated with GPT-4.1 and verified by legal experts: evaluation only (R10)",
)
_DOC = re.compile(r"<DOC>\s*<DOCNO>(?P<docno>[^<]+)</DOCNO>\s*<TEXT>(?P<text>.*?)</TEXT>\s*</DOC>", re.S)
_DOCNO = re.compile(r"^(?P<n>\d+[a-z]*)_(?P<slug>[\w-]+)$")


def parse_topics(text: str) -> list[Query]:
    rows = [line.split("\t", 1) for line in text.splitlines()[1:] if line.strip()]
    return [Query(id=qid.strip(), text=" ".join(q.split())) for qid, q in rows]


def parse_qrels(text: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for line in text.splitlines()[1:]:
        if line.strip():
            qid, _, docno, label = line.split("\t")
            qrels.setdefault(qid.strip(), {})[docno.strip()] = int(label)
    return qrels


def iter_corpus(trec: str) -> Iterator[Passage]:
    """Passages of the TREC corpus (whitespace normalised line by line)."""
    for m in _DOC.finditer(trec):
        lines = (" ".join(line.split()) for line in m.group("text").split("\n"))
        yield Passage(id=m.group("docno").strip(), text="\n".join(line for line in lines if line))


def gii_id(docno: str) -> str | None:
    """``812_bgb`` -> ``gii:bgb:812``: the same provision in the GII collector's corpus."""
    m = _DOCNO.match(docno)
    return f"gii:{m.group('slug')}:{m.group('n')}" if m else None


def select(
    queries: Iterable[Query], qrels: dict, passages: Iterable[Passage], limit: int | None
) -> RetrievalSet:
    """The first ``limit`` queries that have judgements, their qrels and their relevant passages
    (every passage when ``limit`` is None)."""
    chosen = [q for q in queries if qrels.get(q.id)][:limit]
    ids = {q.id for q in chosen}
    rel = {qid: r for qid, r in qrels.items() if qid in ids}
    wanted = {docno for r in rel.values() for docno in r}
    corpus = [p for p in passages if limit is None or p.id in wanted]
    return RetrievalSet(info=INFO, queries=chosen, qrels=rel, corpus=corpus)


def fetch(limit: int | None = None) -> RetrievalSet:
    topics = http_get(FILE_URL.format(name="topics.txt")).text
    qrels = http_get(FILE_URL.format(name="qrels.txt")).text
    trec = gzip.decompress(http_get(FILE_URL.format(name="corpus.trec.gz"), timeout=300).content).decode(
        "utf-8"
    )
    return select(parse_topics(topics), parse_qrels(qrels), iter_corpus(trec), limit)


if __name__ == "__main__":
    raise SystemExit(cli("gerlerb", fetch))
