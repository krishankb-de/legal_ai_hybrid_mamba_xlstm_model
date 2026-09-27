"""GerDaLIR, the German Dataset for Legal Information Retrieval (plan P3-P): case-law retrieval.

Source: the MTEB copy ``mteb/GerDaLIRSmall`` on the Hugging Face Hub (MIT, like the original
``lavis-nlp/GerDaLIR``), pinned revision, read line by line with ``huggingface_hub``'s file system:
``queries.jsonl`` (``_id``, ``text``: a passage of a decision whose citation is masked as
``[REF]``), ``qrels/test.jsonl`` (``query-id``, ``corpus-id``, ``score``) and ``corpus.jsonl``
(``_id``, ``title``, ``text``: the cited decisions, from Open Legal Data, dates masked as
``[DATE]``). With a limit, reading stops once the chosen queries and their passages are found
(the corpus lists passages in qrels order), so a smoke run does not read the 214 MB corpus.
"""

import json
from collections.abc import Iterable, Iterator

from lexhybrid.data.evalsets.base import EvalSetInfo, Passage, Query, RetrievalSet, cli

REPO = "mteb/GerDaLIRSmall"
REVISION = "b199f38071bc06a2cb86c4c10d57ecee6c46056a"  # main on 2026-09-27
INFO = EvalSetInfo(
    name="gerdalir",
    task="retrieval",
    licence="MIT",
    source=f"https://huggingface.co/datasets/{REPO}@{REVISION[:7]}",
    note="queries and decisions from Open Legal Data; citations masked as [REF], dates as [DATE]",
)


def parse_query(line: str) -> Query:
    row = json.loads(line)
    return Query(id=str(row["_id"]), text=row["text"])


def parse_qrel(line: str) -> tuple[str, str, int]:
    row = json.loads(line)
    return str(row["query-id"]), str(row["corpus-id"]), int(row["score"])


def parse_passage(line: str) -> Passage:
    row = json.loads(line)
    return Passage(id=str(row["_id"]), text=row["text"], title=row.get("title") or "")


def select(
    qrel_lines: Iterable[str], query_lines: Iterable[str], corpus_lines: Iterable[str], limit: int | None
) -> RetrievalSet:
    qrels: dict[str, dict[str, int]] = {}
    for qid, pid, score in map(parse_qrel, qrel_lines):
        if limit is not None and qid not in qrels and len(qrels) >= limit:
            break
        qrels.setdefault(qid, {})[pid] = score
    queries: list[Query] = []
    for q in map(parse_query, query_lines):
        if q.id in qrels:
            queries.append(q)
            if limit is not None and len(queries) == len(qrels):
                break
    wanted = {pid for rel in qrels.values() for pid in rel}
    corpus: list[Passage] = []
    for line in corpus_lines:
        p = parse_passage(line)
        if limit is None or p.id in wanted:
            corpus.append(p)
        if limit is not None and len(corpus) == len(wanted):
            break
    return RetrievalSet(info=INFO, queries=queries, qrels=qrels, corpus=corpus)


def _lines(path: str) -> Iterator[str]:
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(
        f"datasets/{REPO}@{REVISION}/{path}", "r", encoding="utf-8", block_size=1 << 20
    ) as f:
        yield from f


def fetch(limit: int | None = None) -> RetrievalSet:
    return select(_lines("qrels/test.jsonl"), _lines("queries.jsonl"), _lines("corpus.jsonl"), limit)


if __name__ == "__main__":
    raise SystemExit(cli("gerdalir", fetch))
