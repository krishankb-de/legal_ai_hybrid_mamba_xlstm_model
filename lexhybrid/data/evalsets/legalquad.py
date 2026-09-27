"""LegalQuAD: extractive question answering over German court decisions (plan P3-Q).

Source: the authors' release (Hoppe et al., AIKE 2021), ``data/LegalQuAD.json`` in
``Christoph911/AIKE2021_Appendix`` at a pinned commit: SQuAD format, 200 decisions from Open Legal
Data, one hand-written question per decision with its answer span. The original repository states
no licence; MTEB's retrieval copy (``mteb/LegalQuAD``) declares CC BY 4.0. Evaluation only.
"""

from lexhybrid.data.corpus.collectors.base import http_get
from lexhybrid.data.evalsets.base import EvalSetInfo, QAItem, QASet, cli

COMMIT = "6eed75f066c98a7b3e43044b764ff8e0273ad1f3"
URL = f"https://raw.githubusercontent.com/Christoph911/AIKE2021_Appendix/{COMMIT}/data/LegalQuAD.json"
INFO = EvalSetInfo(
    name="legalquad",
    task="qa",
    licence="none stated by the authors (MTEB's copy: CC-BY-4.0)",
    source=f"https://github.com/Christoph911/AIKE2021_Appendix@{COMMIT[:7]}",
    note="200 questions with extractive answers over Open Legal Data decisions; one answer differs from its "
    "context by one character",
)


def parse_legalquad(squad: dict) -> list[QAItem]:
    """SQuAD-format JSON -> one item per question; answers are the annotated spans."""
    items = []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                items.append(
                    QAItem(
                        id=f"legalquad:{qa['id']}",
                        question=" ".join(qa["question"].split()),
                        answers=[a["text"].strip() for a in qa["answers"]],
                        context=para["context"],
                        context_id=f"oldp:{para['document_id']}",
                    )
                )
    return items


def fetch(limit: int | None = None) -> QASet:
    return QASet(info=INFO, items=parse_legalquad(http_get(URL, timeout=120).json())[:limit])


if __name__ == "__main__":
    raise SystemExit(cli("legalquad", fetch))
