"""GerLayQA: laypeople's legal questions answered by lawyers, grounded in BGB paragraphs.

Source: Büttner and Habernal (EACL 2024), ``data/bgb_eval.json`` in
``trusthlt/eacl24-german-legal-questions`` at a pinned commit: the paper's evaluation split, 2,154
items ``{Question_text, Answer_text, Paragraphs}`` where ``Paragraphs`` names the BGB provisions the
lawyer's answer rests on (``§ 573b``, mapped to ``BGB §573b``). The GPT-3.5 answers of
``bgb_eval_qa.json`` are not used.

The repository's notice restricts the data to non-commercial scientific research, and it is
excluded from training explicitly: evaluation only, ``research_only``.
"""

import html
import re

from lexhybrid.data.corpus.collectors.base import http_get
from lexhybrid.data.evalsets.base import EvalSetInfo, QAItem, QASet, cli

COMMIT = "cbc465942f0c1eab25ec8864bc358466e4d88e0e"
URL = f"https://raw.githubusercontent.com/trusthlt/eacl24-german-legal-questions/{COMMIT}/data/bgb_eval.json"
INFO = EvalSetInfo(
    name="gerlayqa",
    task="qa",
    licence="non-commercial scientific research only (repository notice); the authors claim no copyright",
    source=f"https://github.com/trusthlt/eacl24-german-legal-questions@{COMMIT[:7]}",
    note="evaluation split bgb_eval.json; never training data",
    research_only=True,
)
_PARAGRAPH = re.compile(r"^§\s*(\d+[a-z]*)$")
_TAG = r"(?:a|b|i|u|p|br|div|span|strong|em|blockquote|ul|ol|li|sup|sub)"
# The texts carry the forum's HTML, mostly with every "<" lost (a href="..." title="§ 573b BGB: ...">§ 573b BGB/a>)
_MARKUP = re.compile(
    rf"<{_TAG}\b[^>]*>|(?<![\w]){_TAG}(?:\s+[\w-]+=(?:\"[^\"]*\"|'[^']*'))*\s*/?>|<?/{_TAG}>|<?!--.*?-->",
    re.IGNORECASE,
)


def plain_text(text: str) -> str:
    """What a reader of the page sees: entities decoded, then tags and comments dropped."""
    return " ".join(_MARKUP.sub(" ", html.unescape(text)).split())


def bgb_citation(paragraph: str) -> str | None:
    """``§ 573b`` -> ``BGB §573b``."""
    m = _PARAGRAPH.match(" ".join(paragraph.split()))
    return f"BGB §{m.group(1)}" if m else None


def parse_gerlayqa(rows: list[dict]) -> list[QAItem]:
    items = []
    for i, row in enumerate(rows):
        gold = [c for c in (bgb_citation(p) for p in row.get("Paragraphs") or []) if c]
        items.append(
            QAItem(
                id=f"gerlayqa:bgb_eval:{i}",
                question=plain_text(row["Question_text"]),
                answers=[plain_text(row["Answer_text"])],
                gold=gold,
            )
        )
    return items


def fetch(limit: int | None = None) -> QASet:
    return QASet(info=INFO, items=parse_gerlayqa(http_get(URL, timeout=300).json())[:limit])


if __name__ == "__main__":
    raise SystemExit(cli("gerlayqa", fetch))
