"""Corpus assembly gate (plan P3-B, rule R9): which documents may enter which training arm.

Two arms (decision 2): ``commercial_safe`` admits only documents whose licence and flags permit a
shippable model; ``research`` additionally admits ``research_only`` documents (the non-commercial
research-exception arm). Every document must carry a registered, known licence consistent with its
flags, or the whole build stops: a licence problem is never filtered away silently.
"""

from collections.abc import Iterable, Iterator

from lexhybrid.data.corpus.licences import LicenceError, check_document
from lexhybrid.data.schema import Document

ARMS = ("commercial_safe", "research")


def admits(arm: str, doc: Document) -> bool:
    """Whether ``doc`` (already licence-checked) belongs in ``arm``."""
    if arm == "commercial_safe":
        return doc.commercial_safe and not doc.research_only
    return doc.commercial_safe or doc.research_only


def build_corpus(documents: Iterable[Document], arm: str = "commercial_safe") -> Iterator[Document]:
    """Yield the documents of ``arm``; raise ``LicenceError`` on the first licence problem."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    for doc in documents:
        problems = check_document(doc)
        if problems:
            raise LicenceError("; ".join(problems))
        if admits(arm, doc):
            yield doc
