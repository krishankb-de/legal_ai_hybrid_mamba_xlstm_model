"""Deduplication of the training corpus (plan P3-S; decision 13): exact, then near-duplicate.

1. **Exact.** Documents with the same text hash (``Document.sha256``) are one document.
2. **Near.** ``datasketch`` MinHash over lowercased word 5-gram shingles, 128 permutations, and an LSH
   index at Jaccard 0.8; a candidate the index returns is a duplicate when its MinHash estimate of
   the Jaccard similarity is at least 0.8.

Which copy survives is fixed before any comparison: documents are visited commercial-safe first,
then research-only, then by source and id, and the first copy of a group is kept. So a Multi Legal
Pile copy of an Open Legal Data decision is dropped and the commercial-safe OLDP copy stays, and a
run gives the same answer every time. The log (``DedupLog``) names every dropped document, the
document it duplicates, the kind and the estimate.

This deduplicates **training** data. Versions of one provision (RIS, Fedlex) are near-duplicates by
design and are collapsed here; the retrieval index (P7) is built from the collected documents, not
from this output, so every version stays retrievable.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from lexhybrid.data.schema import Document

SHINGLE = 5
NUM_PERM = 128
THRESHOLD = 0.8
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class DedupLog:
    id: str
    duplicate_of: str
    kind: str  # "exact" or "near"
    jaccard: float  # 1.0 for exact duplicates; the MinHash estimate for near ones


def shingles(text: str, n: int = SHINGLE) -> set[str]:
    """Lowercased word n-grams; a text shorter than n words is its own single shingle."""
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def minhash(text: str, num_perm: int = NUM_PERM, seed: int = 1):
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm, seed=seed)
    m.update_batch([s.encode("utf-8") for s in shingles(text)])
    return m


def keep_order(doc: Document) -> tuple:
    """The visiting order: commercial-safe first, research-only last, then source and id."""
    return (not doc.commercial_safe, doc.research_only, doc.source, doc.id)


def deduplicate(
    documents: Iterable[Document], threshold: float = THRESHOLD, num_perm: int = NUM_PERM
) -> tuple[list[Document], list[DedupLog]]:
    """The documents to keep (in visiting order) and the log of the dropped ones."""
    from datasketch import MinHashLSH

    docs = sorted(documents, key=keep_order)
    kept: list[Document] = []
    log: list[DedupLog] = []
    by_hash: dict[str, str] = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sketches = {}
    for doc in docs:
        if doc.sha256 in by_hash:
            log.append(DedupLog(doc.id, by_hash[doc.sha256], "exact", 1.0))
            continue
        m = minhash(doc.text, num_perm)
        best = max(((m.jaccard(sketches[c]), c) for c in lsh.query(m)), default=None)
        if best is not None and best[0] >= threshold:
            log.append(DedupLog(doc.id, best[1], "near", round(best[0], 4)))
            continue
        by_hash[doc.sha256] = doc.id
        lsh.insert(doc.id, m)
        sketches[doc.id] = m
        kept.append(doc)
    return kept, log
