"""Deduplication of the training corpus (plan P3-S; decision 13): exact, then near-duplicate.

1. **Exact.** Documents with the same text hash (``Document.sha256``) are one document.
2. **Near.** ``datasketch`` MinHash over lowercased word 5-gram shingles, 128 permutations, and an LSH
   index at Jaccard 0.8; a candidate the index returns is a duplicate when its MinHash estimate of
   the Jaccard similarity is at least 0.8.

**Within one source** a duplicate is dropped. Which copy survives is fixed before any comparison:
documents are visited commercial-safe first, then research-only, then by source and id, and the
first copy of a group is kept, so a run gives the same answer every time.

**Across sources every copy is kept** (decision 13 as amended by the user on 2026-09-29: "keep both
and when cited cite both"): the RII and the OLDP copy of a decision, a web page in FineWeb that
reproduces a statute, the Multi Legal Pile copy of an OLDP decision (research arm only) -- each is
packed with its own source, and the log links it to the copy it matches, so the retrieval index can
group the copies and a citation can name every source (P7). The log (``DedupLog``) names every
dropped document and every linked copy (``dropped`` False), the document it matches, the kind and
the estimate.

This deduplicates **training** data. Versions of one provision (RIS, Fedlex) are near-duplicates by
design and are collapsed here (they share a source); the retrieval index (P7) is built from the
collected documents, not from this output, so every version stays retrievable.

At corpus scale (P4-L) the texts do not fit in memory: ``signature`` reduces a document to what
the decisions need -- its visiting order, id, source, text hash and 128 MinHash values (512 bytes) -- and
``deduplicate_stream`` computes the signatures in worker processes, then runs the same decisions
(``deduplicate_signatures``) over them. ``deduplicate`` is that procedure over a list.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

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
    dropped: bool = True  # False: a copy in another source, kept and linked (decision 13 as amended)


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


@dataclass(frozen=True)
class Signature:
    """A document as deduplication sees it: no text."""

    order: tuple  # keep_order(doc)
    id: str
    source: str
    sha256: str
    hashvalues: bytes  # the NUM_PERM MinHash values, uint32


def signature(doc: Document, num_perm: int = NUM_PERM) -> Signature:
    values = minhash(doc.text, num_perm).hashvalues.tobytes()
    return Signature(keep_order(doc), doc.id, doc.source, doc.sha256, values)


def deduplicate_signatures(
    signatures: Iterable[Signature], threshold: float = THRESHOLD, num_perm: int = NUM_PERM
) -> tuple[list[str], list[DedupLog]]:
    """The ids to keep, in visiting order, and the log: dropped duplicates (within a source) and
    linked copies (across sources, kept)."""
    import numpy as np
    from datasketch import LeanMinHash, MinHash, MinHashLSH

    scheme = MinHash(num_perm=num_perm, seed=1).scheme  # the scheme ``minhash`` hashes with
    kept: list[str] = []
    log: list[DedupLog] = []
    by_hash: dict[str, dict[str, str]] = {}  # text hash -> {source: the kept id}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sketches: dict[str, tuple[str, bytes]] = {}  # kept id -> (source, MinHash values)
    for s in sorted(signatures, key=lambda s: s.order):
        copies = by_hash.get(s.sha256, {})
        if s.source in copies:
            log.append(DedupLog(s.id, copies[s.source], "exact", 1.0))
            continue
        values = np.frombuffer(s.hashvalues, dtype=np.uint32)
        m = LeanMinHash(seed=1, hashvalues=values, scheme=scheme)
        scored = [
            (float(np.mean(values == np.frombuffer(sketches[c][1], dtype=np.uint32))), c)
            for c in lsh.query(m)
        ]
        scored = [(j, c) for j, c in scored if j >= threshold]
        same = max(((j, c) for j, c in scored if sketches[c][0] == s.source), default=None)
        if same is not None:
            log.append(DedupLog(s.id, same[1], "near", round(same[0], 4)))
            continue
        if copies:  # the same text in another source: kept, linked
            log.append(DedupLog(s.id, next(iter(copies.values())), "exact", 1.0, dropped=False))
        elif scored:
            best = max(scored)
            log.append(DedupLog(s.id, best[1], "near", round(best[0], 4), dropped=False))
        by_hash.setdefault(s.sha256, {})[s.source] = s.id
        lsh.insert(s.id, m)
        sketches[s.id] = (s.source, s.hashvalues)
        kept.append(s.id)
    return kept, log


def deduplicate_stream(
    documents: Iterable[Document],
    workers: int = 1,
    threshold: float = THRESHOLD,
    num_perm: int = NUM_PERM,
    chunksize: int = 64,
) -> tuple[list[str], list[DedupLog]]:
    """``deduplicate_signatures`` over documents streamed once, their MinHashes computed by
    ``workers`` processes; returns ids (the texts are never all in memory) and the log."""
    sign = partial(signature, num_perm=num_perm)
    if workers <= 1:
        signatures = [sign(d) for d in documents]
    else:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            signatures = list(pool.imap(sign, documents, chunksize=chunksize))
    return deduplicate_signatures(signatures, threshold, num_perm)


def deduplicate(
    documents: Iterable[Document], threshold: float = THRESHOLD, num_perm: int = NUM_PERM
) -> tuple[list[Document], list[DedupLog]]:
    """The documents to keep (in visiting order) and the log: dropped duplicates and linked copies."""
    docs = list(documents)
    kept_ids, log = deduplicate_signatures([signature(d, num_perm) for d in docs], threshold, num_perm)
    by_id = {d.id: d for d in docs}
    return [by_id[i] for i in kept_ids], log
