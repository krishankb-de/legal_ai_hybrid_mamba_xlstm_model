"""Scrub leaks into the index: no scrubbed name in a retrievable chunk.

The scrub (``lexhybrid/data/corpus/scrub_ler.py``) replaces every tagged person, judge, lawyer
and street by a placeholder and logs a keyed hash of the entity text, never the text. A chunk that
still holds one of those texts is a leak: the renderer could quote it. ``leaked_entities`` finds them
without the names: every candidate surface form in a chunk -- each capitalised word, and each run of
two or three capitalised words separated by whitespace or a period and whitespace (``Hans Müller``,
``Dr. Müller``, ``Hans-Peter Müller``) -- is hashed with ``entity_hash(form, key)``, and a hash among
the log's ``entity_hmac`` values is a leak, whichever document the log row came from.

The candidates are read from the chunk's header, hierarchy and sentences (all of ``Chunk.text``).
The function is pure: the log rows and the key are passed in. Logged forms with other separators
(a comma, a line break inside the name) are not reconstructed.
"""

import re
from collections.abc import Iterable, Mapping, Sequence

from lexhybrid.data.corpus.scrub_ler import entity_hash
from lexhybrid.retrieval.chunking import Chunk

_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")  # letters, with inner hyphens or apostrophes
_SEPARATOR = re.compile(r"\.?\s+")
MAX_RUN = 3


def candidate_forms(text: str) -> list[str]:
    """Each capitalised word of ``text`` and each run of 2..``MAX_RUN`` capitalised words, as written."""
    words = list(_WORD.finditer(text))
    out: list[str] = []
    for i, first in enumerate(words):
        if not first.group()[0].isupper():
            continue
        out.append(first.group())
        for last in range(i + 1, min(i + MAX_RUN, len(words))):
            between = text[words[last - 1].end() : words[last].start()]
            if not words[last].group()[0].isupper() or not _SEPARATOR.fullmatch(between):
                break
            out.append(text[first.start() : words[last].end()])
    return out


def leaked_entities(
    chunks: Sequence[Chunk], scrub_log_rows: Iterable[Mapping], key: bytes
) -> list[tuple[str, str]]:
    """``(chunk id, entity_hmac)`` of every logged entity a chunk still holds, once per chunk, in order."""
    logged = {row["entity_hmac"] for row in scrub_log_rows}
    if not logged:
        return []
    hashes: dict[str, str] = {}
    leaks: list[tuple[str, str]] = []
    for chunk in chunks:
        found: set[str] = set()
        for text in (chunk.citation_id, *chunk.hierarchy, *chunk.sentences):
            for form in candidate_forms(text):
                h = hashes.get(form)
                if h is None:
                    h = hashes[form] = entity_hash(form, key)
                if h in logged and h not in found:
                    found.add(h)
                    leaks.append((chunk.id, h))
    return leaks
