"""Cross-source copies of a document, from the pack's dedup log.

A document whose text another source also holds is kept,
not dropped. Examples are one decision in rii and oldp, or an oldp decision in the research arm's
Multi Legal Pile. The pack's ``dedup.jsonl`` links each such copy to the document it matches
(``dropped: false``). Here those links become groups: linked copies are grouped transitively, so A–B
and B–C put A, B and C together. Every chunk of a grouped document then carries the group's other
documents in ``Chunk.copies``, which the renderer turns into one citation naming every source.
Rows with ``dropped: true`` are duplicates removed within one source and link nothing.
"""

import json
from collections.abc import Iterable
from pathlib import Path


def copy_groups(rows: Iterable[dict]) -> dict[str, tuple[str, ...]]:
    """``{document id: the other documents of its copy group, sorted}`` for every linked document."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in rows:
        if row.get("dropped", True):
            continue
        a, b = find(row["id"]), find(row["duplicate_of"])
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[str, set[str]] = {}
    for x in list(parent):
        groups.setdefault(find(x), set()).add(x)
    return {doc: tuple(sorted(members - {doc})) for members in groups.values() for doc in members}


def read_dedup_logs(paths: Iterable[Path]) -> list[dict]:
    return [
        json.loads(line) for p in paths for line in Path(p).read_text(encoding="utf-8").splitlines() if line
    ]
