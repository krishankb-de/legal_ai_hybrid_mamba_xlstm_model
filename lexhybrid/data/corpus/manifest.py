"""Corpus manifests (plan P3-E): one JSON line per document with every schema field but the text.

``data/manifests/<source>.jsonl`` is versioned (``.gitignore`` allowlists ``data/manifests/``) while
the texts under ``data/raw/`` are not: the manifest is the auditable record of what was collected,
from where, under which licence, with which content hash -- enough to recount tokens per source
and licence (P4-M) and to prove what a model saw.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from lexhybrid.data.schema import Document


def manifest_record(doc: Document) -> dict:
    """Every field of ``doc`` except ``text`` (and each section's text); sections keep their labels."""
    record = doc.to_dict(with_text=False)
    record["n_chars"] = len(doc.text)
    return record


def write_manifest(docs: Iterable[Document], path: Path) -> int:
    """Write the manifest lines for ``docs`` to ``path`` (replacing it); return the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(manifest_record(doc), ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
