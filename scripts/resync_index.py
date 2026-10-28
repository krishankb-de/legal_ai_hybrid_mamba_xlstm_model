#!/usr/bin/env python3
"""Nightly re-sync stub: record what rebuilding an index would change; change nothing.

    .venv/bin/python scripts/resync_index.py --index data/index/smoke --report data/manifests/resync_smoke.json
    .venv/bin/python scripts/resync_index.py --index data/index/smoke --docs data/scrubbed/gii.jsonl \
        --report data/manifests/resync_smoke.json

The index's ``documents.jsonl`` (what ``scripts/build_index.py`` built it from) is compared with the
current documents (default: every ``data/scrubbed/*.jsonl``, as the build reads them; a file under a
``raw`` directory is refused, as the build refuses it), counting only documents of an indexed type
(statute, regulation, decision):

    added          ids in the current documents and not in the index
    removed        ids in the index and not in the current documents
    changed        ids in both whose record differs (``DocumentDigest.record_sha256``)
    metadata_only  the changed ids whose text is the same (``sha256``) -- e.g. a new ``valid_to``

Id lists are sorted. The stub never rebuilds, writes or deletes anything in the index directory (a
``--report`` inside it is refused). It prints one line ``RESYNC added=<n> removed=<n> changed=<n>``
and writes the report JSON (atomically); it exits 0 whether or not anything would change, since the
report is the result.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from lexhybrid.data.schema import Document
from lexhybrid.retrieval.index import (
    INDEXED_TYPES,
    DocumentDigest,
    document_digest,
    indexed_documents,
    read_manifest,
    write_atomic,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def current_documents(paths: list[Path]) -> dict[str, DocumentDigest]:
    """``{id: digest}`` of the documents of an indexed type in ``paths``; a repeated id is refused
    (the build would fail on its duplicate chunk ids)."""
    out: dict[str, DocumentDigest] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                doc = Document.from_json(line)
                if doc.doc_type not in INDEXED_TYPES:
                    continue
                if doc.id in out:
                    raise SystemExit(f"{path}:{n}: document {doc.id!r} appears twice in the inputs")
                out[doc.id] = document_digest(doc)
    return out


def compare(indexed: dict[str, DocumentDigest], current: dict[str, DocumentDigest]) -> dict:
    both = indexed.keys() & current.keys()
    changed = sorted(i for i in both if indexed[i].record_sha256 != current[i].record_sha256)
    return {
        "added": sorted(current.keys() - indexed.keys()),
        "removed": sorted(indexed.keys() - current.keys()),
        "changed": changed,
        "metadata_only": [i for i in changed if indexed[i].sha256 == current[i].sha256],
        "unchanged": len(both) - len(changed),
    }


def resync(index_dir: Path, paths: list[Path]) -> dict:
    """The report: what a rebuild of ``index_dir`` from ``paths`` would change. Reads only."""
    if not paths:
        raise SystemExit("no document files")
    raw = [str(p) for p in paths if "raw" in p.parts]
    if raw:
        raise SystemExit(f"not scrubbed (an index is built from scrubbed documents only): {raw}")
    try:
        manifest, indexed = read_manifest(index_dir), indexed_documents(index_dir)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    diff = compare(indexed, current_documents(paths))
    return {
        "index": str(index_dir),
        "built_at": manifest.get("built_at"),
        "checked_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "docs": [str(p) for p in paths],
        "indexed_documents": len(indexed),
        **diff,
        "would_change": bool(diff["added"] or diff["removed"] or diff["changed"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--index", type=Path, required=True, help="an index directory (scripts/build_index.py)"
    )
    parser.add_argument(
        "--docs", type=Path, nargs="*", help="scrubbed JSONL files (default: data/scrubbed/*.jsonl)"
    )
    parser.add_argument("--report", type=Path, required=True, help="where the JSON report goes")
    args = parser.parse_args(argv)
    if args.report.resolve().is_relative_to(args.index.resolve()):
        raise SystemExit(
            f"--report {args.report} is inside the index {args.index}; the re-sync never writes there"
        )
    paths = args.docs if args.docs else sorted((REPO_ROOT / "data" / "scrubbed").glob("*.jsonl"))
    report = resync(args.index, paths)
    write_atomic(args.report, json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(
        f"RESYNC added={len(report['added'])} removed={len(report['removed'])} "
        f"changed={len(report['changed'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
