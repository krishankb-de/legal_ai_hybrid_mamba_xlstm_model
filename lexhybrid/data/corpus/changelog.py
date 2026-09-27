"""The currency changelog (plan P3-T): which provisions changed since a date; P9's currency subset.

A provision's version date is its ``valid_from``, and what that date means depends on the source:

* **RIS** (Austria): one document per version with its own ``Inkrafttretensdatum`` -- the date is the
  provision's (granularity ``provision``), and the version it replaced, when the corpus holds it
  (same citation, ``valid_to`` the day before or earlier), is linked as ``previous_id``.
* **Fedlex** (Switzerland): the latest in-force date in the article's footnotes (P3-K) -- the
  provision's own date, conservative (granularity ``provision``).
* **GII** (Germany): the date of the law's last amendment from its ``Stand`` comment -- the whole
  act's, not the provision's (granularity ``act``: the provision may be unchanged).

``changes_since`` lists every statute or regulation whose version date is on or after ``since``,
newest first; ``write_changelog`` writes ``data/manifests/changelog.jsonl``.

    python -m lexhybrid.data.corpus.changelog --since 2025-01-01
"""

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from lexhybrid.data.schema import Document

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCES = {"ris": "provision", "fedlex": "provision", "gii": "act"}


@dataclass(frozen=True)
class Change:
    citation_id: str
    doc_id: str
    source: str
    jurisdiction: str
    valid_from: str
    valid_to: str | None
    granularity: str  # "provision" or "act"
    previous_id: str | None  # the version this one replaced, when the corpus has it
    sha256: str


def changes_since(documents: Iterable[Document], since: str) -> list[Change]:
    docs = [
        d
        for d in documents
        if d.source in SOURCES and d.doc_type in ("statute", "regulation") and d.valid_from
    ]
    by_citation: dict[tuple[str, str], list[Document]] = {}
    for d in docs:
        by_citation.setdefault((d.source, d.citation_id), []).append(d)
    out = []
    for d in docs:
        if d.valid_from < since:
            continue
        earlier = [
            e
            for e in by_citation[(d.source, d.citation_id)]
            if e.valid_to is not None and e.valid_to < d.valid_from
        ]
        previous = max(earlier, key=lambda e: e.valid_to, default=None)
        out.append(
            Change(
                citation_id=d.citation_id,
                doc_id=d.id,
                source=d.source,
                jurisdiction=d.jurisdiction,
                valid_from=d.valid_from,
                valid_to=d.valid_to,
                granularity=SOURCES[d.source],
                previous_id=previous.id if previous else None,
                sha256=d.sha256,
            )
        )
    out.sort(key=lambda c: (c.source, c.citation_id, c.doc_id))
    return sorted(out, key=lambda c: c.valid_from, reverse=True)  # newest first; stable within a date


def write_changelog(changes: list[Change], path: Path, since: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"since": since, "changes": len(changes)}) + "\n")
        for c in changes:
            f.write(json.dumps(asdict(c), ensure_ascii=False, sort_keys=True) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Provisions changed since a date (the currency subset).")
    parser.add_argument("--since", required=True, help="ISO date")
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "manifests" / "changelog.jsonl")
    args = parser.parse_args(argv)
    docs = []
    for source in SOURCES:
        path = args.raw / source / f"{source}.jsonl"
        if path.exists():
            docs += [
                Document.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
    changes = changes_since(docs, args.since)
    write_changelog(changes, args.out, args.since)
    print(f"{len(changes)} provisions changed since {args.since} ({len(docs)} statutes read) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
