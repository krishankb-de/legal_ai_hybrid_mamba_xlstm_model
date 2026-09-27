"""What every evaluation set shares (plan P3-P, P3-Q): records, the info block, writers, the CLI."""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lexhybrid.data.schema import sha256_text

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Query:
    id: str
    text: str


@dataclass
class Passage:
    id: str
    text: str
    title: str = ""


@dataclass
class QAItem:
    """A question with its reference answers; ``context`` is the passage an extractive answer comes
    from, ``gold`` the provisions a lawyer's answer rests on (``BGB §573b``)."""

    id: str
    question: str
    answers: list[str]
    context: str = ""
    context_id: str = ""
    gold: list[str] = field(default_factory=list)


@dataclass
class EvalSetInfo:
    name: str
    task: str  # "retrieval" or "qa"
    licence: str  # as the publisher states it
    source: str  # where it was fetched, with the pinned revision
    note: str = ""
    eval_only: bool = True
    research_only: bool = False


@dataclass
class RetrievalSet:
    info: EvalSetInfo
    queries: list[Query]
    qrels: dict[str, dict[str, int]]  # query id -> {passage id: relevance}
    corpus: list[Passage]


@dataclass
class QASet:
    info: EvalSetInfo
    items: list[QAItem]


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write(evalset: RetrievalSet | QASet, out_dir: Path, manifest_dir: Path) -> int:
    """The set's files under ``out_dir`` and its manifest (ids and hashes, no text); returns the
    number of queries or items."""
    info = asdict(evalset.info)
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
    if isinstance(evalset, RetrievalSet):
        _write_jsonl(out_dir / "queries.jsonl", (asdict(q) for q in evalset.queries))
        _write_jsonl(out_dir / "corpus.jsonl", (asdict(p) for p in evalset.corpus))
        _write_jsonl(
            out_dir / "qrels.jsonl",
            (
                {"query_id": q, "passage_id": p, "relevance": r}
                for q, rel in evalset.qrels.items()
                for p, r in rel.items()
            ),
        )
        manifest = [
            {"id": q.id, "sha256": sha256_text(q.text), "relevant": sorted(evalset.qrels.get(q.id, {}))}
            for q in evalset.queries
        ]
        n = len(evalset.queries)
    else:
        _write_jsonl(out_dir / "items.jsonl", (asdict(i) for i in evalset.items))
        manifest = [{"id": i.id, "sha256": sha256_text(i.question), "gold": i.gold} for i in evalset.items]
        n = len(evalset.items)
    _write_jsonl(manifest_dir / f"evalset_{evalset.info.name}.jsonl", [{"info": info}, *manifest])
    return n


def cli(name: str, fetch: Callable[[int | None], RetrievalSet | QASet], argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"Fetch the {name} evaluation set.")
    parser.add_argument("--limit", type=int, default=20, help="queries or items (default 20; 0 = all)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "evalsets" / name)
    parser.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "data" / "manifests")
    args = parser.parse_args(argv)
    evalset = fetch(args.limit or None)
    n = write(evalset, args.out, args.manifest_dir)
    extra = f", {len(evalset.corpus)} passages" if isinstance(evalset, RetrievalSet) else ""
    print(f"{name}: {n} {'queries' if isinstance(evalset, RetrievalSet) else 'items'}{extra} -> {args.out}")
    return 0 if n else 1
