"""The German legal-entity scrub (plan P3-R; decisions 13 and 22): names out before training.

Entities come from ``flair/ner-german-legal`` (LER German F1 96.35), which runs in the scrub
environment ``envs/scrub`` through ``scripts/scrub_ner_worker.py``; this module never imports flair.
Of the model's 19 tags four identify people or where they live -- ``PER`` (person), ``RR`` (judge),
``AN`` (lawyer), ``STR`` (street) -- and are replaced by typed placeholders numbered per document
(``[PER_1]``, ``[RR_1]``, ``[AN_1]``, ``[STR_1]``): one surface form, one placeholder, so a decision
still reads "[PER_1] verklagte [PER_2]; [PER_1] trägt die Kosten". Besides the tagged spans, every
other occurrence of a tagged form of at least three letters is replaced too (a name the model
missed once is still caught); shorter forms (``W.``) are replaced only where tagged, so headings
such as ``A. Problem`` survive. Punctuation the model's whitespace tokens carry (``Haag,``) is
trimmed off first. Section texts get the same replacements.

The scrub log ``data/manifests/scrub_<source>.jsonl`` has one line per tagged entity: document id,
tag, placeholder, a keyed hash of the entity text (never the text), offsets and score. The hash is
HMAC-SHA256 under a local key (``LEXHYBRID_SCRUB_KEY``, else ``data/cache/scrub_hmac.key``, created
once and git-ignored): a plain hash of a name could be confirmed by anyone who guesses the name,
and the log is versioned.

    python -m lexhybrid.data.corpus.scrub_ler --source bger     # data/raw/bger/bger.jsonl -> data/scrubbed/

At corpus scale (P4-L; ~1,000 characters/s on a laptop CPU, so a GPU array on the cluster) a source
is split into ``--parts`` stride shares, one array task each (``--part k``: lines k, k+n, ...),
streamed through ONE worker process that batches ``--docs-per-batch`` documents per NER call; the
outputs are ``data/scrubbed/<source>.part-<k>-of-<n>.jsonl`` and the matching scrub log, renamed
from ``*.partial`` only when the part completes. ``scripts/build_shards.py`` reads the parts back
in their original order.
"""

import argparse
import dataclasses
import hashlib
import hmac
import json
import os
import queue
import re
import secrets
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from lexhybrid.data.schema import Document

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRUB_PYTHON = REPO_ROOT / "envs" / "scrub" / ".venv" / "bin" / "python"
WORKER = REPO_ROOT / "scripts" / "scrub_ner_worker.py"
SCRUB_TAGS = ("PER", "RR", "AN", "STR")
MIN_PROPAGATE = 3  # letters a tagged form needs before its untagged repeats are replaced too
KEY_PATH = REPO_ROOT / "data" / "cache" / "scrub_hmac.key"


def scrub_key() -> bytes:
    """The HMAC key of the scrub log: ``LEXHYBRID_SCRUB_KEY`` (hex), else the local key file."""
    env = os.environ.get("LEXHYBRID_SCRUB_KEY")
    if env:
        return bytes.fromhex(env)
    if not KEY_PATH.exists():
        KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        KEY_PATH.write_text(secrets.token_hex(32) + "\n")
        KEY_PATH.chmod(0o600)
    return bytes.fromhex(KEY_PATH.read_text().strip())


def entity_hash(text: str, key: bytes) -> str:
    return hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str
    score: float


_LEADING = "(„\"'»"
_TRAILING = ",;:)“\"'«"


def trim(entity: Entity, text: str) -> Entity:
    """The model tags whitespace tokens, so ``Haag,`` or ``(Müller`` carries punctuation: cut it off.
    A final period stays with an initial (``W.``) and goes with a name (``Baur.`` -> ``Baur``)."""
    start, end = entity.start, entity.end
    while start < end and text[start] in _LEADING:
        start += 1
    while end > start:
        if text[end - 1] in _TRAILING:
            end -= 1
        elif text[end - 1] == "." and sum(ch.isalpha() for ch in text[start : end - 1]) >= MIN_PROPAGATE:
            end -= 1
        else:
            break
    return dataclasses.replace(entity, start=start, end=end)


def _pattern(surface: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(surface) + r"(?!\w)")


def placeholders(text: str, entities: Iterable[Entity]) -> dict[str, str]:
    """Surface form -> placeholder, numbered per tag in order of first appearance."""
    mapping: dict[str, str] = {}
    counts: dict[str, int] = {}
    for e in sorted(entities, key=lambda e: e.start):
        surface = text[e.start : e.end]
        if e.label in SCRUB_TAGS and surface.strip() and surface not in mapping:
            counts[e.label] = counts.get(e.label, 0) + 1
            mapping[surface] = f"[{e.label}_{counts[e.label]}]"
    return mapping


def replace_forms(text: str, mapping: dict[str, str]) -> str:
    """Replace every occurrence of the propagating forms (longest first, whole words only)."""
    for surface in sorted(mapping, key=len, reverse=True):
        if sum(ch.isalpha() for ch in surface) >= MIN_PROPAGATE:
            text = _pattern(surface).sub(mapping[surface], text)
    return text


def scrub_text(text: str, entities: Iterable[Entity], mapping: dict[str, str]) -> str:
    """The tagged spans replaced exactly (right to left), then the propagating forms."""
    spans = sorted((e for e in entities if e.label in SCRUB_TAGS), key=lambda e: e.start, reverse=True)
    out, last = text, len(text) + 1
    for e in spans:
        if e.end <= last:  # the model's spans do not overlap; skip one that would
            out = out[: e.start] + mapping[text[e.start : e.end]] + out[e.end :]
            last = e.start
    return replace_forms(out, mapping)


def scrub_document(
    doc: Document, entities: list[Entity], key: bytes | None = None
) -> tuple[Document, list[dict]]:
    """The document with its people replaced, and its scrub-log lines.

    A section found verbatim in the text gets the tagged spans that fall inside it; every section
    gets the propagating forms."""
    entities = [trim(e, doc.text) for e in entities if e.label in SCRUB_TAGS]
    entities = [e for e in entities if e.end > e.start]
    mapping = placeholders(doc.text, entities)
    if not mapping:
        return doc, []
    text = scrub_text(doc.text, entities, mapping)
    sections, cursor = [], 0
    for s in doc.sections:
        at = doc.text.find(s.text, cursor) if s.text else -1
        if at >= 0:
            inside = [dataclasses.replace(e, start=e.start - at, end=e.end - at) for e in entities
                      if at <= e.start and e.end <= at + len(s.text)]  # fmt: skip
            stext, cursor = scrub_text(s.text, inside, mapping), at
        else:
            stext = replace_forms(s.text, mapping)
        sections.append(dataclasses.replace(s, text=stext))
    scrubbed = dataclasses.replace(doc, text=text, sha256="", sections=sections)  # sha256 recomputed
    log = [
        {
            "doc_id": doc.id,
            "label": e.label,
            "placeholder": mapping[doc.text[e.start : e.end]],
            "entity_hmac": entity_hash(doc.text[e.start : e.end], key if key is not None else scrub_key()),
            "start": e.start,
            "end": e.end,
            "score": e.score,
        }
        for e in sorted(entities, key=lambda e: e.start)
        if e.label in SCRUB_TAGS and doc.text[e.start : e.end] in mapping
    ]
    return scrubbed, log


def _require(python: Path) -> None:
    if not Path(python).exists():
        raise FileNotFoundError(
            f"{python}: build the scrub environment with `uv sync --locked --project envs/scrub`"
        )


def stream_worker(
    docs: Iterable[Document],
    python: Path = SCRUB_PYTHON,
    args: tuple[str, ...] = (),
    worker: Path = WORKER,
    in_flight: int = 1024,
    stats: dict | None = None,
) -> Iterator[tuple[Document, list[Entity]]]:
    """Each document with its entities, in order, from ONE worker process: the documents are fed on
    a thread while the results are read here, so memory stays bounded by ``in_flight`` documents
    whatever the corpus size (P4-L). ``in_flight`` must exceed the worker's ``--docs-per-batch``.
    The worker's log goes to a temporary file; its throughput line lands in ``stats``."""
    _require(python)
    err = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    proc = subprocess.Popen(
        [str(python), str(worker), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
        text=True, encoding="utf-8",
    )  # fmt: skip
    pending: queue.Queue = queue.Queue(maxsize=in_flight)
    feed_error: list[BaseException] = []

    def feed():
        try:
            for doc in docs:
                pending.put(doc)
                proc.stdin.write(json.dumps({"id": doc.id, "text": doc.text}, ensure_ascii=False) + "\n")
                proc.stdin.flush()  # a buffered line would never reach a worker waiting for its batch
        except BaseException as e:  # a worker that died (BrokenPipeError) or a bad input line
            feed_error.append(e)
        finally:
            pending.put(None)
            try:
                proc.stdin.close()
            except OSError:
                pass

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    try:
        while (doc := pending.get()) is not None:
            line = proc.stdout.readline()
            if not line:
                break
            row = json.loads(line)
            if row["id"] != doc.id:
                raise RuntimeError(f"scrub worker answered {row['id']!r} for {doc.id!r}")
            yield doc, [Entity(*e) for e in row["entities"]]
    finally:
        if proc.poll() is None and doc is not None:  # stopped early: do not wait for the rest
            proc.kill()
        code = proc.wait()
        while feeder.is_alive():  # a feeder blocked on a full queue: take its documents until it ends
            try:
                pending.get(timeout=0.1)
            except queue.Empty:
                pass
        err.seek(0)
        log = err.read()
        err.close()
    if code != 0 or feed_error or doc is not None:
        tail = "\n".join(log.strip().splitlines()[-20:])
        raise RuntimeError(
            f"scrub worker failed (exit {code}; {feed_error or 'no error in the feeder'}):\n{tail}"
        )
    if stats is not None:
        stats.update(json.loads(log.strip().splitlines()[-1]))


def run_worker(
    docs: list[Document], python: Path = SCRUB_PYTHON, args: tuple[str, ...] = (), worker: Path = WORKER
) -> tuple[dict, dict]:
    """Entities per document id from the NER worker in the scrub environment, and its throughput."""
    stats: dict = {}
    entities = {doc.id: ents for doc, ents in stream_worker(docs, python, args, worker, stats=stats)}
    return entities, stats


def iter_part(path: Path, part: int = 0, parts: int = 1) -> Iterator[Document]:
    """The documents on lines ``part, part + parts, part + 2 * parts, ...`` of ``path``: a source
    split across ``parts`` array tasks, each streaming its own share (P4-L)."""
    if not 0 <= part < parts:
        raise ValueError(f"part {part} is not in 0..{parts - 1}")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % parts == part and line.strip():
                yield Document.from_json(line)


def part_name(source: str, part: int, parts: int) -> str:
    """``<source>.jsonl``, or ``<source>.part-<k>-of-<n>.jsonl`` for one part of a split scrub."""
    return f"{source}.jsonl" if parts == 1 else f"{source}.part-{part}-of-{parts}.jsonl"


def scrub_file(
    source: str,
    in_path: Path,
    out_path: Path,
    log_path: Path,
    python: Path = SCRUB_PYTHON,
    part: int = 0,
    parts: int = 1,
    worker_args: tuple[str, ...] = (),
    worker: Path = WORKER,
    in_flight: int = 1024,
    progress_every: int = 10_000,
) -> dict:
    """Scrub one source (or one part of it) as a stream: documents in, scrubbed documents and the
    scrub log out, both written to ``*.partial`` and renamed only when the part is complete."""
    t0 = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_tmp, log_tmp = (p.with_name(p.name + ".partial") for p in (out_path, log_path))
    n_docs, n_entities, key, worker_stats = 0, 0, scrub_key(), {}
    docs = iter_part(in_path, part, parts)
    with open(out_tmp, "w", encoding="utf-8") as out, open(log_tmp, "w", encoding="utf-8") as log:
        for doc, entities in stream_worker(docs, python, worker_args, worker, in_flight, worker_stats):
            scrubbed, rows = scrub_document(doc, entities, key)
            out.write(scrubbed.to_json() + "\n")
            for row in rows:
                log.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n_docs, n_entities = n_docs + 1, n_entities + len(rows)
            if progress_every and n_docs % progress_every == 0:
                print(f"{source} part {part}/{parts}: {n_docs} docs, {n_entities} entities, "
                      f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)  # fmt: skip
    out_tmp.replace(out_path)
    log_tmp.replace(log_path)
    seconds = time.perf_counter() - t0
    return {"source": source, "part": part, "parts": parts, "docs": n_docs, "entities": n_entities,
            "seconds": round(seconds, 1), "docs_per_s": round(n_docs / seconds, 3) if seconds else None,
            "worker": worker_stats}  # fmt: skip


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scrub person names from a collected source.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--in", dest="in_path", type=Path)
    parser.add_argument("--out", type=Path, help="default data/scrubbed/<source>[.part-k-of-n].jsonl")
    parser.add_argument("--log", type=Path, help="default data/manifests/scrub_<source>[.part-k-of-n].jsonl")
    parser.add_argument("--part", type=int, default=0, help="this task's share: lines part, part+parts, ...")
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--docs-per-batch", type=int, default=1, help="documents per NER call (GPU: 64)")
    parser.add_argument("--batch-size", type=int, default=32, help="sentences per NER mini-batch")
    parser.add_argument("--cache-dir", help="the worker's model cache (default: its HF_HOME rule)")
    args = parser.parse_args(argv)
    name = part_name(args.source, args.part, args.parts)
    in_path = args.in_path or REPO_ROOT / "data" / "raw" / args.source / f"{args.source}.jsonl"
    out = args.out or REPO_ROOT / "data" / "scrubbed" / name
    log = args.log or REPO_ROOT / "data" / "manifests" / f"scrub_{name}"
    worker_args = ("--docs-per-batch", str(args.docs_per_batch), "--batch-size", str(args.batch_size))
    if args.cache_dir:
        worker_args += ("--cache-dir", args.cache_dir)
    in_flight = max(1024, 4 * args.docs_per_batch)
    stats = scrub_file(
        args.source, in_path, out, log, part=args.part, parts=args.parts, worker_args=worker_args,
        in_flight=in_flight,
    )  # fmt: skip
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
