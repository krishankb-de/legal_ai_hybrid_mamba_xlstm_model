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
"""

import argparse
import dataclasses
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import time
from collections.abc import Iterable
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


def run_worker(
    docs: list[Document], python: Path = SCRUB_PYTHON, args: tuple[str, ...] = ()
) -> tuple[dict, dict]:
    """Entities per document id from the NER worker in the scrub environment, and its throughput."""
    if not Path(python).exists():
        raise FileNotFoundError(
            f"{python}: build the scrub environment with `uv sync --locked --project envs/scrub`"
        )
    payload = "".join(json.dumps({"id": d.id, "text": d.text}, ensure_ascii=False) + "\n" for d in docs)
    proc = subprocess.run(
        [str(python), str(WORKER), *args], input=payload, capture_output=True, text=True, check=True
    )
    entities = {}
    for line in proc.stdout.splitlines():
        row = json.loads(line)
        entities[row["id"]] = [Entity(*e) for e in row["entities"]]
    stats = json.loads(proc.stderr.strip().splitlines()[-1])
    return entities, stats


def scrub_file(
    source: str, in_path: Path, out_path: Path, log_path: Path, python: Path = SCRUB_PYTHON
) -> dict:
    docs = [
        Document.from_json(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    t0 = time.perf_counter()
    entities, worker = run_worker(docs, python)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    n_entities, key = 0, scrub_key()
    with open(out_path, "w", encoding="utf-8") as out, open(log_path, "w", encoding="utf-8") as log:
        for doc in docs:
            scrubbed, rows = scrub_document(doc, entities.get(doc.id, []), key)
            out.write(scrubbed.to_json() + "\n")
            for row in rows:
                log.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n_entities += len(rows)
    seconds = time.perf_counter() - t0
    return {"source": source, "docs": len(docs), "entities": n_entities, "seconds": round(seconds, 1),
            "docs_per_s": round(len(docs) / seconds, 3), "worker": worker}  # fmt: skip


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scrub person names from a collected source.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--in", dest="in_path", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args(argv)
    in_path = args.in_path or REPO_ROOT / "data" / "raw" / args.source / f"{args.source}.jsonl"
    out = args.out or REPO_ROOT / "data" / "scrubbed" / f"{args.source}.jsonl"
    log = args.log or REPO_ROOT / "data" / "manifests" / f"scrub_{args.source}.jsonl"
    print(json.dumps(scrub_file(args.source, in_path, out, log)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
