"""Index versions: a registry of the retrieval indexes that have been built.

``scripts/build_index.py`` writes an index directory (``chunks.jsonl``, ``bm25/``, ``index.json``,
``documents.jsonl``); ``register`` records it in a registry, a JSON file
``{"versions": [<entry>, ...]}``, oldest first. An entry (``IndexVersion``) is

    version        1, 2, 3, ... in registration order (the largest so far plus one)
    kind           index.json's ``kind`` ("bm25")
    built_at       index.json's ``built_at`` (UTC, ISO)
    index_dir      the index directory, resolved to an absolute path
    chunks         index.json's chunk count
    chunks_sha256  hex SHA-256 of the bytes of ``chunks.jsonl``: what was indexed, exactly
    documents      index.json's document count
    inputs         index.json's input files

Registering a directory again while its ``chunks.jsonl`` is byte-identical to that directory's
latest entry is a no-op that returns that entry; a rebuild into the same directory that changed the
chunks is a new version (the earlier entry stays as the record of what was served before), also
when it restores the chunks of an older entry, so the directory's latest entry is what it holds. The registry is rewritten atomically
(``write_atomic``: a temporary file in the same directory, flushed, fsynced and ``os.replace``d), so
a reader sees the old or the new registry, never a partial one. It assumes one writer at a time
(builds are run by hand or by one nightly job).

``documents.jsonl`` holds one ``DocumentDigest`` row per input document of an indexed type
(``INDEXED_TYPES``: statute, regulation, decision; general text and parliamentary papers add no
chunks, so they are not part of what an index was built from): ``{"id", "sha256", "record_sha256"}``.
``sha256`` is the document's own hash of its text; ``record_sha256`` hashes the whole record except
``retrieved_at`` (which changes on every collection) and the derived ``sha256``, so it also changes
when only the metadata the chunks carry changes -- a repealed provision that gains a ``valid_to``, a
moved heading, a licence. ``indexed_documents`` reads it for the nightly re-sync
(``scripts/resync_index.py``).
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from lexhybrid.data.schema import Document
from lexhybrid.retrieval.chunking import STATUTE_TYPES

MANIFEST_NAME = "index.json"
CHUNKS_NAME = "chunks.jsonl"
DOCUMENTS_NAME = "documents.jsonl"
INDEXED_TYPES = (*STATUTE_TYPES, "decision")  # what chunk_document chunks; other types add nothing
_NOT_RECORDED = ("retrieved_at", "sha256")  # left out of record_sha256


@dataclass(frozen=True)
class IndexVersion:
    """One registered index build. See the module docstring."""

    version: int
    kind: str
    built_at: str
    index_dir: str
    chunks: int
    chunks_sha256: str
    documents: int
    inputs: tuple[str, ...]

    def to_dict(self) -> dict:
        return {**asdict(self), "inputs": list(self.inputs)}

    @classmethod
    def from_dict(cls, d: dict) -> "IndexVersion":
        return cls(**{**d, "inputs": tuple(d["inputs"])})


@dataclass(frozen=True)
class DocumentDigest:
    """What an index records of one input document (a ``documents.jsonl`` row)."""

    id: str
    sha256: str  # of the text (``Document.sha256``)
    record_sha256: str  # of the record without retrieved_at and sha256

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def document_digest(doc: Document) -> DocumentDigest:
    record = {k: v for k, v in doc.to_dict().items() if k not in _NOT_RECORDED}
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return DocumentDigest(doc.id, doc.sha256, hashlib.sha256(canonical.encode("utf-8")).hexdigest())


def file_sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(block):
            h.update(data)
    return h.hexdigest()


def read_manifest(index_dir: str | Path) -> dict:
    """``index.json`` of an index directory; ``FileNotFoundError`` when it is not an index."""
    path = Path(index_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{index_dir}: no {MANIFEST_NAME} (not built by scripts/build_index.py)")
    return json.loads(path.read_text(encoding="utf-8"))


def indexed_documents(index_dir: str | Path) -> dict[str, DocumentDigest]:
    """``{document id: digest}`` of the documents an index was built from (``documents.jsonl``)."""
    path = Path(index_dir) / DOCUMENTS_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{index_dir}: no {DOCUMENTS_NAME} (built before document storage; rebuild with scripts/build_index.py)"
        )
    out: dict[str, DocumentDigest] = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if line.strip():
                row = DocumentDigest(**json.loads(line))
                if row.id in out:
                    raise ValueError(f"{path}:{n}: document {row.id!r} listed twice")
                out[row.id] = row
    return out


def versions(registry_path: str | Path) -> list[IndexVersion]:
    """Every registered version, oldest first; ``[]`` when the registry does not exist yet."""
    path = Path(registry_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("versions"), list):
        raise ValueError(f'{registry_path}: not an index registry ({{"versions": [...]}})')
    return [IndexVersion.from_dict(e) for e in data["versions"]]


def latest(registry_path: str | Path) -> IndexVersion | None:
    """The highest version, or ``None`` when nothing is registered."""
    return max(versions(registry_path), key=lambda v: v.version, default=None)


def write_atomic(path: str | Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one step: a reader sees the old file or the new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    umask = os.umask(0)
    os.umask(umask)
    try:
        os.fchmod(fd, 0o666 & ~umask)  # mkstemp's 0600 would make the file owner-only, unlike open()
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def register(index_dir: str | Path, registry_path: str | Path) -> IndexVersion:
    """Record the index in ``index_dir`` as the next version, or return the directory's latest
    entry when that entry has the same ``chunks.jsonl``."""
    index_dir = Path(index_dir).resolve()
    manifest = read_manifest(index_dir)
    chunks_path = index_dir / CHUNKS_NAME
    if not chunks_path.is_file():
        raise FileNotFoundError(f"{index_dir}: no {CHUNKS_NAME}")
    digest = file_sha256(chunks_path)
    known = versions(registry_path)
    # against this directory's latest entry only: chunks X, Y, then X again is a third version
    current = max((v for v in known if v.index_dir == str(index_dir)), key=lambda v: v.version, default=None)
    if current is not None and current.chunks_sha256 == digest:
        return current
    entry = IndexVersion(
        version=max((v.version for v in known), default=0) + 1,
        kind=manifest["kind"],
        built_at=manifest["built_at"],
        index_dir=str(index_dir),
        chunks=int(manifest["chunks"]),
        chunks_sha256=digest,
        documents=int(manifest["documents"]),
        inputs=tuple(manifest.get("inputs", ())),
    )
    registry = {"versions": [v.to_dict() for v in [*known, entry]]}
    write_atomic(registry_path, json.dumps(registry, indent=1, ensure_ascii=False) + "\n")
    return entry
