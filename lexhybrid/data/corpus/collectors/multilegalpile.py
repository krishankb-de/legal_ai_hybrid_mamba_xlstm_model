"""Multi Legal Pile, German parts: the research-exception arm (plan P3-O, decision 2).

Source: the Hugging Face dataset ``joelniklaus/Multi_Legal_Pile`` (not gated, no token needed),
read straight from its ``data/<lang>/<type>/<source>.jsonl.xz`` files at a pinned revision with
``huggingface_hub``'s file system and an incremental xz decoder, so only the rows read are fetched
(the German case-law files are 0.8 and 1.3 GB). Its loading script is not used: ``datasets`` 5
runs no dataset scripts.

A row is ``{language, type, jurisdiction, text}``: no id, date, URL or citation. One ``Document``
per row: id ``multilegalpile:<subset>:<row>`` (stable at the pinned revision), ``type`` caselaw ->
``decision``, legislation -> ``statute``, ``jurisdiction`` Germany -> DE and Switzerland -> CH,
paragraphs as sections, ``valid_from`` unknown (None). The German parts are Swiss cantonal
legislation (lexfind, no licence information), Swiss court decisions (entscheidsuche) and German
statutes and decisions (Open Legal Data); the last two overlap the ``bger``/``oldp`` collectors and
deduplication (P3-S) keeps the commercial-safe copy.

Licence: the dataset is CC BY-NC-SA 4.0, so every document is ``research_only`` and never
``commercial_safe``; it enters only the ``research`` arm (register row ``multilegalpile``).
"""

import itertools
import lzma
from collections.abc import Iterator

from lexhybrid.data.corpus.collectors.base import licence_flags, main, utc_now
from lexhybrid.data.schema import Document, Section

REPO = "joelniklaus/Multi_Legal_Pile"
REVISION = "911e1d214162fd11d2c78d3f1428cbfcbe07782c"  # main on 2026-09-27
# (language, type, source): the German native parts, the content the other collectors lack first.
SUBSETS = (
    ("de", "legislation", "switzerland_lexfind"),
    ("de", "caselaw", "switzerland_entscheidsuche"),
    ("de", "legislation", "germany_openlegaldata"),
    ("de", "caselaw", "germany_openlegaldata"),
)
JURISDICTIONS = {"Germany": "DE", "Switzerland": "CH", "Austria": "AT", "EU": "EU"}
DOC_TYPES = {"caselaw": "decision", "legislation": "statute"}


def subset_name(language: str, typ: str, source: str) -> str:
    return f"{language}_{typ}_{source}"


def parse_mlp_row(row: dict, subset: str, index: int, retrieved_at: str | None = None) -> Document | None:
    """Row ``index`` (0-based) of ``subset`` as a ``Document``; None without text or with a
    jurisdiction or type outside the schema."""
    jurisdiction = JURISDICTIONS.get(row.get("jurisdiction") or "")
    doc_type = DOC_TYPES.get(row.get("type") or "")
    paragraphs = [" ".join(p.replace("\xa0", " ").split()) for p in (row.get("text") or "").split("\n")]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs or jurisdiction is None or doc_type is None:
        return None
    language, typ, source = subset.split("_", 2)
    return Document(
        id=f"multilegalpile:{subset}:{index}",
        source="multilegalpile",
        jurisdiction=jurisdiction,
        doc_type=doc_type,
        text="\n".join(paragraphs),
        url=f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/data/{language}/{typ}/{source}.jsonl.xz",
        retrieved_at=retrieved_at or utc_now(),
        sections=[Section(label="Text", text=p) for p in paragraphs],
        **licence_flags("CC-BY-NC-SA-4.0"),
    )


def iter_rows(language: str, typ: str, source: str, revision: str = REVISION) -> Iterator[dict]:
    """The rows of one ``.jsonl.xz`` file, decompressed as they arrive."""
    import json

    from huggingface_hub import HfFileSystem

    path = f"datasets/{REPO}@{revision}/data/{language}/{typ}/{source}.jsonl.xz"
    with HfFileSystem().open(path, "rb", block_size=1 << 20) as raw, lzma.open(raw) as lines:
        for line in lines:
            yield json.loads(line)


class MultiLegalPileCollector:
    name = "multilegalpile"
    licence = "CC-BY-NC-SA-4.0"
    jurisdiction = "DE"  # and CH

    def __init__(self, subsets: tuple[tuple[str, str, str], ...] = SUBSETS, revision: str = REVISION):
        self.subsets, self.revision = subsets, revision

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        """Each subset in turn; with a limit, an equal share of it from each."""
        share = None if limit is None else -(-limit // len(self.subsets))
        n = 0
        for language, typ, source in self.subsets:
            subset = subset_name(language, typ, source)
            docs = (
                parse_mlp_row(row, subset, i)
                for i, row in enumerate(iter_rows(language, typ, source, self.revision))
            )
            for doc in itertools.islice((d for d in docs if d is not None), share):
                yield doc
                n += 1
                if limit is not None and n >= limit:
                    return


if __name__ == "__main__":
    main(MultiLegalPileCollector())
