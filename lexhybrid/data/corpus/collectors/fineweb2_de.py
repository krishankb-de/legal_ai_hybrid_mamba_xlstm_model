"""FineWeb-2 German (``deu_Latn``): general-German web text, the replay share of the mixture (plan P3-N).

Source: the Hugging Face dataset ``HuggingFaceFW/fineweb-2``, config ``deu_Latn``, streamed with
``datasets`` (nothing is downloaded beyond the rows read). A row is one CommonCrawl page after
FineWeb-2's extraction, filtering and deduplication: ``text`` (paragraphs separated by newlines),
``id`` (``<urn:uuid:...>``), ``url``, ``date`` (the crawl time) and ``dump``.

One ``Document`` per page: ``doc_type=general``, ``jurisdiction=DE`` (the schema's convention for
general-German text), no citation id, one section per paragraph, ``valid_from`` = the crawl date.
Rows come in the dataset's order (oldest crawl first); P4's at-scale job samples across dumps.
Licence: ODC-By 1.0, and CommonCrawl's terms of use apply (register row ``fineweb2_de``).
"""

from collections.abc import Iterator

from lexhybrid.data.corpus.collectors.base import cli, licence_flags, utc_now
from lexhybrid.data.schema import Document, Section

DATASET = "HuggingFaceFW/fineweb-2"
CONFIG = "deu_Latn"


def parse_fineweb_row(row: dict, retrieved_at: str | None = None) -> Document | None:
    """One FineWeb-2 row as a ``Document``; None when it has no text."""
    paragraphs = [" ".join(p.split()) for p in (row.get("text") or "").split("\n")]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return None
    uid = (row.get("id") or "").strip("<>").removeprefix("urn:uuid:")
    return Document(
        id=f"fineweb2_de:{uid}",
        source="fineweb2_de",
        jurisdiction="DE",
        doc_type="general",
        text="\n".join(paragraphs),
        url=row.get("url") or "",
        valid_from=(row.get("date") or "")[:10] or None,
        retrieved_at=retrieved_at or utc_now(),
        sections=[Section(label="Text", text=p) for p in paragraphs],
        **licence_flags("ODC-By-1.0"),
    )


class FineWeb2DECollector:
    name = "fineweb2_de"
    licence = "ODC-By-1.0"
    jurisdiction = "DE"

    def __init__(self, dataset: str = DATASET, config: str = CONFIG, split: str = "train"):
        self.dataset, self.config, self.split = dataset, config, split

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]:
        from datasets import load_dataset  # heavy; only when collecting

        rows = load_dataset(self.dataset, name=self.config, split=self.split, streaming=True)
        n = 0
        for row in rows:
            doc = parse_fineweb_row(row)
            if doc is None:
                continue
            yield doc
            n += 1
            if limit is not None and n >= limit:
                return


if __name__ == "__main__":
    raise SystemExit(cli(FineWeb2DECollector()))
