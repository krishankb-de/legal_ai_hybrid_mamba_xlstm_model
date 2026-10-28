"""Retrieval filters: jurisdiction, validity date and document type.

``filter_chunks`` keeps, in input order, the chunks that pass every filter given; a filter left at
``None`` keeps everything. ``jurisdictions`` and ``doc_types`` are the allowed values, each checked
against ``lexhybrid.data.schema.JURISDICTIONS`` (``DE``, ``AT``, ``CH``, ``EU``) and ``DOC_TYPES``;
an unknown value raises ``ValueError`` (a typo must not silently empty the result), a single string
counts as one value, and an empty collection allows nothing.

``valid_on`` is a day (``datetime.date`` or an ISO date string). A chunk is valid on day ``d`` when
``valid_from <= d <= valid_to``, both bounds inclusive and a missing bound open: a provision is in
force on the day it takes effect and on its last day. A chunk with neither bound -- a decision
without a recorded date, a provision whose source gives no version date -- is always kept; a
decision whose collector records its date as ``valid_from`` is kept from that day on (it cannot be
cited before it was handed down). A chunk whose date is not an ISO date raises ``ValueError`` naming
the chunk, when the date filter reads it.
"""

from collections.abc import Iterable
from datetime import date, datetime
from functools import lru_cache

from lexhybrid.data.schema import DOC_TYPES, JURISDICTIONS
from lexhybrid.retrieval.chunking import Chunk


def _day(value: str | date) -> date:
    """A ``date`` from a date, a datetime (its day) or an ISO date string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"a day is a datetime.date or an ISO date string, got {type(value).__name__}")


@lru_cache(maxsize=4096)
def _parse(value: str) -> date:
    return date.fromisoformat(value)  # versions share stand dates, so the cache hits


def _bound(chunk: Chunk, name: str) -> date | None:
    value = getattr(chunk, name)
    if value is None:
        return None
    try:
        return _parse(value)
    except (TypeError, ValueError):
        raise ValueError(f"{chunk.id}: {name} {value!r} is not an ISO date") from None


def _valid(chunk: Chunk, day: date) -> bool:
    start, end = _bound(chunk, "valid_from"), _bound(chunk, "valid_to")
    return (start is None or start <= day) and (end is None or day <= end)


def valid_on(chunk: Chunk, day: str | date) -> bool:
    """Whether ``chunk`` is valid on ``day``: ``valid_from <= day <= valid_to``, a missing bound open."""
    return _valid(chunk, _day(day))


def _allowed(values: Iterable[str] | str, known: tuple[str, ...], what: str) -> frozenset[str]:
    allowed = frozenset((values,) if isinstance(values, str) else values)
    unknown = sorted(allowed - set(known))
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; known: {known}")
    return allowed


def filter_chunks(
    chunks: Iterable[Chunk],
    jurisdictions: Iterable[str] | str | None = None,
    valid_on: str | date | None = None,
    doc_types: Iterable[str] | str | None = None,
) -> list[Chunk]:
    """The chunks, in input order, whose jurisdiction and doc_type are allowed and that are valid on
    ``valid_on``; a filter left at ``None`` is not applied. See the module docstring."""
    places = None if jurisdictions is None else _allowed(jurisdictions, JURISDICTIONS, "jurisdiction")
    types = None if doc_types is None else _allowed(doc_types, DOC_TYPES, "doc_type")
    day = None if valid_on is None else _day(valid_on)
    return [
        c
        for c in chunks
        if (places is None or c.jurisdiction in places)
        and (types is None or c.doc_type in types)
        and (day is None or _valid(c, day))
    ]
