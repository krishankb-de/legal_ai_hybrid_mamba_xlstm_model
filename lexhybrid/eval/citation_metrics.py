"""Citation metrics of answer records (``lexhybrid.eval.records``), per question and per run.

Per record (``record_metrics``), numeric only; a metric that does not apply is left out, so
``scripts/bootstrap_compare.py`` compares it on the records where both systems have it:

* ``citation_precision``: from the record's verifier log, the share of cited claims that pass
  pointer resolution, the quote check and NLI (absent when no claim cites). **System-internal**: the
  system was tuned against this verifier (best-of-n scores samples with it), so it is not an
  independent measure of faithfulness;
* ``citation_recall`` (records with ``gold_chunk_ids``): the share of the gold passages that the
  rendered answer cites, i.e. the pointers of the claims the verifier kept (a refusal cites none). A
  cited chunk counts for a gold passage when it is that passage or a copy of it (``gold_relation``:
  the same document, or linked through ``Chunk.copies``);
* ``quote_fidelity`` (records whose rendered answer shows a quote): 1.0 when every rendered quote
  equals its source after NFC and whitespace normalisation (``quotes.check_quotes`` on the kept
  segments of the raw answer, the record's chunks and its rendered text), else 0.0. The verifier
  refuses an answer whose quotes drift, so anything below 1.0 is a regression;
* ``refused``, ``should_refuse``, ``refusal_correct``: 0 or 1;
* ``exact_match``, ``char_f1`` (records with ``gold_answers``): the best over the gold answers of the
  probes' scorers on the answer's own words (``answer_words``: the kept text and the quoted source
  sentences, without quotation marks, labels, source lists or the notice; empty for a refusal).

A pointer that does not resolve (a passage outside ``1..len(chunks)``, a sentence outside its
passage) cites nothing and adds no words: it never wraps around to another passage.

``summarise`` pools them per (system, seed, arm): citation precision over all cited claims, mean
recall, fidelity, EM and char-F1, and refusal precision, recall and F1 with "should refuse" as the
positive class (a zero division gives 0; all three are ``None`` when no record should be refused).
"""

from collections.abc import Callable, Iterable, Sequence

from lexhybrid.data.probes.statute_recall import char_f1, exact_match
from lexhybrid.eval.records import AnswerRecord
from lexhybrid.renderer.segments import CitePointer, QuotePointer, Segment, Text, parse_answer
from lexhybrid.retrieval.chunking import Chunk
from lexhybrid.verifier.pointers import check_pointer
from lexhybrid.verifier.quotes import check_quotes, source_text
from lexhybrid.verifier.verifier import split_claims

SYSTEM_INTERNAL = "system-internal: the system was tuned against this verifier"


def kept_segments(record: AnswerRecord) -> list[Segment]:
    """The segments of the claims the verifier kept, i.e. what the rendered answer shows (none when refused)."""
    if record.refused:
        return []
    passed = {c["index"] for c in record.verifier.get("claims", []) if c["passed"]}
    claims = split_claims(parse_answer(record.raw_answer))
    return [s for n, claim in enumerate(claims) if n in passed for _, s in claim]


def doc_of(chunk_id: str) -> str:
    """The document of a chunk id (``<document id>#<n>``)."""
    return chunk_id.rsplit("#", 1)[0]


def gold_relation(docs: Iterable[str], copies: Iterable[str] = ()) -> Callable[[str, Iterable[str]], bool]:
    """``(doc_id, copies) -> bool``: a chunk is a gold passage or a copy of one when it shares a document
    with one of the gold ``docs`` or is linked to one through ``Chunk.copies`` (the gold chunks'
    ``copies``, or its own)."""
    docs = set(docs)
    linked = docs | set(copies)
    return lambda doc_id, chunk_copies: doc_id in linked or not docs.isdisjoint(chunk_copies)


def cited_passages(segments: Sequence[Segment], chunks: Sequence[Chunk]) -> list[int | None]:
    """Each pointer's passage as a 0-based index into ``chunks``, in order; ``None`` for a pointer that
    does not resolve (``pointers.check_pointer``)."""
    return [
        None if check_pointer(i, s, chunks) else s.passage - 1
        for i, s in enumerate(segments)
        if isinstance(s, QuotePointer | CitePointer)
    ]


def answer_words(segments: Sequence[Segment], chunks: Sequence[Chunk]) -> str:
    """The answer's own words: its text with each quote as the source sentences, whitespace-normalised;
    a citation adds nothing (its label and the space before it go)."""
    words = ""
    for s in segments:
        if isinstance(s, Text):
            words += s.text
        elif isinstance(s, QuotePointer) and not check_pointer(0, s, chunks):
            words += source_text(chunks[s.passage - 1], s.sentences)
        elif isinstance(s, CitePointer):
            words = words.rstrip()
    return " ".join(words.split())


def record_metrics(record: AnswerRecord) -> dict[str, float]:
    """One record's metrics (see the module docstring)."""
    m = {
        "refused": float(record.refused),
        "should_refuse": float(record.should_refuse),
        "refusal_correct": float(record.refused == record.should_refuse),
    }
    if record.verifier.get("citation_precision") is not None:
        m["citation_precision"] = float(record.verifier["citation_precision"])
    kept, chunks = kept_segments(record), record.chunk_objects()
    if record.gold_chunk_ids:
        cited = [chunks[k] for k in cited_passages(kept, chunks) if k is not None]
        copies = {c.id: c.copies for c in chunks}
        gold = list(dict.fromkeys(record.gold_chunk_ids))
        relations = [gold_relation([doc_of(g)], copies.get(g, ())) for g in gold]
        m["citation_recall"] = sum(any(rel(c.doc_id, c.copies) for c in cited) for rel in relations) / len(
            gold
        )
    if any(isinstance(s, QuotePointer) for s in kept):
        m["quote_fidelity"] = float(not check_quotes(kept, chunks, record.rendered))
    if record.gold_answers:
        words = answer_words(kept, chunks)
        m["exact_match"] = max(float(exact_match(words, g)) for g in record.gold_answers)
        m["char_f1"] = max(char_f1(words, g) for g in record.gold_answers)
    return m


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise(records: Iterable[AnswerRecord]) -> list[dict]:
    """One summary per (system, seed, arm), in order of first appearance (see the module docstring)."""
    groups: dict[tuple, list[AnswerRecord]] = {}
    for r in records:
        groups.setdefault((r.system, r.seed, r.arm), []).append(r)
    out = []
    for (system, seed, arm), rs in groups.items():
        metrics = [record_metrics(r) for r in rs]
        column = {k: [m[k] for m in metrics if k in m] for k in ("citation_recall", "quote_fidelity",
                                                                  "exact_match", "char_f1", "refusal_correct")}  # fmt: skip
        cited = sum(r.verifier.get("cited_claims", 0) for r in rs)
        passed = sum(r.verifier.get("passed_cited_claims", 0) for r in rs)
        refused, should = sum(r.refused for r in rs), sum(r.should_refuse for r in rs)
        correct = sum(r.refused and r.should_refuse for r in rs)
        precision = recall = f1 = None  # no record should be refused: the positive class is absent
        if should:
            precision, recall = (correct / refused if refused else 0.0), correct / should
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append({
            "system": system, "seed": seed, "arm": arm, "sets": sorted({r.set for r in rs}), "records": len(rs),
            "citation_precision": _ratio(passed, cited), "citation_precision_scope": SYSTEM_INTERNAL,
            "cited_claims": cited, "passed_cited_claims": passed,
            "citation_recall": _mean(column["citation_recall"]), "recall_records": len(column["citation_recall"]),
            "quote_fidelity": _mean(column["quote_fidelity"]), "quote_records": len(column["quote_fidelity"]),
            "refusal_precision": precision, "refusal_recall": recall, "refusal_f1": f1,
            "refusal_accuracy": _mean(column["refusal_correct"]),
            "refused": refused, "should_refuse": should, "correct_refusals": correct,
            "exact_match": _mean(column["exact_match"]), "char_f1": _mean(column["char_f1"]),
            "answer_records": len(column["exact_match"]),
        })  # fmt: skip
    return out
