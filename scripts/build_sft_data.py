#!/usr/bin/env python3
"""SFT data: synthetic pointer-answer items and extractive QA items, as token-id splits.

    .venv/bin/python scripts/build_sft_data.py --synthetic data/sft/synthetic.jsonl \\
        --qa data/qa/items.jsonl --out data/sft --manifest-dir <complete evalset manifests>

Inputs (each flag repeatable; row ids must be unique across all of them):

* ``--synthetic``: one JSON per line, ``{id, question_id, source, kind, chunk_ids, text}``. ``text``
  is ``prompt + target`` exactly as ``sft_format.render`` writes them (``parse`` gives the example
  back; anything else is an error naming the row's id and line). ``kind`` is ``answerable``,
  ``unanswerable``, ``false_premise`` or ``party``; every kind but ``answerable`` has the target
  ``<|unanswerable|>`` + EOS.
* ``--qa``: extractive QA items as the evaluation-set writers store them (``QAItem``: ``{id,
  question, answers, context, context_id}``, ids ``<set>:<n>``; the set name is the item's source),
  with the writer's ``info.json`` beside the file: it must say ``eval_only: false`` and
  ``research_only: false``, or the file is refused. The context, split with ``split_sentences``, is
  passage 1 (at most 64 sentences, the window holding the answer), cited by ``context_id``; the
  target quotes the sentences the first answer span covers. A span not found in its context is
  dropped and counted. The question id is the hash of the normalised question, so a question asked
  twice stays on one side of the split.

**No evaluation item enters training.** An item is refused (dropped, counted as ``eval_overlap``)
when its source names an evaluation-only set, or when any form of its question -- as written, NFC,
NFD, whitespace-normalised, without a final ``?``/``.`` or with one -- has a ``sha256_text`` listed in
an ``evalset_*.jsonl`` manifest (they hash the raw question). A manifest lists only the items fetched
(LegalQuAD's lists 20 of 200), so a question of such a set must not pass on its hash alone.

Outputs under ``--out``:

* ``train.parquet``, ``dev.parquet``: one row per item with ``id``, ``question_id``, ``source``,
  ``kind`` and ``sft_format.encode``'s ``input_ids`` and ``labels`` (-100 over the prompt; its
  ``doc_ids`` are all 0 and not stored).
* ``train.jsonl``, ``dev.jsonl``: the kept synthetic rows of each split in the generator's format,
  for the preference-pair builder.
* ``manifest.json``: counts by split, kind and source, the drops, the non-answerable share, warnings.

The dev split holds out ``--dev-fraction`` of the question ids by seeded hash, so one question's items
never straddle the splits. Every target ends in exactly one EOS (asserted). An item longer than
``--max-len`` tokens is dropped and counted. QA items are all answerable: when they pull the
non-answerable share below 0.2, a seeded subset of them is dropped (``qa_subsampled``) until it is
0.2. A share outside [0.2, 0.3] after that fails the build.
"""

import argparse
import hashlib
import json
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from lexhybrid.data.datasets import _unit
from lexhybrid.data.evalsets.base import question_key
from lexhybrid.data.schema import sha256_text
from lexhybrid.data.sft_format import Example, Passage, Quote, encode, parse, render, split_sentences
from lexhybrid.data.tokenizer import EOS_ID, N_SENTENCES

REPO_ROOT = Path(__file__).resolve().parent.parent
KINDS = ("answerable", "unanswerable", "false_premise", "party")
UNANSWERABLE_SHARE = (0.2, 0.3)
DEV_FRACTION = 0.05
MAX_LEN = 8192  # the model's trained length; a prompt carries up to ~4,096 tokens of passages
SPLITS = ("train", "dev")
ROW_KEYS = ("id", "question_id", "source", "kind", "chunk_ids", "text")


@dataclass(frozen=True)
class Item:
    id: str
    question_id: str
    source: str
    kind: str
    example: Example
    row: dict | None = None  # the generator row of a synthetic item; None for a QA item


def read_jsonl(path) -> list[tuple[int, dict]]:
    """``(line number, row)`` of every non-empty line."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    return [(n, json.loads(line)) for n, line in enumerate(lines, start=1) if line.strip()]


def evalset_guard(manifest_dir: Path) -> tuple[set[str], set[str]]:
    """The question hashes of every ``evalset_*.jsonl`` and the names of the evaluation-only sets."""
    paths = sorted(Path(manifest_dir).glob("evalset_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no evalset_*.jsonl in {manifest_dir}: the overlap check would be empty")
    hashes, names = set(), set()
    for path in paths:
        (_, info), *rows = read_jsonl(path)
        if info["info"].get("eval_only", True):
            names.add(info["info"]["name"])
        hashes.update(r["sha256"] for _, r in rows)
        hashes.update(r["question_key"] for _, r in rows if "question_key" in r)
    return hashes, names


def question_forms(question: str) -> set[str]:
    """Every combination of NFC/NFD, whitespace normalisation and the final ``?``/``.`` (dropped or
    added) of a question: a copy that differs from an evaluation question only so is still that question."""
    forms = {question}
    for change in (
        lambda s: {unicodedata.normalize("NFC", s), unicodedata.normalize("NFD", s)},
        lambda s: {" ".join(s.split())},
        lambda s: {(bare := s.rstrip("?.").rstrip()), bare + "?", bare + "."},
    ):
        forms |= {f for s in forms for f in change(s)}
    return forms


def is_eval(source: str, question: str, hashes: set[str], names: set[str]) -> bool:
    if source in names or question_key(question) in hashes:
        return True
    return not hashes.isdisjoint(
        sha256_text(f) for f in question_forms(question)
    )  # manifests with raw hashes only


def question_id(question: str) -> str:
    """One id per question however it was typed, for every input: the split keeps a question on one side."""
    return "q:" + question_key(question)[:16]


def synthetic_item(row: dict, where: str) -> Item:
    """A generator row; its text must be ``render``'s and its kind must match its target."""
    if missing := [k for k in ROW_KEYS if k not in row]:
        raise ValueError(f"{where}: row {row.get('id')!r}: no {missing}")
    try:
        ex = parse(row["text"])
    except ValueError as e:
        raise ValueError(f"{where}: row {row.get('id')!r}: {e}") from e
    if "".join(render(ex)) != row["text"]:
        raise ValueError(f"{where}: row {row['id']!r}: text is not render(parse(text))")
    if row["kind"] not in KINDS or ex.unanswerable != (row["kind"] != "answerable"):
        raise ValueError(f"{where}: row {row['id']!r}: kind {row['kind']!r} does not match its target")
    qid = question_id(ex.question)
    return Item(
        row["id"], qid, row["source"], row["kind"], ex, {k: row[k] for k in ROW_KEYS} | {"question_id": qid}
    )


def qa_item(row: dict) -> Item | None:
    """An extractive QA item as quotes of the sentences its first answer span covers; None when the
    span is not in the context or covers more sentences than a passage holds."""
    sentences = split_sentences(row["context"])
    context = " ".join(sentences)  # = the context with its whitespace normalised, as the sentences are
    answer = " ".join(row["answers"][0].split()) if row["answers"] else ""
    start = context.find(answer) if answer else -1
    if start < 0:
        return None
    end, offset, hit = start + len(answer), 0, []
    for j, s in enumerate(sentences):
        if offset < end and offset + len(s) > start:
            hit.append(j)
        offset += len(s) + 1
    lo = max(0, min(hit[0], len(sentences) - N_SENTENCES))
    if hit[-1] >= lo + N_SENTENCES:
        return None
    passage = Passage(row.get("context_id") or row["id"], sentences[lo : lo + N_SENTENCES])
    ex = Example(row["question"], (passage,), tuple(Quote(1, j - lo + 1) for j in hit))
    return Item(row["id"], question_id(row["question"]), row["id"].partition(":")[0], "answerable", ex)


def qa_info(path: Path) -> dict:
    """The ``info.json`` beside a QA file; refused unless it frees the items for training."""
    info_path = Path(path).with_name("info.json")
    if not info_path.is_file():
        raise FileNotFoundError(f"{path}: no info.json beside it, so its licence and use are unknown")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("eval_only", True) or info.get("research_only", True):
        raise ValueError(
            f"{path}: info.json has eval_only={info.get('eval_only')}, research_only={info.get('research_only')}; "
            "SFT takes only items free for training (both false)"
        )
    return info


def write_split(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("id", pa.string()), ("question_id", pa.string()), ("source", pa.string()), ("kind", pa.string()),
        ("input_ids", pa.list_(pa.int32())), ("labels", pa.list_(pa.int32())),
    ])  # fmt: skip
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def build(
    synthetic: list[Path],
    qa: list[Path],
    out: Path,
    tok,
    manifest_dir: Path = REPO_ROOT / "data" / "manifests",
    dev_fraction: float = DEV_FRACTION,
    seed: int = 0,
    max_len: int = MAX_LEN,
) -> dict:
    """Write the splits and ``manifest.json`` under ``out``; returns the manifest."""
    hashes, names = evalset_guard(manifest_dir)
    infos = [qa_info(path) for path in qa]
    dropped, items, seen = Counter(), [], set()

    def unique(row_id: str, where: str) -> None:
        if row_id in seen:
            raise ValueError(f"{where}: duplicate row id {row_id!r}")
        seen.add(row_id)

    for path in synthetic:
        for n, row in read_jsonl(path):
            item = synthetic_item(row, f"{path}:{n}")
            unique(item.id, f"{path}:{n}")
            if is_eval(item.source, item.example.question, hashes, names):
                dropped["eval_overlap"] += 1
            else:
                items.append(item)
    for path in qa:
        for n, row in read_jsonl(path):
            unique(row["id"], f"{path}:{n}")
            if is_eval(row["id"].partition(":")[0], row["question"], hashes, names):
                dropped["eval_overlap"] += 1
            elif (item := qa_item(row)) is None:
                dropped["answer_not_in_context"] += 1
            else:
                items.append(item)

    kept = []
    for item in items:
        enc = encode(item.example, tok)
        ids, labels = enc["input_ids"], enc["labels"]
        assert ids[-1] == labels[-1] == EOS_ID and ids.count(EOS_ID) == 1, f"{item.id}: one EOS, at the end"
        if len(ids) > max_len:
            dropped["too_long"] += 1
        else:
            kept.append((item, ids, labels))

    lo, hi = UNANSWERABLE_SHARE
    n_un = sum(item.kind != "answerable" for item, _, _ in kept)
    # QA items are all answerable: while they hold the share below lo, drop a seeded subset of them
    surplus = sorted((it.id for it, _, _ in kept if it.row is None), key=lambda i: _unit(seed, i))
    drop = set()
    while surplus and n_un / (len(kept) - len(drop)) < lo:
        drop.add(surplus.pop())
    if drop:
        dropped["qa_subsampled"] = len(drop)
        kept = [k for k in kept if k[0].id not in drop]
    share = n_un / max(len(kept), 1)
    if not lo <= share <= hi:
        raise ValueError(f"non-answerable share {share:.3f} of {len(kept)} items is outside [{lo}, {hi}]")

    rows, synthetic_rows = {s: [] for s in SPLITS}, {s: [] for s in SPLITS}
    for item, ids, labels in kept:
        split = "dev" if _unit(seed, item.question_id) < dev_fraction else "train"
        rows[split].append({"id": item.id, "question_id": item.question_id, "source": item.source,
                            "kind": item.kind, "input_ids": ids, "labels": labels})  # fmt: skip
        if item.row is not None:
            synthetic_rows[split].append(item.row)
    train_share = sum(r["kind"] != "answerable" for r in rows["train"]) / max(len(rows["train"]), 1)
    notes = []
    if not lo <= train_share <= hi:  # the split is random, so the share training sees can drift past a bound
        notes.append(f"the train split's non-answerable share {train_share:.3f} is outside [{lo}, {hi}]")
        warnings.warn(notes[-1], RuntimeWarning, stacklevel=2)
    if dev_fraction > 0 and not rows["dev"]:
        notes.append(f"the dev split is empty (dev_fraction {dev_fraction}, {len(kept)} items)")
        warnings.warn(notes[-1], RuntimeWarning, stacklevel=2)
    out.mkdir(parents=True, exist_ok=True)
    rows_sha256 = {}
    for s in SPLITS:
        write_split(rows[s], out / f"{s}.parquet")
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in synthetic_rows[s])
        (out / f"{s}.jsonl").write_text(text, encoding="utf-8")
        rows_sha256[f"{s}.jsonl"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest = {
        "items": {s: len(rows[s]) for s in SPLITS},
        "tokens": {s: sum(len(r["input_ids"]) for r in rows[s]) for s in SPLITS},
        "synthetic_rows": {f"{s}.jsonl": len(synthetic_rows[s]) for s in SPLITS},
        "synthetic_rows_sha256": rows_sha256,
        "by_kind": dict(sorted(Counter(item.kind for item, _, _ in kept).items())),
        "by_source": dict(sorted(Counter(item.source for item, _, _ in kept).items())),
        "dropped": dict(sorted(dropped.items())),
        "unanswerable_share": share,
        "train_unanswerable_share": train_share,
        "warnings": notes,
        "dev_fraction": dev_fraction, "seed": seed, "max_len": max_len,
        "inputs": {
            "synthetic": [str(p) for p in synthetic],
            "qa": [{"path": str(p), "name": i.get("name"), "licence": i.get("licence")} for p, i in zip(qa, infos)],
        },
    }  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    return manifest


def main(argv=None) -> int:
    from lexhybrid.data.tokenizer import load_tokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", type=Path, action="append", default=[], help="generator rows (JSONL)")
    ap.add_argument("--qa", type=Path, action="append", default=[], help="extractive QA items (JSONL)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "sft")
    ap.add_argument(
        "--manifest-dir", type=Path, required=True,
        help="manifests of the COMPLETE evaluation sets; data/manifests in the repository lists only samples",
    )  # fmt: skip
    ap.add_argument("--dev-fraction", type=float, default=DEV_FRACTION)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    args = ap.parse_args(argv)
    if not args.synthetic and not args.qa:
        ap.error("give --synthetic and/or --qa")
    m = build(args.synthetic, args.qa, args.out, load_tokenizer(), args.manifest_dir, args.dev_fraction,
              args.seed, args.max_len)  # fmt: skip
    print(
        f"SFT {m['items']} items, unanswerable share {m['unanswerable_share']:.3f}, dropped {m['dropped']} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
