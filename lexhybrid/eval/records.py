"""One system's answer to one evaluation question: the JSONL record every evaluation script reads or writes.

The decoding scripts write one ``AnswerRecord`` per question, system, seed and arm; the scorers
(citation metrics, the judge, the truncation audit, the bootstrap) read them back. A record carries
its prompt's passages in full (``chunks``, ``Chunk`` fields as a dict, passage ``K`` is
``chunks[K - 1]``), so scoring never needs the index that produced them.

``kind`` names what the question asks for: ``answerable``, or one of the questions a correct system
refuses (``unanswerable``, ``false_premise``, ``party``: who a party was).
"""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lexhybrid.retrieval.chunking import Chunk

KINDS = ("answerable", "unanswerable", "false_premise", "party")
REFUSE_KINDS = frozenset(KINDS[1:])


@dataclass
class AnswerRecord:
    id: str  # the question's id within its set
    set: str  # evaluation set name
    system: str  # "hybrid", "transformer", ...
    arm: str  # which pipeline variant produced it ("default" when there is one)
    question: str
    seed: int | None = None
    kind: str = "answerable"
    gold_answers: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)  # relevant passages, when known
    chunks: list[dict] = field(default_factory=list)  # the prompt's passages, in <|cK|> order
    raw_answer: str = ""  # the decoded pointer markup, EOS included when the model produced it
    rendered: str = ""  # what the reader sees
    refused: bool = False
    verifier: dict = field(default_factory=dict)  # VerifierResult.log_row()
    new_tokens: int = 0  # answer tokens decoded
    max_new_tokens: int = 0  # the decoding budget

    def __post_init__(self):
        self.id = str(self.id)  # evaluation sets use numeric ids too; every consumer joins on strings
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: kind {self.kind!r} is not one of {KINDS}")

    @property
    def should_refuse(self) -> bool:
        return self.kind in REFUSE_KINDS

    def chunk_objects(self) -> list[Chunk]:
        return [Chunk.from_json(json.dumps(c)) for c in self.chunks]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "AnswerRecord":
        return cls(**json.loads(line))


def chunk_dict(chunk: Chunk) -> dict:
    return json.loads(chunk.to_json())


def write_records(path: str | Path, records: Iterable[AnswerRecord]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")
            n += 1
    return n


def read_records(path: str | Path) -> list[AnswerRecord]:
    # split on "\n" only: str.splitlines also breaks at U+2028/U+0085, which JSON keeps raw in strings
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    return [AnswerRecord.from_json(line) for line in lines if line.strip()]
