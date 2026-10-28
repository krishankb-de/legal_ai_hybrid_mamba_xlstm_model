"""The answer grammar the renderer and the verifier share (lexhybrid/renderer/segments.py; P7-G/H/J)."""

import pytest

from lexhybrid.renderer.segments import (
    CitePointer,
    PointerSyntaxError,
    QuotePointer,
    Text,
    Unanswerable,
    parse_answer,
    pointers,
)


def test_parse_answer_reads_both_pointer_forms_and_text():
    answer = "Der Vermieter muss kündigen. <|q|><|c1|><|s2|> Siehe <|cite|><|c3|>.<|q|><|c2|><|s1|><|s2|><|cite|><|c1|><|s4|><|endoftext|>"
    assert parse_answer(answer) == [
        Text("Der Vermieter muss kündigen. "),
        QuotePointer(1, (2,)),
        Text(" Siehe "),
        CitePointer(3, ()),
        Text("."),
        QuotePointer(2, (1, 2)),
        CitePointer(1, (4,)),
    ]
    assert pointers(parse_answer(answer)) == [
        QuotePointer(1, (2,)),
        CitePointer(3, ()),
        QuotePointer(2, (1, 2)),
        CitePointer(1, (4,)),
    ]
    assert parse_answer("Nur Text.") == [Text("Nur Text.")] and parse_answer("") == []
    assert parse_answer("<|unanswerable|><|endoftext|>") == [Unanswerable()]


@pytest.mark.parametrize(
    "bad",
    [
        "<|q|> text",  # no passage
        "<|q|><|c1|> text",  # a quote names no sentence
        "<|c1|><|s1|>",  # a passage without q/cite
        "Text <|s3|>",  # a sentence alone
        "<|q|> <|c1|><|s1|>",  # not adjacent
        "<|passage|>x",  # a prompt token in the answer
        "Ja. <|unanswerable|>",  # unanswerable with text
        "<|cite|><|cite|><|c1|>",
    ],
)
def test_parse_answer_refuses_malformed_pointers(bad):
    with pytest.raises(PointerSyntaxError):
        parse_answer(bad)
