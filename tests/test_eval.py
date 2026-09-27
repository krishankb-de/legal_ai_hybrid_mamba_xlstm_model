"""Text metrics ported from the reference (plan P1-T)."""

import pytest

from lexhybrid.eval.text_metrics import _lcs_length, corpus_bleu, rouge_l_score


def test_lcs_length():
    assert _lcs_length("a b c d".split(), "a c d e".split()) == 3


def test_rouge_l_known_values():
    assert rouge_l_score([], ["a"]) == 0.0
    assert rouge_l_score("a b".split(), "c d".split()) == 0.0
    assert rouge_l_score("a b c d".split(), "a b c d".split()) == pytest.approx(1.0)
    p = r = 0.75  # lcs 3 of 4 on both sides
    assert rouge_l_score("a b c d".split(), "a c d e".split()) == pytest.approx(
        (1 + 1.44) * p * r / (r + 1.44 * p)
    )


def test_corpus_bleu_known_values():
    assert corpus_bleu([["a", "b", "c", "d"]], [["a", "b", "c", "d"]]) == pytest.approx(1.0)
    assert corpus_bleu([["x"]], [["y"]]) == 0.0
    assert corpus_bleu([[]], [["a"]], max_n=1) == 0.0
    # BLEU-1 with a brevity penalty: 2 of 2 unigrams match, hyp 2 vs ref 4 -> bp = exp(1 - 2)
    assert corpus_bleu([["a", "b"]], [["a", "b", "c", "d"]], max_n=1) == pytest.approx(0.36787944117144233)
