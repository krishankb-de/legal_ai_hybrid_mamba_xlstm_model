"""The student's tokenizer: Qwen3's own, plus the SFT pointer specials (plan P3-C, decisions 6-7).

``Qwen/Qwen3-1.7B-Base``'s tokenizer (pinned revision; Apache-2.0) has 151,669 entries; its EOS,
``<|endoftext|>`` = 151,643, ends every document, so the teacher's KD target at a document end is
meaningful and no new EOS is added. The student's embedding has ``VOCAB_SIZE`` = 151,936 rows (the
teacher's padded size), which leaves 267 free ids; decision 7's 87 specials take ids 151,669 to
151,755 and appear only in SFT data, never in pretraining or KD:

    <|q|> <|cite|> <|c1|>..<|c16|> <|s1|>..<|s64|> <|unanswerable|> <|passage|> <|/passage|>
    <|question|> <|answer|>

``encode_document`` tokenizes raw document text so that no special token can come from the text:
a literal ``<|endoftext|>`` or ``<|cite|>`` inside a document is spelled out, not an EOS or a
pointer. Files are cached under ``data/hf/hub`` (git-ignored) unless ``HF_HOME``/``HF_HUB_CACHE``
name a cache (the cluster's jobs set ``HF_HOME``). Fetch them once with

    .venv/bin/python -m lexhybrid.data.tokenizer --download
"""

import argparse
import os
from pathlib import Path

QWEN3_TOKENIZER = "Qwen/Qwen3-1.7B-Base"
QWEN3_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"  # main on 2026-09-27
EOS_TOKEN = "<|endoftext|>"
EOS_ID = 151643
BASE_LEN = 151669  # Qwen3's tokenizer before the specials
VOCAB_SIZE = 151936  # the student's embedding rows (decision 6)
N_PASSAGES, N_SENTENCES = 16, 64
SPECIALS = (
    "<|q|>",
    "<|cite|>",
    *(f"<|c{k}|>" for k in range(1, N_PASSAGES + 1)),
    *(f"<|s{j}|>" for j in range(1, N_SENTENCES + 1)),
    "<|unanswerable|>",
    "<|passage|>",
    "<|/passage|>",
    "<|question|>",
    "<|answer|>",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CACHE = REPO_ROOT / "data" / "hf" / "hub"
_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json", "LICENSE"]


def cache_dir() -> str | None:
    """``data/hf/hub`` unless the environment names a Hugging Face cache."""
    if os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE"):
        return None
    return str(LOCAL_CACHE)


def load_tokenizer(with_specials: bool = True, name: str = QWEN3_TOKENIZER, revision: str = QWEN3_REVISION):
    """Qwen3's tokenizer, with decision 7's specials unless ``with_specials=False``.

    Raises ``ValueError`` if the EOS is not id 151,643 or the vocabulary outgrows the student's
    embedding, and ``OSError`` if the files are neither cached nor downloadable (offline)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision, cache_dir=cache_dir())
    if tok.convert_tokens_to_ids(EOS_TOKEN) != EOS_ID or tok.eos_token_id != EOS_ID:
        raise ValueError(
            f"{name}: EOS is {tok.eos_token!r} = {tok.eos_token_id}, expected {EOS_TOKEN} = {EOS_ID}"
        )
    if with_specials:
        tok.add_special_tokens(
            {"additional_special_tokens": list(SPECIALS)}, replace_extra_special_tokens=False
        )
    if len(tok) > VOCAB_SIZE:
        raise ValueError(f"{name}: {len(tok)} tokens do not fit the student's {VOCAB_SIZE} embedding rows")
    return tok


def pointer_ids(tok) -> dict:
    """The special ids by role: ``q``, ``cite``, ``passages`` (c1..c16), ``sentences`` (s1..s64),
    ``unanswerable``, ``passage_open``, ``passage_close``, ``question``, ``answer``."""
    ids = dict(zip(SPECIALS, tok.convert_tokens_to_ids(list(SPECIALS)), strict=True))
    unknown = [t for t, i in ids.items() if i is None or i == tok.unk_token_id or i < BASE_LEN]
    if unknown:
        raise ValueError(f"specials missing from the tokenizer: {unknown[:5]}")
    return {
        "q": ids["<|q|>"],
        "cite": ids["<|cite|>"],
        "passages": [ids[f"<|c{k}|>"] for k in range(1, N_PASSAGES + 1)],
        "sentences": [ids[f"<|s{j}|>"] for j in range(1, N_SENTENCES + 1)],
        "unanswerable": ids["<|unanswerable|>"],
        "passage_open": ids["<|passage|>"],
        "passage_close": ids["<|/passage|>"],
        "question": ids["<|question|>"],
        "answer": ids["<|answer|>"],
    }


def encode_document(tok, text: str) -> list[int]:
    """Token ids of raw document text; special-token strings inside it are split into plain tokens."""
    return tok(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch or inspect the student's tokenizer.")
    parser.add_argument("--download", action="store_true", help=f"fetch {QWEN3_TOKENIZER}'s tokenizer files")
    args = parser.parse_args(argv)
    if args.download:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            QWEN3_TOKENIZER, revision=QWEN3_REVISION, allow_patterns=_FILES, cache_dir=cache_dir()
        )
        print(f"tokenizer files in {path}")
    tok = load_tokenizer()
    print(f"{QWEN3_TOKENIZER}@{QWEN3_REVISION[:7]}: {len(tok)} tokens (<= {VOCAB_SIZE}), eos {EOS_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
