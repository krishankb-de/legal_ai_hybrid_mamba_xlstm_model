"""Reading a JSONL file that a killed job may have been appending to."""

import os
from pathlib import Path


def read_complete_lines(path: str | Path) -> list[str]:
    """The file's non-empty whole lines; a last line left without its newline is cut from the file.
    At byte level, since the cut may fall inside a UTF-8 character, and on the newline alone, since
    JSON keeps U+2028 and U+0085 raw inside strings."""
    data = Path(path).read_bytes()
    keep = data.rfind(b"\n") + 1
    if keep < len(data):
        os.truncate(path, keep)
    return [line for line in data[:keep].decode("utf-8").split("\n") if line.strip()]
