"""HTML to text blocks for collectors that receive HTML fragments (OLDP). Stdlib only.

A block is the text of one block-level element (``p``, ``div``, headings, list items, table rows).
Whitespace inside a block is normalised; ``<br>`` becomes a line break inside the block; ``script``,
``style`` and ``head`` are dropped.
"""

from html.parser import HTMLParser

_BLOCK = {
    "p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "tr", "dt", "dd", "blockquote", "pre", "table", "ul", "ol", "dl", "body",
}  # fmt: skip
_SKIP = {"script", "style", "head", "title", "noscript"}


class _BlockParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._buf: list[str] = []
        self._skip = 0

    def _flush(self):
        text = "".join(self._buf)
        lines = (" ".join(line.split()) for line in text.split("\n"))
        block = "\n".join(line for line in lines if line)
        if block:
            self.blocks.append(block)
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag == "br":
            self._buf.append("\n")
        elif tag in ("td", "th"):
            self._buf.append(" ")
        elif tag in _BLOCK:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK:
            self._flush()

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data.replace("\xa0", " "))

    def close(self):
        super().close()
        self._flush()


def html_blocks(html: str) -> list[str]:
    """The non-empty text blocks of ``html``, in document order."""
    parser = _BlockParser()
    parser.feed(html)
    parser.close()
    return parser.blocks
