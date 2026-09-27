"""What every collector shares (plan P3-E): the protocol, polite HTTP, and the CLI.

A collector turns one public source into ``Document`` records. Each is split in two so the tests
never need the network: a pure ``parse_*`` function from raw bytes (XML, JSON, HTML) to documents,
tested on fixtures under ``tests/fixtures/collectors/<source>/``; and ``iter_documents``, which
fetches and parses, exercised by a ``network``-marked live smoke (``--limit 2``) in the weekly CI job.

    python -m lexhybrid.data.corpus.collectors.gii --limit 20 --out data/raw/gii

writes ``data/raw/gii/gii.jsonl`` (full documents, git-ignored) and ``data/manifests/gii.jsonl``
(every field but the text; versioned).
"""

import argparse
import sys
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import requests

from lexhybrid.data.corpus.licences import require_known
from lexhybrid.data.corpus.manifest import write_manifest
from lexhybrid.data.schema import Document

USER_AGENT = (
    "lexhybrid-corpus/0.1 (research collection of public legal texts; "
    "https://github.com/krishankb-de/legal_ai_hybrid_mamba_xlstm_model)"
)
REPO_ROOT = Path(__file__).resolve().parents[4]


class Collector(Protocol):
    """A public source of documents."""

    name: str  # the ``Document.source`` value
    licence: str  # an id of ``lexhybrid.data.corpus.licences.LICENCES``
    jurisdiction: str

    def iter_documents(self, limit: int | None = None) -> Iterator[Document]: ...


# ---------------------------------------------------------------------------------------------
# polite HTTP
# ---------------------------------------------------------------------------------------------


class RateLimiter:
    """At most one request per ``min_interval`` seconds per host."""

    def __init__(self, min_interval: float = 1.0, clock=time.monotonic, sleep=time.sleep):
        self.min_interval = min_interval
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._clock, self._sleep = clock, sleep

    def wait(self, host: str) -> None:
        with self._lock:
            now = self._clock()
            ready = self._last.get(host, float("-inf")) + self.min_interval
            if ready > now:
                self._sleep(ready - now)
                now = ready
            self._last[host] = now


_LIMITER = RateLimiter()
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = USER_AGENT
RETRY_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """A request failed after every retry."""


def http_get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    retries: int = 4,
    backoff: float = 2.0,
    timeout: float = 60.0,
    limiter: RateLimiter | None = None,
    session: requests.Session | None = None,
    sleep=time.sleep,
) -> requests.Response:
    """GET with a per-host rate limit and exponential backoff on 429/5xx and connection errors.

    ``Retry-After`` is honoured when the server sends it. Any other 4xx fails at once: it will not
    get better by asking again.
    """
    session = session or _SESSION
    send = partial(session.get, url, params=params, headers=headers, timeout=timeout)
    return _with_retries("GET", url, send, retries, backoff, limiter, sleep)


def http_post(
    url: str,
    json: dict | None = None,
    headers: dict | None = None,
    retries: int = 4,
    backoff: float = 2.0,
    timeout: float = 60.0,
    limiter: RateLimiter | None = None,
    session: requests.Session | None = None,
    sleep=time.sleep,
) -> requests.Response:
    """POST a JSON body (a search API's query), with the same rate limit and retries as ``http_get``."""
    session = session or _SESSION
    send = partial(session.post, url, json=json, headers=headers, timeout=timeout)
    return _with_retries("POST", url, send, retries, backoff, limiter, sleep)


def _with_retries(method, url, send, retries, backoff, limiter, sleep) -> requests.Response:
    limiter = limiter or _LIMITER
    host = urlparse(url).netloc
    last_error = None
    for attempt in range(retries + 1):
        limiter.wait(host)
        try:
            response = send()
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
        else:
            if response.status_code < 400:
                return response
            if response.status_code not in RETRY_STATUS:
                raise FetchError(f"{method} {url} -> HTTP {response.status_code}")
            last_error = f"HTTP {response.status_code}"
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit() and attempt < retries:
                sleep(float(retry_after))
                continue
        if attempt < retries:
            sleep(backoff**attempt)
    raise FetchError(f"{method} {url} failed after {retries + 1} attempts ({last_error})")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def licence_flags(licence_id: str) -> dict:
    """``licence``, ``commercial_safe`` and ``research_only`` as the register defines them."""
    lic = require_known(licence_id)
    return {"licence": lic.id, "commercial_safe": lic.commercial_safe, "research_only": lic.research_only}


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def run_collector(collector: Collector, limit: int | None, out_dir: Path, manifest_dir: Path) -> int:
    """Collect up to ``limit`` documents into ``out_dir/<name>.jsonl`` and the manifest; return the count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    with open(out_dir / f"{collector.name}.jsonl", "w", encoding="utf-8") as f:
        for doc in collector.iter_documents(limit=limit):
            f.write(doc.to_json() + "\n")
            docs.append(doc)
    write_manifest(docs, manifest_dir / f"{collector.name}.jsonl")
    return len(docs)


def cli(collector: Collector, argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"Collect documents from {collector.name}.")
    parser.add_argument("--limit", type=int, default=20, help="documents to collect (default 20)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "raw" / collector.name)
    parser.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "data" / "manifests")
    args = parser.parse_args(argv)
    n = run_collector(collector, args.limit, args.out, args.manifest_dir)
    print(
        f"{collector.name}: {n} documents -> {args.out}/{collector.name}.jsonl, manifest {args.manifest_dir}"
    )
    if n == 0:
        print(f"{collector.name}: no documents collected", file=sys.stderr)
        return 1
    return 0
