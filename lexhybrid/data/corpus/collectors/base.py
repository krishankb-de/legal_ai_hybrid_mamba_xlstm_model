"""What every collector shares (plan P3-E): the protocol, polite HTTP, and the CLI.

A collector turns one public source into ``Document`` records. Each is split in two so the tests
never need the network: a pure ``parse_*`` function from raw bytes (XML, JSON, HTML) to documents,
tested on fixtures under ``tests/fixtures/collectors/<source>/``; and ``iter_documents``, which
fetches and parses, exercised by a ``network``-marked live smoke (``--limit 2``) in the weekly CI job.

    python -m lexhybrid.data.corpus.collectors.gii --limit 20 --out data/raw/gii

writes ``data/raw/gii/gii.jsonl`` (full documents, git-ignored) and ``data/manifests/gii.jsonl``
(every field but the text; versioned).

At scale (P4-K) a collection runs for hours to days at one request per second, on a preemptible
partition: ``--limit all`` collects everything, both files are streamed to ``*.partial`` and renamed
only when the collection completes (a killed run never leaves a complete-looking file), and
``--http-cache DIR`` keeps every successful response on disk, so a requeued run replays what it
already fetched without the network or the rate limit and carries on from there.
"""

import argparse
import hashlib
import json
import os
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
from requests.structures import CaseInsensitiveDict

from lexhybrid.data.corpus.licences import require_known
from lexhybrid.data.corpus.manifest import manifest_record
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


class SkipFailedDocuments:
    """Fetch one document, or skip it (logged) when its fetch fails after every retry, so a crawl of
    days does not die on one dropped connection (job 2589358_1: RII, 13,000 documents and 3.8 h in,
    one RemoteDisconnected). ``max_consecutive`` failures in a row still raise: that is an outage,
    not a flaky document. Listing and index requests are not wrapped: without them nothing follows.
    """

    def __init__(self, source: str, max_consecutive: int = 20):
        self.source, self.max_consecutive = source, max_consecutive
        self.in_a_row = self.skipped = 0

    def __call__(self, fetch, url: str):
        """``fetch(url)``'s response, or None when it raised ``FetchError``."""
        try:
            response = fetch(url)
        except FetchError as e:
            self.in_a_row += 1
            self.skipped += 1
            print(f"{self.source}: skipping {url}: {e}", file=sys.stderr)
            if self.in_a_row >= self.max_consecutive:
                raise FetchError(f"{self.source}: {self.in_a_row} document fetches failed in a row") from e
            return None
        self.in_a_row = 0
        return response


class HttpCache:
    """Successful responses on disk, keyed by the request (method, URL, query, JSON body).

    A hit is served without the network and without the rate limiter, so a restarted collection
    replays the requests it already made -- listing pages included, which keeps the replay
    consistent with the first run -- and fetches only what is new. Entries are written to a
    temporary name and renamed, so a killed process never leaves a half-written one.
    """

    _DROP_HEADERS = {"content-encoding", "content-length", "transfer-encoding"}  # the body is decoded

    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def key(method: str, url: str, params: dict | None = None, body: dict | None = None) -> str:
        request = {"method": method, "url": url, "params": sorted((params or {}).items()), "json": body}
        return hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        base = self.root / key[:2] / key
        return base.with_suffix(".body"), base.with_suffix(".json")

    def get(self, key: str) -> requests.Response | None:
        body_path, meta_path = self._paths(key)
        if not (body_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        response = requests.Response()
        response.status_code = meta["status"]
        response._content = body_path.read_bytes()
        response.headers = CaseInsensitiveDict(meta["headers"])
        response.url = meta["url"]
        response.encoding = meta["encoding"]
        response.reason = "OK (http cache)"
        return response

    def put(self, key: str, response: requests.Response) -> None:
        body_path, meta_path = self._paths(key)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        headers = {k: v for k, v in response.headers.items() if k.lower() not in self._DROP_HEADERS}
        meta = {
            "status": response.status_code,
            "url": response.url,
            "encoding": response.encoding,
            "headers": headers,
        }
        for path, data in ((body_path, response.content), (meta_path, json.dumps(meta).encode())):
            tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
            tmp.write_bytes(data)
            tmp.replace(path)  # the .json is written last: an entry exists only when both do


_CACHE: HttpCache | None = None


def set_http_cache(root: Path | None) -> HttpCache | None:
    """Serve every ``http_get``/``http_post`` of this process through a cache under ``root``
    (None switches it off)."""
    global _CACHE
    _CACHE = None if root is None else HttpCache(root)
    return _CACHE


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
    return _cached(
        "GET", url, params, None, lambda: _with_retries("GET", url, send, retries, backoff, limiter, sleep)
    )


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
    return _cached(
        "POST", url, None, json, lambda: _with_retries("POST", url, send, retries, backoff, limiter, sleep)
    )


def _cached(method, url, params, body, fetch) -> requests.Response:
    """``fetch()`` through the process's ``HttpCache`` when one is set."""
    if _CACHE is None:
        return fetch()
    key = HttpCache.key(method, url, params, body)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    response = fetch()
    _CACHE.put(key, response)
    return response


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


def run_collector(
    collector: Collector, limit: int | None, out_dir: Path, manifest_dir: Path, progress_every: int = 1000
) -> int:
    """Collect up to ``limit`` documents (all when None) into ``out_dir/<name>.jsonl`` and the
    manifest; return the count. Both are streamed to ``*.partial`` files -- nothing is held in
    memory -- and renamed only when the collection has finished."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    docs_path = out_dir / f"{collector.name}.jsonl"
    manifest_path = manifest_dir / f"{collector.name}.jsonl"
    docs_tmp, manifest_tmp = (p.with_name(p.name + ".partial") for p in (docs_path, manifest_path))
    n, start = 0, time.monotonic()
    with open(docs_tmp, "w", encoding="utf-8") as f, open(manifest_tmp, "w", encoding="utf-8") as m:
        for doc in collector.iter_documents(limit=limit):
            f.write(doc.to_json() + "\n")
            m.write(json.dumps(manifest_record(doc), ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
            if progress_every and n % progress_every == 0:
                print(
                    f"{collector.name}: {n} documents, {(time.monotonic() - start) / 60:.1f} min", flush=True
                )
    docs_tmp.replace(docs_path)
    manifest_tmp.replace(manifest_path)
    return n


def parse_limit(value: str) -> int | None:
    """``--limit``: a positive number of documents, or ``all`` (every document the source has)."""
    if value.lower() in ("all", "none"):
        return None
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"--limit must be >= 1 or 'all', got {value}")
    return n


def cli(collector: Collector, argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"Collect documents from {collector.name}.")
    parser.add_argument(
        "--limit", type=parse_limit, default=20, help="documents to collect, or 'all' (default 20)"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "raw" / collector.name)
    parser.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "data" / "manifests")
    parser.add_argument(
        "--http-cache", type=Path, default=None, help="keep responses here; a rerun replays them (P4-K)"
    )
    args = parser.parse_args(argv)
    set_http_cache(args.http_cache)
    n = run_collector(collector, args.limit, args.out, args.manifest_dir)
    print(
        f"{collector.name}: {n} documents -> {args.out}/{collector.name}.jsonl, manifest {args.manifest_dir}"
    )
    if n == 0:
        print(f"{collector.name}: no documents collected", file=sys.stderr)
        return 1
    return 0


def main(collector: Collector, argv=None) -> None:
    """A collector module's entry point: ``cli``, then end the process at once with its exit code.

    A ``datasets`` stream that is stopped at ``--limit`` with shuffling on leaves native download
    workers whose teardown never finishes: on 2026-09-30 the FineWeb-2 collector wrote its files,
    printed its summary and was still alive minutes later, past Python's own shutdown (the in-order
    stream exits at once). On the cluster that array task would hold its slot until the time limit.
    By the time ``cli`` returns, ``run_collector`` has closed and renamed every file, so the process
    ends with ``os._exit`` after flushing; an error ends it the same way, with a non-zero code.
    """
    try:
        code = cli(collector, argv)
    except SystemExit as e:  # argparse usage errors
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:  # noqa: BLE001 -- report it, then end without the teardown that can hang
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
