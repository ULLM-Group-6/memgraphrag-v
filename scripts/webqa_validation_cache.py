"""Persistent SHA-256 cache for an exclusively owned, otherwise immutable run.

Reuse requires unchanged device/inode/size/mtime_ns/ctime_ns. This is a metadata
trust policy, not protection against modifications that preserve all metadata.
SQLite uses DELETE journaling (not WAL, which is unsuitable for network mounts).
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import time


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


@contextmanager
def timed(label):
    started = time.monotonic()
    log(f"START {label}")
    try:
        yield
    except BaseException:
        log(f"STOP {label}: {time.monotonic() - started:.1f}s (interrupted/error)")
        raise
    else:
        log(f"DONE {label}: {time.monotonic() - started:.1f}s")


def token(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def absolute(path):
    return os.path.abspath(os.fspath(path))


def descriptor_matches(info, state):
    # Windows Python versions can expose different ctime meanings through stat
    # and fstat. Snellius/Linux always compares all five fields. Path stat is
    # still checked before/after reading on both platforms.
    count = 4 if os.name == "nt" else 5
    return token(info)[:count] == state[:count]


class HashCache:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS hashes (path TEXT PRIMARY KEY, token TEXT NOT NULL, sha TEXT NOT NULL)")
        self.db.commit()
        self.observed = {}
        self.hits = self.reads = self.pending = 0

    def observe(self, path):
        key = absolute(path)
        value = os.stat(key)
        self.observed[key] = token(value)
        return value

    def _store(self, path, state, digest):
        self.db.execute("INSERT OR REPLACE INTO hashes VALUES (?, ?, ?)",
                        (absolute(path), json.dumps(state), digest))
        self.pending += 1
        if self.pending >= 64:
            self.flush()

    def flush(self):
        self.db.commit()
        self.pending = 0

    def digest(self, path):
        key = absolute(path)
        try:
            before = self.observe(key)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Not a regular file: {key}")
        state = token(before)
        row = self.db.execute("SELECT token, sha FROM hashes WHERE path=?", (key,)).fetchone()
        if row and json.loads(row[0]) == state:
            self.hits += 1
            return row[1]
        if before.st_size >= 256 * 1024 * 1024:
            log(f"Hashing changed/uncached large file ({before.st_size / 2**20:.0f} MiB): {key}")
        digest = hashlib.sha256()
        with open(key, "rb") as file:
            if not descriptor_matches(os.fstat(file.fileno()), state):
                raise RuntimeError(f"File changed before hashing: {key}")
            for block in iter(lambda: file.read(1 << 20), b""):
                digest.update(block)
            if not descriptor_matches(os.fstat(file.fileno()), state):
                raise RuntimeError(f"File changed during hashing: {key}")
        if token(self.observe(key)) != state:
            raise RuntimeError(f"File replaced during hashing: {key}")
        value = digest.hexdigest()
        self._store(key, state, value)
        self.reads += 1
        if self.reads % 256 == 0:
            log(f"Hash cache: {self.reads} files read, {self.hits} hashes reused")
        return value

    def require(self, path, expected):
        if not expected or self.digest(path) != expected:
            raise RuntimeError(f"Missing or changed file: {path}")

    def read_json(self, path):
        # Metadata is parsed only while creating a preparation receipt. Receipts
        # hold the parsed records, avoiding thousands of JSON opens next time.
        before = token(self.observe(path))
        data = Path(path).read_bytes()
        if token(self.observe(path)) != before:
            raise RuntimeError(f"Metadata changed while reading: {path}")
        return json.loads(data)

    def close(self):
        self.flush()
        self.db.close()
        log(f"Hash cache totals: {self.reads} files read, {self.hits} hashes reused")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def receipt_valid(receipt, context):
    if receipt.get("context") != context or not receipt.get("dependencies"):
        return False
    for path, expected in receipt["dependencies"].items():
        try:
            if token(os.stat(path)) != expected:
                return False
        except OSError:
            return False
    return True
