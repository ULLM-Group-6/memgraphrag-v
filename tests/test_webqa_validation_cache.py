#!/usr/bin/env python3
"""No models or scheduler: cache invalidation and receipt lifetime tests."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from webqa_validation_cache import HashCache, receipt_valid


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.tmp.name)
        self.file = self.root / "data"
        self.file.write_bytes(b"first")
        self.db = self.root / "hashes.sqlite3"

    def tearDown(self):
        self.tmp.cleanup()

    def test_persistent_reuse_without_reading_contents(self):
        with HashCache(self.db) as cache:
            expected = cache.digest(self.file)
            self.assertEqual(cache.reads, 1)
        with HashCache(self.db) as cache:
            cache.require(self.file, expected)
            self.assertEqual((cache.reads, cache.hits), (0, 1))

    def test_changed_same_size_and_restored_mtime_rehashed(self):
        with HashCache(self.db) as cache:
            old = cache.digest(self.file)
            info = self.file.stat()
            self.file.write_bytes(b"other")
            os.utime(self.file, ns=(info.st_atime_ns, info.st_mtime_ns))
            # ctime changes on Linux; on Windows add a size change if its ctime
            # reports creation time rather than Unix metadata-change time.
            if self.file.stat().st_ctime_ns == info.st_ctime_ns:
                self.file.write_bytes(b"other!")
            with self.assertRaises(RuntimeError):
                cache.require(self.file, old)

    def test_new_expected_hash_never_accepted_from_cache(self):
        with HashCache(self.db) as cache:
            cache.digest(self.file)
            with self.assertRaises(RuntimeError):
                cache.require(self.file, "0" * 64)

    def test_missing_file_rejected(self):
        with HashCache(self.db) as cache:
            with self.assertRaises(RuntimeError):
                cache.require(self.root / "absent", "abc")

    def test_completed_checks_committed_even_when_stage_fails(self):
        with self.assertRaisesRegex(RuntimeError, "stage"):
            with HashCache(self.db) as cache:
                cache.digest(self.file)
                raise RuntimeError("stage interrupted")
        with HashCache(self.db) as cache:
            self.assertEqual(cache.digest(self.file), hashlib.sha256(b"first").hexdigest())
            self.assertEqual(cache.reads, 0)

    def test_receipt_invalidated_by_context_or_dependency_change(self):
        with HashCache(self.db) as cache:
            cache.digest(self.file)
            receipt = {"context": {"run": 1}, "dependencies": dict(cache.observed)}
        self.assertTrue(receipt_valid(receipt, {"run": 1}))
        self.assertFalse(receipt_valid(receipt, {"run": 2}))
        self.file.write_bytes(b"longer data")
        self.assertFalse(receipt_valid(receipt, {"run": 1}))

    def test_receipt_missing_dependency_rejected(self):
        self.assertFalse(receipt_valid({"context": {}, "dependencies": {str(self.root / "absent"): []}}, {}))


if __name__ == "__main__":
    unittest.main()
