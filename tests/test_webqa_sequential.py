#!/usr/bin/env python3
"""CPU-only recovery and scheduling checks. No Slurm submission or model imports."""
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embed = load("webqa_embed_checkpointed")
seq = load("webqa_sequential")


class FakeEmbeddings:
    """A committed store is represented by its contents and input identity."""
    def __init__(self):
        self.saved = {}
        self.encoded = []
        self.corrupt = set()

    def saved_set(self, path, identity):
        if path in self.corrupt:
            raise RuntimeError("Corrupt shard")
        value = self.saved.get(path)
        if value is None:
            return None
        key, result = value
        if key != identity:
            raise RuntimeError("Different inputs")
        return deepcopy(result)

    def save_set(self, path, result, identity):
        self.saved[path] = (deepcopy(identity), deepcopy(result))

    def check_set(self, result, items, *args):
        assert result == [item.item_id for item in items]

    def encode(self, items, kind):
        self.encoded.append([item.item_id for item in items])
        return self.encoded[-1]

    def merge_sets(self, parts, expected):
        merged = sum(parts, [])
        assert merged == expected
        return merged


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.api = FakeEmbeddings()
        self.items = [NS(item_id=str(i)) for i in range(300)]
        self.directory = Path("worker/image")
        self.segments = Path("segments")

    def run_kind(self, clock=lambda: 0, identity=None, items=None, empty=None):
        return embed.resume_kind(self.api, self.items if items is None else items,
            "image", None, None, self.directory, self.segments,
            identity or {"plan": "p", "kind": "image"}, self.api.encode,
            empty or (lambda kind: []), 100, clock)

    def test_deadline_then_resume_skips_committed_segments(self):
        ticks = iter([0, 0, 101])
        with self.assertRaises(embed.SliceFinished):
            self.run_kind(clock=lambda: next(ticks))
        self.assertEqual([len(x) for x in self.api.encoded], [128, 128])
        self.assertNotIn(self.directory, self.api.saved)
        self.assertEqual(self.run_kind(), [i.item_id for i in self.items])
        self.assertEqual([len(x) for x in self.api.encoded], [128, 128, 44])

    def test_interrupted_uncommitted_segment_is_recomputed(self):
        original = self.api.save_set
        def fail_second(path, result, identity):
            if path == self.segments / "00000128":
                raise OSError("simulated termination before commit marker")
            original(path, result, identity)
        with patch.object(self.api, "save_set", side_effect=fail_second):
            with self.assertRaises(OSError):
                self.run_kind()
        self.run_kind()
        self.assertEqual([batch[0] for batch in self.api.encoded], ["0", "128", "128", "256"])

    def test_complete_worker_reused_even_after_deadline(self):
        result = self.run_kind()
        self.api.encoded.clear()
        self.assertEqual(self.run_kind(clock=lambda: 200), result)
        self.assertEqual(self.api.encoded, [])

    def test_changed_identity_rejected(self):
        self.run_kind()
        with self.assertRaisesRegex(RuntimeError, "Different inputs"):
            self.run_kind(identity={"plan": "changed"})

    def test_corrupt_committed_segment_rejected(self):
        self.api.corrupt.add(self.segments / "00000000")
        with self.assertRaisesRegex(RuntimeError, "Corrupt"):
            self.run_kind()
        self.assertEqual(self.api.encoded, [])

    def test_empty_crop_set_needs_no_encoder(self):
        self.assertEqual(self.run_kind(items=[]), [])
        self.assertEqual(self.api.encoded, [])

    def test_yield_before_merge_reuses_all_segments(self):
        ticks = iter([0, 0, 0, 101])
        with self.assertRaises(embed.SliceFinished):
            self.run_kind(clock=lambda: next(ticks))
        self.assertEqual(len(self.api.encoded), 3)
        self.run_kind()
        self.assertEqual(len(self.api.encoded), 3)


class Predecessors(unittest.TestCase):
    def check(self, state, code="0:0", owner="me"):
        result = NS(stdout=f"123|{owner}|{state}|{code}|\n")
        with patch.dict(os.environ, {"USER": "me"}), patch.object(seq.subprocess, "run", return_value=result):
            seq.check_previous("123")

    def test_completed_and_timeout_allowed(self):
        self.check("COMPLETED")
        self.check("TIMEOUT", "0:15")

    def test_failure_cancel_oom_refused(self):
        for state in ("FAILED", "CANCELLED by 123", "OUT_OF_MEMORY", "NODE_FAIL"):
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                self.check(state, "1:0")

    def test_wrong_owner_refused(self):
        with self.assertRaisesRegex(RuntimeError, "different user"):
            self.check("COMPLETED", owner="other")

    def test_missing_accounting_fails_closed(self):
        with patch.object(seq.subprocess, "run", return_value=NS(stdout="")), patch.object(seq.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "No completed"):
                seq.check_previous("123")


class Workflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        (self.root / "index").mkdir()
        self.manifest = self.root / "index/manifest.json"
        self.set_stages("text", "ground")
        self.c = object.__new__(seq.Coordinator)
        self.c.root = self.root
        self.c.args = NS(mode="index", run_id="webqa-tuned-v1", last_index=0, previous="-",
                         registration=str(self.root / "registration.json"))
        self.c.deadline = seq.time.time() + 10000
        self.calls = []
        self.c.step = lambda action, worker=0, candidate=0: self.calls.append((action, worker, candidate)) or True

    def tearDown(self):
        self.temp.cleanup()

    def set_stages(self, *names):
        self.manifest.write_text(json.dumps({"stages": {name: {} for name in names}}))

    def test_index_order(self):
        self.c.index()
        self.assertEqual([x[0] for x in self.calls], ["embed-checkpointed", "embed-finalize", "assemble"])

    def test_finished_index_is_reused(self):
        self.set_stages("text", "ground", "embed", "assemble")
        self.c.index()
        self.assertEqual([x[0] for x in self.calls], ["assemble"])

    def test_yield_last_index_blocks_tuning(self):
        self.c.step = lambda *args, **kwargs: False
        self.c.index()  # Intermediate slice is a successful checkpointed yield.
        self.c.args.last_index = 1
        with self.assertRaisesRegex(RuntimeError, "Index still incomplete"):
            self.c.index()

    def test_canonical_writes_wait_for_time(self):
        self.set_stages("text", "ground", "embed")
        self.c.deadline = seq.time.time() + 100
        self.c.index()
        self.assertEqual(self.calls, [])

    def run_mode(self, mode):
        self.c.args.mode = mode
        Path(self.c.args.registration).write_text("{}")
        with patch.object(seq, "signature", return_value={}), patch.object(seq, "check_previous"), \
             patch.object(seq, "chain_lock", return_value=nullcontext()):
            self.c.run()

    def test_tuning_all_candidates_serial_then_select(self):
        self.set_stages("text", "ground", "embed", "assemble")
        self.run_mode("tune")
        expected = [("prepare-auto", 0, 0)]
        for candidate in range(3):
            expected += [("dev-retrieve", 0, candidate), ("dev-retrieve", 1, candidate),
                         ("dev-collect", 0, candidate)]
        expected += [("select-beta", 0, 0)]
        self.assertEqual(self.calls, expected)

    def test_test_workers_serial_then_collect(self):
        self.set_stages("text", "ground", "embed", "assemble")
        self.run_mode("test")
        self.assertEqual(self.calls, [("prepare-auto", 0, 0), ("test-qa", 0, 0),
                                     ("test-qa", 1, 0), ("test-collect", 0, 0)])

    def test_no_qa_with_incomplete_index(self):
        with self.assertRaisesRegex(RuntimeError, "Index is incomplete"):
            self.run_mode("tune")
        self.assertEqual(self.calls, [("prepare-auto", 0, 0)])

    def test_step_requests_one_gpu_nine_cpus_and_correct_python(self):
        self.c.logs = self.root
        self.c.number = 0
        env = {"MGRV_TEXT_PYTHON": "text-python", "MGRV_VISION_PYTHON": "vision-python",
               "MGRV_RUN_CONFIG": "config.yaml"}
        with patch.dict(os.environ, env), patch.object(seq.subprocess, "run", return_value=NS(returncode=0)) as run:
            seq.Coordinator.step(self.c, "embed-checkpointed")
            command = run.call_args.args[0]
            self.assertIn("--gpus=1", command)
            self.assertIn("--cpus-per-task=9", command)
            self.assertIn("vision-python", command)
            seq.Coordinator.step(self.c, "dev-retrieve", 1, 2)
            command = run.call_args.args[0]
            self.assertIn("text-python", command)
            self.assertEqual(command[command.index("--worker") + 1], "1")


if __name__ == "__main__":
    unittest.main()
