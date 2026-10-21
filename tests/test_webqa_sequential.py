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

    def test_timeout_allowed_but_success_must_skip_backup(self):
        self.check("TIMEOUT", "0:15")
        with self.assertRaises(RuntimeError):
            self.check("COMPLETED")

    def test_failure_cancel_oom_refused(self):
        for state in ("FAILED", "CANCELLED by 123", "OUT_OF_MEMORY", "NODE_FAIL"):
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                self.check(state, "1:0")

    def test_wrong_owner_refused(self):
        with self.assertRaisesRegex(RuntimeError, "different user"):
            self.check("COMPLETED", owner="other")

    def test_missing_accounting_fails_closed(self):
        with patch.object(seq.subprocess, "run", return_value=NS(stdout="")), patch.object(seq.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "No eligible"):
                seq.check_previous("123")

    def test_checkpoint_exit_needs_matching_intent_marker(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            registration = root / 'registration.json'
            registration.write_text('{}')
            path = root / 'index/parallel-v2/sequential-v1/attempts/123.json'
            path.parent.mkdir(parents=True)
            result = NS(stdout='123|me|FAILED|75:0|\n')
            with patch.dict(os.environ, {'USER': 'me'}), patch.object(seq.subprocess, 'run', return_value=result):
                with self.assertRaises(RuntimeError):
                    seq.check_previous('123', root, registration)
                marker = {'job_id': '123', 'attempt': 1, 'state': 'INCOMPLETE_TIME_BUDGET',
                          'registration_sha256': seq.sha(registration)}
                path.write_text(json.dumps(marker))
                seq.check_previous('123', root, registration)
                marker['registration_sha256'] = 'different submission'
                path.write_text(json.dumps(marker))
                with self.assertRaises(RuntimeError):
                    seq.check_previous('123', root, registration)


class Workflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        (self.root / "index").mkdir()
        self.manifest = self.root / "index/manifest.json"
        self.set_stages("text", "ground")
        self.c = object.__new__(seq.Coordinator)
        self.c.root = self.root
        self.c.args = NS(mode="embed", run_id="webqa-tuned-v1", attempt=1, previous="-",
                         registration=str(self.root / "registration.json"))
        self.c.deadline = seq.time.time() + 10000
        self.calls = []
        self.c.step = lambda action, worker=0, candidate=0: self.calls.append((action, worker, candidate)) or True
        self.statuses = []
        self.c.status = lambda state, message: self.statuses.append((state, message))

    def tearDown(self):
        self.temp.cleanup()

    def set_stages(self, *names):
        self.manifest.write_text(json.dumps({"stages": {name: {} for name in names}}))

    def test_embedding_success_exits_without_assembly(self):
        self.assertEqual(self.c.embed(), 0)
        self.assertEqual([x[0] for x in self.calls], ["embed-checkpointed"])
        self.assertEqual(self.statuses[-1][0], 'EMBED_COMPLETE')

    def test_finished_index_is_reused(self):
        self.set_stages("text", "ground", "embed", "assemble")
        self.c.embed()
        self.assertEqual([x[0] for x in self.calls], ["embed-checkpointed"])

    def test_both_incomplete_attempts_exit_nonzero_and_block_assembly(self):
        self.c.step = lambda *args, **kwargs: False
        self.assertEqual(self.c.embed(), 75)
        self.c.args.attempt = 2
        self.assertEqual(self.c.embed(), 75)
        self.assertIn('BOTH embedding windows exhausted', self.statuses[-1][1])
        self.assertIn('MUST NOT START', self.statuses[-1][1])

    def test_assembly_allocation_never_encodes(self):
        self.assertEqual(self.c.assemble(), 0)
        self.assertEqual([x[0] for x in self.calls], ["embed-finalize", "assemble"])
        self.assertEqual(self.statuses[-1][0], 'INDEX_COMPLETE')

    def test_finalization_error_prevents_assembly(self):
        def fail(action, *args):
            self.calls.append(action)
            raise RuntimeError('missing worker checkpoint')
        self.c.step = fail
        with self.assertRaisesRegex(RuntimeError, 'missing worker'):
            self.c.assemble()
        self.assertEqual(self.calls, ['embed-finalize'])
        self.assertEqual(self.statuses, [])

    def test_status_marker_persists_intent(self):
        Path(self.c.args.registration).write_text('{}')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}):
            seq.Coordinator.status(self.c, 'INCOMPLETE_TIME_BUDGET', 'checkpoint saved')
        value = json.loads((self.root / 'index/parallel-v2/sequential-v1/attempts/123.json').read_text())
        self.assertEqual(value['state'], 'INCOMPLETE_TIME_BUDGET')
        self.assertEqual(value['registration_sha256'], seq.sha(self.c.args.registration))

    def test_timeout_warning_for_second_attempt(self):
        self.c.args.attempt = 2
        with patch.object(seq, 'log') as log:
            self.c.timeout_warning()
        self.assertIn('BOTH embedding windows', log.call_args.args[0])

    def test_second_attempt_termination_logs_blocked_assembly(self):
        self.c.args.attempt = 2
        with patch.object(seq.os, 'write') as write, self.assertRaises(SystemExit) as stopped:
            self.c.interrupted(15, None)
        self.assertEqual(stopped.exception.code, 143)
        self.assertIn(b'Both embedding attempts ended without success', write.call_args.args[1])

    def test_backup_rejects_real_failure_before_preparation(self):
        self.c.args.attempt = 2
        self.c.args.previous = '123'
        Path(self.c.args.registration).write_text('{}')
        with patch.object(seq, 'signature', return_value={}), patch.object(seq, 'check_previous', side_effect=RuntimeError('failed predecessor')):
            with self.assertRaises(RuntimeError):
                self.c.run()
        self.assertEqual(self.calls, [])

    def run_mode(self, mode):
        self.c.args.mode = mode
        Path(self.c.args.registration).write_text("{}")
        with patch.object(seq, "signature", return_value={}), patch.object(seq, "check_previous"), patch.object(seq.signal, 'signal'), \
             patch.object(seq, "chain_lock", return_value=nullcontext()):
            return self.c.run()

    def test_run_propagates_incomplete_exit_code(self):
        self.c.step = lambda action, *args: action != 'embed-checkpointed'
        self.assertEqual(self.run_mode('embed'), 75)

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
