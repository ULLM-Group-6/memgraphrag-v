"""CPU-only tests. No models, network, GPU, or real dataset are used."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from PIL import Image
import webqa_parallel as parallel
import webqa_run as runner
from memgraphrag_v.embeddings import EmbeddingMeta, EmbeddingRow, EmbeddingSet
from memgraphrag_v.config import ExperimentConfig, Sam3Config
from memgraphrag_v.paths import Artifacts
from memgraphrag_v.schemas import ImageRecord, ImageGrounding, EntityRecord


def embedding(ids, failed=(), kind="image"):
    rows, vectors = [], []
    for item in ids:
        row = None if item in failed else len(vectors)
        rows.append(EmbeddingRow(row=row, kind=kind, item_id=item,
                                  status="failed" if row is None else "ok"))
        if row is not None:
            vectors.append([float(item), float(item) + 10])
    meta = EmbeddingMeta(encoder="test", revision="r", processor="p", processor_revision="r",
                         kind=kind, dim=2, n_rows=len(rows), n_failed=len(failed))
    return EmbeddingSet(meta, rows, np.array(vectors, dtype=np.float32).reshape(-1, 2))


class ParallelTests(unittest.TestCase):
    def test_shards_disjoint_exhaustive_even_and_odd(self):
        for size in (2, 3, 9, 480):
            rows = list(range(size))
            a, b = parallel.partition(rows, 0), parallel.partition(rows, 1)
            self.assertFalse(set(a) & set(b))
            self.assertEqual(sorted(a + b), rows)

    def test_merge_restores_order_and_remaps_rows(self):
        result = parallel.merge_sets([embedding(["0", "2", "4"]), embedding(["1", "3"])],
                                     ["0", "1", "2", "3", "4"])
        self.assertEqual([r.row for r in result.rows], list(range(5)))
        np.testing.assert_array_equal(result.vectors[:, 0], np.arange(5))

    def test_merge_preserves_failed_rows(self):
        result = parallel.merge_sets([embedding(["0", "2"], failed=["2"]), embedding(["1"])],
                                     ["0", "1", "2"])
        self.assertEqual([r.row for r in result.rows], [0, 1, None])
        self.assertEqual((result.meta.n_rows, result.meta.n_failed), (3, 1))

    def test_empty_crop_shards(self):
        result = parallel.merge_sets([embedding([], kind="crop"), embedding([], kind="crop")], [])
        self.assertEqual(result.vectors.shape, (0, 2))

    def test_merge_rejects_duplicate_missing_extra_and_metadata(self):
        cases = [([embedding(["0"]), embedding(["0"])], ["0"]),
                 ([embedding(["0"])], ["0", "1"]),
                 ([embedding(["0", "1"])], ["0"]),
                 ([embedding(["0"]), embedding(["1"], kind="crop")], ["0", "1"])]
        for parts, expected in cases:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                parallel.merge_sets(parts, expected)

    def test_embedding_completion_integrity_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            embedding(["0"]).save(directory)
            self.assertIsNone(parallel.saved_set(directory, {"plan": "x"}))
            parallel.save_set(directory, embedding(["0"]), {"plan": "x"})
            self.assertEqual(parallel.saved_set(directory, {"plan": "x"}).rows[0].item_id, "0")
            with self.assertRaises(RuntimeError):
                parallel.saved_set(directory, {"plan": "different"})
            (directory / "vectors.npy").write_bytes(b"corrupt")
            with self.assertRaises(RuntimeError):
                parallel.saved_set(directory, {"plan": "x"})

    def test_merge_questions_order_and_exact_coverage(self):
        result = parallel.merge_rows([[{"question_id": "a"}, {"question_id": "c"}],
                                      [{"question_id": "b"}]], ["a", "b", "c"])
        self.assertEqual([r["question_id"] for r in result], ["a", "b", "c"])
        for rows in ([{"question_id": "a"}], [{"question_id": "a"}, {"question_id": "a"}],
                     [{"question_id": "a"}, {"question_id": "b"}, {"question_id": "x"}]):
            with self.assertRaises(ValueError):
                parallel.merge_rows([rows], ["a", "b"])

    def test_question_selection_unchanged_by_default(self):
        # Avoid a second schema fixture: return already validated objects.
        questions = [SimpleNamespace(qid=str(i), split="dev") for i in range(5)]
        with patch.object(runner, "read_rows", return_value=questions), \
                patch("memgraphrag_v.schemas.QuestionRecord.model_validate", side_effect=lambda x: x):
            artifacts = SimpleNamespace(questions=Path("unused"))
            self.assertEqual(runner.selection(artifacts, "dev", 0), questions)
            self.assertEqual(runner.selection(artifacts, "dev", 4, 1, 2), questions[1:4:2])
            with self.assertRaises(ValueError):
                runner.selection(artifacts, "dev", 0, 2, 2)

    def test_grounding_reuses_existing_valid_records_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Artifacts(Path(tmp))
            artifacts.images_dir.mkdir(parents=True)
            image = ImageRecord(image_id="i", doc_id="d", path="i.png", passage_ids=[], ok=True)
            Image.new("RGB", (20, 20)).save(artifacts.image_path(image))
            entity = EntityRecord(entity_id="e", name="object", chunk_ids=["c"])
            config = Sam3Config()
            fingerprint = parallel.sam._fingerprint(artifacts.image_path(image), [entity], config, "g")
            record = ImageGrounding(image_id="i", doc_id="d", status="grounded",
                                    fingerprint=fingerprint, prompts=["object"], entity_ids=["e"])
            path = artifacts.grounding_dir / "records" / f"{parallel.sam._key('i')}.json"
            path.parent.mkdir(parents=True)
            parallel.sam._write_record(path, record)
            before = path.read_bytes()
            def forbidden():
                raise AssertionError("Completed image must not load SAM3")
            rows = parallel.sam.ground_images([image], {"i": ["c"]}, [entity], artifacts, config, forbidden, "g")
            self.assertEqual(rows, [record])
            self.assertEqual(path.read_bytes(), before)

    def test_finalizer_refuses_failed_or_missing_grounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Artifacts(Path(tmp))
            artifacts.images_dir.mkdir(parents=True)
            image = ImageRecord(image_id="i", doc_id="d", path="i.png", passage_ids=[], ok=True)
            Image.new("RGB", (20, 20)).save(artifacts.image_path(image))
            with patch.object(parallel, "grounding_context", return_value=({"i": []}, [])):
                with self.assertRaises(ValueError):
                    parallel.validate_grounding(artifacts, ExperimentConfig(), [image], "g")
                record = ImageGrounding(image_id="i", doc_id="d", status="failed", fingerprint="bad", error="x")
                (artifacts.grounding_dir / "records").mkdir(parents=True)
                parallel.sam._write_record(artifacts.grounding_dir / "records" / f"{parallel.sam._key('i')}.json", record)
                with self.assertRaises(RuntimeError):
                    parallel.validate_grounding(artifacts, ExperimentConfig(), [image], "g")

    def test_completed_embedding_stage_is_not_overwritten(self):
        with patch.object(parallel.p, "_read_manifest", return_value={"stages": {"embed": {}}}), \
                patch.object(parallel.p, "index_with_memory_v") as index, \
                patch.object(parallel, "embedding_items", side_effect=AssertionError("Must not remerge")):
            parallel.finalize_embed(None, "config", "artifacts", [], None, {})
            index.assert_called_once_with("artifacts", "config", stages=["embed"], device="cpu")

    def test_collect_validates_contracts_and_produces_ordered_full_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Artifacts(Path(tmp))
            runner.atomic_json(artifacts.index_manifest, {"stages": {}})
            config = ExperimentConfig()
            plan = {"workers": 2}
            args = SimpleNamespace(run_id="full", split="dev")
            questions = [SimpleNamespace(qid=str(i), question=f"Question {i}") for i in range(3)]
            for worker in range(2):
                directory = artifacts.run_dir(f"full-part-{worker}")
                selected = parallel.partition(questions, worker)
                contract = {"questions": [[q.qid, q.question] for q in selected],
                            "split": "dev", "config_hash": config.config_hash(),
                            "index_manifest_sha256": runner.sha(artifacts.index_manifest),
                            "shard": {"index": worker, "count": 2}}
                runner.atomic_json(directory / "run.json", contract)
                for name in parallel.RESULTS:
                    key = "qid" if name == "labels.jsonl" else "question_id"
                    runner.write_rows(directory / name, [{key: q.qid} for q in selected])
                runner.atomic_json(directory / "reader_settings.json", {
                    "model": "same-model", "contract_sha256": runner.stamp(contract)})
                runner.atomic_json(directory / "retrieval_provenance.json", {"test": True})
                runner.atomic_json(directory / "completed.json", {
                    "questions": len(selected), "run_sha256": runner.stamp(contract),
                    "answers_sha256": runner.sha(directory / "answers.jsonl")})
                files = (*parallel.RESULTS, "run.json", "completed.json", "reader_settings.json",
                         "retrieval_provenance.json")
                runner.atomic_json(directory / "parallel_complete.json", {
                    "plan": runner.stamp(plan), "files": {f: runner.sha(directory / f) for f in files}})
            with patch.object(runner, "selection", return_value=questions):
                parallel.collect(args, config, artifacts, [], None, plan)
                out = artifacts.run_dir("full")
                self.assertEqual([r["question_id"] for r in runner.read_rows(out / "answers.jsonl")],
                                 ["0", "1", "2"])
                self.assertEqual(runner.read_json(out / "completed.json")["questions"], 3)
                # A changed worker file must never be accepted as a completed result.
                runner.write_rows(artifacts.run_dir("full-part-0") / "answers.jsonl", [])
                with self.assertRaises(ValueError):
                    parallel.collect(args, config, artifacts, [], None, plan)

    def test_embedding_workers_merge_and_normal_stage_reuses_without_encoder(self):
        class FakeEncoder:
            dim = 2
            def encode_images(self, images):
                return np.tile(np.array([[0.6, 0.8]], dtype=np.float32), (len(images), 1))
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Artifacts(Path(tmp))
            artifacts.images_dir.mkdir(parents=True)
            (artifacts.grounding_dir / "records").mkdir(parents=True)
            images = []
            for i in range(3):
                image = ImageRecord(image_id=f"i{i}", doc_id="d", path=f"i{i}.png", passage_ids=[], ok=True)
                Image.new("RGB", (20, 20)).save(artifacts.image_path(image))
                images.append(image)
                record = ImageGrounding(image_id=image.image_id, doc_id="d", status="no_prompts", fingerprint="x")
                parallel.sam._write_record(artifacts.grounding_dir / "records" / f"{parallel.sam._key(image.image_id)}.json", record)
            runner.write_rows(artifacts.image_manifest, images)
            config, work, plan = ExperimentConfig(), Path(tmp) / "workers", {"same": "plan"}
            with patch.object(parallel, "checked_stage", return_value={"outputs": {"grounding": "x"}}), \
                    patch.object(parallel.visual, "load_encoder", return_value=FakeEncoder()) as load:
                for worker in range(2):
                    parallel.embed_worker(SimpleNamespace(worker=worker), config, artifacts, images, work, plan)
                self.assertEqual(load.call_count, 2)
                load.side_effect = AssertionError("Resume/finalization must not reload encoder")
                for worker in range(2):
                    parallel.embed_worker(SimpleNamespace(worker=worker), config, artifacts, images, work, plan)
                with patch.object(parallel.p, "_read_manifest", return_value={"stages": {}}), \
                        patch.object(parallel.p, "index_with_memory_v", side_effect=lambda a, c, **kw: parallel.p._stage_embed(a, c, False, "cpu")):
                    parallel.finalize_embed(None, config, artifacts, images, work, plan)
            result = EmbeddingSet.load(artifacts.embeddings_dir(config.visual_encoder.name, "image"))
            self.assertEqual([r.item_id for r in result.rows], ["i0", "i1", "i2"])
            crops = EmbeddingSet.load(artifacts.embeddings_dir(config.visual_encoder.name, "crop"))
            self.assertEqual(crops.vectors.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
