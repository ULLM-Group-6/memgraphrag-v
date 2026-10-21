#!/usr/bin/env python3
"""Real schemas/files and legacy checkpoints, with no GPU/model loading."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import numpy as np
from PIL import Image
import webqa_parallel as api
from memgraphrag_v.config import ExperimentConfig
from memgraphrag_v.paths import Artifacts
from memgraphrag_v.schemas import Detection, ImageGrounding
from webqa_validation_cache import HashCache
from webqa_fast_validation import ValidatedRun, CachedEmbeddings
from webqa_fast_stage import finalize, assemble


class TinyEncoder:
    dim = 2
    def encode_images(self, images):
        return np.tile(np.array([[0.6, 0.8]], dtype=np.float32), (len(images), 1))


class Integration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.tmp.name)
        self.art = Artifacts(self.root)
        self.work = self.art.index_dir / "parallel-v2"
        self.work.mkdir(parents=True)
        self.config = ExperimentConfig.model_validate({"text": {"llm_name": "test"}, "retrieval": {"beta": .5}})
        self.images = [api.ImageRecord(image_id=str(i), doc_id="doc", path=f"{i}.png", passage_ids=[], ok=True) for i in range(2)]
        self.art.images_dir.mkdir(parents=True)
        for image in self.images:
            Image.new("RGB", (4, 4), "red").save(self.art.image_path(image))
        self.art.image_manifest.parent.mkdir(parents=True)
        self.art.image_manifest.write_text("".join(i.model_dump_json() + "\n" for i in self.images))
        for path in (self.art.text_manifest, self.art.captions, self.art.questions, self.art.texts,
                     self.art.chunks, self.art.entities, self.art.memory_path, self.root / "preparation.json"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        checkpoint = self.art.models_dir / "sam3/sam3.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fake checkpoint; never loaded")
        self.env = patch.dict("os.environ", {"MGRV_SAM3_CHECKPOINT": str(checkpoint)})
        self.env.start()
        self.patch_prepare = patch.object(api.tuning, "prepare_session")
        self.patch_verify = patch.object(api.tuning, "verify_session")
        self.patch_prep = patch.object(api.runner, "preparation")
        self.patch_prepare.start(); self.patch_verify.start(); self.patch_prep.start()
        text = api.p.effective_text(self.config)
        settings = {key: getattr(text, key) for key in ("llm_name", "embedding_model", "chunk_size", "chunk_overlap",
                      "ontology_filter_mode", "ontology_min_frequency", "ontology_low_percent")}
        sources = {name: api.required_hash(getattr(self.art, name)) for name in ("text_manifest", "captions", "image_manifest", "questions")}
        store = self.root / "chunk-store"
        store.write_bytes(b"vectors")
        outputs = {key: api.required_hash(path) for key, path in {"texts": self.art.texts, "chunks": self.art.chunks,
                   "entities": self.art.entities, "memory": self.art.memory_path}.items()}
        outputs["text_embedding_files"] = {"chunk": {"path": "chunk-store", "sha256": api.required_hash(store)}}
        self.manifest = {"version": 1, "stages": {"text": {"settings": settings, "outputs": outputs,
            "inputs": {"text_manifest": sources["text_manifest"], "captions": sources["captions"],
                       "image_identities": api.p._json_sha256([[i.image_id, i.doc_id] for i in self.images])}}}}
        api.p._write_manifest(self.art, self.manifest)
        self.plan = api.snapshot(self.config, self.art, self.images)
        api.runner.atomic_json(self.work / "plan.json", self.plan)
        self.records = []
        for image in self.images:
            mask = self.art.grounding_dir / "masks" / f"{image.image_id}.npz"
            mask.parent.mkdir(parents=True, exist_ok=True)
            mask.write_bytes(b"mask bytes are only hashed, never opened by embedding")
            crop = self.art.grounding_dir / "crops" / f"{image.image_id}.png"
            crop.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), "blue").save(crop)
            det = Detection(detection_id=f"det{image.image_id}", image_id=image.image_id, entity_id="entity", prompt="x",
                            prompt_index=0, index=0, score=.9, box=[0., 0., 4., 4.], mask_index=0,
                            crop_id=f"crop{image.image_id}", crop_path=self.art.relative(crop),
                            crop_hash=api.required_hash(crop), crop_status="saved")
            record = ImageGrounding(image_id=image.image_id, doc_id=image.doc_id, status="grounded", fingerprint="test",
                mask_file=self.art.relative(mask), mask_hash=api.required_hash(mask), detections=[det])
            self.records.append(record)
            api.runner.atomic_json(self.art.grounding_dir / "records" / f"{api.sam._key(image.image_id)}.json", record.model_dump(mode="json"))
        self.manifest["stages"]["ground"] = {
            "inputs": {"image_manifest": sources["image_manifest"], "fingerprint": api.p._json_sha256(self.plan["images"]),
                       **{key: outputs[key] for key in ("texts", "chunks", "entities")}},
            "settings": {"sam3": self.config.sam3.model_dump(mode="json"), "grounder": self.plan["grounder"]},
            "outputs": {"grounding": api.p._json_sha256([r.model_dump(mode="json") for r in self.records])}}
        api.p._write_manifest(self.art, self.manifest)
        self.db = self.work / "sequential-v1/validation-v2/hashes.sqlite3"

    def tearDown(self):
        self.env.stop(); self.patch_prepare.stop(); self.patch_verify.stop(); self.patch_prep.stop()
        self.tmp.cleanup()

    def run_for(self, cache):
        return ValidatedRun(api, self.config, self.art, "test-run", cache).prepare()

    def test_prepare_never_hashes_source_images_masks_or_crops(self):
        with HashCache(self.db) as cache:
            original = cache.digest
            def guarded(path):
                self.assertNotIn(Path(path).suffix, (".png", ".npz"))
                return original(path)
            with patch.object(cache, "digest", side_effect=guarded):
                run = self.run_for(cache)
            self.assertEqual(len(run.records), 2)

    def test_second_process_reuses_receipt_without_rereading_records(self):
        with HashCache(self.db) as cache:
            self.run_for(cache)
        with HashCache(self.db) as cache, patch.object(cache, "read_json", side_effect=AssertionError("receipt should be reused")):
            run = self.run_for(cache)
            self.assertEqual(len(run.images), 2)

    def test_changed_record_invalidates_receipt_and_is_rejected(self):
        with HashCache(self.db) as cache:
            self.run_for(cache)
        record_path = self.art.grounding_dir / "records" / f"{api.sam._key('0')}.json"
        value = json.loads(record_path.read_text()); value["fingerprint"] = "changed fingerprint"
        api.runner.atomic_json(record_path, value)
        with HashCache(self.db) as cache, self.assertRaisesRegex(ValueError, "Grounding records changed"):
            self.run_for(cache)

    def test_changed_crop_rejected_when_consumed(self):
        with HashCache(self.db) as cache:
            run = self.run_for(cache)
            item = run.items(0)["crop"][0]
            run.validate_inputs([item], "crop")
            Image.new("RGB", (5, 5), "green").save(self.root / item.path)
            with self.assertRaisesRegex(RuntimeError, "changed file"):
                run.validate_inputs([item], "crop")

    def test_bad_mask_deferred_to_finalization_not_silently_accepted(self):
        (self.root / self.records[0].mask_file).write_bytes(b"changed mask")
        with HashCache(self.db) as cache:
            run = self.run_for(cache)
            with self.assertRaisesRegex(RuntimeError, "changed file"):
                run.verify_outputs("ground")

    def test_old_worker_checkpoints_finalize_with_unchanged_schema(self):
        with HashCache(self.db) as cache:
            run = self.run_for(cache)
            for worker in range(2):
                for kind, items in run.items(worker).items():
                    result = api.embed_files(items, kind, TinyEncoder(), self.root, api.visual.encoder_meta(self.config.visual_encoder))
                    identity = {"plan": api.runner.stamp(self.plan), "ground": self.manifest["stages"]["ground"], "worker": worker, "kind": kind}
                    api.save_set(self.work / "embeddings" / str(worker) / kind, result, identity)
            finalize(run)
            for kind in ("image", "crop"):
                expected = api.p._embedding_sha256(self.art.embeddings_dir(self.config.visual_encoder.name, kind))
                self.assertEqual(run.manifest["stages"]["embed"]["outputs"][f"{kind}_embeddings"], expected)
            api.p._verify_outputs(self.art, self.config, "embed", run.manifest["stages"]["embed"]["outputs"])
            # The original graph builder is called exactly once, no algorithm replacement.
            with patch.object(api.p, "_stage_assemble", return_value={"test": "stub graph builder"}) as build:
                assemble(run)
                build.assert_called_once()
            self.assertEqual(run.manifest["stages"]["assemble"]["inputs"], run.expected_inputs("assemble"))
            api.p._validate_chain(run.manifest)


if __name__ == "__main__":
    unittest.main()
