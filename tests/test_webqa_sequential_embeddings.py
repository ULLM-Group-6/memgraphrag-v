#!/usr/bin/env python3
"""Real file-format/cache integration with a tiny CPU encoder, never EVA/torch."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from PIL import Image
import webqa_parallel as api
import webqa_embed_checkpointed as helper
from memgraphrag_v.embeddings import EmbedItem


class TinyEncoder:
    dim = 2
    def __init__(self):
        self.count = 0

    def encode_images(self, images):
        self.count += len(images)
        return np.tile(np.array([[0.6, 0.8]], dtype=np.float32), (len(images), 1))


class Integration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.tmp.name)
        Image.new("RGB", (4, 4), "red").save(self.root / "source.png")
        self.items = [EmbedItem(str(i), "source.png", image_id=str(i), doc_id="doc") for i in range(258)]
        self.encoder = TinyEncoder()
        self.config = NS(visual_encoder=NS(name="tiny", revision="v1", processor="tiny", processor_revision="v1"))
        self.identity = {"plan": "test", "ground": {}, "worker": 0, "kind": "image"}

    def tearDown(self):
        self.tmp.cleanup()

    def run_kind(self, clock=lambda: 0):
        return helper.resume_kind(api, self.items, "image", self.config, NS(root=self.root),
            self.root / "worker", self.root / "segments", self.identity,
            lambda batch, kind: api.embed_files(batch, kind, self.encoder, self.root,
                                                api.visual.encoder_meta(self.config.visual_encoder)),
            lambda kind: None, 100, clock)

    def test_real_files_resume_and_canonical_order(self):
        ticks = iter([0, 101])
        with self.assertRaises(helper.SliceFinished):
            self.run_kind(clock=lambda: next(ticks))
        self.assertEqual(self.encoder.count, 128)
        result = self.run_kind()
        self.assertEqual(self.encoder.count, 258)
        self.assertEqual([row.item_id for row in result.rows], [str(i) for i in range(258)])
        self.assertEqual([row.row for row in result.rows], list(range(258)))
        direct = api.embed_files(self.items, "image", TinyEncoder(), self.root,
                                api.visual.encoder_meta(self.config.visual_encoder))
        np.testing.assert_array_equal(result.vectors, direct.vectors)
        reloaded = api.saved_set(self.root / "worker", self.identity)
        api.check_set(reloaded, self.items, "image", self.config, NS(root=self.root))

    def test_source_change_is_rejected(self):
        self.run_kind()
        Image.new("RGB", (4, 4), "blue").save(self.root / "source.png")
        with self.assertRaisesRegex(RuntimeError, "file changed"):
            self.run_kind()

    def test_vector_tampering_is_rejected(self):
        self.run_kind()
        with (self.root / "worker/vectors.npy").open("ab") as file:
            file.write(b"changed")
        with self.assertRaisesRegex(RuntimeError, "Corrupt"):
            self.run_kind()


if __name__ == "__main__":
    unittest.main()
