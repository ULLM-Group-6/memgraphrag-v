"""CPU-only runner checks. Run with unittest; no model downloads/API calls."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "webqa_run.py"
spec = importlib.util.spec_from_file_location("webqa_run", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class RunnerTests(unittest.TestCase):
    def test_shared_evidence_groups_are_excluded_from_dev_tuning(self):
        questions = [SimpleNamespace(qid=q, group_id=g, split=s) for q, g, s in (
            ("dev-overlap", "g1", "dev"), ("dev-independent", "g2", "dev"),
            ("test", "g1", "test"))]
        result = runner.split_overlap(questions)
        self.assertEqual(result["exclude_from_dev_tuning"], ["dev-overlap"])
        self.assertEqual(result["test_questions_in_shared_groups"], ["test"])

    def test_reader_never_includes_gold_or_generated_descriptions(self):
        row = {"question": "What colour?", "answers": ["SECRET GOLD"],
               "raw_evidence": [{"doc_id": "D", "image_id": "I", "image_path": "/image.jpg",
                 "image_sha256": "hash", "records": [
                     {"record_id": "cap_I", "kind": "passage", "text": "one two three four"},
                     {"record_id": "caption:I:0", "kind": "caption", "text": "GENERATED"}]}]}
        messages, documents = runner.reader_bundle(row, Tokenizer(), 3)
        self.assertEqual(documents[0]["supplied_text"], "one two three")
        self.assertEqual(documents[0]["passage_ids"], ["cap_I"])
        self.assertEqual(documents[0]["image_id"], "I")
        self.assertNotIn("SECRET GOLD", json.dumps(messages))
        self.assertNotIn("GENERATED", json.dumps(messages))

    def test_reader_text_only_evidence_and_order(self):
        row = {"question": "Q", "raw_evidence": [
            {"doc_id": d, "image_id": None, "image_path": None, "image_sha256": None,
             "records": [{"record_id": d, "kind": "passage", "text": d}]}
            for d in ("B", "A")]}
        messages, docs = runner.reader_bundle(row, Tokenizer(), 256)
        self.assertEqual([d["doc_id"] for d in docs], ["B", "A"])
        self.assertFalse(any(x["type"] == "image" for x in messages[1]["content"]))

    def test_reader_cache_resolves_refs_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "models--Qwen--Qwen2.5-VL-7B-Instruct"
            snapshot = folder / "snapshots" / "abc"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            (folder / "refs").mkdir()
            (folder / "refs" / "main").write_text("abc\n")
            self.assertEqual(runner.resolve_reader(tmp), snapshot.resolve())

    def test_reader_cache_refuses_ambiguous_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "models--Qwen--Qwen2.5-VL-7B-Instruct"
            for commit in ("a", "b"):
                snapshot = folder / "snapshots" / commit
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text("{}")
            with self.assertRaises(ValueError):
                runner.resolve_reader(tmp)

    def test_atomic_records_and_content_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            runner.atomic_json(path, {"a": "é"})
            first = runner.sha(path)
            self.assertEqual(runner.read_json(path), {"a": "é"})
            runner.atomic_json(path, {"a": "changed"})
            self.assertNotEqual(first, runner.sha(path))
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_cached_query_requires_exact_text(self):
        import numpy as np
        encoder = runner.CachedQuery("correct", np.array([1., 0.]), True)
        values, cut = encoder.encode_texts(["correct"])
        self.assertEqual(values.shape, (1, 2))
        self.assertEqual(cut, [True])
        with self.assertRaises(ValueError):
            encoder.encode_texts(["wrong"])

    def test_prepare_real_schema_and_refuse_overwrite(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, artifacts = root / "dataset", root / "artifacts"
            source.mkdir()
            images = artifacts / "corpus" / "images"
            images.mkdir(parents=True)
            Image.new("RGB", (20, 20)).save(images / "1.jpg")
            runner.write_rows(source / "text_manifest.jsonl", [
                {"doc_id": "Document A", "passage_id": "cap_1", "text": "Original caption"}])
            runner.write_rows(source / "image_manifest.jsonl", [
                {"image_id": "1", "doc_id": "Document A", "passage_ids": ["cap_1"],
                 "commons_url": "https://example.com"}])
            runner.write_rows(source / "caption_manifest.jsonl", [
                {"image_id": "1", "generated_description": "Generated description"}])
            question = {"qid": "q", "question": "What?", "answers": ["Answer"],
                        "gold_documents": [{"title": "Document A"}], "gold_image_ids": ["1"],
                        "gold_snippet_ids": [], "distractor_image_ids": [],
                        "distractor_snippet_ids": [], "distractor_documents": [],
                        "evidence_class": "image", "topic": "objects"}
            runner.atomic_json(source / "build_qa.json", [question])
            runner.atomic_json(source / "test_qa.json", [])
            args = SimpleNamespace(artifacts=str(artifacts), dataset=str(source),
                                   images=str(images), smoke_questions=0)
            runner.prepare(args)
            generated = runner.read_rows(artifacts / "manifests" / "captions.jsonl")
            self.assertEqual(generated[0]["text"], "Generated description")
            questions = runner.read_rows(artifacts / "manifests" / "questions.jsonl")
            self.assertEqual(questions[0]["gold_doc_ids"], ["Document A"])
            self.assertEqual(questions[0]["topic"], "objects")
            with self.assertRaises(ValueError):
                runner.prepare(args)


if __name__ == "__main__":
    unittest.main()
