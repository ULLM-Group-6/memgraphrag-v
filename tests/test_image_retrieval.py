import json
from pathlib import Path

import numpy as np
import pytest

from memgraphrag_v import image_retrieval
from memgraphrag_v.dummy import COLOURS, make_dummy_dataset
from memgraphrag_v.embeddings import EmbeddingSet, embed_files, embed_queries, items_from_images
from memgraphrag_v.image_retrieval import image_passage_edges, retrieve_direct, select_documents
from memgraphrag_v.io import read_jsonl
from memgraphrag_v.paths import Artifacts
from memgraphrag_v.schemas import ImageRecord, QuestionRecord, RetrievalResult

CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"
META = {"encoder": "BAAI/EVA-CLIP-8B", "revision": None, "processor": "colour", "processor_revision": None}


class ColourEncoder:
    """Understands colours only: an image maps to its saturated pixels'
    nearest colour, a text to the colour words in it. Enough to check that
    retrieval finds the right document, without a GPU."""

    dim = len(COLOURS)

    def encode_images(self, images):
        return np.stack([self._image(np.asarray(image, dtype=float)) for image in images])

    def encode_texts(self, texts):
        return np.stack([self._unit([c in t.lower() for c in COLOURS]) for t in texts]), [False] * len(texts)

    def _image(self, pixels):
        saturated = pixels[pixels.max(-1) - pixels.min(-1) > 60]
        if not len(saturated):
            return self._unit([1] * self.dim)
        palette = np.array(list(COLOURS.values()), dtype=float)
        nearest = np.linalg.norm(saturated.mean(0) - palette, axis=1).argmin()
        return self._unit(np.arange(self.dim) == nearest)

    def _unit(self, v):
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)


def _images(*rows):
    return {i.image_id: i for i in (ImageRecord(image_id=a, doc_id=d, path=f"{a}.png", passage_ids=[f"p-{d}"], ok=True)
                                    for a, d in rows)}


def test_select_documents_keeps_best_image_per_document():
    images = _images(("a1", "A"), ("a2", "A"), ("b1", "B"), ("c1", "C"))
    evidence = select_documents({"a1": 0.2, "a2": 0.9, "b1": 0.5, "c1": 0.1}, images, n_docs=2)
    assert [(e.doc_id, e.image_id, e.score) for e in evidence] == [("A", "a2", 0.9), ("B", "b1", 0.5)]
    assert evidence[0].passage_ids == ["p-A"]


def test_select_documents_breaks_ties_by_image_id():
    images = _images(("x", "X"), ("y", "Y"))
    assert [e.doc_id for e in select_documents({"y": 0.5, "x": 0.5}, images)] == ["X", "Y"]


@pytest.fixture
def dummy(tmp_path, monkeypatch):
    """The dummy dataset, embedded with ColourEncoder where the CLI expects it."""
    artifacts = Artifacts(tmp_path)
    make_dummy_dataset(artifacts)
    images = read_jsonl(artifacts.image_manifest, ImageRecord)
    questions = read_jsonl(artifacts.questions, QuestionRecord)
    encoder = ColourEncoder()
    items = items_from_images(images, artifacts.images_dir.relative_to(tmp_path).as_posix())
    image_set = embed_files(items, "image", encoder, tmp_path, META)
    query_set = embed_queries(questions, encoder, META)
    image_set.save(artifacts.embeddings_dir(META["encoder"], "image"))
    query_set.save(artifacts.embeddings_dir(META["encoder"], "query"))
    monkeypatch.setenv("MGRV_ARTIFACTS", str(tmp_path))
    return artifacts, images, questions, image_set, query_set


def test_dummy_dataset_records_the_broken_picture(dummy):
    _, images, _, image_set, _ = dummy
    broken = next(i for i in images if i.doc_id == "Broken picture")
    assert not broken.ok
    assert image_set.row_for(broken.image_id).status == "failed"
    assert image_set.meta.n_failed == 1


def test_edges_link_embedded_images_to_their_passages(dummy):
    _, images, _, image_set, _ = dummy
    edges = image_passage_edges(images, image_set)
    by_doc = {i.image_id: i.doc_id for i in images}
    assert all(w == 1.0 for _, _, w in edges)
    assert {by_doc[a] for a, _, _ in edges}.isdisjoint({"Broken picture", "Grey frame"})
    assert sum(by_doc[a] == "Red circle" for a, _, _ in edges) == 2


def test_direct_retrieval_finds_the_gold_document(dummy):
    _, images, questions, image_set, query_set = dummy
    results = retrieve_direct(questions, query_set, image_set, images)
    for q, r in zip(questions, results):
        docs = [e.doc_id for e in r.evidence]
        assert q.gold_doc_ids[0] in docs[:3]  # the three shapes of its colour score equally
        assert r.route == "direct" and r.system == "D"
        assert len(docs) == 5 and "Broken picture" not in docs and "Toy corpus" not in docs


def test_cli_writes_a_valid_run(dummy, capsys):
    artifacts, *_ = dummy
    image_retrieval.main(["--config", str(CONFIG), "--run-id", "dummy-d", "--split", "dev"])
    results = read_jsonl(artifacts.run_dir("dummy-d") / "retrieval.jsonl", RetrievalResult)
    summary = json.loads(capsys.readouterr().out)
    assert len(results) == summary["questions"] == 6
    assert summary["gold_doc_in_top_5"] == 6
    assert (artifacts.run_dir("dummy-d") / "config.yaml").exists()
