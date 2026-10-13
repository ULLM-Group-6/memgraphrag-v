import numpy as np
import pytest
from PIL import Image

from memgraphrag_v.embeddings import EmbeddingSet, EmbedItem, embed_files, embed_queries, items_from_images
from memgraphrag_v.schemas import ImageRecord, QuestionRecord

from conftest import IMAGE_ROW, QUESTION_ROW

META = {"encoder": "fake", "revision": "r1", "processor": "fake-proc", "processor_revision": "p1"}


@pytest.fixture
def corpus(tmp_path):
    """Three corpus images; the middle one is corrupt."""
    images_dir = tmp_path / "corpus" / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (32, 24), "red").save(images_dir / "a.png")
    (images_dir / "b.jpg").write_bytes(b"not an image")
    Image.new("L", (10, 10), 128).save(images_dir / "c.png")  # grayscale, converted to RGB
    records = [
        ImageRecord(**{**IMAGE_ROW, "image_id": name, "doc_id": f"Doc {name}", "path": f"{name}.{ext}"})
        for name, ext in [("a", "png"), ("b", "jpg"), ("c", "png")]
    ]
    return tmp_path, records


def test_failed_image_is_recorded_not_blanked(corpus, encoder):
    root, records = corpus
    result = embed_files(items_from_images(records), "image", encoder, root, META, batch_size=2)

    assert [r.item_id for r in result.rows] == ["a", "b", "c"]  # input order
    assert [r.status for r in result.rows] == ["ok", "failed", "ok"]
    assert [r.row for r in result.rows] == [0, None, 1]
    assert result.rows[1].error.startswith("UnidentifiedImageError")
    assert len(encoder.images_seen) == 2  # the corrupt file never reached the encoder
    assert result.vectors.shape == (2, encoder.dim)
    assert result.meta.n_rows == 3 and result.meta.n_failed == 1


def test_rows_carry_metadata(corpus, encoder):
    root, records = corpus
    result = embed_files(items_from_images(records), "image", encoder, root, META)
    a = result.row_for("a")
    assert (a.kind, a.image_id, a.doc_id, a.path) == ("image", "a", "Doc a", "corpus/images/a.png")
    assert (a.width, a.height) == (32, 24)
    assert len(a.sha256) == 64
    assert result.meta.encoder == "fake" and result.meta.dim == encoder.dim


def test_vectors_line_up_with_rows(corpus, encoder):
    root, records = corpus
    result = embed_files(items_from_images(records), "image", encoder, root, META, batch_size=1)
    expected = encoder.encode_images([Image.open(root / "corpus/images/c.png").convert("RGB")])[0]
    np.testing.assert_allclose(result.vector("c"), expected)
    with pytest.raises(KeyError):
        result.vector("b")


def test_save_load_round_trip(tmp_path, corpus, encoder):
    root, records = corpus
    result = embed_files(items_from_images(records), "image", encoder, root, META)
    result.save(tmp_path / "out")
    loaded = EmbeddingSet.load(tmp_path / "out")
    assert loaded.meta == result.meta
    assert loaded.rows == result.rows
    np.testing.assert_array_equal(loaded.vectors, result.vectors)


def test_crop_rows_carry_entity(corpus, encoder):
    root, _ = corpus
    crop = EmbedItem(item_id="crop-1", path="corpus/images/a.png", image_id="a", doc_id="Doc a", entity_id="entity-x")
    row = embed_files([crop], "crop", encoder, root, META).rows[0]
    assert (row.kind, row.image_id, row.entity_id) == ("crop", "a", "entity-x")


def test_top_k_orders_by_score_then_id():
    from memgraphrag_v.embeddings import EmbeddingMeta, EmbeddingRow

    vectors = np.array([[1, 0], [0, 1], [1, 0], [0.6, 0.8]], dtype=np.float32)
    rows = [EmbeddingRow(row=i, kind="image", item_id=name, status="ok") for i, name in enumerate("zyxw")]
    meta = EmbeddingMeta(**META, dim=2, kind="image", n_rows=4, n_failed=0)
    result = EmbeddingSet(meta, rows, vectors)
    top = result.top_k(np.array([1, 0], dtype=np.float32), k=3)
    assert [(r.item_id, round(s, 3)) for r, s in top] == [("x", 1.0), ("z", 1.0), ("w", 0.6)]


def test_misaligned_set_rejected():
    from memgraphrag_v.embeddings import EmbeddingMeta, EmbeddingRow

    rows = [EmbeddingRow(row=0, kind="image", item_id="a", status="ok")]
    meta = EmbeddingMeta(**META, dim=2, kind="image", n_rows=1, n_failed=0)
    with pytest.raises(ValueError):
        EmbeddingSet(meta, rows, np.zeros((2, 2), dtype=np.float32))


def test_queries_mark_truncation(encoder):
    questions = [
        QuestionRecord(**QUESTION_ROW),
        QuestionRecord(**{**QUESTION_ROW, "qid": "short", "question": "Who is this?"}),
    ]
    result = embed_queries(questions, encoder, META)
    assert [(r.item_id, r.status, r.row) for r in result.rows] == [("abc123", "truncated", 0), ("short", "ok", 1)]
    assert result.meta.kind == "query"
