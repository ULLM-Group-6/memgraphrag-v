"""Pure-logic checks for the combined pipeline: IDs, associations, score
handling and crops. No models, graph or upstream code is involved."""

import numpy as np
import pytest

from memgraphrag_v.pipeline import chunk_texts, link_images, normalize_texts
from memgraphrag_v.retrieval import check_reset, fuse_seeds, minmax_normalize, normalize_sum
from memgraphrag_v.sam3_processing import crop_masked, image_prompts
from memgraphrag_v.schemas import CaptionRecord, EntityRecord, ImageRecord, PassageRecord


class WordTokenizer:
    """One token per word, enough for the windowing."""

    model_max_length = 512

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return " ".join(ids)


def _corpus():
    passages = [
        PassageRecord(doc_id="D", passage_id="p1", text="alpha beta gamma delta epsilon zeta"),
        PassageRecord(doc_id="E", passage_id="p2", text="alpha beta gamma delta epsilon zeta"),
        PassageRecord(doc_id="D", passage_id="p3", text="unrelated"),
    ]
    images = [ImageRecord(image_id="i1", doc_id="D", path="i1.png", passage_ids=["p1"], ok=True)]
    captions = [CaptionRecord(image_id="i1", text="a caption", model="m", prompt_version="v")]
    return passages, captions, images


def test_identical_text_in_two_records_keeps_separate_chunks():
    passages, captions, images = _corpus()
    texts = normalize_texts(passages, captions, images)
    chunks = chunk_texts(texts, WordTokenizer(), chunk_size=4, overlap=1)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.record_id for c in chunks if c.text.startswith("alpha")] == ["p1", "p2"]
    assert ids == [c.chunk_id for c in chunk_texts(texts, WordTokenizer(), chunk_size=4, overlap=1)]


def test_image_links_follow_explicit_associations_only():
    passages, captions, images = _corpus()
    texts = normalize_texts(passages, captions, images)
    chunks = chunk_texts(texts, WordTokenizer(), chunk_size=4, overlap=1)
    (link,) = link_images(images, texts, chunks)
    assert link.record_ids == ["p1", "caption:i1:0"]  # p3 shares the document but is not linked
    assert {c.record_id for c in chunks if c.chunk_id in link.chunk_ids} == {"p1", "caption:i1:0"}
    unknown = images[0].model_copy(update={"passage_ids": ["missing"]})
    with pytest.raises(ValueError):
        link_images([unknown], texts, chunks)


def test_min_max_edge_cases():
    assert minmax_normalize(np.array([])).size == 0
    assert minmax_normalize(np.array([0.3, 0.3])).tolist() == [1.0, 1.0]
    assert minmax_normalize(np.array([-1.0, -1.0])).tolist() == [0.0, 0.0]
    assert minmax_normalize(np.array([1.0, 3.0, 2.0])).tolist() == [0.0, 1.0, 0.5]
    with pytest.raises(ValueError):
        minmax_normalize(np.array([1.0, np.nan]))


def test_fusion_with_missing_channels():
    t, i, c = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(fuse_seeds(t, i, c, beta=0.25), [0.75, 0.125, 0.125])
    assert fuse_seeds(t, None, None, beta=0.9) is t
    np.testing.assert_allclose(fuse_seeds(None, i, None, beta=0.1), i)
    assert fuse_seeds(None, None, None, beta=0.5) is None
    assert normalize_sum(np.zeros(3), "seeds") is None
    np.testing.assert_allclose(normalize_sum(np.array([1.0, 3.0]), "seeds"), [0.25, 0.75])
    with pytest.raises(ValueError):
        normalize_sum(np.array([1.0, -1.0]), "seeds")
    check_reset(np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        check_reset(np.array([0.5, 0.2]))


def test_crop_follows_mg2_rules():
    image = np.full((40, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[5:15, 10:20] = 1
    crop, box, status = crop_masked(image, mask, [-5.7, 3.2, 20.9, 25.0], min_size=10)
    assert (status, box) == ("saved", [0, 3, 20, 25])
    assert crop.size == (20, 22)
    pixels = np.asarray(crop)
    assert pixels[2:12, 10:20].min() == 200  # the mask's pixels survive
    assert pixels[0, 0].tolist() == [0, 0, 0]  # everything else is black
    assert crop_masked(image, mask, [0, 0, 9.9, 30], min_size=10)[2] == "too_small"
    assert crop_masked(image, mask, [45, 0, 80, 30], min_size=10)[2] == "too_small"  # clipped first


def test_prompts_are_unique_and_in_chunk_order():
    alice = EntityRecord(entity_id="e1", name="alice", chunk_ids=["c1", "c2"])
    bob = EntityRecord(entity_id="e2", name="bob", chunk_ids=["c2"])
    by_chunk = {"c1": [alice], "c2": [alice, bob]}
    assert [e.name for e in image_prompts(["c1", "c2", "c3"], by_chunk)] == ["alice", "bob"]
