import hashlib

import numpy as np
import pytest

from memgraphrag_v.schemas import ImageRecord, PassageRecord, QuestionRecord

# Rows in the exact shape build_mmqa_manifest.py writes.
PASSAGE_ROW = {
    "doc_id": "Barack Obama",
    "passage_id": "0a1b2c3d4e5f60718293a4b5c6d7e8f9",
    "text": "Barack Obama is an American politician.",
    "source_url": "https://en.wikipedia.org/wiki/Barack_Obama",
}
IMAGE_ROW = {
    "image_id": "f0e1d2c3b4a5968778695a4b3c2d1e0f",
    "doc_id": "Barack Obama",
    "path": "obama.jpg",
    "source_url": None,
    "passage_ids": ["0a1b2c3d4e5f60718293a4b5c6d7e8f9"],
    "width": 300,
    "height": 400,
    "ok": True,
}
QUESTION_ROW = {
    "qid": "abc123",
    "question": "What colour is the tie Barack Obama wears in his portrait?",
    "answers": ["blue"],
    "gold_doc_ids": ["Barack Obama"],
    "gold_image_ids": ["f0e1d2c3b4a5968778695a4b3c2d1e0f"],
    "gold_passage_ids": [],
    "distractor_image_ids": [],
    "distractor_passage_ids": ["0a1b2c3d4e5f60718293a4b5c6d7e8f9"],
    "q_type": "ImageQ",
    "modalities": ["image"],
    "rephrasing_confidence": None,
    "split": "dev",
    "group_id": 0,
}


@pytest.fixture
def passage():
    return PassageRecord(**PASSAGE_ROW)


@pytest.fixture
def image():
    return ImageRecord(**IMAGE_ROW)


@pytest.fixture
def question():
    return QuestionRecord(**QUESTION_ROW)


class FakeEncoder:
    """Deterministic stand-in for EvaClip: vectors depend only on the input."""

    dim = 8
    max_text_tokens = 5

    def __init__(self):
        self.images_seen = []

    def encode_images(self, images):
        self.images_seen += images
        return np.stack([self._vector(image.tobytes()) for image in images])

    def encode_texts(self, texts):
        truncated = [len(t.split()) > self.max_text_tokens for t in texts]
        return np.stack([self._vector(t.encode()) for t in texts]), truncated

    def _vector(self, data: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(data).digest()[:4], "little")
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)


@pytest.fixture
def encoder():
    return FakeEncoder()
