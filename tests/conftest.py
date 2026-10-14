import hashlib

import numpy as np
import pytest

from memgraphrag_v.schemas import ImageRecord, PassageRecord, QuestionRecord

# Rows in the shape webqa.py writes (from the WebQA builder on the dataset branch).
PASSAGE_ROW = {
    "doc_id": "Beale Street",
    "passage_id": "cap_30193116",
    "text": "Memphis, TN, Beale St. April 4, 2008",
    "source_url": "https://commons.wikimedia.org/wiki/File:Memphis_Beale_St.jpg",
}
IMAGE_ROW = {
    "image_id": "30193116",
    "doc_id": "Beale Street",
    "path": "30193116.jpg",
    "source_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Memphis_Beale_St.jpg/960px-Memphis_Beale_St.jpg",
    "commons_url": "https://commons.wikimedia.org/wiki/File:Memphis_Beale_St.jpg",
    "passage_ids": ["cap_30193116"],
    "width": 960,
    "height": 640,
    "ok": True,
}
QUESTION_ROW = {
    "qid": "d5bc0d5a0dba11ecb1e81171463288e9",
    "question": "Does Beale St have fewer than four light-up neon signs?",
    "answers": ["No, Beale St has more than four light-up neon signs."],
    "gold_doc_ids": ["Beale Street"],
    "gold_image_ids": ["30193116"],
    "gold_passage_ids": [],
    "distractor_image_ids": ["30242216"],
    "distractor_passage_ids": ["txt_d5bc0d5a0dba11ecb1e81171463288e9_15"],
    "q_type": "YesNo",
    "topic": "streets",
    "modalities": ["image"],
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
