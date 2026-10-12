"""Records that retrieval reads and writes.

The manifests are built from Misha's WebQA subset (``subsetcheck4.py`` on the
``dataset`` branch) by ``webqa.py``: passages are copied as they are, images
gain the local file and whether it opens, and questions are normalised.
Records are frozen and reject unknown fields, so a producer that drifts from
this contract fails on read instead of corrupting a later stage. Files are
JSON Lines (see ``io.py``).

Records that only graph construction needs (entity export, passage-chunk map,
SAM3 detections) are added with those tasks.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_DOCS = 5
"""Documents returned per question, each with at most one whole image."""

Id = Annotated[str, Field(min_length=1)]
"""A dataset ID. WebQA doc IDs are page or file titles, so spaces are allowed."""


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Manifests (plan task 1) -------------------------------------------------


class PassageRecord(Record):
    """manifests/text_manifest.jsonl: one WebQA source text, either an image
    caption (passage_id ``cap_<image_id>``) or a text snippet
    (``txt_<snippet_id>``)."""

    doc_id: Id
    """Title of the source page; shared by every source from that page."""
    passage_id: Id
    text: str
    source_url: str | None = None


class ImageRecord(Record):
    """manifests/image_manifest.jsonl: one corpus image."""

    image_id: Id
    doc_id: Id
    """Title of the image's page. Several images can share one (e.g. a
    Wikipedia article with many pictures)."""
    path: str
    """Relative to the corpus images directory (``Artifacts.images_dir``)."""
    source_url: str | None = None
    """Where the file was downloaded from."""
    commons_url: str | None = None
    """The page the image belongs to."""
    passage_ids: list[Id]
    """The image's caption; edges to these get weight 1."""
    width: int | None = None
    height: int | None = None
    ok: bool
    """False when the file is missing or cannot be opened."""


class QuestionRecord(Record):
    """manifests/questions.jsonl. Gold and distractor fields are labels:
    retrieval reads only ``qid`` and ``question``. Gold documents are the
    titles of the gold images and snippets."""

    qid: Id
    question: str
    answers: list[str]
    gold_doc_ids: list[Id]
    gold_image_ids: list[Id]
    gold_passage_ids: list[Id]
    distractor_image_ids: list[Id]
    distractor_passage_ids: list[Id]
    q_type: str | None = None
    """WebQA's Qcate: YesNo, Others, number, choose, color, shape or text."""
    topic: str | None = None
    modalities: list[Literal["image", "text"]]
    """Modalities of the gold evidence."""
    split: Literal["dev", "test"]
    """dev: the graph-building questions; test: the held-out questions."""
    group_id: int
    """Questions linked by a shared gold image or snippet, directly or
    through other questions, share a group; the bootstrap resamples groups.
    Groups can span splits, because test evidence is drawn from the corpus
    the dev questions built."""


class CaptionRecord(Record):
    """manifests/captions.jsonl: a generated, question-independent image
    description. Used only by the caption systems (C, G-caption), never by
    the native-image method."""

    image_id: Id
    text: str
    model: str
    prompt_version: str


# --- Retrieval output (plan tasks 3 and 5) -----------------------------------


class Evidence(Record):
    """One retrieved document and what the reader gets from it."""

    doc_id: Id
    score: float
    image_id: Id | None = None
    """The document's highest-scoring image; None for text-only systems."""
    passage_ids: list[Id] = Field(default_factory=list)
    """The evidence's associated passages, in source order, untrimmed. The
    reader input keeps only their first ``retrieval.text_budget_tokens``
    tokens, cut in one place for every system."""


class RetrievalResult(Record):
    """runs/<run_id>/retrieval.jsonl: one line per question and system."""

    question_id: Id
    system: str
    """T, C, D, G, G-caption or an ablation (G-image, G-ground, G0)."""
    route: Literal["graph", "direct", "fallback"]
    evidence: list[Evidence] = Field(max_length=MAX_DOCS)
    """Best first: unique documents, scores non-increasing."""

    @model_validator(mode="after")
    def _ranked_unique_docs(self) -> RetrievalResult:
        docs = [e.doc_id for e in self.evidence]
        if len(set(docs)) != len(docs):
            raise ValueError("evidence must not repeat a document")
        scores = [e.score for e in self.evidence]
        if not all(math.isfinite(s) for s in scores):
            raise ValueError("scores must be finite")
        if any(a < b for a, b in zip(scores, scores[1:])):
            raise ValueError("evidence must be sorted by score, best first")
        return self
