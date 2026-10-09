"""Records that retrieval reads and writes.

The manifest records mirror the files written by Misha's MMQA builder
(``build_mmqa_manifest.py``) field for field, so they are read without any
conversion step. Records are frozen and reject unknown fields, so a producer
that drifts from this contract fails on read instead of corrupting a later
stage. Files are JSON Lines (see ``io.py``).

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
"""A dataset ID. MMQA doc IDs are Wikipedia titles, so spaces are allowed."""


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Manifests (plan task 1) -------------------------------------------------


class PassageRecord(Record):
    """manifests/text_manifest.jsonl: one Wikipedia paragraph."""

    doc_id: Id
    """Wikipedia page title."""
    passage_id: Id
    text: str
    source_url: str | None = None


class ImageRecord(Record):
    """manifests/image_manifest.jsonl: one corpus image."""

    image_id: Id
    doc_id: Id
    path: str
    """Relative to the corpus images directory (``Artifacts.images_dir``)."""
    source_url: str | None = None
    passage_ids: list[Id]
    """Passages of the same document; edges to them get weight 1."""
    width: int | None = None
    height: int | None = None
    ok: bool
    """False when the builder could not open the file."""


class QuestionRecord(Record):
    """manifests/questions.jsonl. Gold and distractor fields are labels:
    retrieval reads only ``qid`` and ``question``."""

    qid: Id
    question: str
    answers: list[str]
    gold_doc_ids: list[Id]
    gold_image_ids: list[Id]
    gold_passage_ids: list[Id]
    distractor_image_ids: list[Id]
    distractor_passage_ids: list[Id]
    q_type: str | None = None
    modalities: list[str]
    rephrasing_confidence: float | None = None
    split: Literal["dev", "test"]
    group_id: int
    """Questions sharing a gold document share a group (kept in one split)."""


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
    """Passages whose text goes to the reader, in source order."""


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
