"""Records that retrieval reads and writes.

The manifests are built from our WebQA subset (``subsetcheck4.py`` on the
``dataset`` branch) by ``webqa.py``: passages are copied as they are, images
gain the local file (unpacked from WebQA's image TSV) and whether it opens,
questions are normalised, and the generated captions get the model and prompt
that made them.
Records are frozen and reject unknown fields, so a producer that drifts from
this contract fails on read instead of corrupting a later stage. Files are
JSON Lines (see ``io.py``).

The index records (normalized texts, chunks, entity export, image links, SAM3
detections, query diagnostics) are written by ``pipeline.py`` under
``Artifacts.index_dir``.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_DOCS = 5
"""Documents returned per question, each with at most one whole image."""

VERTEX_LAYERS = ("entity", "type", "passage", "image")

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
    """The original file on Wikimedia. Our copy comes from WebQA's TSV."""
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
    description, one per image (``caption_manifest.jsonl``). Not the
    same as the image's WebQA caption, which is a passage. Used only by the
    caption systems (C, G-caption), never by the native-image method."""

    image_id: Id
    text: str
    model: str
    prompt_version: str


# --- Index records (combined pipeline) ----------------------------------------


class TextRecord(Record):
    """index/texts.jsonl: one passage or caption, in source order (passages in
    manifest order, then captions in manifest order)."""

    record_id: Id
    """The passage_id, or ``caption:<image_id>:<n>`` for the n-th caption of an image."""
    kind: Literal["passage", "caption"]
    doc_id: Id
    text: str
    image_id: Id | None = None
    """Captions only: the parent image."""


class ChunkRecord(Record):
    """index/chunks.jsonl: one chunk of one text record. The chunk ID depends
    on the record ID, position and text, so identical text in two records
    never collapses into one vertex."""

    chunk_id: Id
    record_id: Id
    doc_id: Id
    position: int
    """Chunk number within its text record."""
    text: str


class EntityRecord(Record):
    """index/entities.jsonl: an entity of the final memory and the chunks that
    support it, in chunk source order."""

    entity_id: Id
    name: str
    chunk_ids: list[Id]


class IndexedImage(Record):
    """index/images.jsonl: an image, its explicitly associated text and its
    graph vertex."""

    image_id: Id
    doc_id: Id
    path: str
    record_ids: list[Id]
    """Associated passages and captions, in source order."""
    chunk_ids: list[Id]
    """Their chunks, in source order."""
    vertex: str | None = None
    """Graph vertex name; None when the image is not usable."""
    usable: bool
    error: str | None = None


class Detection(Record):
    """A retained SAM3 detection (confidence at or above the threshold)."""

    detection_id: Id
    image_id: Id
    entity_id: Id
    prompt: str
    prompt_index: int
    index: int
    """Position in SAM3's output for the image."""
    score: float
    """Full-precision confidence. The image-entity edge weight is the max SAM3 confidence score
    per (image, entity) pair."""
    box: list[float]
    """x1, y1, x2, y2 in the original image's pixel coordinates."""
    mask_index: int
    """Row in the image's mask file."""
    crop_id: Id | None = None
    crop_path: str | None = None
    """Relative to the artifacts root."""
    crop_hash: str | None = None
    """SHA-256 hash of the crop file, for integrity verification."""
    crop_box: list[int] | None = None
    """The clipped integer box the crop was cut from."""
    crop_status: Literal["saved", "too_small", "failed"]
    error: str | None = None


class ImageGrounding(Record):
    """index/grounding/records/<md5(image_id)>.json: everything SAM3 did for
    one image."""

    image_id: Id
    doc_id: Id
    status: Literal["grounded", "no_prompts", "failed"]
    """no_prompts: no entity is linked to the image's text; failed: see error.
    Either way the image keeps its whole-image vertex and passage edges."""
    fingerprint: str
    """Hash of the image bytes, prompts and grounding settings, to validate reuse."""
    entity_ids: list[Id] = Field(default_factory=list)
    """Prompt index -> existing entity ID."""
    prompts: list[str] = Field(default_factory=list)
    n_raw: int = 0
    n_dropped: int = 0
    """Raw detections below the threshold or invalid."""
    mask_file: str | None = None
    """.npz with ``masks`` (n_retained, H, W), relative to the artifacts root."""
    mask_hash: str | None = None
    detections: list[Detection] = Field(default_factory=list) # retained
    error: str | None = None


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


class QueryDiagnostics(Record):
    """runs/<run_id>/diagnostics.jsonl: why one question got its evidence.
    Compact: seed and score maps keep only the strongest entries."""

    question_id: Id
    route: Literal["graph", "fallback"]
    fallback_reason: str | None = None
    beta: float
    text_available: bool
    image_available: bool
    crop_available: bool
    text_reason: str | None = None
    """Why the text channel is unavailable, when it is."""
    query_truncated: bool
    """The question was cut to the visual encoder's token limit."""
    facts: list[str] = Field(default_factory=list)
    """Facts that passed selection."""
    n_facts_scored: int = 0
    text_seeds: dict[str, float] = Field(default_factory=dict) # vertex name: score
    """Strongest raw text seeds (before normalization), by vertex name."""
    image_seeds: dict[str, float] = Field(default_factory=dict) # vertex name: score
    crop_entity_seeds: dict[str, float] = Field(default_factory=dict) # vertex name: score
    image_ppr: dict[str, float] = Field(default_factory=dict) # vertex name: score
    """Highest PPR scores of image vertices, by image ID."""
    selected_doc_ids: list[Id] = Field(default_factory=list)
    selected_image_ids: list[Id | None] = Field(default_factory=list)
