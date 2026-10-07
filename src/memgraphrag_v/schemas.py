"""Record schemas shared by every pipeline stage.

All records are frozen and reject unknown fields, so a producer that drifts
from the contract fails at write time instead of corrupting a later stage.
Files are JSON Lines, one record per line (see ``io.py``).
"""

from __future__ import annotations

import math
import re
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from . import ids

MAX_DOCS = 5
"""Documents returned per question (method.md §5.3)."""

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def check_rel_path(value: str) -> str:
    """Paths are POSIX and relative to the artifacts root, so manifests written
    on a laptop resolve unchanged on Snellius."""
    if not value or "\\" in value:
        raise ValueError(f"path must be non-empty POSIX (no backslashes): {value!r}")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"path must be relative to the artifacts root: {value!r}")
    if ".." in PurePosixPath(value).parts:
        raise ValueError(f"path must not escape the artifacts root: {value!r}")
    return value


def _check_sha256(value: str) -> str:
    if not _SHA256.match(value):
        raise ValueError(f"not a lowercase sha256 hex digest: {value!r}")
    return value


def _check_entity_id(value: str) -> str:
    if not ids.is_hash_id(value, ids.ENTITY_PREFIX):
        raise ValueError(f"not an upstream entity ID: {value!r}")
    return value


def _check_chunk_id(value: str) -> str:
    if not ids.is_hash_id(value, ids.CHUNK_PREFIX):
        raise ValueError(f"not an upstream chunk ID: {value!r}")
    return value


def _check_crop_id(value: str) -> str:
    if not ids.is_hash_id(value, ids.CROP_PREFIX):
        raise ValueError(f"not a crop ID: {value!r}")
    return value


NativeId = Annotated[str, AfterValidator(ids.check_native_id)]
EntityId = Annotated[str, AfterValidator(_check_entity_id)]
ChunkId = Annotated[str, AfterValidator(_check_chunk_id)]
CropId = Annotated[str, AfterValidator(_check_crop_id)]
RelPath = Annotated[str, AfterValidator(check_rel_path)]
Sha256 = Annotated[str, AfterValidator(_check_sha256)]
NonEmptyStr = Annotated[str, Field(min_length=1)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Manifests (method.md §2), built by [09] -------------------------------


class PassageRecord(Record):
    """manifests/text.jsonl"""

    doc_id: NativeId
    passage_id: NativeId
    text: NonEmptyStr
    source_location: str | None = None
    """Where the passage comes from, e.g. URL plus section; free text."""


class ImageRecord(Record):
    """manifests/images.jsonl"""

    image_id: NativeId
    doc_id: NativeId
    passage_ids: list[NativeId]
    """Passages that contain or reference the image, in source order."""
    path: RelPath
    page_or_figure: str | None = None
    caption: str | None = None
    """Original caption from the source, never a generated description."""

    @model_validator(mode="after")
    def _unique_passages(self) -> ImageRecord:
        # Image–passage edges are deduplicated (method.md §3.2).
        if len(set(self.passage_ids)) != len(self.passage_ids):
            raise ValueError("passage_ids must not repeat")
        return self


class QuestionRecord(Record):
    """manifests/questions.jsonl"""

    question_id: NativeId
    text: NonEmptyStr
    answers: list[str] = Field(min_length=1)
    gold_doc_ids: list[NativeId] = Field(min_length=1)
    split: Literal["dev", "test"]
    gold_image_ids: list[NativeId] | None = None
    group_id: str | None = None
    """Dependency group for splitting and bootstrap resampling ([10])."""


# --- Text index exports (method.md §3.1), written by [11] --------------------


class PassageMapRecord(Record):
    """index/text/exports/passage_map.jsonl: benchmark passage ↔ upstream chunk."""

    passage_id: NativeId
    doc_id: NativeId
    chunk_id: ChunkId


class EntityRecord(Record):
    """index/text/exports/entities.jsonl: final MemGraphRAG entities."""

    entity_id: EntityId
    name: NonEmptyStr
    type: str | None = None
    passage_ids: list[NativeId]

    @model_validator(mode="after")
    def _id_matches_name(self) -> EntityRecord:
        if self.entity_id != ids.upstream_entity_id(self.name):
            raise ValueError(f"entity_id does not hash from name {self.name!r}")
        return self


# --- SAM3 grounding (method.md §3.3–3.4), written by [13]/[18] --------------


class GroundingRecord(Record):
    """grounding/sam3/attempts.jsonl: one line per (image, entity) prompt,
    including prompts with zero kept detections."""

    image_id: NativeId
    entity_id: EntityId
    prompt: NonEmptyStr
    threshold: float = Field(ge=0.0, le=1.0)
    n_kept: int = Field(ge=0)
    """Detections with confidence >= threshold. The edge exists iff n_kept > 0."""
    n_crops: int = Field(ge=0)
    """Crops actually saved; can be fewer than n_kept because MG²'s crop code
    drops boxes under 10 px."""
    max_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """Edge weight of the image–entity pair (max over kept detections, never a
    sum); None when n_kept == 0."""

    @model_validator(mode="after")
    def _consistent(self) -> GroundingRecord:
        if (self.n_kept == 0) != (self.max_confidence is None):
            raise ValueError("max_confidence must be set iff n_kept > 0")
        if self.max_confidence is not None and self.max_confidence < self.threshold:
            raise ValueError("max_confidence is below the threshold")
        if self.n_crops > self.n_kept:
            raise ValueError("n_crops cannot exceed n_kept")
        return self


class CropRecord(Record):
    """manifests/crops.jsonl: one saved crop of a kept detection. Not a graph
    vertex. Its crop_id rank orders the pair's saved crops by confidence."""

    crop_id: CropId
    image_id: NativeId
    entity_id: EntityId
    prompt: NonEmptyStr
    bbox: tuple[float, float, float, float]
    """Pixel coordinates (x1, y1, x2, y2) in the parent image."""
    confidence: float = Field(ge=0.0, le=1.0)
    path: RelPath
    mask_path: RelPath | None = None

    @model_validator(mode="after")
    def _valid_bbox(self) -> CropRecord:
        x1, y1, x2, y2 = self.bbox
        if not all(math.isfinite(v) for v in self.bbox):
            raise ValueError("bbox must be finite")
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox must be non-negative x1<x2, y1<y2: {self.bbox}")
        return self


# --- Visual embeddings (method.md §4), written by [12] -----------------------


class EmbeddingKey(Record):
    """Cache key: a vector is reusable only if every field matches."""

    encoder: NonEmptyStr
    model_revision: NonEmptyStr
    processor_revision: NonEmptyStr
    kind: Literal["image", "crop", "query"]
    input_sha256: Sha256


class EmbeddingRow(Record):
    """embeddings/<encoder>/<kind>/rows.jsonl: row i describes vector i."""

    row: int = Field(ge=0)
    item_id: NonEmptyStr
    """image_id, crop_id or question_id, depending on kind."""
    key: EmbeddingKey
    status: Literal["ok", "failed", "truncated"]
    note: str | None = None


# --- Generated descriptions (method.md §6, baseline C), written by [14] ------


class DescriptionRecord(Record):
    """descriptions/descriptions.jsonl: one question-independent description
    per image, used by C and G-caption only."""

    image_id: NativeId
    text: str
    """Empty when status == "failed"."""
    model: NonEmptyStr
    model_revision: NonEmptyStr
    prompt_version: NonEmptyStr
    max_tokens: int = Field(ge=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    status: Literal["ok", "failed", "truncated"]
    """truncated = stopped at max_tokens."""
    note: str | None = None

    @model_validator(mode="after")
    def _text_matches_status(self) -> DescriptionRecord:
        if (self.status == "failed") != (not self.text):
            raise ValueError("text must be empty iff status is failed")
        return self


# --- Retrieval output (method.md §5), written by [16]/[24] -------------------

SystemId = Literal["T", "C", "D", "G", "G-caption", "G-image", "G-ground", "G0"]
Route = Literal["graph", "direct", "fallback"]


class SeedSummary(Record):
    text_available: bool
    image_available: bool
    crop_available: bool
    beta: float | None = Field(default=None, ge=0.0, le=1.0)
    """Fusion weight actually applied; None unless both channels were used."""


class EvidenceItem(Record):
    rank: int = Field(ge=1)
    doc_id: NativeId
    score: float
    image_id: NativeId | None = None
    """Selected image (highest-scoring in the document) for image systems."""
    passage_id: NativeId | None = None
    """Selected passage for passage-based systems and fallback."""
    attached_passage_ids: list[NativeId] = Field(default_factory=list)
    """Associated text given to the reader, in source order."""

    @model_validator(mode="after")
    def _has_evidence(self) -> EvidenceItem:
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if self.image_id is None and self.passage_id is None:
            raise ValueError("evidence needs an image_id or a passage_id")
        return self


class RetrievalResult(Record):
    """runs/<run_id>/retrieval.jsonl: one line per question and system."""

    question_id: NativeId
    system: SystemId
    route: Route
    seeds: SeedSummary | None = None
    evidence: list[EvidenceItem] = Field(max_length=MAX_DOCS)
    config_hash: Sha256
    latency_ms: float | None = Field(default=None, ge=0.0)
    """Retrieval time on the reporting hardware, excluding generation (§7).
    None for G-caption, which reuses G's selection without retrieving."""

    @model_validator(mode="after")
    def _ranked_unique_docs(self) -> RetrievalResult:
        ranks = [e.rank for e in self.evidence]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"ranks must be 1..n in order, got {ranks}")
        docs = [e.doc_id for e in self.evidence]
        if len(set(docs)) != len(docs):
            raise ValueError("doc_ids must be unique (one evidence item per document)")
        scores = [e.score for e in self.evidence]
        if any(a < b for a, b in zip(scores, scores[1:])):
            raise ValueError("scores must be non-increasing with rank")
        return self
