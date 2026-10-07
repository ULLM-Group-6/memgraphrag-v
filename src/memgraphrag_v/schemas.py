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


def _check_bbox(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = value
    if not all(math.isfinite(v) for v in value):
        raise ValueError("bbox must be finite")
    if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox must be non-negative x1<x2, y1<y2: {value}")
    return value


def _check_finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    return value


NativeId = Annotated[str, AfterValidator(ids.check_native_id)]
EntityId = Annotated[str, AfterValidator(_check_entity_id)]
ChunkId = Annotated[str, AfterValidator(_check_chunk_id)]
CropId = Annotated[str, AfterValidator(_check_crop_id)]
RelPath = Annotated[str, AfterValidator(check_rel_path)]
Sha256 = Annotated[str, AfterValidator(_check_sha256)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
BBox = Annotated[tuple[float, float, float, float], AfterValidator(_check_bbox)]
"""Pixel coordinates (x1, y1, x2, y2) in the parent image."""
Score = Annotated[float, AfterValidator(_check_finite)]


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


class CaptionChunkMapRecord(Record):
    """index/caption/exports/chunk_map.jsonl, written by [22]: which passage
    and/or image description each chunk of baseline C's index came from."""

    chunk_id: ChunkId
    doc_id: NativeId
    passage_id: NativeId | None = None
    image_id: NativeId | None = None
    """Set when the chunk contains this image's generated description."""

    @model_validator(mode="after")
    def _has_source(self) -> CaptionChunkMapRecord:
        if self.passage_id is None and self.image_id is None:
            raise ValueError("chunk needs a passage_id, an image_id or both")
        return self


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


class Detection(Record):
    """One kept SAM3 detection (confidence >= threshold), saved whether or not
    a crop was produced for it."""

    rank: int = Field(ge=0)
    """0-based position among the pair's kept detections, by confidence
    descending (ties: SAM3's box order)."""
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    mask_path: RelPath
    crop_id: CropId | None = None
    """None when MG²'s crop code dropped the box (under 10 px)."""


class GroundingRecord(Record):
    """grounding/sam3/attempts.jsonl: one line per (image, entity) prompt,
    including prompts with zero kept detections."""

    image_id: NativeId
    entity_id: EntityId
    prompt: NonEmptyStr
    threshold: float = Field(ge=0.0, le=1.0)
    detections: list[Detection]
    """Every kept detection, in rank order. The edge exists iff this is non-empty."""

    @property
    def n_kept(self) -> int:
        return len(self.detections)

    @property
    def n_crops(self) -> int:
        """Crops actually saved; can be fewer than n_kept because of the 10 px filter."""
        return sum(d.crop_id is not None for d in self.detections)

    @property
    def max_confidence(self) -> float | None:
        """Edge weight of the image–entity pair (max over kept detections,
        never a sum); None when nothing was kept."""
        return self.detections[0].confidence if self.detections else None

    @model_validator(mode="after")
    def _consistent(self) -> GroundingRecord:
        ranks = [d.rank for d in self.detections]
        if ranks != list(range(len(ranks))):
            raise ValueError(f"detection ranks must be 0..n-1 in order, got {ranks}")
        confidences = [d.confidence for d in self.detections]
        if any(a < b for a, b in zip(confidences, confidences[1:])):
            raise ValueError("detections must be ordered by confidence, descending")
        if confidences and confidences[-1] < self.threshold:
            raise ValueError("a kept detection is below the threshold")
        for d in self.detections:
            expected = ids.crop_id(self.image_id, self.entity_id, d.rank)
            if d.crop_id is not None and d.crop_id != expected:
                raise ValueError(f"detection {d.rank}: crop_id does not match its rank")
        return self


class CropRecord(Record):
    """manifests/crops.jsonl: one saved crop of a kept detection. Not a graph
    vertex. The rank in its crop_id is the detection's rank, so it joins to
    exactly one ``Detection``; ranks of saved crops can have gaps."""

    crop_id: CropId
    image_id: NativeId
    entity_id: EntityId
    prompt: NonEmptyStr
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    path: RelPath
    mask_path: RelPath


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
Selection = Literal["image", "passage"]
"""How documents are ranked (§5.3): by their best image node or best passage."""

SYSTEM_SELECTIONS: dict[str, frozenset[str]] = {
    "T": frozenset({"passage"}),
    "C": frozenset({"passage"}),
    "D": frozenset({"image"}),
    "G": frozenset({"image"}),
    "G-caption": frozenset({"image"}),
    "G-image": frozenset({"image"}),
    "G-ground": frozenset({"image"}),
    # Image selection for the ablation table, passage selection for the
    # T-vs-G0 graph-augmentation diagnostic (§6).
    "G0": frozenset({"image", "passage"}),
}

_NO_VISUAL_SEEDS = {"T", "C", "G0"}
_NO_CROP_SEEDS = {"G-image", "G-ground"}


class SeedSummary(Record):
    """Which components put restart mass into the seed vector. A component
    that was computed but not allowed to seed (e.g. visual seeds in G0) is
    False."""

    text_used: bool
    image_used: bool
    crop_used: bool
    beta: float | None = Field(default=None, ge=0.0, le=1.0)
    """Fusion weight actually applied; None unless text and a visual
    component were both used."""

    @model_validator(mode="after")
    def _beta_iff_fused(self) -> SeedSummary:
        fused = self.text_used and (self.image_used or self.crop_used)
        if fused != (self.beta is not None):
            raise ValueError("beta must be set iff text and visual seeds were both used")
        return self

    @property
    def any_used(self) -> bool:
        return self.text_used or self.image_used or self.crop_used


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
    selection: Selection
    route: Route
    seeds: SeedSummary | None = None
    """Required on the graph route (except G-caption, which copies G);
    None for D."""
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

    @model_validator(mode="after")
    def _matches_system(self) -> RetrievalResult:
        system = self.system
        if self.selection not in SYSTEM_SELECTIONS[system]:
            raise ValueError(f"{system} cannot use {self.selection} selection")
        if (self.route == "direct" and system != "D") or (system == "D" and self.route == "graph"):
            raise ValueError(f"{system} cannot take the {self.route} route")

        # Fallback selects passages and may attach an image (§5.2).
        needs = "passage_id" if self.route == "fallback" or self.selection == "passage" else "image_id"
        if any(getattr(e, needs) is None for e in self.evidence):
            raise ValueError(f"{system} ({self.selection}, {self.route}) evidence needs {needs}")

        seeds = self.seeds
        if system == "D":
            if seeds is not None:
                raise ValueError("D does not build graph seeds")
            return self
        if seeds is None:
            if self.route == "graph" and system != "G-caption":
                raise ValueError(f"{system} on the graph route must record its seeds")
            return self
        if system in _NO_VISUAL_SEEDS and (seeds.image_used or seeds.crop_used):
            raise ValueError(f"{system} must not use visual seeds")
        if system in _NO_CROP_SEEDS and seeds.crop_used:
            raise ValueError(f"{system} must not use crop seeds")
        if self.route == "fallback" and seeds.any_used:
            raise ValueError("the fallback runs only when no seed component is available")
        if self.route == "graph" and not seeds.any_used:
            raise ValueError("the graph route needs at least one seed component")
        return self


# --- Error-analysis diagnostics (method.md §7), written by [16]/[24] ---------


class ScoredItem(Record):
    item_id: NativeId
    """image_id or passage_id."""
    score: Score


class CropCandidate(Record):
    crop_id: CropId
    image_id: NativeId
    entity_id: EntityId
    score: Score


class SeedWeight(Record):
    node_id: NonEmptyStr
    """Graph node: an upstream entity/chunk/fact node or an image node."""
    channel: Literal["text", "image", "crop"]
    weight: float = Field(gt=0.0)
    """This channel's share of the node's restart mass after normalisation
    and fusion. A node seeded by two channels appears once per channel."""


def _descending(items: list[ScoredItem] | list[CropCandidate], name: str) -> None:
    scores = [i.score for i in items]
    if any(a < b for a, b in zip(scores, scores[1:])):
        raise ValueError(f"{name} must be sorted by score, descending")


class DiagnosticsRecord(Record):
    """runs/<run_id>/diagnostics.jsonl: one line per question and system, with
    what the error analysis needs to tell encoder, propagation and grounding
    failures apart. Gold labels are not stored here; gold ranks are computed
    at analysis time by joining with questions.jsonl ("> N" when absent)."""

    question_id: NativeId
    system: SystemId
    selection: Selection
    image_candidates: list[ScoredItem] = Field(default_factory=list)
    """Top retrieval.diagnostics_top_n images by SigLIP2 similarity; the first
    top_k_candidates are the ones that seeded the graph."""
    crop_candidates: list[CropCandidate] = Field(default_factory=list)
    """Top crops by SigLIP2 similarity, same cut-off."""
    seeds: list[SeedWeight] = Field(default_factory=list)
    """Every node with non-zero restart mass."""
    ppr_ranking: list[ScoredItem] = Field(default_factory=list)
    """Top image_ids (image selection) or passage_ids (passage selection)
    after PPR. Empty for D."""

    @model_validator(mode="after")
    def _sorted(self) -> DiagnosticsRecord:
        _descending(self.image_candidates, "image_candidates")
        _descending(self.crop_candidates, "crop_candidates")
        _descending(self.ppr_ranking, "ppr_ranking")
        return self
