"""Validated JSON Lines I/O and cross-manifest checks.

Every ``write_jsonl`` also writes ``<file>.meta.json`` with the schema version,
record type, count and sha256, so later stages and the frozen experiment
config can pin exactly which artifact they consumed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

from . import SCHEMA_VERSION
from .schemas import CropRecord, GroundingRecord, ImageRecord, PassageRecord, QuestionRecord

M = TypeVar("M", bound=BaseModel)


def file_sha256(path: str | Path) -> str:
    """sha256 of a file with CRLF normalised to LF, so the same text artifact
    hashes identically on Windows and on Snellius."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def meta_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".meta.json")


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> dict:
    """Write records (all of one type) and their sidecar; returns the sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    record_type = None
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            name = type(record).__name__
            if record_type is None:
                record_type = name
            elif name != record_type:
                raise TypeError(f"mixed record types in {path}: {record_type} and {name}")
            f.write(record.model_dump_json() + "\n")
            count += 1
    meta = {
        "schema_version": SCHEMA_VERSION,
        "record_type": record_type,
        "count": count,
        "sha256": file_sha256(path),
    }
    meta_path(path).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta


def read_jsonl(path: str | Path, model: type[M], *, check_meta: bool = True) -> list[M]:
    """Read and validate every line; with ``check_meta`` the sidecar must match."""
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    if check_meta:
        meta = json.loads(meta_path(path).read_text(encoding="utf-8"))
        if meta["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {meta['schema_version']} != {SCHEMA_VERSION}")
        if meta["count"] and meta["record_type"] != model.__name__:
            raise ValueError(f"{path}: holds {meta['record_type']}, not {model.__name__}")
        if meta["count"] != len(records) or meta["sha256"] != file_sha256(path):
            raise ValueError(f"{path}: content does not match its .meta.json")
    return records


@dataclass
class ManifestReport:
    """Result of ``validate_manifests``. Errors block indexing; warnings
    (e.g. gold evidence missing from the corpus) are reported as coverage."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValueError("manifest validation failed:\n" + "\n".join(self.errors))


def _duplicates(values: Iterable[str]) -> list[str]:
    return sorted(v for v, n in Counter(values).items() if n > 1)


def validate_manifests(
    passages: list[PassageRecord],
    images: list[ImageRecord],
    questions: list[QuestionRecord] | None = None,
    crops: list[CropRecord] | None = None,
    grounding: list[GroundingRecord] | None = None,
    artifacts_root: str | Path | None = None,
) -> ManifestReport:
    """Cross-file checks for [09]: unique IDs, resolvable references and, when
    ``artifacts_root`` is given, that every referenced file exists."""
    report = ManifestReport()
    questions = questions or []
    crops = crops or []

    for label, values in [
        ("passage_id", [p.passage_id for p in passages]),
        ("image_id", [i.image_id for i in images]),
        ("question_id", [q.question_id for q in questions]),
        ("crop_id", [c.crop_id for c in crops]),
    ]:
        for dup in _duplicates(values):
            report.errors.append(f"duplicate {label}: {dup}")

    passage_doc = {p.passage_id: p.doc_id for p in passages}
    image_ids = {i.image_id for i in images}
    doc_ids = set(passage_doc.values()) | {i.doc_id for i in images}

    for image in images:
        for pid in image.passage_ids:
            if pid not in passage_doc:
                report.errors.append(f"image {image.image_id}: unknown passage {pid}")
            elif passage_doc[pid] != image.doc_id:
                report.errors.append(
                    f"image {image.image_id} (doc {image.doc_id}): passage {pid} "
                    f"belongs to doc {passage_doc[pid]}"
                )
        if not image.passage_ids:
            report.warnings.append(f"image {image.image_id}: no associated passages")

    for crop in crops:
        if crop.image_id not in image_ids:
            report.errors.append(f"crop {crop.crop_id}: unknown image {crop.image_id}")

    if grounding is not None:
        _check_grounding(report, grounding, crops, image_ids)

    for q in questions:
        missing = [d for d in q.gold_doc_ids if d not in doc_ids]
        if missing:
            report.warnings.append(f"question {q.question_id}: gold docs not in corpus {missing}")
        missing_img = [i for i in (q.gold_image_ids or []) if i not in image_ids]
        if missing_img:
            report.warnings.append(f"question {q.question_id}: gold images not in corpus {missing_img}")

    if artifacts_root is not None:
        root = Path(artifacts_root)
        files = [(f"image {i.image_id}", i.path) for i in images]
        for c in crops:
            files.append((f"crop {c.crop_id}", c.path))
            if c.mask_path:
                files.append((f"crop {c.crop_id} mask", c.mask_path))
        for label, rel in files:
            if not (root / rel).is_file():
                report.errors.append(f"{label}: missing file {rel}")

    return report


def _check_grounding(
    report: ManifestReport,
    grounding: list[GroundingRecord],
    crops: list[CropRecord],
    image_ids: set[str],
) -> None:
    """Each (image, entity) is prompted once, and its crops agree with the attempt."""
    pair_counts = Counter((g.image_id, g.entity_id) for g in grounding)
    for pair, n in sorted(pair_counts.items()):
        if n > 1:
            report.errors.append(f"grounding: {pair} prompted more than once")
    attempts = {(g.image_id, g.entity_id): g for g in grounding}
    for g in grounding:
        if g.image_id not in image_ids:
            report.errors.append(f"grounding: unknown image {g.image_id}")

    crops_by_pair: dict[tuple[str, str], list[CropRecord]] = {}
    for c in crops:
        crops_by_pair.setdefault((c.image_id, c.entity_id), []).append(c)
    for pair, pair_crops in crops_by_pair.items():
        g = attempts.get(pair)
        if g is None or g.n_kept == 0:
            report.errors.append(f"crops of {pair} have no grounding attempt with kept detections")
            continue
        if len(pair_crops) != g.n_crops:
            report.errors.append(f"{pair}: {len(pair_crops)} crops but n_crops={g.n_crops}")
        for c in pair_crops:
            if not g.threshold <= c.confidence <= g.max_confidence + 1e-9:
                report.errors.append(
                    f"crop {c.crop_id}: confidence {c.confidence} outside "
                    f"[threshold {g.threshold}, max {g.max_confidence}]"
                )
    for pair, g in attempts.items():
        if g.n_crops and pair not in crops_by_pair:
            report.errors.append(f"{pair}: n_crops={g.n_crops} but no crops in the manifest")
