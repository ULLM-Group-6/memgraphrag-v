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
from typing import TYPE_CHECKING, Iterable, TypeVar

from pydantic import BaseModel

from . import SCHEMA_VERSION
from .schemas import (
    CropRecord,
    DescriptionRecord,
    GroundingRecord,
    ImageRecord,
    PassageRecord,
    QuestionRecord,
)

if TYPE_CHECKING:
    from .config import ExperimentConfig

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
    cross_doc = _check_duplicate_text(report, passages)

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
        shared = [pid for pid in image.passage_ids if pid in cross_doc]
        if shared:
            report.warnings.append(
                f"image {image.image_id}: passages {shared} share a chunk node with other "
                f"documents, so its image–passage edges reach those documents too"
            )

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
            files.append((f"crop {c.crop_id} mask", c.mask_path))
        for g in grounding or []:
            for d in g.detections:
                if d.crop_id is None:  # masks of saved crops are listed above
                    files.append((f"detection {g.image_id}/{g.entity_id}#{d.rank} mask", d.mask_path))
        for label, rel in files:
            if not (root / rel).is_file():
                report.errors.append(f"{label}: missing file {rel}")

    return report


def _check_duplicate_text(report: ManifestReport, passages: list[PassageRecord]) -> set[str]:
    """Upstream chunk IDs hash the passage text, so identical passages become one
    graph node whose score is credited to each of them. Returns the passage IDs
    whose shared node spans more than one document."""
    by_text: dict[str, list[PassageRecord]] = {}
    for p in passages:
        by_text.setdefault(p.text, []).append(p)
    cross_doc: set[str] = set()
    for group in by_text.values():
        if len(group) < 2:
            continue
        pids = [p.passage_id for p in group]
        docs = sorted({p.doc_id for p in group})
        where = f" across docs {docs}" if len(docs) > 1 else ""
        report.warnings.append(
            f"passages {pids} have identical text{where}: they share one chunk node, "
            f"whose score is credited to each passage"
        )
        if len(docs) > 1:
            cross_doc.update(pids)
    return cross_doc


def _check_grounding(
    report: ManifestReport,
    grounding: list[GroundingRecord],
    crops: list[CropRecord],
    image_ids: set[str],
) -> None:
    """Each (image, entity) is prompted once, and every saved crop matches
    exactly one kept detection."""
    pair_counts = Counter((g.image_id, g.entity_id) for g in grounding)
    for pair, n in sorted(pair_counts.items()):
        if n > 1:
            report.errors.append(f"grounding: {pair} prompted more than once")
    for g in grounding:
        if g.image_id not in image_ids:
            report.errors.append(f"grounding: unknown image {g.image_id}")

    detections = {
        d.crop_id: (g, d) for g in grounding for d in g.detections if d.crop_id is not None
    }
    crop_ids = {c.crop_id for c in crops}
    for c in crops:
        found = detections.get(c.crop_id)
        if found is None:
            report.errors.append(f"crop {c.crop_id}: no matching detection in the grounding attempts")
            continue
        g, d = found
        if (c.image_id, c.entity_id) != (g.image_id, g.entity_id):
            report.errors.append(f"crop {c.crop_id}: image/entity differ from its detection")
        if c.confidence != d.confidence or tuple(c.bbox) != tuple(d.bbox):
            report.errors.append(f"crop {c.crop_id}: confidence or bbox differ from its detection")
    for crop_id, (g, d) in detections.items():
        if crop_id not in crop_ids:
            report.errors.append(
                f"({g.image_id}, {g.entity_id}) detection {d.rank}: crop {crop_id} "
                f"is not in the crop manifest"
            )


def check_config_consistency(
    config: ExperimentConfig,
    grounding: list[GroundingRecord] | None = None,
    descriptions: list[DescriptionRecord] | None = None,
) -> ManifestReport:
    """Check that cached grounding and descriptions were made with the settings
    the experiment config pins (SAM3 threshold; reader VLM and caption prompt)."""
    report = ManifestReport()
    thresholds = sorted({g.threshold for g in grounding or []})
    if any(t != config.sam3.threshold for t in thresholds):
        report.errors.append(
            f"grounding thresholds {thresholds} differ from sam3.threshold {config.sam3.threshold}"
        )
    expected = {
        "model": config.reader.model,
        "model_revision": config.reader.revision,
        "max_tokens": config.caption.description_max_tokens,
        "prompt_version": config.caption.prompt_version,
    }
    for name, want in expected.items():
        found = sorted({str(getattr(d, name)) for d in descriptions or []})
        if found and found != [str(want)]:
            report.errors.append(f"descriptions: {name} {found} differs from the config ({want})")
    return report
