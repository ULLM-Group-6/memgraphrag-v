"""SAM3 grounding of existing entities: prompts, detections, masks and crops.

For each image, the entities of the final text memory that are supported by
the image's explicitly associated chunks are the prompts (their names, one
prompt per entity). SAM3 only locates them: no entity is created, and the
prompt index maps back to an existing entity ID. Detections at or above the
confidence threshold are kept with their original-coordinate boxes, masks and
full-precision scores. Each also gets a crop, cut the way MG2's
``SAM3Grounding._process_single_crop`` does (commit ``MG2_COMMIT``): clip the
integer box to the image, skip it when narrower or shorter than the minimum
size, keep the mask's pixels on a black background, save a PNG.

Results are written per image (``<grounding>/records/<md5(image_id)>.json``,
last, after its masks and crops), so an interrupted run resumes where it
stopped. A record is reused only if its fingerprint (image bytes, prompts,
settings, model) still matches. An image whose grounding fails, or finds
nothing, keeps its whole-image vertex and passage edges: failures are only
recorded here.

``MG2Sam3`` wraps MG2's model the way ``component-verification/smoke_sam3.py``
does. torch and SAM3 are imported there only, never at module import.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import logging
import os
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image

from .config import Sam3Config
from .paths import SAM3_CHECKPOINT_ENV, Artifacts
from .schemas import Detection, EntityRecord, ImageGrounding, ImageRecord

logger = logging.getLogger(__name__)

MG2_COMMIT = "91f0eed6fc5f4383f0bae746d12850f05e381e77"

@dataclass(frozen=True)
class RawDetections:
    """One image's SAM3 output as numpy arrays in original-image coordinates."""

    masks: np.ndarray
    """(n, H, W) or (n, 1, H, W)."""
    boxes: np.ndarray
    """(n, 4): x1, y1, x2, y2."""
    scores: np.ndarray
    """(n,)."""
    labels: np.ndarray
    """(n,): index into the prompt list."""


class Grounder(Protocol):
    def predict(self, image: Image.Image, prompts: list[str]) -> RawDetections: ...


def release_gpu() -> None:
    """Free cached GPU memory between stages. Drop references to models first."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Grounding ---------------------------------------------------------------


def image_prompts(chunk_ids: Sequence[str], entities_by_chunk: Mapping[str, Sequence[EntityRecord]]) -> list[EntityRecord]:
    """Entities supported by any of the chunks, once each, in chunk order."""
    prompts: dict[str, EntityRecord] = {}
    for chunk_id in chunk_ids:
        for entity in entities_by_chunk.get(chunk_id, ()):
            prompts.setdefault(entity.entity_id, entity) # the prompt is only the entity name
    return list(prompts.values())

def grounding_artifacts_valid(
    record: ImageGrounding,
    artifacts: Artifacts,
) -> bool:
    def file_matches(path: str | None, expected_hash: str | None) -> bool:
        if path is None or expected_hash is None:
            return False

        full_path = artifacts.root / path
        try:
            current_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
        except OSError:
            return False  # Missing, unreadable, or not a regular file.

        return current_hash == expected_hash

    # Retained detections must have a saved mask.
    if record.detections:
        if not file_matches(record.mask_file, record.mask_hash):
            return False
    elif record.mask_file is not None or record.mask_hash is not None:
        return False  # No detections should mean no mask or mask hash.

    for detection in record.detections:
        if detection.crop_status == "saved":
            if detection.crop_id is None:
                return False
            if not file_matches(detection.crop_path, detection.crop_hash):
                return False
        else:
            # Too-small or failed crops should not claim saved artifacts.
            if (
                detection.crop_id is not None
                or detection.crop_path is not None
                or detection.crop_hash is not None
            ):
                return False

    return True

def ground_images(
    images: Sequence[ImageRecord],
    image_chunks: Mapping[str, Sequence[str]],
    entities: Sequence[EntityRecord],
    artifacts: Artifacts,
    config: Sam3Config,
    get_grounder: Callable[[], Grounder],
    grounder_id: str,
) -> list[ImageGrounding]:
    """One ``ImageGrounding`` per image, in image order. ``get_grounder`` runs
    only when an image needs the model, so a complete cache never loads SAM3."""
    entities_by_chunk: dict[str, list[EntityRecord]] = {}
    for entity in entities:
        for chunk_id in entity.chunk_ids:
            entities_by_chunk.setdefault(chunk_id, []).append(entity)

    records_dir = artifacts.grounding_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    grounder: Grounder | None = None
    results: list[ImageGrounding] = []
    attempted = failures = 0
    first_error = ""
    for image in images:
        prompts = image_prompts(image_chunks.get(image.image_id, ()), entities_by_chunk)
        key = _key(image.image_id)
        record_path = records_dir / f"{key}.json"
        path = artifacts.image_path(image)

        def failed(error: str, fingerprint: str = "") -> ImageGrounding:
            return ImageGrounding(
                image_id=image.image_id, doc_id=image.doc_id, status="failed", fingerprint=fingerprint,
                entity_ids=[e.entity_id for e in prompts], prompts=[e.name for e in prompts], error=error,
            )

        if not image.ok:
            record = failed("the manifest marks the image as unreadable")
        else:
            try:
                fingerprint = _fingerprint(path, prompts, config, grounder_id)
            except OSError as e:
                record = failed(f"{type(e).__name__}: {e}")
            else:
                cached = _read_record(record_path)
                if cached is not None and cached.fingerprint == fingerprint and cached.status != "failed" and grounding_artifacts_valid(cached, artifacts):
                    results.append(cached)
                    continue
                if not prompts:
                    record = ImageGrounding(
                        image_id=image.image_id, doc_id=image.doc_id, status="no_prompts", fingerprint=fingerprint,
                    )
                else:
                    attempted += 1
                    if grounder is None:
                        grounder = get_grounder()  # a model that cannot load is fatal, not an image failure
                    try:
                        record = _ground_one(image, path, prompts, fingerprint, grounder, artifacts, key, config)
                    except Exception as e:
                        failures += 1
                        first_error = first_error or f"{image.image_id}: {type(e).__name__}: {e}"
                        logger.warning("SAM3 failed on %s: %s", image.image_id, e)
                        record = failed(f"{type(e).__name__}: {e}", fingerprint)
        _write_record(record_path, record)
        results.append(record)
    if attempted and failures == attempted:
        raise RuntimeError(f"SAM3 failed on every image it was given; first error: {first_error}")
    return results


def _ground_one(
    image: ImageRecord,
    path: Path,
    prompts: Sequence[EntityRecord],
    fingerprint: str,
    grounder: Grounder,
    artifacts: Artifacts,
    key: str,
    config: Sam3Config,
) -> ImageGrounding:
    with Image.open(path) as source:
        pil = source.convert("RGB")
    width, height = pil.size
    raw = grounder.predict(pil, [e.name for e in prompts])
    masks = np.asarray(raw.masks)
    if masks.ndim == 4:
        masks = masks[:, 0]
    boxes = np.asarray(raw.boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(raw.scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(raw.labels).reshape(-1).astype(np.int64)
    n = len(scores)
    if not (len(boxes) == len(labels) == len(masks) == n):
        raise ValueError(f"inconsistent SAM3 output: {len(masks)} masks, {len(boxes)} boxes, {n} scores, {len(labels)} labels")
    if n and masks.shape[1:] != (height, width):
        raise ValueError(f"mask size {masks.shape[1:]} differs from the image size {(height, width)}")

    crops_dir = artifacts.grounding_dir / "crops" / key
    masks_path = artifacts.grounding_dir / "masks" / f"{key}.npz"
    shutil.rmtree(crops_dir, ignore_errors=True)
    masks_path.unlink(missing_ok=True)

    image_np = np.asarray(pil)
    detections: list[Detection] = []
    kept_masks: list[np.ndarray] = []
    seen: dict[str, int] = {}
    dropped = 0
    for i in range(n):
        label = int(labels[i])
        valid = (
            0 <= label < len(prompts)
            and bool(np.isfinite(scores[i]))
            and scores[i] >= config.threshold
            and bool(np.isfinite(boxes[i]).all())
        )
        if not valid:
            dropped += 1
            continue
        entity = prompts[label]
        ordinal = seen.get(entity.entity_id, 0)
        seen[entity.entity_id] = ordinal + 1
        stem = hashlib.md5(f"{image.image_id}\x1f{entity.entity_id}\x1f{ordinal}".encode("utf-8")).hexdigest()
        mask = masks[i].astype(np.uint8)  # as MG2: masks.astype(np.uint8)
        crop, crop_box, status = crop_masked(image_np, mask, boxes[i], config.min_crop_size)
        crop_id = crop_path = crop_hash = error = None
        if crop is not None:
            crop_id = f"crop-{stem}"
            crops_dir.mkdir(parents=True, exist_ok=True)
            crop_file = crops_dir / f"{crop_id}.png"
            try:
                crop.save(crop_file, compress_level=1)
                crop_path = artifacts.relative(crop_file)
                crop_hash = hashlib.sha256(crop_file.read_bytes()).hexdigest()
            except Exception as e:
                crop_id= crop_path = crop_hash = None
                status = "failed"
                error =f"{type(e).__name__}: {e}"
        kept_masks.append(mask.astype(bool))
        detections.append(Detection(
            detection_id=f"det-{stem}", image_id=image.image_id, entity_id=entity.entity_id, prompt=entity.name,
            prompt_index=label, index=i, score=float(scores[i]), box=[float(v) for v in boxes[i]],
            mask_index=len(kept_masks) - 1, crop_id=crop_id, crop_path=crop_path, crop_hash=crop_hash, crop_box=crop_box,
            crop_status=status, error=error,
        ))
    mask_file = None
    mask_hash = None
    if kept_masks:
        masks_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(masks_path, masks=np.stack(kept_masks))
        mask_file = artifacts.relative(masks_path)
        mask_hash = hashlib.sha256(masks_path.read_bytes()).hexdigest()

    return ImageGrounding(
        image_id=image.image_id, doc_id=image.doc_id, status="grounded", fingerprint=fingerprint,
        entity_ids=[e.entity_id for e in prompts], prompts=[e.name for e in prompts],
        n_raw=n, n_dropped=dropped, mask_file=mask_file, mask_hash=mask_hash, detections=detections,
    )


def crop_masked(
    image_np: np.ndarray, mask: np.ndarray, box: Sequence[float], min_size: int
) -> tuple[Image.Image | None, list[int], str]:
    """MG2's crop: the integer box (truncated, as ``astype(int)``) clipped to
    the image, the mask's pixels on black. Returns (crop or None, clipped box,
    "saved" or "too_small")."""
    height, width = image_np.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    clipped = [x1, y1, x2, y2]
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None, clipped, "too_small"
    keep = mask[y1:y2, x1:x2].astype(bool)[:, :, None]
    out = np.where(keep, image_np[y1:y2, x1:x2], 0).astype(np.uint8)
    return Image.fromarray(out), clipped, "saved"


def load_grounding(artifacts: Artifacts, image_ids: Sequence[str]) -> dict[str, ImageGrounding]:
    """The saved record of every image; a missing one means the ground stage has not finished."""
    records_dir = artifacts.grounding_dir / "records"
    result = {}
    for image_id in image_ids:
        record = _read_record(records_dir / f"{_key(image_id)}.json")
        if record is None:
            raise ValueError(f"no grounding record for image {image_id}: run the ground stage")
        result[image_id] = record
    return result


def _key(image_id: str) -> str:
    return hashlib.md5(image_id.encode("utf-8")).hexdigest()


def _fingerprint(path: Path, prompts: Sequence[EntityRecord], config: Sam3Config, grounder_id: str) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    payload = json.dumps(
        [digest.hexdigest(), [[e.entity_id, e.name] for e in prompts], config.threshold, config.min_crop_size,
         config.prompt_batch_size, grounder_id],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_record(path: Path) -> ImageGrounding | None:
    if not path.is_file():
        return None
    try:
        return ImageGrounding.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _write_record(path: Path, record: ImageGrounding) -> None:
    """Atomic, so a crash never leaves a half-written record."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(record.model_dump_json(), encoding="utf-8", newline="\n")
    tmp.replace(path)


# --- MG2's SAM3 wrapper ------------------------------------------------------


def default_checkpoint(artifacts: Artifacts) -> Path:
    value = os.environ.get(SAM3_CHECKPOINT_ENV)
    return Path(value).expanduser() if value else artifacts.models_dir / "sam3" / "sam3.pt"


def grounder_identity(checkpoint: Path) -> str:
    """Names the model in fingerprints: the MG2 commit and the checkpoint file."""
    return f"mg2-sam3@{MG2_COMMIT[:7]}:{checkpoint.name}:{checkpoint.stat().st_size}"


class MG2Sam3:
    """MG2's ``SAM3Grounding``, loaded from a checkout of MG2-RAG.

    The wrapper lives in a package whose ``__init__`` imports unrelated graph
    and CuPy code, so (as in the smoke test) it is imported under an alias
    package that points at ``src/mmgraphrag`` and skips that ``__init__``. The
    checkout also stays on ``sys.path``: it provides the bundled ``sam3``
    package. Needs CUDA. Whether this coexists in one process with
    MemGraphRAG's own ``src`` package is unverified; use the staged
    execution (``pipeline`` stages) if it does not.
    """

    def __init__(self, repo: Path, checkpoint: Path, device: int, threshold: float, prompt_batch_size: int, work_dir: Path):
        repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
        for required in (
            checkpoint,
            repo / "assets" / "bpe_simple_vocab_16e6.txt.gz",
            repo / "src" / "mmgraphrag" / "grounding_model" / "sam3Model.py",
            repo / "sam3" / "__init__.py",
        ):
            if not required.exists():
                raise FileNotFoundError(f"SAM3 needs {required}")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("MG2's SAM3Grounding requires CUDA; run on an allocated GPU")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        alias = "mgrv_mg2"
        if alias not in sys.modules:
            package = types.ModuleType(alias)
            package.__path__ = [str(repo / "src" / "mmgraphrag")]
            sys.modules[alias] = package
        config_module = importlib.import_module(f"{alias}.utils.config_utils")
        wrapper = importlib.import_module(f"{alias}.grounding_model.sam3Model")
        sam3 = importlib.import_module("sam3")
        if Path(sam3.__file__).resolve().parent != repo / "sam3":
            raise RuntimeError("the imported sam3 is not the copy bundled with the MG2 checkout")
        work_dir = Path(work_dir)
        config = config_module.BaseConfig(
            grounding_model_name=str(checkpoint),
            grounding_model_work_dir="grounding",
            grounding_model_prompt_batch_size=prompt_batch_size,
            grounding_detection_threshold=threshold,
            sam3_device=device,
            save_dir=str(work_dir / "config"),
        )
        self._batch = prompt_batch_size
        self._model = wrapper.SAM3Grounding(global_config=config, global_work_dir=str(work_dir))

    def predict(self, image: Image.Image, prompts: list[str]) -> RawDetections:
        result = self._model.predict(image=image, prompt=prompts, prompt_batch_size=self._batch)

        def array(tensor) -> np.ndarray:
            return tensor.detach().cpu().numpy()

        return RawDetections(
            masks=array(result["masks"]), boxes=array(result["boxes"]),
            scores=array(result["scores"]), labels=array(result["labels"]),
        )

    def close(self) -> None:
        self._model = None
        release_gpu()
