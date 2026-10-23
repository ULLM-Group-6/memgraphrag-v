"""Visual indexing: EVA-CLIP embeddings of whole images and SAM3 crops, and the
image vertices and edges added to the text graph.

Embeddings reuse ``embeddings.py`` (``EmbedItem``, ``EmbeddingSet``,
``items_from_images``, ``embed_files``) and ``eva_clip.py``. ``embed_files``
is not an incremental store, so ``embed_cached`` saves each set explicitly
and reuses a saved one only after checking encoder and preprocessing revisions,
item IDs, file paths and file hashes. Crops are external records linked to
their parent image and existing entity; they are embedded here but never
become vertices.

``add_visual_structure`` adds to the finished text graph:

- one vertex per usable image (a whole image with a vector), including images
  without detections;
- ``image_passage`` edges of weight 1 to the chunks of the image's explicitly
  associated passages and captions;
- ``image_entity`` edges to entities SAM3 grounded, weighted by the maximum
  retained confidence of the pair.

Every new pair is added once; text edges, parallel ones included, are not
touched. The functions take the graph as an argument, so nothing here needs
igraph at import.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
from PIL import Image

from . import sam3_processing
from .config import VisualEncoderConfig
from .embeddings import EmbedItem, EmbeddingMeta, EmbeddingSet, Encoder, embed_files, items_from_images
from .image_retrieval import image_passage_edges
from .paths import Artifacts
from .schemas import VERTEX_LAYERS, ImageGrounding, ImageRecord, IndexedImage

logger = logging.getLogger(__name__)

IMAGE_LAYER = "image"
IMAGE_PASSAGE = "image_passage"
IMAGE_ENTITY = "image_entity"


def image_vertex(image_id: str) -> str:
    """Vertex name of an image; the prefix keeps it apart from chunk-, entity- and type- names."""
    return f"image-{image_id}"


# --- Embeddings --------------------------------------------------------------


def encoder_meta(enc: VisualEncoderConfig) -> dict:
    """The fields of ``EmbeddingMeta`` that identify the encoder and preprocessing."""
    return {
        "encoder": enc.name,
        "revision": enc.revision,
        "processor": enc.processor,
        "processor_revision": enc.processor_revision,
    }


def load_encoder(enc: VisualEncoderConfig, artifacts: Artifacts, device: str = "cuda") -> Encoder:
    """EVA-CLIP from ``<artifacts>/models``. Loads the checkpoint: call it only when needed."""
    from .eva_clip import EvaClip

    return EvaClip(
        artifacts.models_dir / enc.name.rsplit("/", 1)[-1],
        artifacts.models_dir / enc.processor.rsplit("/", 1)[-1],
        device=device,
        batch_size=enc.batch_size,
        max_text_tokens=enc.max_text_tokens,
    )


def crop_items(records: Sequence[ImageGrounding]) -> list[EmbedItem]:
    """The saved crops, as embedding items carrying their parent image and entity."""
    return [
        EmbedItem(item_id=d.crop_id, path=d.crop_path, image_id=d.image_id, doc_id=g.doc_id, entity_id=d.entity_id)
        for g in records
        for d in g.detections
        if d.crop_status == "saved" and d.crop_id and d.crop_path
    ]


def cache_problem(cached: EmbeddingSet, items: Sequence[EmbedItem], kind: str, meta: dict, root: Path) -> str | None:
    """Why a saved set cannot be reused for ``items``; None when it can."""
    if cached.meta.kind != kind:
        return f"kind {cached.meta.kind!r}, wanted {kind!r}"
    for key, value in meta.items():
        if getattr(cached.meta, key) != value:
            return f"{key} is {getattr(cached.meta, key)!r}, wanted {value!r}"
    if [r.item_id for r in cached.rows] != [i.item_id for i in items]:
        return "the item IDs differ"
    for row, item in zip(cached.rows, items):
        if (row.path, row.image_id, row.doc_id, row.entity_id) != (item.path, item.image_id, item.doc_id, item.entity_id):
            return f"{item.item_id}: metadata differs"
        file = root / item.path
        if row.status == "failed":
            if _readable(file):
                return f"{item.item_id}: failed before but opens now"
        elif not file.is_file() or _sha256(file) != row.sha256:
            return f"{item.item_id}: the file changed"
    return None


def embed_cached(
    kind: Literal["image", "crop"],
    items: Sequence[EmbedItem],
    enc: VisualEncoderConfig,
    artifacts: Artifacts,
    get_encoder: Callable[[], Encoder],
    reuse: bool = True,
    dim: int | None = None,
) -> EmbeddingSet:
    """Embed ``items`` and save the set under ``embeddings/<encoder>/<kind>/``,
    unless a valid one is already there. ``get_encoder`` runs only when the
    model is needed; with no items and a known ``dim`` it never does."""
    out = artifacts.embeddings_dir(enc.name, kind)
    meta = encoder_meta(enc)
    if reuse and (out / "meta.json").is_file():
        cached = EmbeddingSet.load(out)
        problem = cache_problem(cached, items, kind, meta, artifacts.root)
        if problem is None:
            logger.info("reusing the saved %s embeddings in %s", kind, out)
            return cached
        logger.info("re-embedding %s: %s", kind, problem)
    if items:
        result = embed_files(items, kind, get_encoder(), artifacts.root, meta)
    else:
        dim = dim or get_encoder().dim
        empty = EmbeddingMeta(kind=kind, dim=dim, n_rows=0, n_failed=0, **meta)
        result = EmbeddingSet(empty, [], np.zeros((0, dim), dtype=np.float32))
    result.save(out)
    return result


def embed_visual(
    artifacts: Artifacts,
    enc: VisualEncoderConfig,
    images: Sequence[ImageRecord],
    get_encoder: Callable[[], Encoder],
    reuse: bool = True,
) -> tuple[EmbeddingSet, EmbeddingSet]:
    """Whole-image and crop embeddings (both saved). Every manifest image is
    embedded, so the set matches what ``embeddings.py image`` writes; the ones
    that fail stay as failed rows. Needs the saved grounding records."""
    images_dir = artifacts.images_dir.relative_to(artifacts.root).as_posix()
    image_set = embed_cached("image", items_from_images(images, images_dir), enc, artifacts, get_encoder, reuse)
    grounding = sam3_processing.load_grounding(artifacts, [i.image_id for i in images])
    crops = crop_items([grounding[i.image_id] for i in images])
    crop_set = embed_cached("crop", crops, enc, artifacts, get_encoder, reuse, dim=image_set.meta.dim)
    return image_set, crop_set


# --- Graph -------------------------------------------------------------------


def add_visual_structure(
    graph,
    links: Sequence[IndexedImage],
    chunks_by_record: Mapping[str, Sequence[str]],
    image_set: EmbeddingSet,
    grounding: Mapping[str, ImageGrounding],
) -> tuple[list[IndexedImage], dict[str, int]]:
    """Add image vertices and both edge types to ``graph`` (an igraph graph
    with ``name`` and ``layer`` vertex attributes and ``weight`` and ``type``
    edge attributes). Returns the links with vertex and usability filled in,
    and counts.

    ``image_passage_edges`` gives (image, text record, 1.0) for each usable
    image's associated records; they are translated to the records' chunk
    vertices through ``chunks_by_record``, so a caption's or passage's image
    edges land on its chunks and nowhere else."""
    attributes = graph.vs.attributes()
    if graph.vcount() and not {"name", "layer"} <= set(attributes):
        raise ValueError("the text graph needs name and layer vertex attributes")
    names = list(graph.vs["name"]) if graph.vcount() else []
    layers = list(graph.vs["layer"]) if graph.vcount() else []
    bad = sorted({layer for layer in layers if layer not in VERTEX_LAYERS})
    if bad:
        raise ValueError(f"unexpected vertex layers in the text graph: {bad}")
    layer_of = dict(zip(names, layers))
    index = {name: i for i, name in enumerate(names)}
    if len(index) != len(names):
        raise ValueError("vertex names must be unique")

    base = graph.vcount()
    new_names: list[str] = []
    new_content: list[str] = []
    updated: list[IndexedImage] = []
    for link in links:
        row = image_set.row_for(link.image_id) if link.image_id in image_set else None
        if row is None or row.row is None:
            reason = "no embedding row" if row is None else f"embedding failed: {row.error}"
            updated.append(link.model_copy(update={"usable": False, "vertex": None, "error": reason}))
            continue
        vertex = image_vertex(link.image_id)
        if vertex in index:
            raise ValueError(f"vertex name {vertex!r} already exists")
        index[vertex] = base + len(new_names)
        new_names.append(vertex)
        new_content.append(link.image_id)
        updated.append(link.model_copy(update={"usable": True, "vertex": vertex, "error": None}))
    if new_names:
        graph.add_vertices(
            len(new_names),
            attributes={"name": new_names, "layer": [IMAGE_LAYER] * len(new_names), "content": new_content},
        )
    usable = {link.image_id: link for link in updated if link.usable}

    pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    kinds: list[str] = []
    seen: set[tuple[int, int]] = set()
    records = [
        ImageRecord(image_id=u.image_id, doc_id=u.doc_id, path=u.path, passage_ids=list(u.record_ids), ok=True)
        for u in usable.values()
    ]
    for image_id, record_id, weight in image_passage_edges(records, image_set):
        for chunk_id in chunks_by_record.get(record_id, ()):
            if layer_of.get(chunk_id) != "passage":
                raise ValueError(f"image {image_id} is linked to {chunk_id}, which is not a passage vertex")
            pair = (index[usable[image_id].vertex], index[chunk_id])
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                weights.append(weight)
                kinds.append(IMAGE_PASSAGE)
    n_passage = len(pairs)

    n_entity = n_detections = 0
    for link in usable.values():
        best: dict[int, float] = {}
        for detection in grounding[link.image_id].detections:
            if layer_of.get(detection.entity_id) != "entity":
                raise ValueError(
                    f"image {link.image_id}: grounded entity {detection.entity_id} is not an entity vertex; "
                    "the grounding does not match the final memory"
                )
            if not np.isfinite(detection.score) or detection.score <= 0:
                raise ValueError(f"detection {detection.detection_id} has an invalid confidence {detection.score}")
            entity = index[detection.entity_id]
            best[entity] = max(best.get(entity, 0.0), detection.score)
            n_detections += 1
        for entity, weight in best.items():
            pairs.append((index[link.vertex], entity))
            weights.append(float(weight))
            kinds.append(IMAGE_ENTITY)
            n_entity += 1
    if pairs:
        graph.add_edges(pairs, attributes={"weight": weights, "type": kinds})
    stats = {
        "images": len(links),
        "image_vertices": len(new_names),
        "unusable_images": len(links) - len(new_names),
        "image_passage_edges": n_passage,
        "image_entity_edges": n_entity,
        "detections_used": n_detections,
    }
    return updated, stats


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _readable(path: Path) -> bool:
    try:
        with Image.open(path) as source:
            source.convert("RGB")
        return True
    except Exception:
        return False
