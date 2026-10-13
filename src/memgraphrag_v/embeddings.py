"""Visual embeddings: whole images, SAM3 crops and questions.

One directory per kind, ``embeddings/<encoder>/<kind>/``:

- ``vectors.npy``  float32 (n_ok, dim), L2-normalised, so a dot product is the
  cosine similarity.
- ``rows.jsonl``   one ``EmbeddingRow`` per input item, including failures.
  ``row`` is the item's index into ``vectors.npy`` and is None when the item
  failed, so a broken file never gets a (blank) vector.
- ``meta.json``    ``EmbeddingMeta``: encoder, revisions, dim, counts.

Image and crop rows carry image_id and doc_id, and crop rows carry
entity_id, so direct retrieval (image -> document) and crop seeds
(crop -> entity) work without joins.

Run on a GPU node:
    python -m memgraphrag_v.embeddings image   [--limit N]
    python -m memgraphrag_v.embeddings query   [--limit N]
Crops are embedded from Python with ``embed_files(..., kind="crop")`` once
SAM3 preprocessing (plan task 4) exports them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Protocol, Sequence

import numpy as np
from PIL import Image

from .io import read_jsonl, write_jsonl
from .schemas import Id, ImageRecord, QuestionRecord, Record

Kind = Literal["image", "crop", "query"]


class EmbeddingMeta(Record):
    """meta.json: what produced the vectors. Reuse vectors only if this matches."""

    encoder: str
    revision: str | None
    processor: str
    processor_revision: str | None
    dim: int
    kind: Kind
    normalized: bool = True
    n_rows: int
    n_failed: int


class EmbeddingRow(Record):
    """rows.jsonl: one input item."""

    row: int | None
    """Index into vectors.npy; None iff status is "failed"."""
    kind: Kind
    item_id: Id
    """image_id, crop_id or qid."""
    image_id: Id | None = None
    """The whole image, or the crop's parent image."""
    doc_id: Id | None = None
    entity_id: Id | None = None
    """Crops only: the MemGraphRAG entity SAM3 grounded."""
    path: str | None = None
    """Input file, relative to the artifacts root."""
    sha256: str | None = None
    """Of the input file's bytes, to detect stale vectors."""
    width: int | None = None
    height: int | None = None
    status: Literal["ok", "truncated", "failed"]
    """truncated: a query cut to the encoder's token limit (still embedded)."""
    error: str | None = None


class Encoder(Protocol):
    """What the pipeline needs from an encoder; ``eva_clip.EvaClip`` is one."""

    dim: int

    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> tuple[np.ndarray, list[bool]]: ...


@dataclass
class EmbeddingSet:
    meta: EmbeddingMeta
    rows: list[EmbeddingRow]
    vectors: np.ndarray
    _embedded: list[EmbeddingRow] = field(init=False, repr=False)
    _by_id: dict[str, EmbeddingRow] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._embedded = [r for r in self.rows if r.row is not None]
        self._by_id = {r.item_id: r for r in self.rows}
        if len(self._by_id) != len(self.rows):
            raise ValueError("item_ids must be unique")
        if [r.row for r in self._embedded] != list(range(len(self._embedded))):
            raise ValueError("row indices must be 0..n-1 in file order")
        if self.vectors.shape != (len(self._embedded), self.meta.dim):
            raise ValueError(
                f"vectors {self.vectors.shape} do not match {len(self._embedded)} rows of dim {self.meta.dim}"
            )

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors.astype(np.float32))
        write_jsonl(directory / "rows.jsonl", self.rows)
        (directory / "meta.json").write_text(
            self.meta.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )

    @classmethod
    def load(cls, directory: str | Path) -> EmbeddingSet:
        directory = Path(directory)
        meta = EmbeddingMeta.model_validate_json((directory / "meta.json").read_text(encoding="utf-8"))
        rows = read_jsonl(directory / "rows.jsonl", EmbeddingRow)
        return cls(meta, rows, np.load(directory / "vectors.npy"))

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._by_id

    def row_for(self, item_id: str) -> EmbeddingRow:
        return self._by_id[item_id]

    def vector(self, item_id: str) -> np.ndarray:
        r = self.row_for(item_id)
        if r.row is None:
            raise KeyError(f"{item_id} has no vector ({r.error})")
        return self.vectors[r.row]

    def scores(self, query: np.ndarray) -> dict[str, float]:
        """Cosine similarity of every embedded item to ``query``, by item_id."""
        return {r.item_id: float(s) for r, s in zip(self._embedded, self.vectors @ query)}

    def top_k(self, query: np.ndarray, k: int) -> list[tuple[EmbeddingRow, float]]:
        """The k most similar items, best first; ties broken by item_id."""
        scores = self.vectors @ query
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self._embedded[i].item_id))
        return [(self._embedded[i], float(scores[i])) for i in order[:k]]


# --- Embedding pipeline ------------------------------------------------------


@dataclass(frozen=True)
class EmbedItem:
    """A file to embed and the metadata its row will carry."""

    item_id: str
    path: str
    """Relative to the artifacts root."""
    image_id: str | None = None
    doc_id: str | None = None
    entity_id: str | None = None


def items_from_images(images: Iterable[ImageRecord], images_dir: str = "corpus/images") -> list[EmbedItem]:
    """Whole-image items; ``images_dir`` is relative to the artifacts root."""
    return [
        EmbedItem(item_id=i.image_id, path=f"{images_dir}/{i.path}", image_id=i.image_id, doc_id=i.doc_id)
        for i in images
    ]


def embed_files(
    items: Sequence[EmbedItem],
    kind: Literal["image", "crop"],
    encoder: Encoder,
    root: str | Path,
    meta: dict,
    batch_size: int = 64,
) -> EmbeddingSet:
    """Embed image or crop files under ``root``. Each file is opened here;
    one that cannot be read becomes a failed row without a vector."""
    root = Path(root)
    rows: list[EmbeddingRow] = []
    chunks: list[np.ndarray] = []
    n_ok = 0
    for start in range(0, len(items), batch_size):
        loaded = [_load(root, item) for item in items[start : start + batch_size]]
        images = [image for _, image, _ in loaded if image is not None]
        if images:
            chunks.append(encoder.encode_images(images))
        for item, image, error in loaded:
            info = {"kind": kind, **item.__dict__}
            if image is None:
                rows.append(EmbeddingRow(row=None, status="failed", error=error, **info))
            else:
                rows.append(EmbeddingRow(row=n_ok, status="ok", width=image.width, height=image.height,
                                         sha256=_sha256(root / item.path), **info))
                n_ok += 1
    return _build_set(kind, rows, chunks, encoder.dim, meta)


def _load(root: Path, item: EmbedItem) -> tuple[EmbedItem, Image.Image | None, str | None]:
    """Open a file as RGB; an unreadable or missing file is reported, never
    replaced by a blank image."""
    try:
        with Image.open(root / item.path) as source:
            return item, source.convert("RGB"), None
    except Exception as e:
        return item, None, f"{type(e).__name__}: {e}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embed_queries(questions: Sequence[QuestionRecord], encoder: Encoder, meta: dict) -> EmbeddingSet:
    """Embed question texts as they are, without query expansion."""
    vectors, truncated = encoder.encode_texts([q.question for q in questions])
    rows = [
        EmbeddingRow(row=i, kind="query", item_id=q.qid, status="truncated" if cut else "ok")
        for i, (q, cut) in enumerate(zip(questions, truncated))
    ]
    return _build_set("query", rows, [vectors], encoder.dim, meta)


def _build_set(kind: Kind, rows: list[EmbeddingRow], chunks: list[np.ndarray], dim: int, meta: dict) -> EmbeddingSet:
    vectors = np.concatenate(chunks) if chunks else np.zeros((0, dim), dtype=np.float32)
    n_failed = sum(r.row is None for r in rows)
    full_meta = EmbeddingMeta(kind=kind, dim=dim, n_rows=len(rows), n_failed=n_failed, **meta)
    return EmbeddingSet(full_meta, rows, vectors.astype(np.float32))


# --- Command line ------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    from .config import ExperimentConfig
    from .eva_clip import EvaClip
    from .paths import Artifacts

    parser = argparse.ArgumentParser(description="Embed corpus images or questions with EVA-CLIP.")
    parser.add_argument("kind", choices=["image", "query"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, help="embed only the first N items (smoke runs)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    config = ExperimentConfig.load(args.config)
    enc = config.visual_encoder
    artifacts = Artifacts.from_env(config.artifacts_root)
    encoder = EvaClip(
        artifacts.models_dir / enc.name.rsplit("/", 1)[-1],
        artifacts.models_dir / enc.processor.rsplit("/", 1)[-1],
        device=args.device,
        batch_size=enc.batch_size,
        max_text_tokens=enc.max_text_tokens,
    )
    meta = {
        "encoder": enc.name,
        "revision": enc.revision,
        "processor": enc.processor,
        "processor_revision": enc.processor_revision,
    }
    if args.kind == "image":
        images = read_jsonl(artifacts.image_manifest, ImageRecord)[: args.limit]
        images_dir = artifacts.images_dir.relative_to(artifacts.root).as_posix()
        result = embed_files(items_from_images(images, images_dir), "image", encoder, artifacts.root, meta)
    else:
        questions = read_jsonl(artifacts.questions, QuestionRecord)[: args.limit]
        result = embed_queries(questions, encoder, meta)

    out = artifacts.embeddings_dir(enc.name, args.kind)
    result.save(out)
    print(json.dumps({"out": str(out), **result.meta.model_dump()}, indent=2))


if __name__ == "__main__":
    main()
