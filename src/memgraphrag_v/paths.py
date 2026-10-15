"""Artifact locations.

Everything lives under one root outside the repo, given by ``$MGRV_ARTIFACTS``
(on Snellius: project space or /scratch-shared) or the config's
``artifacts_root``. Copy the builder's output into ``manifests/`` and the
MMQA ``final_dataset_images/`` folder into ``corpus/images/``.

External code and checkpoints are located through environment variables, only
when the requested operation needs them (``external_path``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import ImageRecord

ENV_VAR = "MGRV_ARTIFACTS"
MEMGRAPHRAG_ENV = "MGRV_MEMGRAPHRAG"
"""Checkout of XMUDeepLIT/MemGraphRAG at the pinned commit (its ``code/`` folder is imported)."""
MG2_ENV = "MGRV_MG2_RAG"
"""Checkout of Daboolu/MG2-RAG at the pinned commit (provides ``sam3/`` and the SAM3 wrapper)."""
SAM3_CHECKPOINT_ENV = "MGRV_SAM3_CHECKPOINT"
"""SAM3 weights; default ``<artifacts>/models/sam3/sam3.pt``."""
TEXT_EMBEDDING_ENV = "MGRV_TEXT_EMBEDDING_MODEL"
"""Overrides ``text.embedding_model`` (e.g. with an offline snapshot path)."""
EMBEDDING_KINDS = ("image", "crop", "query")


def external_path(env_var: str, what: str) -> Path:
    """An existing path from an environment variable, with an error that says
    which variable to set."""
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(f"{what}: set {env_var}")
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{what}: {path} (from {env_var}) does not exist")
    return path


@dataclass(frozen=True)
class Artifacts:
    root: Path

    @classmethod
    def from_env(cls, root: str | Path | None = None) -> Artifacts:
        """Use ``root`` if given (e.g. from the config), else ``$MGRV_ARTIFACTS``."""
        value = root or os.environ.get(ENV_VAR)
        if not value:
            raise RuntimeError(f"no artifacts root: set {ENV_VAR} or artifacts_root in the config")
        return cls(Path(value).expanduser())

    # WebQA subset as the dataset branch's builder writes it (plan task 1)
    @property
    def webqa_dir(self) -> Path:
        """text_manifest.jsonl, image_manifest.jsonl, caption_manifest.jsonl,
        build_qa.json, test_qa.json from the ``dataset`` branch."""
        return self.root / "webqa"

    # Manifests (plan task 1)
    @property
    def text_manifest(self) -> Path:
        return self.root / "manifests" / "text_manifest.jsonl"

    @property
    def image_manifest(self) -> Path:
        return self.root / "manifests" / "image_manifest.jsonl"

    @property
    def questions(self) -> Path:
        return self.root / "manifests" / "questions.jsonl"

    @property
    def captions(self) -> Path:
        return self.root / "manifests" / "captions.jsonl"

    @property
    def images_dir(self) -> Path:
        return self.root / "corpus" / "images"

    def image_path(self, image: ImageRecord) -> Path:
        return self.images_dir / image.path

    # Models and embeddings (plan tasks 3 and 4)
    @property
    def models_dir(self) -> Path:
        """Downloaded checkpoints, e.g. models/EVA-CLIP-8B."""
        return self.root / "models"

    def embeddings_dir(self, encoder: str, kind: str) -> Path:
        """e.g. embeddings/eva-clip-8b/image for encoder BAAI/EVA-CLIP-8B."""
        if kind not in EMBEDDING_KINDS:
            raise ValueError(f"unknown embedding kind {kind!r}")
        slug = encoder.rsplit("/", 1)[-1].lower()
        return self.root / "embeddings" / slug / kind

    # Runs (plan tasks 5 and 6)
    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ValueError(f"run_id must match [A-Za-z0-9._-]+: {run_id!r}")
        return self.root / "runs" / run_id

    # Combined index: MemGraphRAG text memory plus the visual additions.
    # Image and crop vectors stay under embeddings/.
    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def index_manifest(self) -> Path:
        """Stage stamps (input/setting/output hashes) and provenance."""
        return self.index_dir / "manifest.json"

    @property
    def text_dir(self) -> Path:
        """MemGraphRAG's ``save_dir``: OpenIE cache, ``memory.json`` and, under
        ``<llm>_<embedding>/``, the text stores and the final ``graph.graphml``."""
        return self.index_dir / "text"

    @property
    def memory_path(self) -> Path:
        return self.text_dir / "memory.json"

    @property
    def texts(self) -> Path:
        return self.index_dir / "texts.jsonl"

    @property
    def chunks(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def entities(self) -> Path:
        return self.index_dir / "entities.jsonl"

    @property
    def indexed_images(self) -> Path:
        return self.index_dir / "images.jsonl"

    @property
    def grounding_dir(self) -> Path:
        """``records/`` (one ImageGrounding per image), ``masks/`` and ``crops/``."""
        return self.index_dir / "grounding"

    def relative(self, path: Path) -> str:
        """``path`` relative to the root with forward slashes (as stored in records)."""
        return Path(path).relative_to(self.root).as_posix()
