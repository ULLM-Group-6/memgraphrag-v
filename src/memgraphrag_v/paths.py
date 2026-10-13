"""Artifact locations.

Everything lives under one root outside the repo, given by ``$MGRV_ARTIFACTS``
(on Snellius: project space or /scratch-shared) or the config's
``artifacts_root``. Copy the builder's output into ``manifests/`` and the
MMQA ``final_dataset_images/`` folder into ``corpus/images/``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import ImageRecord

ENV_VAR = "MGRV_ARTIFACTS"
EMBEDDING_KINDS = ("image", "crop", "query")


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
