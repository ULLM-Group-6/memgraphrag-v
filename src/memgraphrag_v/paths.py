"""Artifact locations.

All artifacts live under one root outside the repo, given by the config's
``artifacts_root`` or the ``MGRV_ARTIFACTS`` environment variable (on Snellius,
project space or /scratch-shared). Manifests store paths relative to this root.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import check_rel_path

ENV_VAR = "MGRV_ARTIFACTS"


def encoder_slug(encoder: str) -> str:
    """Directory name for an encoder, e.g. google/siglip2-so400m-patch14-384
    -> google--siglip2-so400m-patch14-384."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", encoder.replace("/", "--"))


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path

    @classmethod
    def from_env(cls, root: str | Path | None = None) -> ArtifactLayout:
        """Use ``root`` if given (e.g. from the config), else ``$MGRV_ARTIFACTS``."""
        value = root if root is not None else os.environ.get(ENV_VAR)
        if not value:
            raise RuntimeError(f"no artifacts root: set {ENV_VAR} or artifacts_root in the config")
        return cls(Path(value).expanduser())

    def resolve(self, rel_path: str) -> Path:
        """Turn a manifest path into a local path."""
        return self.root / check_rel_path(rel_path)

    def relative(self, path: str | Path) -> str:
        """Turn a local path under the root into the form manifests store."""
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    # manifests/ ([09])
    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def text_manifest(self) -> Path:
        return self.manifests / "text.jsonl"

    @property
    def image_manifest(self) -> Path:
        return self.manifests / "images.jsonl"

    @property
    def crop_manifest(self) -> Path:
        return self.manifests / "crops.jsonl"

    @property
    def question_manifest(self) -> Path:
        return self.manifests / "questions.jsonl"

    @property
    def images_dir(self) -> Path:
        """Corpus image files, referenced by ImageRecord.path."""
        return self.root / "corpus" / "images"

    # splits/ ([10])
    @property
    def splits(self) -> Path:
        return self.root / "splits"

    # index/ ([11], [22])
    @property
    def memgraphrag_dir(self) -> Path:
        """Upstream MemGraphRAG save_dir for the unaugmented text index."""
        return self.root / "index" / "text" / "memgraphrag"

    @property
    def passage_map(self) -> Path:
        return self.root / "index" / "text" / "exports" / "passage_map.jsonl"

    @property
    def entities(self) -> Path:
        return self.root / "index" / "text" / "exports" / "entities.jsonl"

    @property
    def caption_index_dir(self) -> Path:
        """Separate index for caption baseline C."""
        return self.root / "index" / "caption"

    @property
    def caption_chunk_map(self) -> Path:
        """Chunk -> passage/image description map of the caption index ([22])."""
        return self.caption_index_dir / "exports" / "chunk_map.jsonl"

    @property
    def descriptions(self) -> Path:
        """Cached question-independent image descriptions ([14])."""
        return self.root / "descriptions" / "descriptions.jsonl"

    # grounding/ ([13], [18])
    @property
    def grounding_attempts(self) -> Path:
        return self.root / "grounding" / "sam3" / "attempts.jsonl"

    @property
    def crops_dir(self) -> Path:
        return self.root / "grounding" / "sam3" / "crops"

    @property
    def masks_dir(self) -> Path:
        return self.root / "grounding" / "sam3" / "masks"

    # embeddings/ ([12], [19])
    def embeddings_dir(self, encoder: str, kind: str) -> Path:
        if kind not in ("image", "crop", "query"):
            raise ValueError(f"unknown embedding kind {kind!r}")
        return self.root / "embeddings" / encoder_slug(encoder) / kind

    # graph/ ([20], [21])
    @property
    def graph_dir(self) -> Path:
        return self.root / "graph"

    # runs/ ([15], [17], [25])
    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ValueError(f"run_id must be [A-Za-z0-9._-]+: {run_id!r}")
        return self.root / "runs" / run_id

    def run_config(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "config.yaml"

    def retrieval_output(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "retrieval.jsonl"

    def diagnostics_output(self, run_id: str) -> Path:
        """Candidates, seeds and PPR rankings for the error analysis (§7)."""
        return self.run_dir(run_id) / "diagnostics.jsonl"

    def reader_output(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "reader.jsonl"

    def scores_output(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "scores.jsonl"
