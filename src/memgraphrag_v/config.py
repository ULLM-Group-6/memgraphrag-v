"""Experiment configuration.

One YAML file pins every setting. Settings not decided yet are ``null``;
``require_frozen()`` refuses to start a test evaluation while any remain.
Sections for later tasks (text index, reader, evaluation) are added when
those tasks need them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .schemas import MAX_DOCS


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(_Section):
    name: str = "mmqa"
    split_seed: int = 42
    n_dev: int = 100
    n_test: int = 400


class VisualEncoderConfig(_Section):
    """MG2's EVA-CLIP setup, as smoke-tested on Snellius."""

    name: str = "BAAI/EVA-CLIP-8B"
    revision: str | None = None
    processor: str = "openai/clip-vit-large-patch14"
    processor_revision: str | None = None
    batch_size: int = Field(default=16, ge=1)
    max_text_tokens: int = Field(default=77, ge=1)


class Sam3Config(_Section):
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class RetrievalConfig(_Section):
    top_k_candidates: int = Field(default=20, ge=1)
    """Whole images and crops taken as visual candidates per question."""
    visual_mix: float = Field(default=0.5, ge=0.0, le=1.0)
    """Weight of the image component in the visual seeds (crops get the rest)."""
    beta_grid: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75])
    beta: float | None = Field(default=None, ge=0.0, le=1.0)
    """Text/visual mixing weight, tuned on dev and frozen before test."""
    n_docs: int = Field(default=MAX_DOCS, ge=1, le=MAX_DOCS)
    text_budget_tokens: int = 256


class ExperimentConfig(_Section):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    visual_encoder: VisualEncoderConfig = Field(default_factory=VisualEncoderConfig)
    sam3: Sam3Config = Field(default_factory=Sam3Config)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    artifacts_root: str | None = None
    """Machine-specific: leave null in committed configs and set MGRV_ARTIFACTS."""

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        Path(path).write_text(text, encoding="utf-8", newline="\n")

    def unresolved(self) -> list[str]:
        """Dotted names of settings that are still null."""
        data = self.model_dump(mode="json", exclude={"artifacts_root"})
        return [f"{s}.{k}" for s, section in data.items() for k, v in section.items() if v is None]

    def require_frozen(self) -> None:
        missing = self.unresolved()
        if missing:
            raise ValueError("config is not frozen; unresolved: " + ", ".join(missing))

    def config_hash(self) -> str:
        """sha256 of the canonical JSON; ignores key order and artifacts_root."""
        data = self.model_dump(mode="json", exclude={"artifacts_root"})
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
