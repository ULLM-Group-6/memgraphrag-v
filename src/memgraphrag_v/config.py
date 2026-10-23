"""Experiment configuration.

One YAML file pins every setting. Settings not decided yet are ``null``;
``require_frozen()`` refuses to start a test evaluation while any remain.
Sections for later tasks (reader, evaluation) are added when those tasks need
them. Machine-specific paths never live here: see ``paths.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .schemas import MAX_DOCS


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(_Section):
    name: str = "webqa"
    """Our WebQA subset; its builder fixes the dev and test questions."""


class VisualEncoderConfig(_Section):
    """MG2's EVA-CLIP setup, as smoke-tested on Snellius."""

    name: str = "BAAI/EVA-CLIP-8B"
    revision: str | None = None
    processor: str = "openai/clip-vit-large-patch14"
    processor_revision: str | None = None
    batch_size: int = Field(default=16, ge=1)
    max_text_tokens: int = Field(default=77, ge=1)


class Sam3Config(_Section):
    """Entity-prompt grounding with MG2's SAM3 wrapper (``sam3_processing.py``)."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Detections with a lower confidence are dropped."""
    min_crop_size: int = Field(default=10, ge=1)
    """MG2 skips crops narrower or shorter than this many pixels."""
    prompt_batch_size: int = Field(default=1, ge=1)
    """Prompts per SAM3 forward pass; 1 is what the smoke test used."""
    device: int = Field(default=0, ge=0) #cuda:0


class TextConfig(_Section):
    """MemGraphRAG text index. Everything not listed keeps the upstream default."""

    llm_name: str | None = None # LLM for OpenIE, schema and ontology extraction and conflict resolution
    """Name the OpenAI-compatible endpoint serves; still to be decided."""
    llm_base_url: str = ""
    """Empty: the provider default. The API key comes from OPENAI_API_KEY."""
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    """Also the chunking tokenizer. MGRV_TEXT_EMBEDDING_MODEL overrides it
    (e.g. with an offline snapshot path); the same value must be used for
    indexing and retrieval because it names the stores."""
    embedding_batch_size: int = Field(default=8, ge=1)
    chunk_size: int = Field(default=256, ge=1)
    chunk_overlap: int = Field(default=32, ge=0)
    memory_max_workers: int = Field(default=8, ge=1)
    ontology_filter_mode: Literal["absolute", "percentile"] = "absolute"
    ontology_min_frequency: int = Field(default=1, ge=0)
    """Schemas rarer than this are removed (mode absolute)."""
    ontology_low_percent: float = Field(default=20.0, ge=0.0, le=100.0)
    """Share of rarest schemas removed (mode percentile)."""


class RetrievalConfig(_Section):
    top_k_candidates: int = Field(default=20, ge=1)
    """Whole images and crops taken as visual candidates per question."""
    visual_mix: float = Field(default=0.5, ge=0.0, le=1.0)
    """Weight of the image component in the visual seeds (crops get the rest)."""
    beta_grid: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75])
    """Candidate values for tuning; nothing in the pipeline tunes."""
    beta: float | None = Field(default=None, ge=0.0, le=1.0)
    """Text/visual mixing weight: an explicit run setting (``--beta`` overrides)."""
    n_docs: int = Field(default=MAX_DOCS, ge=1, le=MAX_DOCS)
    text_budget_tokens: int = 256
    passage_node_weight: float = Field(default=0.05, ge=0.0)
    """Weight of the dense-passage scores in the text seeds (upstream default)."""
    linking_top_k: int = Field(default=5, ge=1)
    """Entities kept as text seeds (upstream default)."""
    skip_fact_rerank: bool = True
    """Select facts by similarity threshold instead of the LLM reranker.""" # we did not implement the LLM reranker
    fact_similarity_threshold: float = 0.4
    """Raw query-fact similarity a fact needs to be selected."""
    use_raw_threshold_filter: bool = True
    """Threshold all facts, not only the top ``linking_top_k``."""
    damping: float = Field(default=0.5, gt=0.0, lt=1.0)
    """PPR damping factor, as upstream. The restart probability is ``1 - damping``."""


class ExperimentConfig(_Section):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    visual_encoder: VisualEncoderConfig = Field(default_factory=VisualEncoderConfig)
    sam3: Sam3Config = Field(default_factory=Sam3Config)
    text: TextConfig = Field(default_factory=TextConfig)
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
