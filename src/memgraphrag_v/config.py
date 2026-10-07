"""Experiment configuration (method.md §8).

One YAML file pins every setting. Values still being decided in [02] (and
revisions pinned in [04], [05], [12]) are ``null`` until resolved;
``require_frozen()`` refuses to start a test evaluation while any remain.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import MAX_DOCS


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(_Section):
    name: str | None = None  # [03]
    revision: str | None = None
    corpus_sha256: str | None = None  # [09] sha256 of manifests/text.jsonl + images.jsonl
    split_sha256: str | None = None  # [10]
    split_seed: int = 42
    n_dev: int = 100
    n_test: int = 400
    distractor_set: str | None = None  # [03]/[09] source and rule of the fixed distractors


class UpstreamConfig(_Section):
    memgraphrag_repo: str = "https://github.com/XMUDeepLIT/MemGraphRAG"
    memgraphrag_commit: str | None = None  # [04]
    mg2_repo: str = "https://github.com/Daboolu/MG2-RAG"
    mg2_commit: str | None = None  # [13] source of the SAM3 wrapper and crop code


class TextIndexConfig(_Section):
    llm: str | None = None  # [02]
    text_encoder: str | None = None  # [02]; also used by the dense-passage fallback
    chunking: str | None = None  # [04]/[11] e.g. one benchmark passage per chunk, no re-splitting
    upstream_settings: dict[str, Any] | None = None
    """[04]/[11] The upstream BaseConfig used for indexing, minus machine paths:
    extraction, ontology filtering, conflict resolution, linking_top_k,
    passage_node_weight and so on."""


class VisualEncoderConfig(_Section):
    name: str = "google/siglip2-so400m-patch14-384"  # or EVA-CLIP, pending [05]/[12]
    model_revision: str | None = None
    processor_revision: str | None = None


class Sam3Config(_Section):
    checkpoint: str | None = None  # [05]
    revision: str | None = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    crop_preprocessing: str | None = None  # [13] masking/background treatment, frozen


class RetrievalConfig(_Section):
    top_k_candidates: int = Field(default=20, ge=1)
    visual_mix: float = Field(default=0.5, ge=0.0, le=1.0)
    """Weight of the image component in ŝ_visual (crop gets 1 − visual_mix)."""
    beta_grid: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75])
    beta: float | None = Field(default=None, ge=0.0, le=1.0)  # tuned on dev
    ppr_restart_alpha: float | None = Field(default=None, gt=0.0, lt=1.0)  # [02]
    """Restart probability α of method.md §5.3. Upstream passes igraph's
    ``damping`` (probability of following an edge), so damping = 1 − α."""
    n_docs: int = Field(default=MAX_DOCS, ge=1, le=MAX_DOCS)
    text_budget_tokens: int = 256
    text_budget_tokenizer: str | None = None  # [02] tokenizer the 256-token budgets are counted with
    fallback_top_k: int | None = Field(default=None, ge=1)  # [02] passages from upstream dense_passage_retrieval
    diagnostics_top_n: int = Field(default=100, ge=1)
    """Candidates and PPR ranks kept per question in diagnostics.jsonl."""


class CaptionConfig(_Section):
    description_max_tokens: int = 256
    prompt_version: str | None = None  # [07]


class ReaderConfig(_Section):
    model: str | None = None  # [02]
    revision: str | None = None
    prompt_version: str | None = None  # [17]
    max_new_tokens: int | None = None  # [02]
    temperature: float = 0.0
    image_settings: str | None = None  # [02]


class JudgeConfig(_Section):
    model: str | None = None  # [02]
    rubric_version: str | None = None  # [02]


class EvaluationConfig(_Section):
    recall_k: int = MAX_DOCS
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 42
    scorer_version: str | None = None  # [03]/[15]


# Machine-specific, so excluded from the hash and from require_frozen().
_LOCAL_FIELDS = {"artifacts_root"}


class ExperimentConfig(_Section):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    text_index: TextIndexConfig = Field(default_factory=TextIndexConfig)
    visual_encoder: VisualEncoderConfig = Field(default_factory=VisualEncoderConfig)
    sam3: Sam3Config = Field(default_factory=Sam3Config)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    caption: CaptionConfig = Field(default_factory=CaptionConfig)
    reader: ReaderConfig = Field(default_factory=ReaderConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    artifacts_root: str | None = None
    """Leave null in committed configs; MGRV_ARTIFACTS supplies it per machine."""

    @model_validator(mode="after")
    def _damping_matches_alpha(self) -> ExperimentConfig:
        # Upstream's run_ppr hands `damping` to igraph's personalized_pagerank,
        # where it is the edge-following probability, i.e. 1 − α.
        alpha = self.retrieval.ppr_restart_alpha
        damping = (self.text_index.upstream_settings or {}).get("damping")
        if alpha is not None and damping is not None and not math.isclose(damping, 1 - alpha):
            raise ValueError(
                f"upstream damping {damping} must equal 1 - ppr_restart_alpha ({1 - alpha})"
            )
        return self

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
        missing = []
        for section, value in self.model_dump(mode="json").items():
            if section in _LOCAL_FIELDS:
                continue
            if isinstance(value, dict):
                missing += [f"{section}.{k}" for k, v in value.items() if v is None]
            elif value is None:
                missing.append(section)
        return missing

    def require_frozen(self) -> None:
        missing = self.unresolved()
        if missing:
            raise ValueError("config is not frozen; unresolved: " + ", ".join(missing))

    def config_hash(self) -> str:
        """sha256 of the canonical JSON, independent of key order and machine."""
        data = self.model_dump(mode="json", exclude=_LOCAL_FIELDS)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
