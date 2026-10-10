from pathlib import Path

import pytest
from pydantic import ValidationError

from memgraphrag_v.config import ExperimentConfig
from memgraphrag_v.paths import Artifacts

DEFAULT = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_default_config_loads():
    config = ExperimentConfig.load(DEFAULT)
    assert config.visual_encoder.name == "BAAI/EVA-CLIP-8B"
    assert config.retrieval.n_docs == 5


def test_unresolved_lists_open_settings():
    config = ExperimentConfig.load(DEFAULT)
    assert config.unresolved() == ["sam3.threshold", "retrieval.beta"]
    with pytest.raises(ValueError, match="not frozen"):
        config.require_frozen()


def test_frozen_config_passes():
    config = ExperimentConfig.load(DEFAULT)
    config = config.model_copy(update={
        "sam3": config.sam3.model_copy(update={"threshold": 0.5}),
        "retrieval": config.retrieval.model_copy(update={"beta": 0.5}),
    })
    config.require_frozen()


def test_hash_ignores_machine_and_round_trips(tmp_path):
    config = ExperimentConfig.load(DEFAULT)
    local = config.model_copy(update={"artifacts_root": "/scratch-shared/me/mgrv"})
    assert local.config_hash() == config.config_hash()
    config.save(tmp_path / "config.yaml")
    assert ExperimentConfig.load(tmp_path / "config.yaml").config_hash() == config.config_hash()


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"retrieval": {"top_k": 20}})


def test_artifacts_from_env(tmp_path, monkeypatch, image):
    monkeypatch.setenv("MGRV_ARTIFACTS", str(tmp_path))
    artifacts = Artifacts.from_env()
    assert artifacts.image_manifest == tmp_path / "manifests" / "image_manifest.jsonl"
    assert artifacts.image_path(image) == tmp_path / "corpus" / "images" / "obama.jpg"
    assert artifacts.embeddings_dir("BAAI/EVA-CLIP-8B", "image") == tmp_path / "embeddings" / "eva-clip-8b" / "image"
    with pytest.raises(ValueError):
        artifacts.embeddings_dir("BAAI/EVA-CLIP-8B", "images")
    with pytest.raises(ValueError):
        artifacts.run_dir("../escape")


def test_artifacts_root_required(monkeypatch):
    monkeypatch.delenv("MGRV_ARTIFACTS", raising=False)
    with pytest.raises(RuntimeError):
        Artifacts.from_env()
