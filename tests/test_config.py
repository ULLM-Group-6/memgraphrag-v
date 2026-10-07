from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from memgraphrag_v.config import ExperimentConfig

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_yaml_matches_model_defaults():
    assert ExperimentConfig.load(DEFAULT) == ExperimentConfig()


def test_known_method_values():
    cfg = ExperimentConfig.load(DEFAULT)
    assert cfg.sam3.threshold == 0.5
    assert cfg.retrieval.top_k_candidates == 20
    assert cfg.retrieval.beta_grid == [0.25, 0.5, 0.75]
    assert cfg.retrieval.n_docs == 5
    assert cfg.evaluation.bootstrap_resamples == 2000


def test_require_frozen_lists_open_settings():
    cfg = ExperimentConfig.load(DEFAULT)
    missing = cfg.unresolved()
    assert "retrieval.ppr_restart_alpha" in missing
    assert "reader.model" in missing
    for name in ["text_index.chunking", "text_index.upstream_settings", "retrieval.fallback_top_k",
                 "retrieval.text_budget_tokenizer", "dataset.distractor_set"]:
        assert name in missing
    assert "artifacts_root" not in missing
    with pytest.raises(ValueError, match="ppr_restart_alpha"):
        cfg.require_frozen()


def test_fully_set_config_is_frozen():
    data = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    fill = {"ppr_restart_alpha": 0.5, "beta": 0.5, "max_new_tokens": 512, "fallback_top_k": 50,
            "upstream_settings": {"linking_top_k": 5}}
    for section in data.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if value is None:
                    section[key] = fill.get(key, "x")
    ExperimentConfig.model_validate(data).require_frozen()


def test_hash_ignores_key_order_and_machine():
    a = ExperimentConfig.load(DEFAULT)
    data = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    reordered = dict(reversed(list(data.items())))
    reordered["artifacts_root"] = "/scratch-shared/u/mgrv"
    assert ExperimentConfig.model_validate(reordered).config_hash() == a.config_hash()
    changed = a.model_copy(update={"retrieval": a.retrieval.model_copy(update={"beta": 0.5})})
    assert changed.config_hash() != a.config_hash()


def test_save_load_round_trip(tmp_path):
    cfg = ExperimentConfig.load(DEFAULT)
    cfg.save(tmp_path / "run" / "config.yaml")
    assert ExperimentConfig.load(tmp_path / "run" / "config.yaml") == cfg


def test_unknown_keys_and_bad_values_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"retrieval": {"top_k": 20}})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"retrieval": {"n_docs": 6}})
