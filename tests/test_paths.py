from pathlib import Path

import pytest

from memgraphrag_v.paths import ENV_VAR, ArtifactLayout, encoder_slug


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert ArtifactLayout.from_env().root == tmp_path
    assert ArtifactLayout.from_env("/scratch-shared/u/mgrv").root == Path("/scratch-shared/u/mgrv")
    monkeypatch.delenv(ENV_VAR)
    with pytest.raises(RuntimeError):
        ArtifactLayout.from_env()


def test_resolve_and_relative_round_trip(tmp_path):
    layout = ArtifactLayout(tmp_path)
    local = layout.resolve("corpus/images/a.jpg")
    assert local == tmp_path / "corpus" / "images" / "a.jpg"
    assert layout.relative(local) == "corpus/images/a.jpg"
    for bad in ["/etc/passwd", "..\\x", "../x", "C:/x"]:
        with pytest.raises(ValueError):
            layout.resolve(bad)


def test_layout_is_under_root_and_distinct(tmp_path):
    layout = ArtifactLayout(tmp_path)
    paths = [
        layout.text_manifest, layout.image_manifest, layout.crop_manifest, layout.question_manifest,
        layout.images_dir, layout.splits, layout.memgraphrag_dir, layout.passage_map, layout.entities,
        layout.caption_index_dir, layout.descriptions, layout.grounding_attempts, layout.crops_dir,
        layout.masks_dir, layout.graph_dir,
        layout.embeddings_dir("google/siglip2-so400m-patch14-384", "crop"),
        layout.run_config("run-1"), layout.retrieval_output("run-1"),
        layout.reader_output("run-1"), layout.scores_output("run-1"),
    ]
    assert all(tmp_path in p.parents for p in paths)
    assert len(set(paths)) == len(paths)


def test_encoder_slug_and_bad_inputs(tmp_path):
    assert encoder_slug("google/siglip2-so400m-patch14-384") == "google--siglip2-so400m-patch14-384"
    with pytest.raises(ValueError):
        ArtifactLayout(tmp_path).embeddings_dir("x", "text")
    with pytest.raises(ValueError):
        ArtifactLayout(tmp_path).run_dir("../escape")
