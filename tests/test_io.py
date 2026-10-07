import pytest

from memgraphrag_v import ids
from memgraphrag_v.config import ExperimentConfig
from memgraphrag_v.io import (
    check_config_consistency,
    file_sha256,
    meta_path,
    read_jsonl,
    validate_manifests,
    write_jsonl,
)
from memgraphrag_v.schemas import (
    DescriptionRecord,
    Detection,
    GroundingRecord,
    ImageRecord,
    PassageRecord,
)


def test_round_trip_and_meta(tmp_path, corpus):
    for name, records in zip("piqc", corpus):
        path = tmp_path / f"{name}.jsonl"
        meta = write_jsonl(path, records)
        assert meta["count"] == len(records)
        assert read_jsonl(path, type(records[0])) == records
    assert b"\r\n" not in (tmp_path / "p.jsonl").read_bytes()


def test_tampered_file_detected(tmp_path, corpus):
    path = tmp_path / "p.jsonl"
    write_jsonl(path, corpus[0])
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(corpus[0][0].model_dump_json() + "\n")
    with pytest.raises(ValueError, match="does not match"):
        read_jsonl(path, PassageRecord)


def test_wrong_model_detected(tmp_path, corpus):
    path = tmp_path / "p.jsonl"
    write_jsonl(path, corpus[0])
    with pytest.raises(ValueError):
        read_jsonl(path, ImageRecord)


def test_mixed_types_rejected(tmp_path, corpus):
    with pytest.raises(TypeError):
        write_jsonl(tmp_path / "x.jsonl", [corpus[0][0], corpus[1][0]])


def test_hash_ignores_crlf(tmp_path):
    (tmp_path / "a").write_bytes(b"x\ny\n")
    (tmp_path / "b").write_bytes(b"x\r\ny\r\n")
    assert file_sha256(tmp_path / "a") == file_sha256(tmp_path / "b")


def test_meta_sidecar_name(tmp_path):
    assert meta_path(tmp_path / "text.jsonl").name == "text.jsonl.meta.json"


def test_valid_manifests(tmp_path, corpus):
    passages, images, questions, crops = corpus
    grounding = [attempt(crops[0].entity_id, [detection(crops[0]), dropped(1)])]
    rels = [i.path for i in images] + [c.path for c in crops] + [c.mask_path for c in crops]
    for rel in rels + ["grounding/sam3/masks/1.png"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    report = validate_manifests(passages, images, questions, crops, grounding, artifacts_root=tmp_path)
    assert report.ok and not report.warnings, report
    (tmp_path / "grounding/sam3/masks/1.png").unlink()
    report = validate_manifests(passages, images, questions, crops, grounding, artifacts_root=tmp_path)
    assert any("missing file grounding/sam3/masks/1.png" in e for e in report.errors), report.errors


def test_duplicate_passage_text_warns(corpus):
    passages, images, _, _ = corpus
    copy = PassageRecord(doc_id="d2", passage_id="d2#p0001", text=passages[0].text)
    report = validate_manifests(passages + [copy], images)
    assert report.ok
    text = "\n".join(report.warnings)
    assert "['d1#p0000', 'd2#p0001'] have identical text across docs ['d1', 'd2']" in text
    assert "image d1#img000: passages ['d1#p0000'] share a chunk node" in text


def test_manifest_errors(tmp_path, corpus):
    passages, images, questions, crops = corpus
    bad_images = images + [
        images[0],  # duplicate
        ImageRecord(image_id="d1#img001", doc_id="d1", passage_ids=["missing"], path="x.jpg"),
        ImageRecord(image_id="d1#img002", doc_id="d1", passage_ids=["d2#p0000"], path="y.jpg"),
    ]
    report = validate_manifests(passages, bad_images, questions, crops, artifacts_root=tmp_path)
    text = "\n".join(report.errors)
    assert "duplicate image_id: d1#img000" in text
    assert "unknown passage missing" in text
    assert "belongs to doc d2" in text
    assert "missing file corpus/images/d1_0.jpg" in text
    with pytest.raises(ValueError):
        report.raise_if_errors()


def test_missing_gold_is_a_warning(corpus):
    passages, images, questions, _ = corpus
    q = questions[0].model_copy(update={"gold_doc_ids": ["d1", "d9"], "gold_image_ids": ["nope"]})
    report = validate_manifests(passages, images, [q])
    assert report.ok
    assert any("d9" in w for w in report.warnings)
    assert any("nope" in w for w in report.warnings)


def detection(crop, **kw):
    base = dict(rank=0, bbox=crop.bbox, confidence=crop.confidence, mask_path=crop.mask_path,
                crop_id=crop.crop_id)
    return Detection(**{**base, **kw})


def dropped(rank, confidence=0.6):
    """A kept detection whose crop MG²'s 10 px filter dropped."""
    return Detection(rank=rank, bbox=(0, 0, 5, 5), confidence=confidence,
                     mask_path=f"grounding/sam3/masks/{rank}.png")


def attempt(entity, detections, image_id="d1#img000"):
    return GroundingRecord(image_id=image_id, entity_id=entity, prompt="Eiffel Tower",
                           threshold=0.5, detections=detections)


def test_grounding_consistent_with_crops(corpus):
    passages, images, questions, crops = corpus
    entity = crops[0].entity_id
    other = "entity-" + "1" * 32
    ok = [attempt(entity, [detection(crops[0]), dropped(1)]), attempt(other, [])]
    assert validate_manifests(passages, images, questions, crops, grounding=ok).ok


def _other_crop(c):
    return ids.crop_id(c.image_id, c.entity_id, 1)


@pytest.mark.parametrize("attempts, expected", [
    (lambda c: [attempt(c.entity_id, [detection(c)])] * 2, "prompted more than once"),
    (lambda c: [], "no matching detection"),
    (lambda c: [attempt(c.entity_id, [detection(c), dropped(1).model_copy(update={"crop_id": _other_crop(c)})])],
     "is not in the crop manifest"),
    (lambda c: [attempt(c.entity_id, [detection(c, confidence=0.95)])], "differ from its detection"),
    (lambda c: [attempt(c.entity_id, [], image_id="nope")], "unknown image nope"),
])
def test_grounding_errors(corpus, attempts, expected):
    passages, images, questions, crops = corpus
    report = validate_manifests(passages, images, questions, crops, grounding=attempts(crops[0]))
    assert any(expected in e for e in report.errors), report.errors


def test_config_consistency():
    cfg = ExperimentConfig()
    cfg = cfg.model_copy(update={
        "reader": cfg.reader.model_copy(update={"model": "vlm", "revision": "r1"}),
        "caption": cfg.caption.model_copy(update={"prompt_version": "v1"}),
    })
    desc = DescriptionRecord(image_id="i", text="A chart.", model="vlm", model_revision="r1",
                             prompt_version="v1", max_tokens=256, status="ok")
    entity = ids.upstream_entity_id("Eiffel Tower")
    g = GroundingRecord(image_id="i", entity_id=entity, prompt="x", threshold=0.5, detections=[])
    assert check_config_consistency(cfg, [g], [desc]).ok
    report = check_config_consistency(
        cfg,
        [g.model_copy(update={"threshold": 0.4})],
        [desc.model_copy(update={"model": "other-vlm"})],
    )
    text = "\n".join(report.errors)
    assert "sam3.threshold" in text
    assert "model ['other-vlm']" in text
