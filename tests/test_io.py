import pytest

from memgraphrag_v.io import file_sha256, meta_path, read_jsonl, validate_manifests, write_jsonl
from memgraphrag_v.schemas import GroundingRecord, ImageRecord, PassageRecord


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
    for rel in [i.path for i in images] + [c.path for c in crops]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    report = validate_manifests(passages, images, questions, crops, artifacts_root=tmp_path)
    assert report.ok and not report.warnings, report


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


def attempt(entity, **kw):
    base = dict(image_id="d1#img000", entity_id=entity, prompt="Eiffel Tower", threshold=0.5,
                n_kept=1, n_crops=1, max_confidence=0.9)
    return GroundingRecord(**{**base, **kw})


def test_grounding_consistent_with_crops(corpus):
    passages, images, questions, crops = corpus
    entity = crops[0].entity_id
    other = "entity-" + "1" * 32
    ok = [attempt(entity), attempt(other, n_kept=0, n_crops=0, max_confidence=None)]
    assert validate_manifests(passages, images, questions, crops, grounding=ok).ok


@pytest.mark.parametrize("attempts, expected", [
    (lambda e: [attempt(e), attempt(e)], "prompted more than once"),
    (lambda e: [], "no grounding attempt"),
    (lambda e: [attempt(e, n_kept=3, n_crops=2)], "1 crops but n_crops=2"),
    (lambda e: [attempt(e, max_confidence=0.6)], "outside"),
    (lambda e: [attempt(e, image_id="nope")], "unknown image nope"),
])
def test_grounding_errors(corpus, attempts, expected):
    passages, images, questions, crops = corpus
    report = validate_manifests(passages, images, questions, crops, grounding=attempts(crops[0].entity_id))
    assert any(expected in e for e in report.errors), report.errors
