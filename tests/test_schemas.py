import pytest
from pydantic import ValidationError

from memgraphrag_v import ids
from memgraphrag_v.schemas import (
    CropRecord,
    DescriptionRecord,
    EntityRecord,
    EvidenceItem,
    GroundingRecord,
    ImageRecord,
    PassageMapRecord,
    QuestionRecord,
    RetrievalResult,
    SeedSummary,
)

ENTITY = ids.upstream_entity_id("Eiffel Tower")
HASH = "0" * 64


def crop(**kw):
    base = dict(crop_id=ids.crop_id("i", ENTITY, 0), image_id="i", entity_id=ENTITY,
                prompt="Eiffel Tower", bbox=(0, 0, 10, 10), confidence=0.7, path="c/a.png")
    return CropRecord(**{**base, **kw})


def evidence(rank, doc, score, image="img"):
    return EvidenceItem(rank=rank, doc_id=doc, score=score, image_id=f"{doc}-{image}")


def result(evidence_items, **kw):
    base = dict(question_id="q1", system="G", route="graph",
                seeds=SeedSummary(text_available=True, image_available=True, crop_available=False, beta=0.5),
                evidence=evidence_items, config_hash=HASH)
    return RetrievalResult(**{**base, **kw})


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        ImageRecord(image_id="i", doc_id="d", passage_ids=[], path="a.jpg", alt="x")


@pytest.mark.parametrize("path", ["/abs/a.jpg", "C:/x/a.jpg", "corpus\\a.jpg", "../a.jpg", ""])
def test_paths_must_be_relative_posix(path):
    with pytest.raises(ValidationError):
        ImageRecord(image_id="i", doc_id="d", passage_ids=[], path=path)


def test_records_are_frozen():
    rec = crop()
    with pytest.raises(ValidationError):
        rec.confidence = 0.1


@pytest.mark.parametrize("kw", [
    dict(confidence=1.2),
    dict(confidence=-0.1),
    dict(bbox=(5, 0, 5, 10)),
    dict(bbox=(-1, 0, 5, 10)),
    dict(bbox=(0, 0, float("nan"), 10)),
    dict(entity_id="Eiffel Tower"),
    dict(crop_id="crop-x"),
])
def test_bad_crops_rejected(kw):
    with pytest.raises(ValidationError):
        crop(**kw)


def test_question_needs_gold_docs_and_answers():
    with pytest.raises(ValidationError):
        QuestionRecord(question_id="q", text="?", answers=["a"], gold_doc_ids=[], split="dev")
    with pytest.raises(ValidationError):
        QuestionRecord(question_id="q", text="?", answers=[], gold_doc_ids=["d"], split="dev")
    with pytest.raises(ValidationError):
        QuestionRecord(question_id="q", text="?", answers=["a"], gold_doc_ids=["d"], split="train")


def test_entity_id_must_hash_from_name():
    EntityRecord(entity_id=ENTITY, name="Eiffel Tower", passage_ids=[])
    with pytest.raises(ValidationError):
        EntityRecord(entity_id=ENTITY, name="eiffel tower", passage_ids=[])


def test_passage_map_needs_chunk_id():
    PassageMapRecord(passage_id="p", doc_id="d", chunk_id=ids.upstream_chunk_id("t"))
    with pytest.raises(ValidationError):
        PassageMapRecord(passage_id="p", doc_id="d", chunk_id=ENTITY)


def grounding(**kw):
    base = dict(image_id="i", entity_id=ENTITY, prompt="x", threshold=0.5, n_kept=2, n_crops=1, max_confidence=0.8)
    return GroundingRecord(**{**base, **kw})


def test_grounding_record_zero_detections_allowed():
    grounding(n_kept=0, n_crops=0, max_confidence=None)
    grounding()


@pytest.mark.parametrize("kw", [
    dict(max_confidence=None),  # kept detections need a confidence
    dict(max_confidence=0.3),  # below threshold
    dict(n_kept=0, n_crops=0),  # confidence without detections
    dict(n_kept=1, n_crops=2),  # more crops than detections
])
def test_bad_grounding_rejected(kw):
    with pytest.raises(ValidationError):
        grounding(**kw)


def test_image_passages_must_be_unique():
    with pytest.raises(ValidationError):
        ImageRecord(image_id="i", doc_id="d", passage_ids=["p", "p"], path="a.jpg")


def test_description_record():
    base = dict(image_id="i", model="m", model_revision="r", prompt_version="v1", max_tokens=256)
    DescriptionRecord(**base, text="A bar chart.", status="ok", output_tokens=4)
    DescriptionRecord(**base, text="", status="failed", note="timeout")
    with pytest.raises(ValidationError):
        DescriptionRecord(**base, text="", status="ok")
    with pytest.raises(ValidationError):
        DescriptionRecord(**base, text="partial", status="failed")


def test_valid_retrieval_result_round_trips():
    r = result([evidence(i + 1, f"d{i}", 1.0 - i / 10) for i in range(5)])
    assert RetrievalResult.model_validate_json(r.model_dump_json()) == r


def test_fallback_with_passages_and_no_seeds():
    # Native-image fallback attaches the passage's first image (method.md §5.2).
    item = EvidenceItem(rank=1, doc_id="d1", score=0.3, passage_id="d1#p0000", image_id="d1#img000")
    result([item], route="fallback", seeds=None, latency_ms=12.5)
    with pytest.raises(ValidationError):
        result([item], latency_ms=-1)


@pytest.mark.parametrize("items", [
    [evidence(i + 1, f"d{i}", 1.0) for i in range(6)],  # 6 documents
    [evidence(1, "d1", 1.0), evidence(2, "d1", 0.5, image="b")],  # same document twice
    [evidence(1, "d1", 1.0), evidence(3, "d2", 0.5)],  # rank gap
    [evidence(1, "d1", 0.1), evidence(2, "d2", 0.5)],  # scores increase
])
def test_bad_retrieval_results_rejected(items):
    with pytest.raises(ValidationError):
        result(items)


def test_evidence_needs_image_or_passage_and_finite_score():
    with pytest.raises(ValidationError):
        EvidenceItem(rank=1, doc_id="d", score=1.0)
    with pytest.raises(ValidationError):
        EvidenceItem(rank=1, doc_id="d", score=float("inf"), image_id="i")


def test_unknown_system_rejected():
    with pytest.raises(ValidationError):
        result([], system="G-full")
