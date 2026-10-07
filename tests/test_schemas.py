import pytest
from pydantic import ValidationError

from memgraphrag_v import ids
from memgraphrag_v.schemas import (
    CaptionChunkMapRecord,
    CropRecord,
    DescriptionRecord,
    Detection,
    DiagnosticsRecord,
    EntityRecord,
    EvidenceItem,
    GroundingRecord,
    ImageRecord,
    PassageMapRecord,
    QuestionRecord,
    RetrievalResult,
    ScoredItem,
    SeedSummary,
    SeedWeight,
)

ENTITY = ids.upstream_entity_id("Eiffel Tower")
HASH = "0" * 64


def crop(**kw):
    base = dict(crop_id=ids.crop_id("i", ENTITY, 0), image_id="i", entity_id=ENTITY,
                prompt="Eiffel Tower", bbox=(0, 0, 10, 10), confidence=0.7, path="c/a.png",
                mask_path="m/a.png")
    return CropRecord(**{**base, **kw})


def evidence(rank, doc, score, image="img"):
    return EvidenceItem(rank=rank, doc_id=doc, score=score, image_id=f"{doc}-{image}")


def seeds(text=True, image=True, crop=False):
    beta = 0.5 if text and (image or crop) else None
    return SeedSummary(text_used=text, image_used=image, crop_used=crop, beta=beta)


def result(evidence_items, **kw):
    base = dict(question_id="q1", system="G", selection="image", route="graph",
                seeds=seeds(), evidence=evidence_items, config_hash=HASH)
    return RetrievalResult(**{**base, **kw})


def passage_item(doc="d1"):
    return EvidenceItem(rank=1, doc_id=doc, score=0.3, passage_id=f"{doc}#p0000")


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
    dict(mask_path=None),  # masks are always saved (§3.3)
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


def detection(rank, confidence, saved=True, image="i"):
    return Detection(rank=rank, bbox=(0, 0, 10, 10), confidence=confidence, mask_path=f"m/{rank}.png",
                     crop_id=ids.crop_id(image, ENTITY, rank) if saved else None)


def grounding(detections, **kw):
    base = dict(image_id="i", entity_id=ENTITY, prompt="x", threshold=0.5, detections=detections)
    return GroundingRecord(**{**base, **kw})


def test_grounding_record_zero_detections_allowed():
    g = grounding([])
    assert (g.n_kept, g.n_crops, g.max_confidence) == (0, 0, None)


def test_dropped_crop_keeps_its_detection():
    # Rank 1 was too small for a crop; its box and mask survive and rank 2's
    # crop keeps its own rank, so saved crop ranks skip 1.
    g = grounding([detection(0, 0.9), detection(1, 0.8, saved=False), detection(2, 0.6)])
    assert (g.n_kept, g.n_crops, g.max_confidence) == (3, 2, 0.9)
    assert g.detections[2].crop_id == ids.crop_id("i", ENTITY, 2)


@pytest.mark.parametrize("detections", [
    [detection(0, 0.3)],  # below threshold
    [detection(0, 0.6), detection(1, 0.9)],  # not sorted by confidence
    [detection(1, 0.9)],  # ranks must start at 0
    [detection(0, 0.9, image="other")],  # crop_id of another image
])
def test_bad_grounding_rejected(detections):
    with pytest.raises(ValidationError):
        grounding(detections)


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
    result([passage_item()], route="fallback", seeds=seeds(False, False, False))
    with pytest.raises(ValidationError):
        result([item], latency_ms=-1)
    with pytest.raises(ValidationError):  # fallback means no seed component was available
        result([passage_item()], route="fallback", seeds=seeds(text=False))
    with pytest.raises(ValidationError):  # fallback evidence is a passage
        result([evidence(1, "d1", 1.0)], route="fallback", seeds=None)


def test_g0_runs_with_either_selection():
    result([evidence(1, "d1", 1.0)], system="G0", seeds=seeds(image=False))
    result([passage_item()], system="G0", selection="passage", seeds=seeds(image=False))


@pytest.mark.parametrize("kw", [
    dict(system="T"),  # T ranks passages
    dict(system="D", selection="passage", route="direct", seeds=None),
    dict(system="G", selection="passage"),
])
def test_system_selection_rules(kw):
    with pytest.raises(ValidationError):
        result([passage_item()], **kw)


@pytest.mark.parametrize("kw", [
    dict(system="G0"),  # G0 must not use visual seeds
    dict(system="G-image", seeds=seeds(crop=True)),
    dict(system="G-ground", seeds=seeds(image=False, crop=True)),
    dict(seeds=seeds(False, False, False)),  # graph route with nothing seeded
    dict(seeds=None),  # graph route must log its seeds
    dict(system="D", route="direct"),  # D builds no seeds
    dict(system="D", route="graph", seeds=None),
    dict(route="direct"),  # only D retrieves directly
])
def test_seed_and_route_rules(kw):
    with pytest.raises(ValidationError):
        result([evidence(1, "d1", 1.0)], **kw)


def test_d_and_g_caption():
    result([evidence(1, "d1", 1.0)], system="D", route="direct", seeds=None)
    result([evidence(1, "d1", 1.0)], system="G-caption", seeds=None, latency_ms=None)


def test_image_selection_needs_image_ids():
    with pytest.raises(ValidationError):
        result([passage_item()])
    with pytest.raises(ValidationError):
        result([evidence(1, "d1", 1.0)], system="T", selection="passage", seeds=seeds(image=False))


def test_beta_only_when_fused():
    with pytest.raises(ValidationError):
        SeedSummary(text_used=True, image_used=False, crop_used=False, beta=0.5)
    with pytest.raises(ValidationError):
        SeedSummary(text_used=True, image_used=True, crop_used=False)
    SeedSummary(text_used=False, image_used=True, crop_used=True)


def test_caption_chunk_map_needs_a_source():
    chunk = ids.upstream_chunk_id("A bar chart.")
    CaptionChunkMapRecord(chunk_id=chunk, doc_id="d1", image_id="d1#img000")
    CaptionChunkMapRecord(chunk_id=chunk, doc_id="d1", passage_id="d1#p0000", image_id="d1#img000")
    with pytest.raises(ValidationError):
        CaptionChunkMapRecord(chunk_id=chunk, doc_id="d1")


def test_diagnostics_record():
    rec = DiagnosticsRecord(
        question_id="q1", system="G", selection="image",
        image_candidates=[ScoredItem(item_id="d1#img000", score=0.4), ScoredItem(item_id="d2#img000", score=0.1)],
        seeds=[SeedWeight(node_id=ENTITY, channel="text", weight=0.3),
               SeedWeight(node_id=ENTITY, channel="crop", weight=0.2)],
        ppr_ranking=[ScoredItem(item_id="d2#img000", score=0.02)],
    )
    assert DiagnosticsRecord.model_validate_json(rec.model_dump_json()) == rec
    with pytest.raises(ValidationError):
        rec.model_validate({**rec.model_dump(), "ppr_ranking": [
            {"item_id": "a", "score": 0.1}, {"item_id": "b", "score": 0.2}]})
    with pytest.raises(ValidationError):
        SeedWeight(node_id=ENTITY, channel="text", weight=0.0)


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
