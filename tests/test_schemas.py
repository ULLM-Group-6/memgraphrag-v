import pytest
from pydantic import ValidationError

from memgraphrag_v.schemas import (
    MAX_DOCS,
    CaptionRecord,
    Evidence,
    ImageRecord,
    QuestionRecord,
    RetrievalResult,
)

from conftest import IMAGE_ROW, QUESTION_ROW


def test_builder_rows_parse(passage, image, question):
    assert passage.doc_id == "Barack Obama"  # Wikipedia titles contain spaces
    assert image.passage_ids == [passage.passage_id]
    assert question.split == "dev"


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        ImageRecord(**IMAGE_ROW, caption="extra")


def test_empty_id_rejected():
    with pytest.raises(ValidationError):
        ImageRecord(**{**IMAGE_ROW, "image_id": ""})


def test_split_must_be_dev_or_test():
    with pytest.raises(ValidationError):
        QuestionRecord(**{**QUESTION_ROW, "split": "train"})


def test_records_are_frozen(image):
    with pytest.raises(ValidationError):
        image.ok = False


def test_caption_record():
    c = CaptionRecord(image_id="i1", text="A man in a suit.", model="reader-vlm", prompt_version="v1")
    assert c.text


def _evidence(doc, score, image_id=None):
    return Evidence(doc_id=doc, score=score, image_id=image_id, passage_ids=["p"])


def test_retrieval_result_valid():
    r = RetrievalResult(
        question_id="q1",
        system="D",
        route="direct",
        evidence=[_evidence("d1", 0.9, "i1"), _evidence("d2", 0.9, "i2"), _evidence("d3", 0.1)],
    )
    assert [e.doc_id for e in r.evidence] == ["d1", "d2", "d3"]


def test_retrieval_result_rejects_repeated_document():
    with pytest.raises(ValidationError, match="repeat"):
        RetrievalResult(question_id="q1", system="G", route="graph",
                        evidence=[_evidence("d1", 0.9, "i1"), _evidence("d1", 0.5, "i2")])


def test_retrieval_result_rejects_more_than_five_docs():
    evidence = [_evidence(f"d{i}", 1.0 - i / 10) for i in range(MAX_DOCS + 1)]
    with pytest.raises(ValidationError):
        RetrievalResult(question_id="q1", system="T", route="graph", evidence=evidence)


def test_retrieval_result_rejects_unsorted_scores():
    with pytest.raises(ValidationError, match="sorted"):
        RetrievalResult(question_id="q1", system="G", route="graph",
                        evidence=[_evidence("d1", 0.1), _evidence("d2", 0.9)])


def test_retrieval_result_rejects_nonfinite_scores():
    with pytest.raises(ValidationError, match="finite"):
        RetrievalResult(question_id="q1", system="G", route="graph", evidence=[_evidence("d1", float("nan"))])


def test_route_is_checked():
    with pytest.raises(ValidationError):
        RetrievalResult(question_id="q1", system="D", route="teleport", evidence=[])
