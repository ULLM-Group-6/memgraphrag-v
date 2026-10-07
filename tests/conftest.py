import pytest

from memgraphrag_v import ids
from memgraphrag_v.schemas import CropRecord, ImageRecord, PassageRecord, QuestionRecord

ENTITY = ids.upstream_entity_id("Eiffel Tower")


@pytest.fixture
def corpus():
    passages = [
        PassageRecord(doc_id="d1", passage_id="d1#p0000", text="The Eiffel Tower is in Paris."),
        PassageRecord(doc_id="d2", passage_id="d2#p0000", text="A bridge."),
    ]
    images = [
        ImageRecord(image_id="d1#img000", doc_id="d1", passage_ids=["d1#p0000"], path="corpus/images/d1_0.jpg"),
        ImageRecord(image_id="d2#img000", doc_id="d2", passage_ids=["d2#p0000"], path="corpus/images/d2_0.jpg"),
    ]
    questions = [
        QuestionRecord(question_id="q1", text="Where is it?", answers=["Paris"], gold_doc_ids=["d1"], split="dev"),
    ]
    crops = [
        CropRecord(
            crop_id=ids.crop_id("d1#img000", ENTITY, 0), image_id="d1#img000", entity_id=ENTITY,
            prompt="Eiffel Tower", bbox=(1, 2, 30, 40), confidence=0.9,
            path="grounding/sam3/crops/a.png", mask_path="grounding/sam3/masks/a.png",
        )
    ]
    return passages, images, questions, crops
