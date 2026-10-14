import base64
import io
import json

import pytest
from PIL import Image

from memgraphrag_v import webqa
from memgraphrag_v.io import read_jsonl
from memgraphrag_v.paths import Artifacts
from memgraphrag_v.schemas import CaptionRecord, ImageRecord, PassageRecord, QuestionRecord

THUMB = ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
         "Memphis%2C_TN%2C_Beale_St.jpg/800px-Memphis%2C_TN%2C_Beale_St.jpg")


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (30, 20), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_unpack_images_writes_each_image_once(tmp_path):
    tsv, images = tmp_path / "imgs.tsv", tmp_path / "images"
    encoded = base64.b64encode(_png()).decode()
    tsv.write_text(f"1\t{encoded}\n2\t{encoded}\n\n")

    assert webqa.unpack_images(tsv, images) == {"written": 2}
    assert sorted(p.name for p in images.iterdir()) == ["1.jpg", "2.jpg"]
    # a PNG saved as .jpg still opens, since Pillow goes by the content
    with Image.open(images / "1.jpg") as image:
        assert image.size == (30, 20)

    assert webqa.unpack_images(tsv, images) == {"skipped": 2}


def _write_source(artifacts: Artifacts) -> None:
    src = artifacts.webqa_dir
    src.mkdir(parents=True)
    passages = [
        {"doc_id": "Beale Street", "passage_id": "cap_1", "text": "Beale St.", "source_url": None},
        {"doc_id": "Beale Street", "passage_id": "cap_2", "text": "Beale St. at night", "source_url": None},
        {"doc_id": "Memphis", "passage_id": "txt_q2_0", "text": "Memphis is a city.", "source_url": None},
    ]
    images = [
        {"image_id": "1", "doc_id": "Beale Street", "passage_ids": ["cap_1"], "source_url": THUMB, "commons_url": None},
        {"image_id": "2", "doc_id": "Beale Street", "passage_ids": ["cap_2"], "source_url": THUMB, "commons_url": None},
    ]
    # only image 1 has a generated caption
    captions = [
        {"image_id": "1", "doc_id": "Beale Street", "passage_ids": ["cap_1"], "source_url": THUMB,
         "commons_url": None, "image_title": "Beale Street",
         "generated_description": "A street with neon signs at dusk."},
    ]
    (src / "text_manifest.jsonl").write_text("".join(json.dumps(p) + "\n" for p in passages))
    (src / "image_manifest.jsonl").write_text("".join(json.dumps(i) + "\n" for i in images))
    (src / "caption_manifest.jsonl").write_text("".join(json.dumps(c) + "\n" for c in captions))

    def question(qid, cls, images=(), snippets=(), titles=()):
        return {"qid": qid, "question": f"{qid}?", "answers": ["a"], "Qcate": "YesNo", "topic": "streets",
                "evidence_class": cls, "gold_image_ids": list(images), "gold_snippet_ids": list(snippets),
                "gold_documents": [{"title": t} for t in titles],
                "distractor_image_ids": [], "distractor_snippet_ids": ["q2_9"], "distractor_documents": []}

    build = [question("q1", "image", ["1"], titles=["Beale Street"]),
             question("q2", "text", snippets=["q2_0"], titles=["Memphis"])]
    test = [question("q3", "both", ["1"], ["q2_0"], ["Beale Street", "Memphis"]),
            question("q4", "image", ["2"], titles=["Beale Street"])]
    (src / "build_qa.json").write_text(json.dumps(build))
    (src / "test_qa.json").write_text(json.dumps(test))


def test_manifests(tmp_path):
    artifacts = Artifacts(tmp_path)
    _write_source(artifacts)
    artifacts.images_dir.mkdir(parents=True)
    (artifacts.images_dir / "1.jpg").write_bytes(_png())
    (artifacts.images_dir / "2.jpg").write_bytes(b"not an image")

    counts = webqa.write_manifests(artifacts)

    assert counts["images_ok"] == 1 and counts["gold_images_not_ok"] == 1
    assert counts["generated_captions"] == 1 and counts["images_without_generated_caption"] == 1
    assert len(read_jsonl(artifacts.text_manifest, PassageRecord)) == 3
    images = {i.image_id: i for i in read_jsonl(artifacts.image_manifest, ImageRecord)}
    assert (images["1"].path, images["1"].width, images["1"].ok) == ("1.jpg", 30, True)
    assert not images["2"].ok
    [caption] = read_jsonl(artifacts.captions, CaptionRecord)
    assert (caption.image_id, caption.text) == ("1", "A street with neon signs at dusk.")
    assert caption.model == "Qwen/Qwen2.5-VL-7B-Instruct"
    q = {q.qid: q for q in read_jsonl(artifacts.questions, QuestionRecord)}
    assert [q[i].split for i in ("q1", "q2", "q3", "q4")] == ["dev", "dev", "test", "test"]
    assert q["q3"].gold_passage_ids == ["txt_q2_0"] and q["q3"].modalities == ["image", "text"]
    assert q["q3"].gold_doc_ids == ["Beale Street", "Memphis"]
    assert q["q1"].distractor_passage_ids == ["txt_q2_9"]
    # q1, q2 and q3 share gold sources through q3; q4 stands alone
    assert q["q1"].group_id == q["q2"].group_id == q["q3"].group_id != q["q4"].group_id


def test_missing_image_file_is_not_ok(tmp_path):
    artifacts = Artifacts(tmp_path)
    _write_source(artifacts)
    webqa.write_manifests(artifacts)
    images = read_jsonl(artifacts.image_manifest, ImageRecord)
    assert [i.ok for i in images] == [False, False]
    assert images[0].path == "1.jpg"


def test_broken_gold_reference_is_rejected(tmp_path):
    artifacts = Artifacts(tmp_path)
    _write_source(artifacts)
    test = json.loads((artifacts.webqa_dir / "test_qa.json").read_text())
    test[1]["gold_image_ids"] = ["404"]
    (artifacts.webqa_dir / "test_qa.json").write_text(json.dumps(test))
    with pytest.raises(ValueError, match="gold image 404 not in corpus"):
        webqa.write_manifests(artifacts)


def test_caption_for_unknown_image_is_rejected(tmp_path):
    artifacts = Artifacts(tmp_path)
    _write_source(artifacts)
    path = artifacts.webqa_dir / "caption_manifest.jsonl"
    path.write_text(path.read_text().replace('"image_id": "1"', '"image_id": "999"'))
    with pytest.raises(ValueError, match="generated caption for unknown image 999"):
        webqa.write_manifests(artifacts)
