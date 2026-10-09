import json

import pytest

from memgraphrag_v.io import read_jsonl, write_jsonl
from memgraphrag_v.schemas import ImageRecord, PassageRecord

from conftest import IMAGE_ROW


def test_round_trip(tmp_path, passage):
    path = tmp_path / "manifests" / "text_manifest.jsonl"
    assert write_jsonl(path, [passage, passage]) == 2
    assert read_jsonl(path, PassageRecord) == [passage, passage]
    assert b"\r\n" not in path.read_bytes()


def test_reads_builder_output_unchanged(tmp_path):
    # The builder writes with json.dumps(ensure_ascii=False); read it as is.
    path = tmp_path / "image_manifest.jsonl"
    path.write_text(json.dumps(IMAGE_ROW, ensure_ascii=False) + "\n\n", encoding="utf-8")
    assert read_jsonl(path, ImageRecord) == [ImageRecord(**IMAGE_ROW)]


def test_bad_line_names_file_and_line(tmp_path, passage):
    path = tmp_path / "text_manifest.jsonl"
    path.write_text(passage.model_dump_json() + "\n" + '{"doc_id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"text_manifest\.jsonl:2"):
        read_jsonl(path, PassageRecord)
