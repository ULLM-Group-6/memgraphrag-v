import hashlib

import pytest

from memgraphrag_v import ids


def upstream_compute_mdhash_id(content, prefix=""):
    # Verbatim from MemGraphRAG code/src/utils/misc_utils.py
    return prefix + hashlib.md5(content.encode()).hexdigest()


def test_matches_upstream_hashes():
    assert ids.upstream_entity_id("Eiffel Tower") == upstream_compute_mdhash_id("Eiffel Tower", "entity-")
    assert ids.upstream_chunk_id("Some passage.") == upstream_compute_mdhash_id("Some passage.", "chunk-")
    assert ids.upstream_entity_id("x") == "entity-9dd4e461268c8034f5c8564e155c67a6"


def test_derived_ids():
    assert ids.derived_passage_id("doc-7", 3) == "doc-7#p0003"
    assert ids.derived_image_id("doc-7", 12) == "doc-7#img012"
    with pytest.raises(ValueError):
        ids.derived_passage_id("has space", 0)
    with pytest.raises(ValueError):
        ids.derived_image_id("d", -1)


def test_image_node_and_crop_ids_are_deterministic():
    ent = ids.upstream_entity_id("Eiffel Tower")
    assert ids.image_node_id("d1#img000") == ids.image_node_id("d1#img000")
    assert ids.is_hash_id(ids.image_node_id("d1#img000"), ids.IMAGE_NODE_PREFIX)
    a, b = ids.crop_id("d1#img000", ent, 0), ids.crop_id("d1#img000", ent, 1)
    assert a != b and a == ids.crop_id("d1#img000", ent, 0)
    assert ids.is_hash_id(a, ids.CROP_PREFIX)


def test_crop_id_requires_upstream_entity():
    with pytest.raises(ValueError):
        ids.crop_id("d1#img000", "Eiffel Tower", 0)


@pytest.mark.parametrize("bad", ["", " ", "a b", "a\tb"])
def test_native_id_rejects_whitespace(bad):
    with pytest.raises(ValueError):
        ids.check_native_id(bad)
