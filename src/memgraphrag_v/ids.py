"""Stable identifiers.

Benchmark IDs (documents, questions, and passages/images when the benchmark
has them) are kept verbatim. MemGraphRAG's own IDs are content hashes
(``compute_mdhash_id`` in upstream ``utils/misc_utils.py``); we reproduce them
only to check and join against upstream outputs, never to create entities.
"""

from __future__ import annotations

import hashlib
import re

ENTITY_PREFIX = "entity-"
CHUNK_PREFIX = "chunk-"
IMAGE_NODE_PREFIX = "image-"
CROP_PREFIX = "crop-"

_NATIVE_ID = re.compile(r"^\S+$")
_HASH_ID = re.compile(r"^(entity|chunk|image|crop)-[0-9a-f]{32}$")


def _md5(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def check_native_id(value: str) -> str:
    """Validate a benchmark-native ID: non-empty, no whitespace."""
    if not isinstance(value, str) or not _NATIVE_ID.match(value):
        raise ValueError(f"invalid ID {value!r}: must be non-empty without whitespace")
    return value


def derived_passage_id(doc_id: str, index: int) -> str:
    """Passage ID for benchmarks without one; ``index`` is 0-based source order."""
    check_native_id(doc_id)
    if index < 0:
        raise ValueError("index must be >= 0")
    return f"{doc_id}#p{index:04d}"


def derived_image_id(doc_id: str, index: int) -> str:
    """Image ID for benchmarks without one; ``index`` is 0-based source order."""
    check_native_id(doc_id)
    if index < 0:
        raise ValueError("index must be >= 0")
    return f"{doc_id}#img{index:03d}"


def upstream_entity_id(name: str) -> str:
    """MemGraphRAG's entity node ID for an entity name (check/join only)."""
    return ENTITY_PREFIX + _md5(name)


def upstream_chunk_id(passage_text: str) -> str:
    """MemGraphRAG's passage node ID for the exact passage text it indexed."""
    return CHUNK_PREFIX + _md5(passage_text)


def image_node_id(image_id: str) -> str:
    """Graph vertex name of a whole-image node."""
    return IMAGE_NODE_PREFIX + _md5(check_native_id(image_id))


def crop_id(image_id: str, entity_id: str, rank: int) -> str:
    """ID of a saved crop record. Crops are never graph vertices.

    ``rank`` is the 0-based position of the detection among those kept for this
    (image, entity) pair, ordered by confidence descending (ties: box order).
    """
    check_native_id(image_id)
    if not is_hash_id(entity_id, ENTITY_PREFIX):
        raise ValueError(f"not an upstream entity ID: {entity_id!r}")
    if rank < 0:
        raise ValueError("rank must be >= 0")
    return CROP_PREFIX + _md5(f"{image_id}|{entity_id}|{rank}")


def is_hash_id(value: str, prefix: str | None = None) -> bool:
    """True if ``value`` is a prefixed 32-hex-digit hash ID (optionally of ``prefix``)."""
    if not isinstance(value, str) or not _HASH_ID.match(value):
        return False
    return prefix is None or value.startswith(prefix)
