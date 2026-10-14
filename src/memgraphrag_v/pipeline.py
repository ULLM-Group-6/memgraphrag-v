"""The combined MemGraphRAG-V pipeline: one multimodal index, then retrieval.

    python -m memgraphrag_v.pipeline index    [--stages text ground embed assemble] [--force]
    python -m memgraphrag_v.pipeline retrieve --run-id ID [--beta B] [--split dev|test]

``index_with_memory_v`` builds **one** graph from corpus inputs only (passages,
captions, images). Its stages, each saving its outputs and a stamp of its
inputs and settings in ``index/manifest.json``:

1. ``text``: normalize passages and supplied captions into text records,
   chunk within each record, run MemGraphRAG's triple extraction, memory
   schema, ontology filter and conflict detection/resolution, then export the
   surviving entities and their supporting chunks from the final memory.
2. ``ground``: SAM3 prompts each image with the entities of its explicitly
   associated chunks (``sam3_processing.py``).
3. ``embed``: EVA-CLIP embeds whole images and crops (``visual_indexing.py``).
4. ``assemble``: upstream ``build_memory_graph`` / ``install_memory_graph``,
   then image vertices and image-passage and image-entity edges; the finished
   graph is saved once, together with the stores and memory it belongs to.

Stages can run in separate environments (``--stages``): MemGraphRAG pins
torch 2.5.1 and transformers 4.45.2, while MG2's SAM3 environment and the
EVA-CLIP setup use torch 2.7.0 and transformers 4.49.0. ``text`` and
``assemble`` need the upstream code; ``ground`` and ``embed`` need MG2/SAM3
and EVA-CLIP; they exchange only the saved artifacts. A stage whose inputs or
settings changed raises ``StaleIndexError`` instead of silently reusing
results; ``--force`` recomputes it.

``load_index`` loads the saved index without rebuilding or grounding anything,
and ``retrieve`` (``retrieval.py``) answers questions from question text and
ID alone.

Upstream MemGraphRAG
--------------------
The text backend is the complete MemGraphRAG at ``MEMGRAPHRAG_COMMIT``,
imported from a checkout (``$MGRV_MEMGRAPHRAG``; its ``code/`` folder is put
on ``sys.path`` because upstream's package is the generic ``src``). Nothing is
imported at module import. ``MVBackend`` subclasses it, and each method it
overrides or adds fixes one upstream behaviour that conflicts with the
requirements; there is no runtime patching:

- ``insert_chunks``: upstream ``insert_strings`` hashes the text, so equal
  text in two records collapses and the caller's IDs are lost. Chunks are
  inserted under their source-aware IDs. The semantic text is never changed
  (upstream's ``index.py`` prefixes chunks with ``<n>:``; not done here).
- ``load_existing_openie`` and ``triple_extraction_chunks``: upstream re-hashes
  cached OpenIE entries from their text and processes new chunks in set order.
  The cache is matched by chunk ID and text, and results stay in source order.
- ``build_entity_to_triples_index``: upstream re-hashes chunk IDs again; the
  index is only for ``get_triples_by_entity``, which is not used.
- ``save_igraph``: upstream saves the text-only graph on installation. Saving
  is deferred until ``write_graph`` stores the completed multimodal graph.
- ``get_top_k_weights``: the same logic without its final assertion, which
  fails when a selected entity has weight zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from . import retrieval, sam3_processing, visual_indexing
from .config import ExperimentConfig, TextConfig
from .embeddings import EmbeddingSet, Encoder
from .io import read_jsonl, write_jsonl
from .paths import MEMGRAPHRAG_ENV, MG2_ENV, TEXT_EMBEDDING_ENV, Artifacts, external_path
from .retrieval import Question, RetrievalIndex
from .schemas import (
    CaptionRecord,
    ChunkRecord,
    EntityRecord,
    ImageRecord,
    IndexedImage,
    PassageRecord,
    QueryDiagnostics,
    QuestionRecord,
    RetrievalResult,
    TextRecord,
)

logger = logging.getLogger(__name__)

MEMGRAPHRAG_COMMIT = "49f20cec41290c95209724d33ec527f72449d300"
STAGES = ("text", "ground", "embed", "assemble")


class StaleIndexError(RuntimeError):
    """Saved results do not match the current inputs, settings or each other."""


# --- Source records ----------------------------------------------------------


def normalize_texts(
    passages: Sequence[PassageRecord], captions: Sequence[CaptionRecord], images: Sequence[ImageRecord]
) -> list[TextRecord]:
    """Passages in manifest order, then supplied captions in manifest order, as
    one kind of text record. A caption's ID is ``caption-<image_id>-<n>`` and
    its document is its parent image's. Nothing is generated."""
    image_by_id: dict[str, ImageRecord] = {}
    for image in images:
        if image.image_id in image_by_id:
            raise ValueError(f"duplicate image_id {image.image_id!r}")
        image_by_id[image.image_id] = image
    texts: list[TextRecord] = []
    seen: set[str] = set()

    def add(record: TextRecord) -> None:
        if record.record_id in seen:
            raise ValueError(f"duplicate text record ID {record.record_id!r}")
        seen.add(record.record_id)
        texts.append(record)

    for p in passages:
        add(TextRecord(record_id=p.passage_id, kind="passage", doc_id=p.doc_id, text=p.text))
    ordinal: Counter[str] = Counter()
    for c in captions:
        image = image_by_id.get(c.image_id)
        if image is None:
            raise ValueError(f"caption for unknown image {c.image_id!r}")
        add(TextRecord(
            record_id=f"caption-{c.image_id}-{ordinal[c.image_id]}", kind="caption",
            doc_id=image.doc_id, text=c.text, image_id=c.image_id,
        ))
        ordinal[c.image_id] += 1
    return texts


def split_text(text: str, tokenizer, chunk_size: int, overlap: int) -> list[str]:
    """Token windows, the algorithm of upstream ``code/index.py::split_text``."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
    previous_limit = getattr(tokenizer, "model_max_length", None)
    try:
        tokenizer.model_max_length = int(1e9)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    finally:
        if previous_limit is not None:
            tokenizer.model_max_length = previous_limit
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + chunk_size]
        if not window:
            break
        chunks.append(tokenizer.decode(window, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        if start + chunk_size >= len(token_ids):
            break
    return chunks


def chunk_texts(texts: Sequence[TextRecord], tokenizer, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    """Chunk within each record, never across records, keeping source order.
    A chunk ID hashes (record ID, position, text): equal text in different
    records gets different IDs, and the same input always the same ID."""
    chunks: list[ChunkRecord] = []
    seen: set[str] = set()
    for record in texts:
        pieces = [p for p in split_text(record.text, tokenizer, chunk_size, overlap) if p.strip()]
        for position, piece in enumerate(pieces):
            payload = json.dumps([record.record_id, position, piece], ensure_ascii=False)
            chunk_id = "chunk-" + hashlib.md5(payload.encode("utf-8")).hexdigest()
            if chunk_id in seen:
                raise ValueError(f"duplicate chunk ID for record {record.record_id!r}")
            seen.add(chunk_id)
            chunks.append(ChunkRecord(
                chunk_id=chunk_id, record_id=record.record_id, doc_id=record.doc_id, position=position, text=piece,
            ))
    return chunks


def link_images(
    images: Sequence[ImageRecord], texts: Sequence[TextRecord], chunks: Sequence[ChunkRecord]
) -> list[IndexedImage]:
    """The text each image is explicitly associated with: the passages in its
    ``passage_ids`` and its own captions, in source order, and their chunks. A
    shared document does not link an image to every passage of the document.
    An unknown passage ID is an error."""
    position = {t.record_id: i for i, t in enumerate(texts)}
    kinds = {t.record_id: t.kind for t in texts}
    captions: dict[str, list[str]] = defaultdict(list)
    for t in texts:
        if t.kind == "caption":
            captions[t.image_id].append(t.record_id)
    by_record: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        by_record[chunk.record_id].append(chunk.chunk_id)
    links = []
    for image in images:
        explicit = list(dict.fromkeys(image.passage_ids))
        unknown = [p for p in explicit if kinds.get(p) != "passage"]
        if unknown:
            raise ValueError(f"image {image.image_id} references unknown passages {unknown[:3]}")
        record_ids = sorted(set(explicit) | set(captions.get(image.image_id, ())), key=position.__getitem__)
        links.append(IndexedImage(
            image_id=image.image_id, doc_id=image.doc_id, path=image.path, record_ids=record_ids,
            chunk_ids=[c for r in record_ids for c in by_record.get(r, ())], usable=image.ok,
        ))
    return links


def export_entities(memory, chunk_order: dict[str, int], entity_id: Callable[[str], str]) -> list[EntityRecord]:
    """Surviving entities of the final memory (subjects and objects of its
    facts) and the chunks supporting them, from the memory itself and not
    from the raw OpenIE results, which still hold removed facts."""
    chunk_of = {p.idx: p.chunk_id for p in memory.passage_layer}
    names: dict[str, str] = {}
    support: dict[str, set[str]] = {}
    for fact in memory.fact_layer:
        if len(fact.content) != 3:
            continue
        chunks = {chunk_of[i] for i in fact.passage_indices if i in chunk_of}
        for name in (str(fact.content[0]), str(fact.content[2])):
            eid = entity_id(name)
            names.setdefault(eid, name)
            support.setdefault(eid, set()).update(chunks)
    unknown = {c for chunks in support.values() for c in chunks} - set(chunk_order)
    if unknown:
        raise ValueError(f"the final memory refers to chunks outside the source: {sorted(unknown)[:3]}")
    return [
        EntityRecord(entity_id=eid, name=names[eid], chunk_ids=sorted(support[eid], key=chunk_order.__getitem__))
        for eid in names
    ]


def _read_sources(artifacts: Artifacts) -> tuple[list[PassageRecord], list[CaptionRecord], list[ImageRecord]]:
    passages = read_jsonl(artifacts.text_manifest, PassageRecord)
    captions = read_jsonl(artifacts.captions, CaptionRecord) if artifacts.captions.is_file() else []
    return passages, captions, read_jsonl(artifacts.image_manifest, ImageRecord)


# --- Upstream text backend ---------------------------------------------------


@lru_cache(maxsize=None)
def _upstream() -> SimpleNamespace:
    """Import upstream MemGraphRAG from its checkout. Its package is the
    generic ``src``, so a different ``src`` already imported is an error."""
    root = external_path(MEMGRAPHRAG_ENV, f"MemGraphRAG checkout at {MEMGRAPHRAG_COMMIT[:8]}")
    code_dir = (root / "code").resolve()
    if not (code_dir / "src" / "MemGraphRAG.py").is_file():
        raise FileNotFoundError(f"{code_dir / 'src' / 'MemGraphRAG.py'} not found: {MEMGRAPHRAG_ENV} must be the repository root")
    existing = sys.modules.get("src")
    if existing is not None and Path(getattr(existing, "__file__", "") or "").resolve().parent != (code_dir / "src"):
        raise RuntimeError("a different top-level 'src' package is already imported; run this in a fresh process")
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from src.MemGraphRAG import MemGraphRAG
    from src.Memory import ThreeLayerMemory
    from src.utils.config_utils import BaseConfig
    from src.utils.misc_utils import compute_mdhash_id

    return SimpleNamespace(
        MemGraphRAG=MemGraphRAG, ThreeLayerMemory=ThreeLayerMemory, BaseConfig=BaseConfig,
        compute_mdhash_id=compute_mdhash_id,
    )


@lru_cache(maxsize=None)
def _backend_class():
    up = _upstream()

    class MVBackend(up.MemGraphRAG):
        """Upstream MemGraphRAG with the adaptations listed in the module docstring."""

        _graph_saving = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._chunk_text: dict[str, str] = {}

        def hash_id(self, content: str, prefix: str) -> str:
            return up.compute_mdhash_id(content, prefix=prefix)

        def entity_id(self, name: str) -> str:
            return self.hash_id(name, "entity-")

        def insert_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
            store = self.chunk_embedding_store
            self._chunk_text = {c.chunk_id: c.text for c in chunks}
            for c in chunks:
                row = store.hash_id_to_row.get(c.chunk_id)
                if row is not None and row["content"] != c.text:
                    raise StaleIndexError(f"chunk store row {c.chunk_id} holds different text")
            new = [c for c in chunks if c.chunk_id not in store.hash_id_to_row]
            if new:
                texts = [c.text for c in new]
                store._upsert([c.chunk_id for c in new], texts, self.embedding_model.batch_encode(texts))
            self.check_chunks(chunks)

        def check_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
            stale = set(self.chunk_embedding_store.hash_ids) - {c.chunk_id for c in chunks}
            if stale:
                raise StaleIndexError(
                    f"the chunk store holds {len(stale)} chunks that are not in the source: rerun the text stage with --force"
                )

        def load_existing_openie(self, chunk_keys):
            cached: list[dict] = []
            path = self.openie_results_path
            if not self.global_config.force_openie_from_scratch and os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    for info in json.load(f).get("docs", []):
                        text = self._chunk_text.get(info["idx"])
                        if text is None:
                            continue  # a chunk that is no longer in the source
                        if info["passage"] != text:
                            raise StaleIndexError(f"cached OpenIE for {info['idx']} has different text")
                        cached.append(info)
            have = {info["idx"] for info in cached}
            return cached, {k for k in chunk_keys if k not in have}

        def triple_extraction_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
            if self.global_config.openie_mode != "online":
                raise ValueError("only online OpenIE is supported")
            self.insert_chunks(chunks)
            rows = {c.chunk_id: {"hash_id": c.chunk_id, "content": c.text} for c in chunks}
            order = {c.chunk_id: i for i, c in enumerate(chunks)}
            all_info, todo = self.load_existing_openie(list(rows))
            new_rows = {k: row for k, row in rows.items() if k in todo}
            if new_rows:
                ner, triples = self.openie.batch_openie(new_rows)
                self.merge_openie_results(all_info, new_rows, ner, triples)
            all_info.sort(key=lambda info: order[info["idx"]])
            self.openie_results = all_info
            if self.global_config.save_openie:
                self.save_openie_results(all_info)

        def build_entity_to_triples_index(self):
            self.entity_to_triples = {}

        def save_igraph(self):
            if self._graph_saving:
                super().save_igraph()
            else:
                logger.info("graph saving deferred until the multimodal graph is complete")

        def write_graph(self) -> None:
            self._graph_saving = True
            try:
                super().save_igraph()
            finally:
                self._graph_saving = False

        def get_top_k_weights(self, link_top_k, all_phrase_weights, linking_score_map):
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])
            keep = {self.entity_id(phrase) for phrase in linking_score_map}
            for key, vertex in self.node_name_to_vertex_idx.items():
                if key not in keep:
                    all_phrase_weights[vertex] = 0.0
            return all_phrase_weights, linking_score_map

    return MVBackend


def effective_text(config: ExperimentConfig, llm_name: str | None = None) -> TextConfig:
    """The text settings with the environment and command-line overrides applied."""
    updates: dict[str, Any] = {}
    if os.environ.get(TEXT_EMBEDDING_ENV):
        updates["embedding_model"] = os.environ[TEXT_EMBEDDING_ENV]
    if llm_name:
        updates["llm_name"] = llm_name
    return config.text.model_copy(update=updates)


def _working_dir(artifacts: Artifacts, text: TextConfig) -> Path:
    """Where upstream keeps stores and graph (mirrors ``MemGraphRAG.__init__``)."""
    return artifacts.text_dir / f"{text.llm_name.replace('/', '_')}_{text.embedding_model.replace('/', '_')}"


def make_backend(config: ExperimentConfig, artifacts: Artifacts, text: TextConfig):
    """The text backend. Loads the embedding model and, if one was saved, the graph."""
    if not text.llm_name:
        raise ValueError("text.llm_name is not set: put it in the config or pass --llm-name")
    r = config.retrieval
    base = _upstream().BaseConfig(
        save_dir=str(artifacts.text_dir),
        llm_name=text.llm_name,
        llm_base_url=text.llm_base_url or None,
        embedding_model_name=text.embedding_model,
        embedding_batch_size=text.embedding_batch_size,
        openie_mode="online",
        save_openie=True,
        memory_artifact_mode="default",
        memory_max_workers=text.memory_max_workers,
        ontology_filter_mode=text.ontology_filter_mode,
        ontology_filter_min_frequency=text.ontology_min_frequency,
        ontology_filter_low_percent=text.ontology_low_percent,
        passage_node_weight=r.passage_node_weight,
        linking_top_k=r.linking_top_k,
        damping=r.damping,
        skip_fact_rerank=r.skip_fact_rerank,
        fact_similarity_threshold=r.fact_similarity_threshold,
        use_raw_threshold_filter=r.use_raw_threshold_filter,
    )
    backend = _backend_class()(global_config=base)
    if Path(backend.working_dir) != _working_dir(artifacts, text):
        raise RuntimeError(f"upstream uses {backend.working_dir}, expected {_working_dir(artifacts, text)}")
    return backend


# --- Stamps ------------------------------------------------------------------


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _embedding_sha256(directory: Path) -> str | None:
    parts = [_file_sha256(directory / name) for name in ("meta.json", "rows.jsonl", "vectors.npy")]
    return None if None in parts else _json_sha256(parts)


def _read_manifest(artifacts: Artifacts) -> dict:
    path = artifacts.index_manifest
    if not path.is_file():
        return {"version": 1, "stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(artifacts: Artifacts, manifest: dict) -> None:
    artifacts.index_dir.mkdir(parents=True, exist_ok=True)
    tmp = artifacts.index_manifest.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    tmp.replace(artifacts.index_manifest)


def _need(manifest: dict, stage: str, key: str) -> str:
    try:
        return manifest["stages"][stage]["outputs"][key]
    except KeyError:
        raise RuntimeError(f"stage {stage!r} has not produced {key!r}: run it first") from None


def _verify_outputs(artifacts: Artifacts, config: ExperimentConfig, stage: str, outputs: dict) -> None:
    """The files a stage produced still have the recorded hashes."""
    files = {
        "texts": artifacts.texts, "chunks": artifacts.chunks, "memory": artifacts.memory_path,
        "entities": artifacts.entities, "images": artifacts.indexed_images,
    }
    for key, path in files.items():
        if key in outputs and _file_sha256(path) != outputs[key]:
            raise StaleIndexError(f"{path} changed after stage {stage!r}: rerun it with --force")
    for kind, key in (("image", "image_embeddings"), ("crop", "crop_embeddings")):
        directory = artifacts.embeddings_dir(config.visual_encoder.name, kind)
        if key in outputs and _embedding_sha256(directory) != outputs[key]:
            raise StaleIndexError(f"{directory} changed after stage {stage!r}: rerun it with --force")
    if "graph" in outputs and _file_sha256(artifacts.root / outputs["graph_file"]) != outputs["graph"]:
        raise StaleIndexError(f"{outputs['graph_file']} changed after stage {stage!r}: rerun it with --force")


def _validate_chain(manifest: dict) -> None:
    """Every stage was built from what the earlier stages currently provide."""
    entries = manifest["stages"]
    for i, later in enumerate(STAGES):
        for earlier in STAGES[:i]:
            if later not in entries or earlier not in entries:
                continue
            for key, value in entries[later]["inputs"].items():
                produced = entries[earlier]["outputs"].get(key)
                if produced is not None and produced != value:
                    raise StaleIndexError(
                        f"stage {later!r} was built from a different {key!r} than stage {earlier!r} now provides: "
                        f"rerun {later!r} with --force"
                    )


def _run_stage(
    artifacts: Artifacts, config: ExperimentConfig, manifest: dict, name: str,
    inputs: dict, settings: dict, force: bool, run: Callable[[], dict],
) -> dict:
    """Reuse a stage whose stamp matches, refuse one that does not, else run it."""
    want = json.loads(json.dumps({"inputs": inputs, "settings": settings}))
    entry = manifest["stages"].get(name)
    if entry is not None and not force:
        have = {"inputs": entry["inputs"], "settings": entry["settings"]}
        if have != want:
            changed = sorted(
                f"{section}.{key}" for section in want for key in set(want[section]) | set(have[section])
                if want[section].get(key) != have[section].get(key)
            )
            raise StaleIndexError(f"stage {name!r} was built with different {', '.join(changed)}: rerun it with --force")
        _verify_outputs(artifacts, config, name, entry["outputs"])
        logger.info("stage %s: reusing saved results", name)
        return entry["outputs"]
    logger.info("stage %s: running", name)
    outputs = run()
    manifest["stages"][name] = {**want, "outputs": outputs}
    _write_manifest(artifacts, manifest)
    return outputs


# --- Stages ------------------------------------------------------------------


def _stage_text(artifacts: Artifacts, config: ExperimentConfig, text: TextConfig, force: bool) -> dict:
    from transformers import AutoTokenizer

    passages, captions, images = _read_sources(artifacts)
    texts = normalize_texts(passages, captions, images)
    chunks = chunk_texts(texts, AutoTokenizer.from_pretrained(text.embedding_model), text.chunk_size, text.chunk_overlap)
    if not chunks:
        raise ValueError("the corpus has no indexable text")
    write_jsonl(artifacts.texts, texts)
    write_jsonl(artifacts.chunks, chunks)
    if force:
        # Stores and graph restart empty; the OpenIE and LLM caches are keyed by content and stay.
        shutil.rmtree(_working_dir(artifacts, text), ignore_errors=True)

    backend = make_backend(config, artifacts, text)
    try:
        backend.triple_extraction_chunks(chunks)
        initial = _upstream().ThreeLayerMemory()
        initial.build_from_raw_openie_results({"docs": backend.openie_results})
        with_schema = backend.extract_memory_schema(initial)
        filtered = backend.filter_memory_ontology(with_schema)
        conflicts = backend.detect_memory_conflicts(filtered)
        final_memory, _ = backend.resolve_memory_conflicts(filtered, conflicts)
        artifacts.text_dir.mkdir(parents=True, exist_ok=True)
        final_memory.save(str(artifacts.memory_path))
        entities = export_entities(final_memory, {c.chunk_id: i for i, c in enumerate(chunks)}, backend.entity_id)
        write_jsonl(artifacts.entities, entities)
    finally:
        del backend
        sam3_processing.release_gpu()
    return {
        "texts": _file_sha256(artifacts.texts), "chunks": _file_sha256(artifacts.chunks),
        "memory": _file_sha256(artifacts.memory_path), "entities": _file_sha256(artifacts.entities),
    }


def _stage_ground(artifacts: Artifacts, config: ExperimentConfig, repo: Path, checkpoint: Path, grounder_id: str) -> dict:
    images = read_jsonl(artifacts.image_manifest, ImageRecord)
    links = link_images(
        images, read_jsonl(artifacts.texts, TextRecord), read_jsonl(artifacts.chunks, ChunkRecord)
    )
    entities = read_jsonl(artifacts.entities, EntityRecord)
    loaded: list[sam3_processing.MG2Sam3] = []

    def get_grounder() -> sam3_processing.MG2Sam3:
        grounder = sam3_processing.MG2Sam3(
            repo, checkpoint, config.sam3.device, config.sam3.threshold, config.sam3.prompt_batch_size,
            artifacts.grounding_dir / "mg2",
        )
        loaded.append(grounder)
        return grounder

    try:
        records = sam3_processing.ground_images(
            images, {link.image_id: link.chunk_ids for link in links}, entities, artifacts, config.sam3,
            get_grounder, grounder_id,
        )
    finally:
        for grounder in loaded:
            grounder.close()
        loaded.clear()
        sam3_processing.release_gpu()
    return {"grounding": _json_sha256([r.model_dump(mode="json") for r in records])}


def _stage_embed(artifacts: Artifacts, config: ExperimentConfig, force: bool, device: str) -> dict:
    images = read_jsonl(artifacts.image_manifest, ImageRecord)
    held: dict[str, Encoder] = {}

    def get_encoder() -> Encoder:
        if "encoder" not in held:
            held["encoder"] = visual_indexing.load_encoder(config.visual_encoder, artifacts, device)
        return held["encoder"]

    try:
        visual_indexing.embed_visual(artifacts, config.visual_encoder, images, get_encoder, reuse=not force)
    finally:
        held.clear()
        sam3_processing.release_gpu()
    name = config.visual_encoder.name
    return {
        "image_embeddings": _embedding_sha256(artifacts.embeddings_dir(name, "image")),
        "crop_embeddings": _embedding_sha256(artifacts.embeddings_dir(name, "crop")),
    }


def _stage_assemble(artifacts: Artifacts, config: ExperimentConfig, text: TextConfig) -> dict:
    images = read_jsonl(artifacts.image_manifest, ImageRecord)
    chunks = read_jsonl(artifacts.chunks, ChunkRecord)
    links = link_images(images, read_jsonl(artifacts.texts, TextRecord), chunks)
    image_set = EmbeddingSet.load(artifacts.embeddings_dir(config.visual_encoder.name, "image"))
    grounding = sam3_processing.load_grounding(artifacts, [i.image_id for i in images])
    memory = _upstream().ThreeLayerMemory.load(str(artifacts.memory_path))

    backend = make_backend(config, artifacts, text)
    try:
        backend.check_chunks(chunks)
        by_record: dict[str, list[str]] = defaultdict(list)
        for chunk in chunks:
            by_record[chunk.record_id].append(chunk.chunk_id)
        memory_graph = backend.build_memory_graph(memory)
        backend.install_memory_graph(memory_graph)  # the text-only graph is not saved
        indexed, stats = visual_indexing.add_visual_structure(backend.graph, links, by_record, image_set, grounding)
        backend.write_graph()
        graph_file = artifacts.relative(Path(backend._graphml_xml_file))
        working_dir = Path(backend.working_dir).name
    finally:
        del backend
        sam3_processing.release_gpu()
    write_jsonl(artifacts.indexed_images, indexed)
    logger.info("visual structure: %s", stats)
    return {
        "graph": _file_sha256(artifacts.root / graph_file), "graph_file": graph_file, "working_dir": working_dir,
        "images": _file_sha256(artifacts.indexed_images), "stats": stats,
    }


def index_with_memory_v(
    artifacts: Artifacts,
    config: ExperimentConfig,
    *,
    stages: Sequence[str] = STAGES,
    force: bool = False,
    device: str = "cuda",
    llm_name: str | None = None,
) -> dict[str, dict]:
    """Build the multimodal index (see the module docstring); returns each
    requested stage's outputs. Only what a stage needs is validated: ``text``
    and ``assemble`` need ``text.llm_name``, ``ground`` the MG2 checkout and
    SAM3 weights, ``embed`` the EVA-CLIP checkpoint."""
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {STAGES}")
    text = effective_text(config, llm_name)
    manifest = _read_manifest(artifacts)
    sources = {
        "text_manifest": _file_sha256(artifacts.text_manifest),
        "image_manifest": _file_sha256(artifacts.image_manifest),
        "captions": _file_sha256(artifacts.captions),
    }
    text_settings = {
        "llm_name": text.llm_name, "embedding_model": text.embedding_model, "chunk_size": text.chunk_size,
        "chunk_overlap": text.chunk_overlap, "ontology_filter_mode": text.ontology_filter_mode,
        "ontology_min_frequency": text.ontology_min_frequency, "ontology_low_percent": text.ontology_low_percent,
        "memgraphrag_commit": MEMGRAPHRAG_COMMIT,
    }
    encoder = visual_indexing.encoder_meta(config.visual_encoder)
    done: dict[str, dict] = {}

    if "text" in stages:
        if not text.llm_name:
            raise ValueError("text.llm_name is not set: put it in the config or pass --llm-name")
        # Of the image manifest, only the image-to-document identity matters here (captions take their image's document).
        identities = _json_sha256([[i.image_id, i.doc_id] for i in read_jsonl(artifacts.image_manifest, ImageRecord)])
        text_inputs = {
            "text_manifest": sources["text_manifest"], "captions": sources["captions"], "image_identities": identities,
        }
        done["text"] = _run_stage(
            artifacts, config, manifest, "text", text_inputs, text_settings, force,
            lambda: _stage_text(artifacts, config, text, force),
        )
    if "ground" in stages:
        repo = external_path(MG2_ENV, "MG2-RAG checkout")
        checkpoint = sam3_processing.default_checkpoint(artifacts)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 weights not found at {checkpoint}: set MGRV_SAM3_CHECKPOINT")
        grounder_id = sam3_processing.grounder_identity(checkpoint)
        inputs = {
            "image_manifest": sources["image_manifest"], "texts": _need(manifest, "text", "texts"),
            "chunks": _need(manifest, "text", "chunks"), "entities": _need(manifest, "text", "entities"),
        }
        done["ground"] = _run_stage(
            artifacts, config, manifest, "ground", inputs,
            {"sam3": config.sam3.model_dump(mode="json"), "grounder": grounder_id}, force,
            lambda: _stage_ground(artifacts, config, repo, checkpoint, grounder_id),
        )
    if "embed" in stages:
        inputs = {"image_manifest": sources["image_manifest"], "grounding": _need(manifest, "ground", "grounding")}
        done["embed"] = _run_stage(
            artifacts, config, manifest, "embed", inputs, {"visual_encoder": encoder}, force,
            lambda: _stage_embed(artifacts, config, force, device),
        )
    if "assemble" in stages:
        if not text.llm_name:
            raise ValueError("text.llm_name is not set: put it in the config or pass --llm-name")
        inputs = {
            "image_manifest": sources["image_manifest"], "memory": _need(manifest, "text", "memory"),
            "texts": _need(manifest, "text", "texts"), "chunks": _need(manifest, "text", "chunks"),
            "grounding": _need(manifest, "ground", "grounding"),
            "image_embeddings": _need(manifest, "embed", "image_embeddings"),
            "crop_embeddings": _need(manifest, "embed", "crop_embeddings"),
        }
        settings = {"llm_name": text.llm_name, "embedding_model": text.embedding_model}
        done["assemble"] = _run_stage(
            artifacts, config, manifest, "assemble", inputs, settings, force,
            lambda: _stage_assemble(artifacts, config, text),
        )
    _validate_chain(manifest)
    manifest["provenance"] = {
        "config_hash": config.config_hash(),
        "config": config.model_dump(mode="json", exclude={"artifacts_root"}),
        "memgraphrag_commit": MEMGRAPHRAG_COMMIT,
        "mg2_commit": sam3_processing.MG2_COMMIT,
    }
    _write_manifest(artifacts, manifest)
    return done


# --- Loading and retrieval ---------------------------------------------------


def load_index(
    artifacts: Artifacts, config: ExperimentConfig, *, llm_name: str | None = None
) -> RetrievalIndex:
    """Load the completed index for retrieval. Nothing is rebuilt, grounded or
    embedded here; every saved piece is checked against the stage stamps and
    against the others, and any mismatch raises."""
    manifest = _read_manifest(artifacts)
    missing = [s for s in STAGES if s not in manifest["stages"]]
    if missing:
        raise RuntimeError(f"the index is incomplete: stages {missing} have not run")
    _validate_chain(manifest)
    for stage in STAGES:
        _verify_outputs(artifacts, config, stage, manifest["stages"][stage]["outputs"])
    assembled = manifest["stages"]["assemble"]
    recorded = assembled["settings"]

    text = effective_text(config, llm_name)
    if text.llm_name is None:
        text = text.model_copy(update={"llm_name": recorded["llm_name"]})
    if (text.llm_name, text.embedding_model) != (recorded["llm_name"], recorded["embedding_model"]):
        raise StaleIndexError(
            f"the index was built with llm {recorded['llm_name']!r} and embedding model "
            f"{recorded['embedding_model']!r}, not {text.llm_name!r} and {text.embedding_model!r}"
        )
    enc = config.visual_encoder
    image_set = EmbeddingSet.load(artifacts.embeddings_dir(enc.name, "image"))
    crop_set = EmbeddingSet.load(artifacts.embeddings_dir(enc.name, "crop"))
    wanted = visual_indexing.encoder_meta(enc)
    for embedded in (image_set, crop_set):
        for key, value in wanted.items():
            if getattr(embedded.meta, key) != value:
                raise StaleIndexError(f"{embedded.meta.kind} vectors were made with {key}={getattr(embedded.meta, key)!r}, not {value!r}")

    backend = make_backend(config, artifacts, text)
    retrieval.prepare_retrieval_objects(backend, artifacts.memory_path)
    provenance = {
        "index_manifest_sha256": _file_sha256(artifacts.index_manifest),
        "graph_sha256": assembled["outputs"]["graph"],
        "config_hash": config.config_hash(),
        "memgraphrag_commit": MEMGRAPHRAG_COMMIT,
        "text": {"llm_name": text.llm_name, "embedding_model": text.embedding_model},
        "visual_encoder": wanted,
    }
    return retrieval.build_index(
        backend, config.retrieval, image_set, crop_set,
        read_jsonl(artifacts.texts, TextRecord), read_jsonl(artifacts.chunks, ChunkRecord),
        read_jsonl(artifacts.indexed_images, IndexedImage), provenance,
    )


def retrieve(
    index: RetrievalIndex,
    questions: Sequence[Question],
    encoder: Encoder,
    *,
    beta: float | None = None,
    out_dir: Path | None = None,
) -> list[tuple[RetrievalResult, QueryDiagnostics]]:
    """Evidence for each question (``retrieval.retrieve_questions``). With
    ``out_dir``, also writes ``retrieval.jsonl``, ``diagnostics.jsonl`` and
    ``provenance.json`` there. Stops at evidence ready for a reader."""
    results = retrieval.retrieve_questions(index, questions, encoder, beta)
    if out_dir is not None:
        out_dir = Path(out_dir)
        write_jsonl(out_dir / "retrieval.jsonl", [r for r, _ in results])
        write_jsonl(out_dir / "diagnostics.jsonl", [d for _, d in results])
        used = index.settings.beta if beta is None else beta
        provenance = {
            **index.provenance, "beta": used,
            "retrieval": index.settings.model_dump(mode="json"),
        }
        (out_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
        )
    return results


# --- Command line ------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the MemGraphRAG-V index, or retrieve from it.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.yaml")
    common.add_argument("--llm-name", help="overrides text.llm_name")
    common.add_argument("--device", default="cuda")
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("index", parents=[common], help="build the index (all stages by default)")
    build.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    build.add_argument("--force", action="store_true", help="recompute the stages even if their stamps no longer match")
    query = modes.add_parser("retrieve", parents=[common], help="retrieve evidence from the saved index")
    query.add_argument("--run-id", required=True)
    query.add_argument("--split", choices=["dev", "test"])
    query.add_argument("--limit", type=int)
    query.add_argument("--beta", type=float, help="text/visual mixing weight in [0, 1]; overrides retrieval.beta")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    config = ExperimentConfig.load(args.config)
    artifacts = Artifacts.from_env(config.artifacts_root)

    if args.mode == "index":
        done = index_with_memory_v(
            artifacts, config, stages=args.stages, force=args.force, device=args.device, llm_name=args.llm_name,
        )
        print(json.dumps(done, indent=2))
        return

    beta = config.retrieval.beta if args.beta is None else args.beta
    if beta is None:
        parser.error("beta is not set: pass --beta or set retrieval.beta")
    index = load_index(artifacts, config, llm_name=args.llm_name)
    # Only the ID and text are read; the labels in the record are never used.
    questions = [
        Question(q.qid, q.question)
        for q in read_jsonl(artifacts.questions, QuestionRecord)
        if args.split is None or q.split == args.split
    ][: args.limit]
    encoder = visual_indexing.load_encoder(config.visual_encoder, artifacts, args.device)
    run_dir = artifacts.run_dir(args.run_id)
    results = retrieve(index, questions, encoder, beta=beta, out_dir=run_dir)
    config.save(run_dir / "config.yaml")
    routes = Counter(r.route for r, _ in results)
    print(json.dumps({"out": str(run_dir), "questions": len(results), "routes": dict(routes)}, indent=2))


if __name__ == "__main__":
    main()
