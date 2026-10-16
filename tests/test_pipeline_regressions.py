"""Regression checks for the WebQA merge: the real dataset files, stage
prerequisites, missing OpenIE results and empty entity/fact stores.

No models, LLM calls or upstream MemGraphRAG are used. The upstream backend is
replaced by a small fake, and the real ``dataset/`` files are only parsed.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from memgraphrag_v import pipeline, webqa
from memgraphrag_v.config import ExperimentConfig
from memgraphrag_v.embeddings import EmbeddingMeta, EmbeddingRow, EmbeddingSet
from memgraphrag_v.io import write_jsonl
from memgraphrag_v.paths import Artifacts
from memgraphrag_v.retrieval import prepare_retrieval_objects, text_seed
from memgraphrag_v.schemas import (
    CaptionRecord,
    ChunkRecord,
    ImageGrounding,
    ImageRecord,
    IndexedImage,
    PassageRecord,
)
from memgraphrag_v.visual_indexing import add_visual_structure

DATASET = Path(__file__).parents[1] / "dataset"
needs_dataset = pytest.mark.skipif(not (DATASET / "build_qa.json").is_file(), reason="dataset/ is not checked out")


class WordTokenizer:
    model_max_length = 512

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return " ".join(ids)


# --- The committed WebQA files are the contract --------------------------------


def _jsonl(name):
    with open(DATASET / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json(name):
    return json.loads((DATASET / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_webqa(tmp_path_factory):
    """The committed dataset converted by webqa.py. No image files are present,
    so every image has ok=False; that does not matter for the checks below."""
    passages = [PassageRecord.model_validate(row) for row in _jsonl("text_manifest.jsonl")]
    images = webqa.image_records(_jsonl("image_manifest.jsonl"), tmp_path_factory.mktemp("no_images"))
    questions = webqa.question_records(_json("build_qa.json"), _json("test_qa.json"))
    captions = webqa.caption_records(_jsonl("caption_manifest.jsonl"))
    counts = webqa.check_references(passages, images, questions, captions)
    return SimpleNamespace(passages=passages, images=images, questions=questions, captions=captions, counts=counts)


@needs_dataset
def test_real_webqa_conversion_counts_and_fixed_split(real_webqa):
    c = real_webqa.counts
    assert (c["passages"], c["images"], c["generated_captions"]) == (12080, 4221, 4221)
    assert c["images_without_generated_caption"] == 0
    # build_qa.json is the 480 dev questions, test_qa.json (with its extra
    # gold_presence_in_corpus field) the 120 test questions; nothing is re-split
    assert (c["questions_dev"], c["questions_test"]) == (480, 120)
    by_split = {"dev": [], "test": []}
    for q in real_webqa.questions:
        by_split[q.split].append(q)
    assert [q.qid for q in by_split["dev"]] == [q["qid"] for q in _json("build_qa.json")]
    assert [q.qid for q in by_split["test"]] == [q["qid"] for q in _json("test_qa.json")]
    modalities = {s: sorted({tuple(q.modalities) for q in qs}) for s, qs in by_split.items()}
    assert modalities == {"dev": [("image",), ("text",)], "test": [("image",)]}
    assert sum(q.modalities == ["text"] for q in by_split["dev"]) == 120
    assert all(q.topic and q.q_type for q in real_webqa.questions)


@needs_dataset
def test_real_webqa_ids_are_titles_with_explicit_caption_passages(real_webqa):
    images = {i.image_id: i for i in real_webqa.images}
    passages = {p.passage_id: p for p in real_webqa.passages}
    # every image has exactly its own cap_ passage, from the same page
    for image in real_webqa.images:
        assert image.passage_ids == [f"cap_{image.image_id}"]
        assert passages[image.passage_ids[0]].doc_id == image.doc_id
    assert images["30000050"].doc_id == "Michigan Wolverines"
    assert images["30000050"].commons_url == "https://en.wikipedia.org/wiki/Michigan_Wolverines"
    assert images["30000050"].path == "30000050.jpg" and images["30000050"].ok is False
    # doc IDs are page titles (with spaces), and gold documents use them
    docs = {p.doc_id for p in real_webqa.passages} | {i.doc_id for i in real_webqa.images}
    assert "LeBron James (March 2011)" in docs
    for q in real_webqa.questions:
        assert q.gold_doc_ids and set(q.gold_doc_ids) <= docs
        assert all(g.startswith("txt_") for g in q.gold_passage_ids)


@needs_dataset
def test_real_webqa_is_indexed_as_passages_plus_generated_descriptions(real_webqa):
    texts = pipeline.normalize_texts(real_webqa.passages, real_webqa.captions, real_webqa.images)
    assert [t.kind for t in texts].count("passage") == 12080
    assert [t.kind for t in texts].count("caption") == 4221
    assert texts[1].doc_id == "LeBron James (March 2011)"  # a title stays a title
    doc_of = {i.image_id: i.doc_id for i in real_webqa.images}
    assert all(t.doc_id == doc_of[t.image_id] for t in texts if t.kind == "caption")

    chunks = pipeline.chunk_texts(texts, WordTokenizer(), chunk_size=256, overlap=32)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    links = pipeline.link_images(real_webqa.images, texts, chunks)
    # an image is linked to its own cap_ passage and its own description, and
    # never to the other passages that share its page title
    assert all(l.record_ids == [f"cap_{l.image_id}", f"caption:{l.image_id}:0"] for l in links)
    by_record = {}
    for c in chunks:
        by_record.setdefault(c.record_id, []).append(c.chunk_id)
    assert all(l.chunk_ids == by_record["cap_" + l.image_id] + by_record["caption:" + l.image_id + ":0"] for l in links)


# --- Stage prerequisites -------------------------------------------------------


def _artifacts_with_manifests(tmp_path):
    artifacts = Artifacts(tmp_path)
    write_jsonl(artifacts.text_manifest, [PassageRecord(doc_id="Page A", passage_id="cap_1", text="A caption.")])
    write_jsonl(artifacts.image_manifest, [
        ImageRecord(image_id="1", doc_id="Page A", path="1.jpg", passage_ids=["cap_1"], ok=False)])
    write_jsonl(artifacts.captions, [CaptionRecord(image_id="1", text="A description.", model="m", prompt_version="v")])
    return artifacts


def _text_outputs(artifacts):
    """Files and hashes a finished text stage would have recorded."""
    sha = pipeline._file_sha256
    artifacts.memory_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (artifacts.texts, artifacts.chunks, artifacts.entities, artifacts.memory_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    store = artifacts.text_dir / "llm_embed" / "chunk_embeddings" / "vdb_chunk.parquet"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_bytes(b"chunk store")
    return {
        "texts": sha(artifacts.texts), "chunks": sha(artifacts.chunks), "entities": sha(artifacts.entities),
        "memory": sha(artifacts.memory_path),
        "text_embedding_files": {"chunk": {"path": artifacts.relative(store), "sha256": sha(store)}},
    }


def _save_manifest(artifacts, stages):
    pipeline._write_manifest(artifacts, {"version": 1, "stages": stages})


def test_single_stage_runs_name_the_missing_earlier_stage(tmp_path):
    artifacts, config = _artifacts_with_manifests(tmp_path), ExperimentConfig()
    for stage in ("ground", "embed", "assemble"):
        with pytest.raises(RuntimeError, match="stage 'text' has not produced"):
            pipeline.index_with_memory_v(artifacts, config, stages=(stage,))

    stamp = {"inputs": {}, "settings": {}}
    _save_manifest(artifacts, {"text": {**stamp, "outputs": {"texts": "t", "chunks": "c", "entities": "e", "memory": "m"}}})
    with pytest.raises(RuntimeError, match="stage 'ground' has not produced 'grounding'"):
        pipeline.index_with_memory_v(artifacts, config, stages=("embed",))
    with pytest.raises(RuntimeError, match="stage 'ground' has not produced 'grounding'"):
        pipeline.index_with_memory_v(artifacts, config, stages=("assemble",))

    stages = json.loads(artifacts.index_manifest.read_text())["stages"]
    stages["ground"] = {**stamp, "outputs": {"grounding": "g"}}
    _save_manifest(artifacts, stages)
    with pytest.raises(RuntimeError, match="stage 'embed' has not produced 'image_embeddings'"):
        pipeline.index_with_memory_v(artifacts, config, stages=("assemble",))


def test_run_stage_checks_earlier_stages_before_running(tmp_path):
    artifacts, config = Artifacts(tmp_path), ExperimentConfig()
    manifest = {"version": 1, "stages": {}}
    expected = {"text": {"text_manifest": "current"}}
    calls = []

    def run():
        calls.append(1)
        return {"done": 1}  # a key _verify_outputs has nothing to check for

    def run_ground():
        return pipeline._run_stage(artifacts, config, manifest, "ground", {"x": 1}, expected, {"s": 1}, False, run)

    with pytest.raises(RuntimeError, match="depends on missing stage 'text'"):
        run_ground()

    outputs = _text_outputs(artifacts)
    manifest["stages"]["text"] = {"inputs": {"text_manifest": "old"}, "settings": {}, "outputs": outputs}
    with pytest.raises(pipeline.StaleIndexError, match="outdated inputs"):
        run_ground()

    manifest["stages"]["text"]["inputs"] = {"text_manifest": "current"}
    artifacts.chunks.write_text("changed\n", encoding="utf-8")
    with pytest.raises(pipeline.StaleIndexError, match="changed after stage 'text'"):
        run_ground()
    assert calls == []  # nothing ran on a stale prerequisite

    artifacts.chunks.write_text("{}\n", encoding="utf-8")
    assert run_ground() == {"done": 1} and calls == [1]
    assert json.loads(artifacts.index_manifest.read_text())["stages"]["ground"]["outputs"] == {"done": 1}
    # an unchanged stage is reused, not rerun
    assert run_ground() == {"done": 1} and calls == [1]


def test_load_index_reports_an_incomplete_index_first(tmp_path):
    artifacts, config = _artifacts_with_manifests(tmp_path), ExperimentConfig()
    with pytest.raises(RuntimeError, match="index is incomplete"):  # no manifest at all
        pipeline.load_index(artifacts, config)
    _save_manifest(artifacts, {"text": {"inputs": {}, "settings": {}, "outputs": {}}})
    with pytest.raises(RuntimeError, match=r"index is incomplete: stages \['ground', 'embed', 'assemble'\]"):
        pipeline.load_index(artifacts, config)


# --- OpenIE results ------------------------------------------------------------


class _Result:
    def __init__(self, metadata=None, unique_entities=("an entity",)):
        self.metadata = metadata if metadata is not None else {}
        self.unique_entities = list(unique_entities)
        self.triples = []


class _OpenIE:
    """Stand-in for upstream's batch_openie; remembers which chunks it was given."""

    def __init__(self, ner_missing=(), triples_missing=(), errored=()):
        self.ner_missing, self.triples_missing, self.errored = set(ner_missing), set(triples_missing), set(errored)
        self.requested = []

    def batch_openie(self, rows):
        self.requested.append(list(rows))
        ner = {k: _Result({"error": "bad json"} if k in self.errored else {}) for k in rows if k not in self.ner_missing}
        triples = {k: _Result() for k in rows if k not in self.triples_missing}
        return ner, triples


class _Store:
    def __init__(self, filename):
        self.filename = str(filename)
        self.hash_ids, self.hash_id_to_row = [], {}

    def _upsert(self, hash_ids, texts, embeddings):
        for hash_id, text in zip(hash_ids, texts):
            self.hash_ids.append(hash_id)
            self.hash_id_to_row[hash_id] = {"hash_id": hash_id, "content": text}

    def _save_data(self):
        Path(self.filename).write_bytes(b"saved by _save_data")


class _FakeUpstream:
    """The part of upstream MemGraphRAG that MVBackend uses for OpenIE and stores."""

    def __init__(self, global_config=None):
        self.global_config = global_config
        save_dir = Path(global_config.save_dir)
        self.openie_results_path = str(save_dir / "openie_results.json")
        self.embedding_model = SimpleNamespace(batch_encode=lambda texts: [[0.0] for _ in texts])
        self.chunk_embedding_store = _Store(save_dir / "chunk.parquet")
        self.entity_embedding_store = _Store(save_dir / "entity.parquet")
        self.fact_embedding_store = _Store(save_dir / "fact.parquet")
        self.openie = _OpenIE()

    def merge_openie_results(self, all_openie_info, rows, ner, triples):
        for chunk_id, row in rows.items():
            all_openie_info.append({
                "idx": chunk_id, "passage": row["content"],
                "extracted_entities": list(ner[chunk_id].unique_entities), "extracted_triples": [],
            })
        return all_openie_info


@pytest.fixture
def make_backend(monkeypatch, tmp_path):
    fake = SimpleNamespace(MemGraphRAG=_FakeUpstream, compute_mdhash_id=lambda content, prefix="": prefix + content)
    monkeypatch.setattr(pipeline, "_upstream", lambda: fake)
    pipeline._backend_class.cache_clear()

    def make(openie=None):
        config = SimpleNamespace(
            openie_mode="online", save_dir=str(tmp_path), force_openie_from_scratch=False, save_openie=True,
        )
        backend = pipeline._backend_class()(global_config=config)
        if openie is not None:
            backend.openie = openie
        return backend

    yield make
    pipeline._backend_class.cache_clear()


def _chunks(n):
    return [
        ChunkRecord(chunk_id=f"chunk-{i}", record_id=f"cap_{i}", doc_id=f"Page {i}", position=0, text=f"text {i}")
        for i in range(n)
    ]


def _failure_report(tmp_path):
    return json.loads((tmp_path / pipeline.OPENIE_FAILURES_FILE).read_text(encoding="utf-8"))


def _cached_ids(tmp_path):
    return [doc["idx"] for doc in json.loads((tmp_path / "openie_results.json").read_text())["docs"]]


def test_openie_failures_names_missing_and_errored_results():
    chunks = {c.chunk_id: c for c in _chunks(4)}
    ner = {k: _Result() for k in chunks if k != "chunk-0"}  # no NER result for chunk-0
    ner["chunk-1"] = _Result({"error": "bad json"})
    triples = {k: _Result() for k in chunks if k != "chunk-2"}  # no triple result for chunk-2
    succeeded, failures = pipeline.openie_failures(chunks, ner, triples)
    assert succeeded == ["chunk-3"]
    assert {f["chunk_id"]: f["errors"] for f in failures} == {
        "chunk-0": ["ner: missing result"],
        "chunk-1": ["ner: bad json"],
        "chunk-2": ["triples: missing result"],
    }
    assert failures[0]["record_id"] == "cap_0" and failures[0]["doc_id"] == "Page 0"


def test_missing_openie_result_is_a_recorded_failure_not_a_crash(make_backend, tmp_path):
    backend = make_backend(_OpenIE(triples_missing={"chunk-3"}))
    backend.triple_extraction_chunks(_chunks(8))
    assert [i["idx"] for i in backend.openie_results] == [f"chunk-{i}" for i in (0, 1, 2, 4, 5, 6, 7)]
    report = _failure_report(tmp_path)
    assert (report["requested"], report["failed"], report["limit"]) == (8, 1, 5)
    assert report["chunks"] == [
        {"chunk_id": "chunk-3", "record_id": "cap_3", "doc_id": "Page 3", "errors": ["triples: missing result"]}
    ]
    assert _cached_ids(tmp_path) == [f"chunk-{i}" for i in (0, 1, 2, 4, 5, 6, 7)]
    assert len(backend.openie_failed) == 1


def test_five_failed_chunks_are_tolerated_and_six_are_fatal(make_backend, tmp_path):
    backend = make_backend(_OpenIE(ner_missing={f"chunk-{i}" for i in range(5)}))
    backend.triple_extraction_chunks(_chunks(8))
    assert _failure_report(tmp_path)["failed"] == 5

    (tmp_path / "openie_results.json").unlink()
    backend = make_backend(_OpenIE(ner_missing={f"chunk-{i}" for i in range(6)}))
    with pytest.raises(RuntimeError, match=r"OpenIE failed for 6 chunks; the limit is 5"):
        backend.triple_extraction_chunks(_chunks(8))
    # the diagnostics and the successful results are saved before the stage stops
    assert _failure_report(tmp_path)["failed"] == 6
    assert _cached_ids(tmp_path) == ["chunk-6", "chunk-7"]


def test_rerun_retries_only_failed_chunks_and_clears_the_report(make_backend, tmp_path):
    make_backend(_OpenIE(triples_missing={"chunk-3"})).triple_extraction_chunks(_chunks(8))
    openie = _OpenIE()
    backend = make_backend(openie)
    backend.triple_extraction_chunks(_chunks(8))
    assert openie.requested == [["chunk-3"]]
    assert _failure_report(tmp_path)["failed"] == 0
    assert [i["idx"] for i in backend.openie_results] == [f"chunk-{i}" for i in range(8)]

    openie = _OpenIE()
    make_backend(openie).triple_extraction_chunks(_chunks(8))  # nothing left to extract
    assert openie.requested == []
    assert _failure_report(tmp_path) == {"requested": 0, "failed": 0, "limit": 5, "chunks": []}


def test_openie_failure_file_has_a_fixed_place(tmp_path):
    assert Artifacts(tmp_path).openie_failures == tmp_path / "index" / "text" / pipeline.OPENIE_FAILURES_FILE


# --- Empty entity and fact stores ----------------------------------------------


def test_missing_stores_are_written_and_existing_ones_are_kept(tmp_path):
    existing, missing = _Store(tmp_path / "chunk.parquet"), _Store(tmp_path / "fact.parquet")
    Path(existing.filename).write_bytes(b"keep")
    pipeline.persist_missing_stores([existing, missing])
    assert Path(existing.filename).read_bytes() == b"keep"
    assert Path(missing.filename).is_file()


def test_backend_persists_all_three_stores_before_they_are_hashed(make_backend, tmp_path):
    backend = make_backend()
    backend.persist_stores()
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == ["chunk.parquet", "entity.parquet", "fact.parquet"]


def test_empty_stores_have_hashes_that_verify(tmp_path):
    artifacts, config = Artifacts(tmp_path), ExperimentConfig()
    files = {}
    for name in ("chunk", "entity", "fact"):
        path = tmp_path / "index" / "text" / "m" / f"vdb_{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"valid but empty store" if name != "chunk" else b"chunk rows")
        files[name] = {"path": artifacts.relative(path), "sha256": pipeline._file_sha256(path)}
    pipeline._verify_outputs(artifacts, config, "assemble", {"text_embedding_files": files})

    files["fact"]["sha256"] = None  # what a store that was never written used to record
    with pytest.raises(pipeline.StaleIndexError, match="no recorded hash"):
        pipeline._verify_outputs(artifacts, config, "assemble", {"text_embedding_files": files})
    (tmp_path / files["entity"]["path"]).unlink()
    files["fact"]["sha256"] = "x"
    with pytest.raises(pipeline.StaleIndexError):
        pipeline._verify_outputs(artifacts, config, "assemble", {"text_embedding_files": files})


def test_an_empty_store_is_a_valid_parquet_file(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    path = tmp_path / "vdb_fact.parquet"
    # what upstream's EmbeddingStore._save_data writes for a store without rows
    pd.DataFrame({"hash_id": [], "content": [], "embedding": []}).to_parquet(path, index=False)
    frame = pd.read_parquet(path)
    assert frame["hash_id"].values.tolist() == [] and frame["embedding"].values.tolist() == []


def test_zero_fact_index_assembles_and_loads(tmp_path):
    ig = pytest.importorskip("igraph")
    # assembled: passages only (no entity or type vertex), then an image with two caption chunks
    graph = ig.Graph()
    graph.add_vertices(2, attributes={"name": ["chunk-a", "chunk-b"], "layer": ["passage", "passage"], "content": ["a", "b"]})
    meta = EmbeddingMeta(encoder="e", revision=None, processor="p", processor_revision=None,
                         dim=2, kind="image", n_rows=1, n_failed=0)
    row = EmbeddingRow(row=0, kind="image", item_id="i1", image_id="i1", doc_id="Page A",
                       path="corpus/images/i1.jpg", status="ok")
    image_set = EmbeddingSet(meta, [row], np.array([[1.0, 0.0]], dtype=np.float32))
    link = IndexedImage(image_id="i1", doc_id="Page A", path="i1.jpg", record_ids=["cap_i1", "caption:i1:0"],
                        chunk_ids=["chunk-a", "chunk-b"], usable=True)
    grounding = {"i1": ImageGrounding(image_id="i1", doc_id="Page A", status="no_prompts", fingerprint="f")}
    [indexed], stats = add_visual_structure(
        graph, [link], {"cap_i1": ["chunk-a"], "caption:i1:0": ["chunk-b"]}, image_set, grounding)
    assert indexed.vertex == "image-i1" and indexed.usable
    assert (stats["image_vertices"], stats["image_passage_edges"], stats["image_entity_edges"]) == (1, 2, 0)

    # loaded: empty entity and fact stores, and a memory without facts
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps({"schema_layer": [], "fact_layer": [], "passage_layer": []}), encoding="utf-8")

    def store(ids, vectors):
        return SimpleNamespace(get_all_ids=lambda: list(ids), get_embeddings=lambda keys: [vectors[k] for k in keys])

    backend = SimpleNamespace(
        graph=graph, global_config=SimpleNamespace(),
        hash_id=lambda content, prefix: prefix + content,
        chunk_embedding_store=store(["chunk-a", "chunk-b"], {"chunk-a": [1.0, 0.0], "chunk-b": [0.0, 1.0]}),
        entity_embedding_store=store([], {}),
        fact_embedding_store=store([], {}),
    )
    prepare_retrieval_objects(backend, memory)
    assert backend.fact_node_keys == [] and backend.entity_node_keys == []
    assert backend.passage_node_keys == ["chunk-a", "chunk-b"] and backend.image_node_keys == ["image-i1"]
    assert backend.ready_to_retrieve
    seed = text_seed(backend, "any question")
    assert seed.weights is None and seed.reason == "the index has no facts"  # visual retrieval still proceeds
