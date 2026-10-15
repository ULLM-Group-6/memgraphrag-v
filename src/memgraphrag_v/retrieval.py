"""Retrieval: text and visual seeds, fusion, CPU PPR, selection and fallback.

Both channels are computed for every question before any routing:

- **Text seeds** are the seed-building half of upstream
  ``graph_search_with_fact_entities`` without the PPR call: query-fact scoring,
  fact selection (``rerank_facts``: by default a raw-score threshold), entity
  weights from the selected facts, the top ``linking_top_k`` entities, plus
  the dense-passage contribution
  ``t_raw = phrase_weights + passage_node_weight * normalized_passage_scores``.
  No selected fact means this channel is unavailable.
- **Visual seeds** use the EVA-CLIP question vector, encoded once, against
  whole-image vectors (image vertices) and crop vectors (existing entity
  vertices, never crop vertices). The text encoder's space and EVA's space are
  never compared with each other.

``fuse_seeds`` combines them (``beta`` weights the visual channel), PPR runs
on the saved multimodal graph, and the highest-scoring image of each of at
most five documents is the evidence. Without any seed, or if no image receives
PPR mass, the shared dense-passage fallback answers instead.

Functions take the text backend (an upstream ``MemGraphRAG`` object, see
``pipeline.py``) as an argument, so this module imports neither upstream code,
torch nor igraph. ``prepare_retrieval_objects`` is the adapted upstream
method of the same name.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import RetrievalConfig
from .embeddings import EmbeddingSet, Encoder
from .image_retrieval import select_documents
from .schemas import (
    VERTEX_LAYERS,
    ChunkRecord,
    Evidence,
    ImageRecord,
    IndexedImage,
    QueryDiagnostics,
    RetrievalResult,
    TextRecord,
)

logger = logging.getLogger(__name__)

TOP_SEEDS = 30
"""Entries kept per seed map in the diagnostics (as upstream logs 30)."""
TOP_PPR = 20
"""Image PPR scores kept in the diagnostics."""


@dataclass(frozen=True)
class Question:
    """What retrieval reads of a question: its ID and text, never labels."""

    qid: str
    text: str


# --- Seeds and fusion (pure numpy) -------------------------------------------


def _require_finite(values: np.ndarray, what: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{what}: nonfinite scores")


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]. Empty stays empty; a constant positive array becomes
    ones and a constant nonpositive one zeros (upstream would divide by zero).
    Nonfinite scores are an error."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s
    _require_finite(s, "min-max normalization")
    low, high = float(s.min()), float(s.max())
    if high == low:
        return np.ones_like(s) if high > 0 else np.zeros_like(s)
    return (s - low) / (high - low)


def normalize_sum(weights: np.ndarray | None, what: str) -> np.ndarray | None:
    """``N(x) = x / sum(x)``; None (the component is unavailable) when the sum
    is not positive. Weights must be finite and nonnegative."""
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64)
    _require_finite(w, what)
    if (w < 0).any():
        raise ValueError(f"{what}: negative weights")
    total = float(w.sum())
    return w / total if total > 0 else None


def fuse_seeds(
    text: np.ndarray | None,
    image: np.ndarray | None,
    crop: np.ndarray | None,
    beta: float,
    visual_mix: float = 0.5,
) -> np.ndarray | None:
    """The PPR reset vector from normalized components (None = unavailable).

    ``v = mix*image + (1-mix)*crop`` if both exist, else the one that does;
    ``s = (1-beta)*text + beta*v`` if both channels exist, else the available
    one; None when neither does (the caller falls back to dense retrieval)."""
    if image is not None and crop is not None:
        visual = visual_mix * image + (1.0 - visual_mix) * crop
    else:
        visual = image if image is not None else crop
    if text is not None and visual is not None:
        return (1.0 - beta) * text + beta * visual
    return text if text is not None else visual


def check_reset(reset: np.ndarray) -> None:
    _require_finite(reset, "reset vector")
    if (reset < 0).any():
        raise ValueError("reset vector: negative weights")
    if abs(float(reset.sum()) - 1.0) > 1e-6:
        raise ValueError(f"reset vector sums to {float(reset.sum())}, not 1")


def personalized_pagerank(graph, reset: np.ndarray, damping: float) -> np.ndarray:
    """CPU igraph PPR over all vertices: undirected, edge ``weight``, PRPACK.
    A graph without edges has nothing to propagate, so the reset is returned."""
    if graph.ecount() == 0:
        return np.asarray(reset, dtype=np.float64).copy()
    scores = np.asarray(
        graph.personalized_pagerank(
            vertices=None, directed=False, damping=damping, weights="weight",
            reset=reset.tolist(), implementation="prpack",
        ),
        dtype=np.float64,
    )
    if scores.shape != reset.shape:
        raise ValueError(f"PPR returned {scores.shape} scores for {reset.shape} vertices")
    _require_finite(scores, "PPR scores")
    return scores


@dataclass
class TextSeed:
    weights: np.ndarray | None
    """``t_raw`` over all graph vertices; None when the channel is unavailable."""
    facts: list[tuple]
    n_scored: int
    reason: str | None = None


def text_seed(backend, question: str) -> TextSeed:
    """Upstream's text seeds for one question (``get_query_embeddings`` must
    have run for it). Raw scores decide which facts pass the threshold;
    min-max normalized scores become the weights."""
    config = backend.global_config
    n_facts = len(backend.fact_node_keys)
    if n_facts == 0:
        return TextSeed(None, [], 0, "the index has no facts")
    raw = np.atleast_1d(np.asarray(backend.get_fact_scores(question, normalize=False), dtype=np.float64))
    if raw.shape != (n_facts,):
        raise ValueError(f"{raw.shape} query-fact scores for {n_facts} facts")
    _require_finite(raw, "query-fact scores")
    normalized = minmax_normalize(raw)
    indices, facts, _ = backend.rerank_facts(question, normalized)
    if not facts:
        return TextSeed(None, [], n_facts, "no fact passed the selection")

    n_vertices = backend.graph.vcount()
    phrase_weights = np.zeros(n_vertices)
    phrase_scores: dict[str, list[float]] = defaultdict(list)
    for rank, fact in enumerate(facts):
        score = float(normalized[indices[rank]])
        for phrase in (fact[0], fact[2]):
            key = backend.entity_id(phrase)
            vertex = backend.node_name_to_vertex_idx.get(key)
            if vertex is not None:
                # As upstream: a later fact overwrites an earlier one for the
                # same entity, then the weight is divided by the entity's passage count.
                phrase_weights[vertex] = score
                n_passages = backend.ent_node_to_num_chunk.get(key, 0)
                if n_passages != 0:
                    phrase_weights[vertex] /= n_passages
            phrase_scores[phrase].append(score)
    linking_scores = {phrase: float(np.mean(scores)) for phrase, scores in phrase_scores.items()}
    if config.linking_top_k:
        phrase_weights, _ = backend.get_top_k_weights(config.linking_top_k, phrase_weights, linking_scores)

    passage_weights = np.zeros(n_vertices)
    if len(backend.passage_node_keys):
        embedding = backend.query_to_embedding["passage"].get(question)
        if embedding is None:
            raise ValueError("the question has no passage embedding: call get_query_embeddings first")
        dense = np.asarray(backend.passage_embeddings @ np.asarray(embedding).reshape(-1), dtype=np.float64)
        _require_finite(dense, "query-passage scores")
        passage_weights[backend.passage_node_idxs] = minmax_normalize(dense) * config.passage_node_weight
    raw_weights = phrase_weights + passage_weights
    if not raw_weights.sum() > 0:
        return TextSeed(None, [tuple(f) for f in facts], n_facts, "no text seed has positive weight")
    return TextSeed(raw_weights, [tuple(f) for f in facts], n_facts)


# --- Index objects -----------------------------------------------------------


@dataclass
class _Candidates:
    """Vectors that can be seeds, with what each seeds."""

    ids: list[str]
    matrix: np.ndarray
    targets: list[Any]

    def top(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """(position, raw cosine similarity) of the k best, ties by stable ID."""
        if not self.ids:
            return []
        scores = np.asarray(self.matrix @ query, dtype=np.float64)
        _require_finite(scores, "query similarities")
        order = sorted(range(len(self.ids)), key=lambda i: (-scores[i], self.ids[i]))[:k]
        return [(i, float(scores[i])) for i in order]


@dataclass
class RetrievalIndex:
    """A loaded index, ready to answer questions. Built by ``build_index``."""

    backend: Any
    settings: RetrievalConfig
    names: list[str]
    """Vertex names by vertex index."""
    dim: int
    image_candidates: _Candidates
    crop_candidates: _Candidates
    images: dict[str, ImageRecord]
    """Usable whole images; ``passage_ids`` are their associated text record IDs."""
    image_vertex: dict[str, int]
    record_images: dict[str, list[str]]
    """Text record -> usable images explicitly associated with it, in source order."""
    record_order: dict[str, int]
    chunks: dict[str, ChunkRecord]
    provenance: dict = field(default_factory=dict)


def prepare_retrieval_objects(backend, memory_path: str | Path) -> None:
    """Adapted upstream ``prepare_retrieval_objects`` for the multimodal graph.

    Upstream asserts that entity, passage and type vertices make up the whole
    graph (images would fail it), keeps stale store rows, and rebuilds an
    entity-to-triples index that re-hashes chunk IDs from their text (which
    would undo source-aware IDs; retrieval does not use that index). Here
    vertex membership comes from the ``layer`` attribute, store rows are
    aligned to IDs with explicit errors on any mismatch, and facts come from
    the final memory."""
    graph = backend.graph
    if graph.vcount() == 0:
        raise ValueError("the saved graph has no vertices")
    names = list(graph.vs["name"])
    layers = list(graph.vs["layer"])
    bad = sorted({layer for layer in layers if layer not in VERTEX_LAYERS})
    if bad:
        raise ValueError(f"unexpected vertex layers {bad}; crops and facts are never vertices")
    if len(set(names)) != len(names):
        raise ValueError("vertex names must be unique")
    layer_of = dict(zip(names, layers))
    by_layer = {layer: [n for n, ln in zip(names, layers) if ln == layer] for layer in VERTEX_LAYERS}

    backend.query_to_embedding = {"triple": {}, "passage": {}}
    backend.entity_node_keys = [k for k in backend.entity_embedding_store.get_all_ids() if layer_of.get(k) == "entity"]
    backend.passage_node_keys = [k for k in backend.chunk_embedding_store.get_all_ids() if layer_of.get(k) == "passage"]
    for layer, keys in (("entity", backend.entity_node_keys), ("passage", backend.passage_node_keys)):
        if set(keys) != set(by_layer[layer]):
            raise ValueError(f"the {layer} vertices of the graph and the {layer} store differ")
    backend.type_node_keys = by_layer["type"]
    backend.image_node_keys = by_layer["image"]

    data = json.loads(Path(memory_path).read_text(encoding="utf-8"))
    fact_keys = list(dict.fromkeys(
        backend.hash_id(str(tuple(map(str, fact["content"]))), "fact-") for fact in data["fact_layer"]
    ))
    stored = set(backend.fact_embedding_store.get_all_ids())
    missing = [k for k in fact_keys if k not in stored]
    if missing:
        raise ValueError(f"{len(missing)} facts of the final memory are not in the fact store")
    backend.fact_node_keys = fact_keys

    backend.node_name_to_vertex_idx = {name: i for i, name in enumerate(names)}
    vertex = backend.node_name_to_vertex_idx
    backend.entity_node_idxs = [vertex[k] for k in backend.entity_node_keys]
    backend.passage_node_idxs = [vertex[k] for k in backend.passage_node_keys]
    backend.image_node_idxs = [vertex[k] for k in backend.image_node_keys]

    if graph.ecount() and "weight" not in graph.es.attributes():
        raise ValueError("the saved graph has no edge weights")
    weights = np.asarray(graph.es["weight"], dtype=np.float64) if graph.ecount() else np.zeros(0)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("edge weights must be finite and nonnegative")
    entity_passages: dict[str, set[str]] = {k: set() for k in backend.entity_node_keys}
    stats: dict[tuple[str, str], float] = {}
    for (s, t), w in zip(graph.get_edgelist(), weights):
        stats[(names[s], names[t])] = float(w)
        if layers[s] == "passage" and layers[t] == "entity":
            entity_passages[names[t]].add(names[s])
        elif layers[t] == "passage" and layers[s] == "entity":
            entity_passages[names[s]].add(names[t])
    backend.node_to_node_stats = stats
    backend.ent_node_to_num_chunk = {k: len(v) for k, v in entity_passages.items()}

    backend.entity_embeddings = np.asarray(backend.entity_embedding_store.get_embeddings(backend.entity_node_keys))
    backend.passage_embeddings = np.asarray(backend.chunk_embedding_store.get_embeddings(backend.passage_node_keys))
    backend.fact_embeddings = np.asarray(backend.fact_embedding_store.get_embeddings(backend.fact_node_keys))
    backend.ready_to_retrieve = True


def build_index(
    backend,
    settings: RetrievalConfig,
    image_set: EmbeddingSet,
    crop_set: EmbeddingSet,
    texts: Sequence[TextRecord],
    chunks: Sequence[ChunkRecord],
    links: Sequence[IndexedImage],
    provenance: Mapping[str, Any] | None = None,
) -> RetrievalIndex:
    """Cross-check the prepared backend, the saved embedding sets and the
    source maps, and derive what retrieval needs. Any inconsistency is an error."""
    vertex = backend.node_name_to_vertex_idx
    names = list(backend.graph.vs["name"])
    layers = list(backend.graph.vs["layer"])
    chunk_map = {c.chunk_id: c for c in chunks}
    if set(chunk_map) != set(backend.passage_node_keys):
        raise ValueError("the graph's passage vertices do not match chunks.jsonl")
    record_order = {t.record_id: i for i, t in enumerate(texts)}
    orphans = [c.chunk_id for c in chunks if c.record_id not in record_order]
    if orphans:
        raise ValueError(f"{len(orphans)} chunks belong to unknown text records (first: {orphans[0]})")

    images: dict[str, ImageRecord] = {}
    image_vertex: dict[str, int] = {}
    record_images: dict[str, list[str]] = defaultdict(list)
    for link in links:
        if not link.usable:
            continue
        if link.vertex not in vertex or layers[vertex[link.vertex]] != "image":
            raise ValueError(f"image {link.image_id}: {link.vertex!r} is not an image vertex of the graph")
        unknown = [r for r in link.record_ids if r not in record_order]
        unknown += [c for c in link.chunk_ids if c not in chunk_map]
        if unknown:
            raise ValueError(f"image {link.image_id} is linked to unknown text: {unknown[:3]}")
        images[link.image_id] = ImageRecord(
            image_id=link.image_id, doc_id=link.doc_id, path=link.path, passage_ids=list(link.record_ids), ok=True,
        )
        image_vertex[link.image_id] = vertex[link.vertex]
        for record_id in link.record_ids:
            record_images[record_id].append(link.image_id)
    if {link.vertex for link in links if link.usable} != set(backend.image_node_keys):
        raise ValueError("the graph's image vertices do not match images.jsonl")

    dim = image_set.meta.dim
    if crop_set.meta.dim != dim:
        raise ValueError(f"crop vectors have dimension {crop_set.meta.dim}, image vectors {dim}")
    embedded = [r for r in image_set.rows if r.row is not None]
    stray = [r.item_id for r in embedded if r.item_id not in images]
    if stray:
        raise ValueError(f"{len(stray)} embedded images are not in the index (first: {stray[0]}): rerun embed and assemble")
    absent = sorted(set(images) - {r.item_id for r in embedded})
    if absent:
        raise ValueError(f"{len(absent)} indexed images have no vector (first: {absent[0]})")
    image_candidates = _Candidates(
        ids=[r.item_id for r in embedded], matrix=image_set.vectors,
        targets=[image_vertex[r.item_id] for r in embedded],
    )

    crop_ids: list[str] = []
    crop_rows: list[int] = []
    crop_targets: list[tuple[str, int]] = []
    for r in crop_set.rows:
        if r.row is None or r.image_id not in images:
            continue  # failed crops and crops of images that are not indexed seed nothing
        entity = vertex.get(r.entity_id)
        if entity is None or layers[entity] != "entity":
            raise ValueError(f"crop {r.item_id} grounds {r.entity_id}, which is not an entity vertex")
        crop_ids.append(r.item_id)
        crop_rows.append(r.row)
        crop_targets.append((r.image_id, entity))
    crop_matrix = crop_set.vectors[crop_rows] if crop_rows else np.zeros((0, dim), dtype=np.float32)

    return RetrievalIndex(
        backend=backend, settings=settings, names=names, dim=dim,
        image_candidates=image_candidates,
        crop_candidates=_Candidates(crop_ids, crop_matrix, crop_targets),
        images=images, image_vertex=image_vertex, record_images=dict(record_images),
        record_order=record_order, chunks=chunk_map, provenance=dict(provenance or {}),
    )


# --- Retrieval ---------------------------------------------------------------


def visual_seeds(index: RetrievalIndex, query: np.ndarray):
    """Raw (unnormalized) image and crop-entity seed arrays over all vertices
    (None when the channel has no candidates), and their readable maps.

    Whole images and crops are ranked independently by raw cosine similarity
    and the top ``top_k_candidates`` of each kept. Negative similarities
    become zero. Crop scores reach existing entity vertices: the maximum per
    (image, entity), then the maximum across images per entity. SAM3
    confidence, entity image counts and min-max scaling play no part."""
    n_vertices = len(index.names)
    k = index.settings.top_k_candidates

    image_weights = np.zeros(n_vertices)
    image_seeds: dict[str, float] = {}
    for position, score in index.image_candidates.top(query, k):
        weight = max(score, 0.0)
        image_weights[index.image_candidates.targets[position]] = weight
        image_seeds[index.image_candidates.ids[position]] = weight

    per_pair: dict[tuple[str, int], float] = {}
    for position, score in index.crop_candidates.top(query, k):
        key = index.crop_candidates.targets[position]
        per_pair[key] = max(per_pair.get(key, 0.0), max(score, 0.0))
    per_entity: dict[int, float] = {}
    for (_, entity), weight in per_pair.items():
        per_entity[entity] = max(per_entity.get(entity, 0.0), weight)
    crop_weights = np.zeros(n_vertices)
    for entity, weight in per_entity.items():
        crop_weights[entity] = weight
    crop_seeds = {index.names[entity]: weight for entity, weight in per_entity.items()}
    return image_weights, crop_weights, image_seeds, crop_seeds


def fallback_evidence(index: RetrievalIndex, question: str) -> tuple[list[Evidence], str | None]:
    """Dense text retrieval over passages and captions (the chunks): the best
    chunk of each of at most ``n_docs`` documents, scored by raw similarity.
    Each document gets the first usable whole image explicitly associated with
    its chunk's text record, in source order, or none; direct image retrieval
    is never substituted."""
    backend = index.backend
    keys = backend.passage_node_keys
    if not keys:
        return [], "the index has no text"
    embedding = backend.query_to_embedding["passage"].get(question)
    if embedding is None:
        raise ValueError("the question has no passage embedding: call get_query_embeddings first")
    scores = np.asarray(backend.passage_embeddings @ np.asarray(embedding).reshape(-1), dtype=np.float64)
    _require_finite(scores, "query-passage scores")
    best: dict[str, tuple[ChunkRecord, float]] = {}
    for position in np.argsort(-scores, kind="stable"):
        chunk = index.chunks[keys[position]]
        best.setdefault(chunk.doc_id, (chunk, float(scores[position])))
        if len(best) >= index.settings.n_docs:
            break
    evidence = []
    for doc_id, (chunk, score) in best.items():
        associated = index.record_images.get(chunk.record_id, [])
        image_id = associated[0] if associated else None
        records = {chunk.record_id}
        if image_id is not None:
            records.update(index.images[image_id].passage_ids)
        evidence.append(Evidence(
            doc_id=doc_id, score=score, image_id=image_id,
            passage_ids=sorted(records, key=index.record_order.__getitem__),
        ))
    return evidence, None


def retrieve_one(
    index: RetrievalIndex, question: Question, query: np.ndarray, truncated: bool, beta: float
) -> tuple[RetrievalResult, QueryDiagnostics]:
    backend, settings = index.backend, index.settings
    text = text_seed(backend, question.text)
    text_part = normalize_sum(text.weights, "text seeds")
    image_weights, crop_weights, image_seeds, crop_seeds = visual_seeds(index, query)
    image_part = normalize_sum(image_weights, "image seeds")
    crop_part = normalize_sum(crop_weights, "crop seeds")
    reset = fuse_seeds(text_part, image_part, crop_part, beta, settings.visual_mix)

    evidence: list[Evidence] = []
    ppr_top: dict[str, float] = {}
    reason: str | None = None
    if reset is None:
        reason = "neither the text nor the visual channel has seeds"
    else:
        check_reset(reset)
        scores = personalized_pagerank(backend.graph, reset, settings.damping)
        all_images = sorted(
            ((image_id, float(scores[vertex])) for image_id, vertex in index.image_vertex.items()),
            key=lambda kv: (-kv[1], kv[0]),
        )
        ppr_top = dict(all_images[:TOP_PPR])
        positive = {image_id: score for image_id, score in all_images if score > 0}
        if positive:
            evidence = select_documents(positive, index.images, settings.n_docs)
        else:
            reason = "no image vertex received PPR mass"
    route = "graph"
    if not evidence:
        route = "fallback"
        evidence, empty_reason = fallback_evidence(index, question.text)
        reason = reason or empty_reason

    result = RetrievalResult(question_id=question.qid, system="G", route=route, evidence=evidence)
    diagnostics = QueryDiagnostics(
        question_id=question.qid, route=route, fallback_reason=reason if route == "fallback" else None,
        beta=beta, text_available=text_part is not None, image_available=image_part is not None,
        crop_available=crop_part is not None, text_reason=text.reason, query_truncated=truncated,
        facts=[str(f) for f in text.facts], n_facts_scored=text.n_scored,
        text_seeds=_top(text.weights, index.names), image_seeds=_top_items(image_seeds),
        crop_entity_seeds=_top_items(crop_seeds), image_ppr=ppr_top,
        selected_doc_ids=[e.doc_id for e in evidence], selected_image_ids=[e.image_id for e in evidence],
    )
    return result, diagnostics


def retrieve_questions(
    index: RetrievalIndex, questions: Sequence[Question], encoder: Encoder, beta: float | None = None
) -> list[tuple[RetrievalResult, QueryDiagnostics]]:
    """Evidence and diagnostics for each question. Questions are encoded once
    with EVA and once with the text embedder; model failures and nonfinite
    scores raise instead of becoming empty evidence."""
    beta = index.settings.beta if beta is None else beta
    if beta is None:
        raise ValueError("beta is not set: pass it explicitly or set retrieval.beta")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if not questions:
        return []
    ids = [q.qid for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("question IDs must be unique")
    texts = [q.text for q in questions]
    vectors, truncated = encoder.encode_texts(texts)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(texts), index.dim):
        raise ValueError(f"the encoder returned {vectors.shape} vectors, expected {(len(texts), index.dim)}")
    _require_finite(vectors, "question vectors")
    index.backend.get_query_embeddings(texts)
    return [
        retrieve_one(index, question, vectors[i], bool(truncated[i]), beta)
        for i, question in enumerate(questions)
    ]


def _top(weights: np.ndarray | None, names: Sequence[str]) -> dict[str, float]:
    if weights is None:
        return {}
    order = sorted(np.flatnonzero(weights), key=lambda i: (-weights[i], names[i]))[:TOP_SEEDS]
    return {names[i]: float(weights[i]) for i in order}


def _top_items(seeds: Mapping[str, float]) -> dict[str, float]:
    return dict(sorted(seeds.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_SEEDS])
