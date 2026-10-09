"""Whole-image retrieval (plan task 3).

- ``image_passage_edges``: graph edges from each image node to its document's
  passages, weight 1, for the PPR graph.
- ``select_documents``: image scores -> top documents, each with its single
  best image. Shared by direct retrieval (D, cosine scores) and the graph
  method (G, PPR image-node scores).
- ``retrieve_direct``: system D, the EVA-CLIP question vector against every
  whole-image vector.

Evidence lists each image's associated passages in source order; trimming
them to the reader's 256-token budget happens when the reader input is built.

Run after embedding images and questions:
    python -m memgraphrag_v.image_retrieval --run-id d-dev [--split dev]
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable, Mapping, Sequence

import numpy as np

from .embeddings import EmbeddingSet
from .io import read_jsonl, write_jsonl
from .schemas import MAX_DOCS, Evidence, ImageRecord, QuestionRecord, RetrievalResult


def image_passage_edges(images: Iterable[ImageRecord], image_set: EmbeddingSet) -> list[tuple[str, str, float]]:
    """(image_id, passage_id, 1.0) for every image that has a vector. Images
    without one stay out of the graph, so PPR never selects an image the
    reader cannot see."""
    return [
        (i.image_id, p, 1.0)
        for i in images
        if i.image_id in image_set and image_set.row_for(i.image_id).row is not None
        for p in i.passage_ids
    ]


def select_documents(
    image_scores: Mapping[str, float], images: Mapping[str, ImageRecord], n_docs: int = MAX_DOCS
) -> list[Evidence]:
    """Score each document by its best image and keep the top ``n_docs``
    documents with that image. Ties are broken by image_id."""
    best: dict[str, tuple[str, float]] = {}
    for image_id, score in sorted(image_scores.items(), key=lambda kv: (-kv[1], kv[0])):
        best.setdefault(images[image_id].doc_id, (image_id, score))
    return [
        Evidence(doc_id=doc_id, score=score, image_id=image_id, passage_ids=images[image_id].passage_ids)
        for doc_id, (image_id, score) in list(best.items())[:n_docs]
    ]


def retrieve_direct(
    questions: Sequence[QuestionRecord],
    query_set: EmbeddingSet,
    image_set: EmbeddingSet,
    images: Iterable[ImageRecord],
    n_docs: int = MAX_DOCS,
    system: str = "D",
) -> list[RetrievalResult]:
    """System D: rank whole images by cosine similarity to the question."""
    by_id = {i.image_id: i for i in images}
    return [
        RetrievalResult(
            question_id=q.qid,
            system=system,
            route="direct",
            evidence=select_documents(image_set.scores(query_set.vector(q.qid)), by_id, n_docs),
        )
        for q in questions
    ]


def gold_doc_hits(results: Sequence[RetrievalResult], questions: Sequence[QuestionRecord]) -> dict[str, int]:
    """Questions whose gold document is ranked first / anywhere. A sanity
    check only; evaluation (plan task 6) computes the real metrics."""
    gold = {q.qid: set(q.gold_doc_ids) for q in questions}
    ranked = [(gold[r.question_id], [e.doc_id for e in r.evidence]) for r in results]
    return {
        "questions": len(ranked),
        "gold_doc_at_1": sum(bool(docs) and docs[0] in g for g, docs in ranked),
        f"gold_doc_in_top_{MAX_DOCS}": sum(bool(g & set(docs)) for g, docs in ranked),
    }


def main(argv: list[str] | None = None) -> None:
    from .config import ExperimentConfig
    from .paths import Artifacts

    parser = argparse.ArgumentParser(description="Direct whole-image retrieval (system D).")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["dev", "test"], help="only questions from this split")
    args = parser.parse_args(argv)

    config = ExperimentConfig.load(args.config)
    artifacts = Artifacts.from_env(config.artifacts_root)
    encoder = config.visual_encoder.name
    image_set = EmbeddingSet.load(artifacts.embeddings_dir(encoder, "image"))
    query_set = EmbeddingSet.load(artifacts.embeddings_dir(encoder, "query"))
    images = read_jsonl(artifacts.image_manifest, ImageRecord)
    questions = [
        q for q in read_jsonl(artifacts.questions, QuestionRecord)
        if q.qid in query_set and (args.split is None or q.split == args.split)
    ]

    results = retrieve_direct(questions, query_set, image_set, images, config.retrieval.n_docs)
    run_dir = artifacts.run_dir(args.run_id)
    write_jsonl(run_dir / "retrieval.jsonl", results)
    config.save(run_dir / "config.yaml")
    print(json.dumps({"out": str(run_dir), **gold_doc_hits(results, questions)}, indent=2))


if __name__ == "__main__":
    main()
