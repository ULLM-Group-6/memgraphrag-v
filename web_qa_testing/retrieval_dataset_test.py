import argparse
import json
import logging
import os
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

def clean_text(text):
    return re.sub(r'^\d+:', '', str(text)).strip().lower()

def main():
    parser = argparse.ArgumentParser(description="Parquet Retrieval with Direct Line-Index Mapping")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--passages", required=True)
    parser.add_argument("--retrieval-top-k", type=int, default=50)
    args = parser.parse_args()

    LOGGER.info("Building text-to-line-index map from %s", args.passages)
    text_to_line_idx = {}
    with open(args.passages, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text") or item.get("content") or ""
            text_to_line_idx[clean_text(text)] = line_idx

    LOGGER.info("Loaded %d passage line indices", len(text_to_line_idx))

    with open(args.questions, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        questions = [q["question"] if isinstance(q, dict) and "question" in q else str(q) for q in data]
    elif isinstance(data, dict):
        questions = [v["question"] if isinstance(v, dict) and "question" in v else str(v) for v in data.values()]
    else:
        raise ValueError("Unsupported questions format.")

    save_dir = Path(args.save_dir)
    parquet_path = save_dir / "chunk_embeddings" / "vdb_chunk.parquet"
    if not parquet_path.exists():
        parquet_path = save_dir / "vdb_chunk.parquet"

    LOGGER.info("Reading parquet directly from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    mapped_doc_strings = []
    unmapped = 0
    for idx, text in enumerate(df["content"]):
        ctext = clean_text(text)
        if ctext in text_to_line_idx:
            line_i = text_to_line_idx[ctext]
            mapped_doc_strings.append(f"{line_i}:{ctext}")
        else:
            mapped_doc_strings.append(f"{idx}:{ctext}")
            unmapped += 1

    LOGGER.info("Mapped %d/%d chunks to line indices", len(mapped_doc_strings) - unmapped, len(mapped_doc_strings))

    passage_matrix = np.vstack(df["embedding"].values)

    LOGGER.info("Loading BGE encoder...")
    encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

    solutions = []
    top_k = args.retrieval_top_k

    for q in questions:
        q_emb = encoder.encode(q, normalize_embeddings=True)
        scores = np.dot(passage_matrix, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]
        doc_ids = [mapped_doc_strings[i] for i in top_indices if i < len(mapped_doc_strings)]
        doc_scores = [float(scores[i]) for i in top_indices]

        solutions.append({
            "question": q,
            "docs": doc_ids,
            "doc_scores": doc_scores
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "questions_path": args.questions,
            "save_dir": args.save_dir,
            "num_questions": len(questions),
            "solutions": solutions
        }, f, indent=2)

    LOGGER.info("Saved %d results to %s", len(solutions), args.output)

if __name__ == "__main__":
    main()
