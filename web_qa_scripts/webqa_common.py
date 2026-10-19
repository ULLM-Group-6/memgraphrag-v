import json, re, sys

def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def ranked_docs(res_path, passages_path, Q, k=5):
    """Per question: top-k unique documents (best passage kept per document) from a MemGraphRAG qa_results.json."""
    P = jl(passages_path)
    sols = json.load(open(res_path, encoding="utf-8"))["solutions"]
    if len(sols) != len(Q):
        sys.exit(f"{res_path}: {len(sols)} results but {len(Q)} questions")
    out = []
    for s, q in zip(sols, Q):
        if s["question"].strip() != q["question"].strip():
            sys.exit(f"order mismatch at question {q['qid']}")
        docs, scores = s["docs"], s.get("doc_scores")
        order = list(range(len(docs)))
        if scores is not None and len(scores) == len(docs):
            order.sort(key=lambda i: -scores[i])          # stable: ties keep retrieval order
        top, seen, best = [], set(), {}
        for i in order:
            m = re.match(r"^(\d+):", docs[i])
            if not m:
                sys.exit("passage without 'idx:' prefix: the one-chunk-per-passage driver was not used")
            p = P[int(m.group(1))]
            if p["doc_id"] not in seen:
                seen.add(p["doc_id"]); top.append(p["doc_id"]); best[p["doc_id"]] = p["passage_id"]
            if len(top) == k:
                break
        out.append({"docs": top, "best_passage": best, "n_unique": len(top), "n_passages_returned": len(docs)})
    return out
