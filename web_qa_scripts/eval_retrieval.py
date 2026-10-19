#!/usr/bin/env python3
"""Document-ID Recall@5 for passage-based systems (T, C) from MemGraphRAG's qa_results.json.
Gold documents = doc_id of gold images and gold snippets. Grouped bootstrap CI (questions sharing a gold document resampled together)."""
import argparse, collections, json, os, random, sys
from webqa_common import jl, ranked_docs

ap = argparse.ArgumentParser()
ap.add_argument("--qa-results", required=True)
ap.add_argument("--qa-results2", help="optional second system on the same questions: paired difference")
ap.add_argument("--questions", required=True, help="build_qa.json or test_qa.json")
ap.add_argument("--passages", required=True, help="passages.jsonl of system 1's corpus")
ap.add_argument("--passages2", help="passages.jsonl of system 2's corpus (default: same as --passages)")
ap.add_argument("--data-dir", required=True, help="folder with image_manifest.jsonl and text_manifest.jsonl")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", help="write per-question results as JSON")
a = ap.parse_args()

img_doc = {r["image_id"]: r["doc_id"] for r in jl(os.path.join(a.data_dir, "image_manifest.jsonl"))}
txt_doc = {r["passage_id"]: r["doc_id"] for r in jl(os.path.join(a.data_dir, "text_manifest.jsonl"))}
Q = json.load(open(a.questions, encoding="utf-8"))

def gold_docs(q):
    d = set()
    for i in q.get("gold_image_ids", []):
        if i not in img_doc: sys.exit(f"{q['qid']}: gold image {i} not in image manifest")
        d.add(img_doc[i])
    for s in q.get("gold_snippet_ids", []):
        if "txt_" + s not in txt_doc: sys.exit(f"{q['qid']}: gold snippet {s} not in text manifest")
        d.add(txt_doc["txt_" + s])
    if not d: sys.exit(f"{q['qid']}: empty gold evidence set")
    return d
G = [gold_docs(q) for q in Q]

def recalls(ranked):
    return [len(set(r["docs"]) & g) / len(g) for r, g in zip(ranked, G)]

# groups: questions sharing any gold document are resampled together
parent = {}
def find(x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for g in G:
    gl = sorted(g)
    for d in gl[1:]: parent[find(d)] = find(gl[0])
    find(gl[0])
groups = collections.defaultdict(list)
for qi, g in enumerate(G): groups[find(sorted(g)[0])].append(qi)
groups = list(groups.values())

def boot_ci(vals):
    rng = random.Random(a.seed); stats = []
    for _ in range(a.n_boot):
        pick = [groups[rng.randrange(len(groups))] for _ in groups]
        n = sum(len(x) for x in pick)
        stats.append(sum(vals[i] for x in pick for i in x) / n)
    stats.sort()
    return stats[int(0.025 * a.n_boot)], stats[int(0.975 * a.n_boot) - 1]

r1 = ranked_docs(a.qa_results, a.passages, Q, a.k); v1 = recalls(r1)
mean = lambda v: sum(v) / len(v)
lo, hi = boot_ci(v1)
print(f"questions: {len(Q)}  independent groups: {len(groups)}  (largest groups: {sorted(map(len, groups), reverse=True)[:5]})")
print(f"Document-ID Recall@{a.k}: {mean(v1):.3f}  95% grouped-bootstrap CI [{lo:.3f}, {hi:.3f}]")
print(f"questions with fewer than {a.k} unique documents returned: {sum(r['n_unique'] < a.k for r in r1)}")
bycat = collections.defaultdict(list)
for q, v in zip(Q, v1): bycat[q.get("Qcate")].append(v)
print("by Qcate:", {k: f"{mean(v):.3f} (n={len(v)})" for k, v in sorted(bycat.items())})
byev = collections.defaultdict(list)
for q, v in zip(Q, v1): byev[q.get("evidence_class")].append(v)
print("by evidence class:", {k: f"{mean(v):.3f} (n={len(v)})" for k, v in sorted(byev.items())})
res = {"recall@k": mean(v1), "ci": [lo, hi], "n": len(Q), "groups": len(groups), "per_question": v1}
if a.qa_results2:
    r2 = ranked_docs(a.qa_results2, a.passages2 or a.passages, Q, a.k); v2 = recalls(r2)
    diff = [x - y for x, y in zip(v2, v1)]
    dlo, dhi = boot_ci(diff)
    print(f"system 2 Recall@{a.k}: {mean(v2):.3f};  paired difference (2 - 1): {mean(diff):+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}]")
    res.update({"recall2@k": mean(v2), "diff": mean(diff), "diff_ci": [dlo, dhi]})
if a.out: json.dump(res, open(a.out, "w"), indent=1)
