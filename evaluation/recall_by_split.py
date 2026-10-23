"""Document-ID Recall@5 of a MemGraphRAG-V retrieval run next to the dense baselines, on one split,
for the paper's build-vs-test table.

Reads retrieval.jsonl and labels.jsonl of a retrieval run (the dev run of one beta, or the test run).
Recall is computed as in evaluate_webqa.py: |gold doc IDs & top-5 doc IDs| / |gold doc IDs|.
Checks that labels.gold_doc_ids equal the gold documents eval_retrieval.py derives from the
manifests, so the baselines and MemGraphRAG-V are scored against the same gold sets.

Reports all questions, the image and text questions, and, with --test-questions, the beta-tuning
subset (build questions whose gold evidence is not shared with a test question, as in
scripts/webqa_tuning.py). Intervals for every system use evaluate_webqa.py's grouping and percentile
bootstrap; paired differences use extra_metrics.py. Standard library only.
"""
import argparse
import json
from pathlib import Path

from evaluate_webqa import clusters, metric, recall, rows, save_json
from extra_metrics import paired


def evidence_groups(questions):
    """Same connected components as memgraphrag_v.webqa._evidence_groups (shared gold image or snippet)."""
    parent = list(range(len(questions)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first = {}
    for i, q in enumerate(questions):
        for source in [f"img:{s}" for s in q["gold_image_ids"]] + [f"txt:{s}" for s in q["gold_snippet_ids"]]:
            if source in first:
                parent[root(i)] = root(first[source])
            else:
                first[source] = i
    return [root(i) for i in range(len(questions))]


def summary(values, gold, baselines, subset, repetitions, seed):
    ours = [values[i] for i in subset]
    groups = clusters([{"gold_doc_ids": gold[i]} for i in subset])
    out = {"questions": len(subset), "gold_document_groups": len(groups),
           "memgraphrag_v": metric(ours, groups, repetitions, seed), "baselines": {}}
    for name, per_question in baselines.items():
        base = [per_question[i] for i in subset]
        out["baselines"][name] = dict(metric(base, groups, repetitions, seed),
                                      paired=paired(ours, base, groups, repetitions, seed))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="retrieval run with retrieval.jsonl and labels.jsonl")
    p.add_argument("--questions", type=Path, required=True, help="build_qa.json or test_qa.json (baseline order)")
    p.add_argument("--test-questions", type=Path, help="test_qa.json; with build questions, adds the tuning subset")
    p.add_argument("--data-dir", type=Path, required=True, help="folder with image_ and text_manifest.jsonl")
    p.add_argument("--baseline", action="append", default=[], metavar="NAME=RECALL_JSON")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    retrieval = {r["question_id"]: r for r in rows(args.run_dir / "retrieval.jsonl")}
    labels = {r["qid"]: r for r in rows(args.run_dir / "labels.jsonl")}
    qids = [q["qid"] for q in questions]
    if set(qids) != set(retrieval) or set(qids) != set(labels) or len(qids) != len(retrieval):
        raise SystemExit("Question IDs differ between the run and --questions")

    img_doc = {r["image_id"]: r["doc_id"] for r in rows(args.data_dir / "image_manifest.jsonl")}
    txt_doc = {r["passage_id"]: r["doc_id"] for r in rows(args.data_dir / "text_manifest.jsonl")}
    gold = []
    for q in questions:
        manifest_gold = {img_doc[i] for i in q["gold_image_ids"]} | {txt_doc["txt_" + s] for s in q["gold_snippet_ids"]}
        if manifest_gold != set(labels[q["qid"]]["gold_doc_ids"]):
            raise SystemExit(f"{q['qid']}: labels.gold_doc_ids differ from the manifest gold documents")
        gold.append(sorted(manifest_gold))

    values, duplicates, image_evidence, evidence = [], 0, 0, 0
    for q, g in zip(questions, gold):
        items = retrieval[q["qid"]]["evidence"]
        top = [e["doc_id"] for e in items]
        if len(top) > 5:
            raise SystemExit(f"{q['qid']}: more than five retrieved documents")
        duplicates += len(top) != len(set(top))
        image_evidence += sum(e.get("image_id") is not None for e in items)
        evidence += len(items)
        values.append(recall(g, top))

    baselines = {}
    for spec in args.baseline:
        name, path = spec.split("=", 1)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if len(data["per_question"]) != len(questions):
            raise SystemExit(f"{name}: {len(data['per_question'])} per-question values, expected {len(questions)}")
        if abs(sum(data["per_question"]) / len(questions) - data["recall@k"]) > 1e-12:
            raise SystemExit(f"{name}: per-question values do not average to recall@k")
        baselines[name] = data["per_question"]

    subsets = {"all": list(range(len(questions)))}
    for kind in ("image", "text"):
        subset = [i for i, q in enumerate(questions) if q["evidence_class"] == kind]
        if subset:
            subsets[kind + "_questions"] = subset
    if args.test_questions:
        test = json.loads(args.test_questions.read_text(encoding="utf-8"))
        groups = evidence_groups(questions + test)
        shared = set(groups[:len(questions)]) & set(groups[len(questions):])
        subsets["tuning_eligible"] = [i for i in range(len(questions)) if groups[i] not in shared]

    result = {"run": args.run_dir.name, "questions_file": args.questions.name,
              "bootstrap": {"resamples": args.bootstrap, "seed": args.seed},
              "questions_with_duplicate_doc_ids": duplicates,
              "retrieved_documents": evidence, "retrieved_documents_with_image": image_evidence}
    result.update({name: summary(values, gold, baselines, subset, args.bootstrap, args.seed)
                   for name, subset in subsets.items()})
    save_json(args.output, result)


if __name__ == "__main__":
    main()
