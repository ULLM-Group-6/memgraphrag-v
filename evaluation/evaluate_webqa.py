"""Offline evaluation of a completed webqa_test_single run. Python standard library only."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import unicodedata


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def stamp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ids(value, name):
    require(isinstance(value, list) and all(isinstance(x, str) and x for x in value),
            f"{name} must be a list of nonempty string IDs")
    require(len(value) == len(set(value)), f"Duplicate IDs in {name}")
    return value


def keyed(records, key):
    result = {}
    for r in records:
        require(r[key] not in result, f"Duplicate question ID: {r[key]}")
        result[r[key]] = r
    return result


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def contains_reference(prediction, references):
    prediction = normalize(prediction)
    return any(ref and re.search(r"(?<!\w)" + re.escape(ref) + r"(?!\w)", prediction)
               for ref in map(normalize, references))


def recall(gold, retrieved):
    return len(set(gold) & set(retrieved)) / len(set(gold)) if gold else None


def clusters(records):
    """Connected components of shared gold documents, including transitive overlap."""
    parent = list(range(len(records)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, r in enumerate(records):
        for doc in r["gold_doc_ids"]:
            if doc in owners:
                parent[root(i)] = root(owners[doc])
            owners[doc] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[root(i)].append(i)
    return list(groups.values())


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def metric(values, groups, repetitions, seed):
    valid = [v for v in values if v is not None]
    if not valid:
        return {"value": None, "n": 0, "ci95": None}
    active = [[i for i in group if values[i] is not None] for group in groups]
    active = [g for g in active if g]
    ci = None
    if len(active) >= 2:
        rng = random.Random(seed)
        samples = []
        for _ in range(repetitions):
            sample = [values[i] for g in rng.choices(active, k=len(active)) for i in g]
            samples.append(statistics.mean(sample))
        ci = [percentile(samples, .025), percentile(samples, .975)]
    return {"value": statistics.mean(valid), "n": len(valid), "ci95": ci}


def load_run(directory, expected):
    data = rows(directory / "evaluation_inputs.jsonl")
    done, contract = read(directory / "completed.json"), read(directory / "run.json")
    answers = keyed(rows(directory / "answers.jsonl"), "question_id")
    retrieval = keyed(rows(directory / "retrieval.jsonl"), "question_id")
    labels = keyed(rows(directory / "labels.jsonl"), "qid")
    require(len(data) == expected == done["questions"], f"Expected {expected} completed questions")
    require(done["answers_sha256"] == sha(directory / "answers.jsonl"), "Answer file hash mismatch")
    require(done["run_sha256"] == stamp(contract), "Run contract hash mismatch")
    require(contract["split"] == "test" and not contract["partial_run"] and not contract["smoke_only"],
            "Only full, non-smoke test runs can be evaluated")
    by_id = keyed(data, "question_id")
    pairs = contract["questions"]
    require(len(pairs) == expected and len({p[0] for p in pairs}) == expected, "Invalid contract coverage")
    require(set(by_id) == set(answers) == set(labels) == set(retrieval) == {p[0] for p in pairs}, "Question coverage mismatch")
    for r, pair in zip(data, pairs):
        qid = r["question_id"]
        a, label = answers[qid], labels[qid]
        require([qid, r["question"]] == pair, "Question text/order mismatch")
        require(r["split"] == "test" and not r["partial_run"], "Partial/non-test record")
        for key in ("gold_doc_ids", "gold_image_ids"):
            ids(r[key], key)
            require(r[key] == label[key], f"{qid}: label mismatch for {key}")
        require(r["gold_doc_ids"], f"{qid}: empty gold document IDs")
        for key in ("top_k_doc_ids", "supplied_image_ids"):
            ids(r[key], key)
            require(len(r[key]) <= 5, f"{qid}: more than five {key}")
        require(r["top_k_doc_ids"] == [e["doc_id"] for e in retrieval[qid]["evidence"]]
                and r["route"] == retrieval[qid]["route"], f"{qid}: retrieval mismatch")
        require(isinstance(r["prediction"], str), "Prediction must be text")
        refs = r["reference_answers"]
        require(isinstance(refs, list) and refs and all(isinstance(x, str) for x in refs)
                and any(normalize(x) for x in refs), f"{qid}: invalid references")
        require(refs == label["answers"], f"{qid}: reference-answer mismatch")
        require(r["prediction"] == a["answer"] and r["status"] == a["status"]
                and r["supplied_image_ids"] == a["supplied_image_ids"], f"{qid}: answer mismatch")
        require(r["route"] in ("graph", "fallback"), f"{qid}: unexpected retrieval route")
    return data, contract, answers


def evaluate(data, repetitions=2000, seed=42):
    scored = []
    for r in data:
        d = recall(r["gold_doc_ids"], r["top_k_doc_ids"])
        image = recall(r["gold_image_ids"], r["supplied_image_ids"])
        ok = r["status"] == "ok" and bool(r["prediction"].strip())
        match = bool(ok and contains_reference(r["prediction"], r["reference_answers"]))
        category = ("execution_failure_or_empty_answer" if not ok else
                    "containment_match" if match else
                    "no_gold_document_retrieved" if d == 0 else
                    "some_gold_documents_missing" if d < 1 else
                    "gold_documents_found_but_gold_images_missing" if image is not None and image < 1 else
                    "gold_evidence_ids_present_reader_or_metric_review")
        scored.append(dict(r, document_recall_at_5=d, supplied_image_recall_at_5=image,
                           containment_match=match, answer_success=ok, review_category=category,
                           missing_gold_doc_ids=sorted(set(r["gold_doc_ids"]) - set(r["top_k_doc_ids"])),
                           missing_gold_image_ids=sorted(set(r["gold_image_ids"]) - set(r["supplied_image_ids"]))))
    groups = clusters(data)
    metrics = {name: metric([r[key] for r in scored], groups, repetitions, seed)
               for name, key in (("Document-ID Recall@5", "document_recall_at_5"),
                                 ("Supplied-image Recall@5", "supplied_image_recall_at_5"),
                                 ("Answer containment accuracy (heuristic)", "containment_match"))}
    return scored, {"questions": len(data), "gold_document_groups": len(groups),
        "group_sizes": [len(g) for g in groups], "metrics": metrics,
        "answer_success_count": sum(r["answer_success"] for r in scored),
        "statuses": dict(Counter(r["status"] for r in scored)),
        "routes": dict(Counter(r["route"] for r in scored)),
        "review_categories": dict(Counter(r["review_category"] for r in scored)),
        "bootstrap": {"resamples": repetitions, "seed": seed,
            "method": "95% percentile; sample shared-gold-document connected components with replacement; question-weighted mean"}}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--expected-questions", type=int, default=120)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--examples", type=int, default=8)
    args = p.parse_args()
    require(args.bootstrap >= 100 and args.examples >= 0, "Use at least 100 bootstrap resamples and nonnegative examples")
    data, contract, answers = load_run(args.run_dir, args.expected_questions)
    scored, summary = evaluate(data, args.bootstrap, args.seed)
    summary.update(selected_beta=contract["beta"], system=contract["system"],
                   input_sha256={name: sha(args.run_dir / name) for name in
                       ("evaluation_inputs.jsonl", "answers.jsonl", "labels.jsonl", "retrieval.jsonl", "run.json", "completed.json")},
                   evaluator_sha256=sha(Path(__file__)))
    out = args.output or args.run_dir / "evaluation_report"
    require(out.resolve() != args.run_dir.resolve(), "Output must be a separate directory")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "summary.json", summary)
    with (out / "per_question.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["question_id", "question", "document_recall_at_5", "supplied_image_recall_at_5",
                  "containment_match", "answer_success", "review_category", "route", "prediction",
                  "reference_answers", "gold_doc_ids", "top_k_doc_ids", "gold_image_ids", "supplied_image_ids"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in scored:
            writer.writerow({k: json.dumps(r[k], ensure_ascii=False) if isinstance(r[k], list) else r[k] for k in fields})
    lines = ["# WebQA test results", "", f"Questions: {len(data)}; beta: {contract['beta']}; "
             f"gold-document groups: {summary['gold_document_groups']}.", "",
             "| Metric | Score (%) | 95% grouped-bootstrap CI (%) | N |",
             "|---|---:|---|---:|"]
    for name, m in summary["metrics"].items():
        value = "N/A" if m["value"] is None else f"{100*m['value']:.2f}"
        ci = "N/A" if m["ci95"] is None else f"[{100*m['ci95'][0]:.2f}, {100*m['ci95'][1]:.2f}]"
        lines.append(f"| {name} | {value} | {ci} | {m['n']} |")
    lines.extend(["", "Recall is the mean per-question fraction of gold IDs recovered, not any-hit accuracy.",
        "Image recall uses images actually supplied to the reader; only questions with gold images are included.",
        "Containment is a heuristic, NOT judged semantic accuracy. Paraphrases can fail it; incorrect answers can pass it.",
        "Failed/empty answers count as zero containment matches; their saved retrieval is still scored.",
        "Bootstrap clusters are connected components of shared gold document IDs, including transitive overlap.",
        "The stored group_id is not used. Match this grouping in baseline evaluation before comparing intervals.",
        "Intervals describe this selected sample; few clusters limit generalization. No paired system comparison is performed.",
        "", f"Successful nonempty answers: {summary['answer_success_count']}/{len(data)}.",
        f"Routes: {summary['routes']}. Statuses: {summary['statuses']}."])
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    review = [r for r in scored if not r["containment_match"]]
    for r in review:
        r["supplied_documents"] = answers[r["question_id"]].get("documents", [])
    (out / "review_candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review), encoding="utf-8")
    # Round-robin categories: avoid printing only one kind of retrieval failure.
    buckets = defaultdict(list)
    for r in sorted(review, key=lambda r: r["question_id"]):
        buckets[r["review_category"]].append(r)
    examples = []
    while len(examples) < args.examples and any(buckets.values()):
        for category in sorted(buckets):
            if buckets[category] and len(examples) < args.examples:
                examples.append(buckets[category].pop(0))
    text = ["# Candidates for manual error analysis", "",
            "These fail reference containment; they are NOT confirmed incorrect answers. Missing evidence IDs suggest a retrieval gap, not proven causation.", ""]
    for r in examples:
        text.extend([f"## {r['question_id']}: {r['review_category']}",
            f"Question: {r['question']}", f"References: {r['reference_answers']}",
            f"Prediction: {r['prediction']}", f"Document Recall@5: {r['document_recall_at_5']:.3f}",
            f"Supplied-image recall: {r['supplied_image_recall_at_5']}",
            f"Missing gold documents: {r['missing_gold_doc_ids']}",
            f"Missing gold images: {r['missing_gold_image_ids']}", f"Route: {r['route']}", ""])
    example_text = "\n".join(text) + "\n"
    (out / "review_examples.md").write_text(example_text, encoding="utf-8")
    print(report)
    print(example_text)
    print(f"Files saved to {out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as e:
        raise SystemExit(f"Evaluation stopped: {e}")
