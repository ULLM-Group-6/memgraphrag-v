"""Extra WebQA test numbers used in the paper, computed from evaluate_webqa.py output.

1. Short-answer match: the normalized prediction appears, word-bounded, in a reference sentence
   (the reverse of evaluate_webqa.py's containment). "Insufficient evidence" counts as wrong.
2. Abstention rate: predictions starting with "Insufficient evidence".
3. Paired Document-ID Recall@5 difference against baselines that saved per-question recall in
   test_qa.json order (web_qa_scripts/eval_retrieval.py --out).

Intervals use evaluate_webqa.py's grouping and percentile bootstrap. Python standard library only.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import re
import statistics

from evaluate_webqa import clusters, metric, normalize, percentile, save_json

ABSTAIN = "insufficient evidence"


def short_answer_match(prediction, references):
    p = normalize(prediction)
    return bool(p) and p != ABSTAIN and any(
        re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", normalize(x)) for x in references)


def paired(ours, base, groups, repetitions, seed):
    d = [x - y for x, y in zip(ours, base)]
    rng = random.Random(seed)
    samples = [statistics.mean([d[i] for g in rng.choices(groups, k=len(groups)) for i in g])
               for _ in range(repetitions)]
    return {"baseline": statistics.mean(base), "ours": statistics.mean(ours), "diff": statistics.mean(d),
            "ci95": [percentile(samples, .025), percentile(samples, .975)],
            "bootstrap_share_diff_le_0": sum(s <= 0 for s in samples) / repetitions,
            "questions_better_same_worse": [sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report_dir", type=Path, help="evaluation_report folder written by evaluate_webqa.py")
    p.add_argument("--questions", type=Path, required=True, help="dataset/test_qa.json (baseline question order)")
    p.add_argument("--baseline", action="append", default=[], metavar="NAME=RECALL_JSON")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, help="default: REPORT_DIR/extra_metrics.json")
    args = p.parse_args()

    rows = {r["question_id"]: r for r in csv.DictReader(
        (args.report_dir / "per_question.csv").open(encoding="utf-8-sig"))}
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if {q["qid"] for q in questions} != set(rows) or len(questions) != len(rows):
        raise SystemExit("Question IDs differ between per_question.csv and --questions")
    if any(q["question"] != rows[q["qid"]]["question"] for q in questions):
        raise SystemExit("Question text differs between per_question.csv and --questions")
    data = [rows[q["qid"]] for q in questions]
    groups = clusters([{"gold_doc_ids": json.loads(r["gold_doc_ids"])} for r in data])
    refs = [json.loads(r["reference_answers"]) for r in data]
    abstained = [normalize(r["prediction"]).startswith(ABSTAIN) for r in data]

    result = {"questions": len(data), "gold_document_groups": len(groups),
              "bootstrap": {"resamples": args.bootstrap, "seed": args.seed},
              "short_answer_match": metric([short_answer_match(r["prediction"], x) for r, x in zip(data, refs)],
                                           groups, args.bootstrap, args.seed),
              "abstention_rate": metric(abstained, groups, args.bootstrap, args.seed),
              "abstained_with_full_document_recall": sum(
                  a and float(r["document_recall_at_5"]) == 1 for a, r in zip(abstained, data)),
              "paired_document_recall_at_5": {}}
    ours = [float(r["document_recall_at_5"]) for r in data]
    for spec in args.baseline:
        name, path = spec.split("=", 1)
        base = json.loads(Path(path).read_text(encoding="utf-8"))["per_question"]
        if len(base) != len(data):
            raise SystemExit(f"{name}: {len(base)} per-question values, expected {len(data)}")
        result["paired_document_recall_at_5"][name] = paired(ours, base, groups, args.bootstrap, args.seed)

    out = args.output or args.report_dir / "extra_metrics.json"
    save_json(out, result)
    print(json.dumps(result, indent=2))
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
