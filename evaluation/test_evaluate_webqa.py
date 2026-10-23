import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import evaluate_webqa as e


def fixture():
    return [dict(question_id=f"q{i}", question=f"Question {i}", reference_answers=["The red car"],
        gold_doc_ids=gold, top_k_doc_ids=top, gold_image_ids=[f"i{i}"], supplied_image_ids=images,
        prediction=prediction, status="ok", route="graph", split="test", partial_run=False)
        for i, (gold, top, images, prediction) in enumerate([
            (["A", "B"], ["A"], ["i0"], "A red car."),
            (["B", "C"], ["B", "C"], [], "blue car"),
            (["C"], [], [], "scarlet vehicle"),
            (["D"], ["D"], ["i3"], "red car")])]


class EvaluationTests(unittest.TestCase):
    def test_macro_recall_and_transitive_groups(self):
        data = fixture()
        self.assertEqual(e.clusters(data), [[0, 1, 2], [3]])
        scored, summary = e.evaluate(data, 100)
        self.assertEqual(summary["metrics"]["Document-ID Recall@5"]["value"], .625)
        self.assertEqual(summary["metrics"]["Supplied-image Recall@5"]["value"], .5)
        self.assertEqual(summary["metrics"]["Answer containment accuracy (heuristic)"]["value"], .5)
        self.assertEqual(scored[1]["review_category"], "gold_documents_found_but_gold_images_missing")
        self.assertEqual(scored[2]["review_category"], "no_gold_document_retrieved")
        self.assertEqual(summary, e.evaluate(data, 100)[1])

    def test_containment_and_failed_answers(self):
        self.assertFalse(e.contains_reference("cart", ["car"]))
        self.assertTrue(e.contains_reference("It is THE red car!", ["red car"]))
        data = fixture()
        data[0]["status"] = "failed"
        scored, summary = e.evaluate(data, 100)
        self.assertFalse(scored[0]["containment_match"])
        self.assertEqual(scored[0]["document_recall_at_5"], .5)

    def test_no_ci_for_one_group_and_no_image_labels(self):
        data = fixture()[:3]
        for r in data:
            r["gold_image_ids"] = []
        _, summary = e.evaluate(data, 100)
        self.assertIsNone(summary["metrics"]["Document-ID Recall@5"]["ci95"])
        self.assertEqual(summary["metrics"]["Supplied-image Recall@5"]["n"], 0)

    def test_cli_and_integrity_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = fixture()
            def write_rows(name, records):
                (root / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            write_rows("evaluation_inputs.jsonl", data)
            write_rows("answers.jsonl", [dict(question_id=r["question_id"], answer=r["prediction"],
                status=r["status"], supplied_image_ids=r["supplied_image_ids"], documents=[]) for r in data])
            write_rows("labels.jsonl", [dict(qid=r["question_id"], answers=r["reference_answers"],
                gold_doc_ids=r["gold_doc_ids"], gold_image_ids=r["gold_image_ids"]) for r in data])
            write_rows("retrieval.jsonl", [dict(question_id=r["question_id"], route=r["route"],
                evidence=[dict(doc_id=d) for d in r["top_k_doc_ids"]]) for r in data])
            contract = dict(split="test", partial_run=False, smoke_only=False, beta=.5, system="G",
                            questions=[[r["question_id"], r["question"]] for r in data])
            e.save_json(root / "run.json", contract)
            e.save_json(root / "completed.json", dict(questions=4, run_sha256=e.stamp(contract),
                answers_sha256=e.sha(root / "answers.jsonl")))
            result = subprocess.run([sys.executable, str(Path(e.__file__)), str(root),
                "--expected-questions", "4", "--bootstrap", "100"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "evaluation_report/report.md").is_file())
            summary = e.read(root / "evaluation_report/summary.json")
            self.assertEqual(summary["metrics"]["Document-ID Recall@5"]["value"], .625)
            data[0]["top_k_doc_ids"] = ["WRONG"]
            write_rows("evaluation_inputs.jsonl", data)
            with self.assertRaisesRegex(ValueError, "retrieval mismatch"):
                e.load_run(root, 4)
            write_rows("evaluation_inputs.jsonl", fixture() + [fixture()[0]])
            with self.assertRaises(ValueError):
                e.load_run(root, 4)
            write_rows("evaluation_inputs.jsonl", fixture())
            (root / "answers.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                e.load_run(root, 4)


if __name__ == "__main__":
    unittest.main()
