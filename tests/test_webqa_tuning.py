"""Synthetic CPU-only tuning-to-test integration; no model/API/Slurm calls."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import webqa_parallel as par
import webqa_tuning as tuning
import webqa_run as runner
from memgraphrag_v.config import ExperimentConfig
from memgraphrag_v.paths import Artifacts
from memgraphrag_v.schemas import QuestionRecord


def question(qid, split, gold):
    return QuestionRecord(qid=qid, question=f"Question {qid}", answers=["irrelevant answer"],
        gold_doc_ids=gold, gold_image_ids=[], gold_passage_ids=[], distractor_image_ids=[],
        distractor_passage_ids=[], modalities=["text"], split=split, group_id=1)


def config():
    c = ExperimentConfig()
    return c.model_copy(update={
        "visual_encoder": c.visual_encoder.model_copy(update={"revision": "r", "processor_revision": "r"}),
        "text": c.text.model_copy(update={"llm_name": "local"}),
        "retrieval": c.retrieval.model_copy(update={"beta": .5, "skip_fact_rerank": True})})


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.art = Artifacts(Path(self.tmp.name))
        self.cfg = config()
        self.cfg.require_frozen()
        self.config_path = self.art.root / "base.yaml"
        self.cfg.save(self.config_path)
        self.plan = {"unchanged_index": True}
        self.dev = [question("d0", "dev", ["A", "B"]), question("d1", "dev", ["C"]),
                    question("excluded", "dev", ["X"])]
        self.test = [question(f"t{i}", "test", ["SECRET_TEST_GOLD"]) for i in range(120)]
        runner.write_rows(self.art.questions, self.dev + self.test)
        runner.atomic_json(self.art.index_manifest, {"stages": {"assemble": "fake complete"}})
        runner.atomic_json(self.art.root / "preparation.json", {
            "normalized_hashes": {"questions.jsonl": runner.sha(self.art.questions)}, "smoke_only": False,
            "split_overlap": {"exclude_from_dev_tuning": ["excluded"]}})
        self.calls = []
        self.rankings = {
            .25: {"d0": ["A"], "d1": ["C"], "excluded": ["X"]},
            .5: {"d0": ["A", "B"], "d1": [], "excluded": ["X"]},
            .75: {"d0": ["A", "B"], "d1": ["C"], "excluded": []}}
        self.env = patch.dict(os.environ, {key: "fake-python" for key in (
            "MGRV_TEXT_PYTHON", "MGRV_VISION_PYTHON", "MGRV_READER_PYTHON")} | {"MGRV_READER_MODEL": "fake-reader"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def args(self, action, worker=0, candidate=0):
        return SimpleNamespace(action=action, run_id="experiment", config=str(self.config_path),
            artifacts=str(self.art.root), worker=worker, candidate=candidate, split="dev")

    def fake_process(self, command, check):
        self.assertTrue(check)
        action = command[2]
        opts = dict(zip(command[3::2], command[4::2]))
        a = argparse.Namespace(config=opts["--config"], artifacts=opts["--artifacts"],
            run_id=opts["--run-id"], split=opts["--split"], limit=0, beta=float(opts["--beta"]),
            shard_index=int(opts["--shard-index"]), shard_count=int(opts["--shard-count"]))
        cfg, art = runner.context(a)
        questions = runner.selection(art, a.split, 0, a.shard_index, a.shard_count)
        out, contract = runner.run_contract(a, cfg, art, questions)
        self.calls.append((action, a.split, a.beta, len(questions), a.config))
        if action == "queries":
            return
        if action == "retrieve":
            results, details = [], []
            for q in questions:
                ids = self.rankings[a.beta][q.qid] if a.split == "dev" else ["arbitrary retrieved doc"]
                result = {"question_id": q.qid, "system": "G-augmented", "route": "graph",
                          "evidence": [{"doc_id": doc, "score": 1.0} for doc in ids]}
                results.append(result)
                details.append({"question_id": q.qid, "result": result, "top_k_doc_ids": ids,
                                "contract_sha256": runner.stamp(contract)})
            runner.write_rows(out / "retrieval.jsonl", results)
            runner.write_rows(out / "retrieval_details.jsonl", details)
            runner.write_rows(out / "diagnostics.jsonl", [{"question_id": q.qid} for q in questions])
            runner.atomic_json(out / "retrieval_provenance.json", {"synthetic": True})
        if action == "read":
            self.assertEqual(a.split, "test", "No dev reader should run during tuning")
            self.assertEqual(cfg.retrieval.beta, a.beta)
            for f in ("answers.jsonl", "reader_inputs.jsonl", "evaluation_inputs.jsonl"):
                runner.write_rows(out / f, [{"question_id": q.qid, "answer": "fake"} for q in questions])
            runner.atomic_json(out / "reader_settings.json", {"model": "fake", "contract_sha256": runner.stamp(contract)})
            runner.atomic_json(out / "completed.json", {"questions": len(questions),
                "run_sha256": runner.stamp(contract), "answers_sha256": runner.sha(out / "answers.jsonl")})

    def make_candidates(self):
        tuning.prepare_session(self.cfg, self.art, self.plan, "experiment")
        with patch.object(par.subprocess, "run", side_effect=self.fake_process):
            for candidate in range(3):
                for worker in range(2):
                    par.automatic_action(self.args("dev-retrieve", worker, candidate), self.cfg, self.art, [], None, self.plan)
                par.automatic_action(self.args("dev-collect", candidate=candidate), self.cfg, self.art, [], None, self.plan)

    def select(self):
        tuning.select_beta(self.cfg, self.art, self.plan, "experiment")
        return runner.read_json(self.art.run_dir("experiment") / "tuning_summary.json")

    def test_full_dev_tuning_to_120_test_results_and_idempotent_selection(self):
        self.make_candidates()
        self.assertFalse(any(call[0] == "read" for call in self.calls))
        summary = self.select()
        self.assertEqual(summary["selected_beta"], .75)
        self.assertEqual([r["mean_document_id_recall_at_5"] for r in summary["scores"]], [.75, .5, 1.0])
        self.assertEqual(summary["eligible_question_ids"], ["d0", "d1"])
        self.assertEqual(summary, self.select())
        with patch.object(par.subprocess, "run", side_effect=self.fake_process):
            for worker in range(2):
                par.automatic_action(self.args("test-qa", worker), self.cfg, self.art, [], None, self.plan)
        par.automatic_action(self.args("test-collect"), self.cfg, self.art, [], None, self.plan)
        out = self.art.run_dir("experiment-test")
        self.assertEqual(runner.read_json(out / "completed.json")["questions"], 120)
        self.assertEqual(len(runner.read_rows(out / "answers.jsonl")), 120)
        self.assertEqual(runner.read_json(out / "run.json")["beta"], .75)
        self.assertEqual(ExperimentConfig.load(self.config_path).retrieval.beta, .5)
        self.assertEqual([c[3] for c in self.calls if c[0] == "read"], [60, 60])

    def test_exact_tie_chooses_smallest_beta(self):
        ranks = {b: {"q": ["A"]} for b in tuning.GRID}
        selected, scores, _ = tuning.score_candidates({"q": ["A", "B", "C"]}, ranks)
        self.assertEqual(selected, .25)
        self.assertTrue(all(row["exact_mean"] == "1/3" for row in scores))

    def test_empty_predictions_count_as_zero_not_missing(self):
        ranks = {b: {"q": []} for b in tuning.GRID}
        selected, scores, _ = tuning.score_candidates({"q": ["A"]}, ranks)
        self.assertEqual(selected, .25)
        self.assertTrue(all(row["exact_mean"] == "0" for row in scores))

    def test_bad_metric_inputs_rejected(self):
        good = {b: {"q": ["A"]} for b in tuning.GRID}
        for gold, candidates in (({}, good), ({"q": []}, good), ({"q": ["A"]}, {.25: {"q": ["A"]}}),
                ({"q": ["A"]}, {b: {"wrong": ["A"]} for b in tuning.GRID}),
                ({"q": ["A"]}, {b: {"q": ["A", "A"]} for b in tuning.GRID})):
            with self.subTest(gold=gold), self.assertRaises(ValueError):
                tuning.score_candidates(gold, candidates)

    def test_no_eligible_questions_stops_preflight(self):
        prep = runner.read_json(self.art.root / "preparation.json")
        prep["split_overlap"]["exclude_from_dev_tuning"] = [q.qid for q in self.dev]
        runner.atomic_json(self.art.root / "preparation.json", prep)
        with self.assertRaisesRegex(ValueError, "No eligible"):
            tuning.prepare_session(self.cfg, self.art, self.plan, "experiment")

    def test_wrong_test_count_stops_preflight(self):
        runner.write_rows(self.art.questions, self.dev + self.test[:119])
        prep = runner.read_json(self.art.root / "preparation.json")
        prep["normalized_hashes"]["questions.jsonl"] = runner.sha(self.art.questions)
        runner.atomic_json(self.art.root / "preparation.json", prep)
        with self.assertRaisesRegex(ValueError, "120"):
            tuning.prepare_session(self.cfg, self.art, self.plan, "experiment")

    def test_missing_candidate_stops_selection_and_test(self):
        tuning.prepare_session(self.cfg, self.art, self.plan, "experiment")
        with self.assertRaises(FileNotFoundError):
            self.select()
        with self.assertRaises(FileNotFoundError):
            tuning.selected_test_config(self.cfg, self.art, self.plan, "experiment")

    def test_changed_retrieval_rejected_before_selection(self):
        self.make_candidates()
        path = self.art.run_dir(tuning.dev_run_id("experiment", .25)) / "retrieval.jsonl"
        runner.write_rows(path, [])
        with self.assertRaisesRegex(ValueError, "file changed"):
            self.select()

    def test_test_gate_rejects_changed_index_and_frozen_config(self):
        self.make_candidates()
        self.select()
        original_index = self.art.index_manifest.read_bytes()
        runner.atomic_json(self.art.index_manifest, {"changed": True})
        with self.assertRaises(ValueError):
            tuning.selected_test_config(self.cfg, self.art, self.plan, "experiment")
        self.art.index_manifest.write_bytes(original_index)
        path = self.art.run_dir("experiment") / "test_config.yaml"
        path.write_text(path.read_text() + "\n# changed\n")
        with self.assertRaises(ValueError):
            tuning.selected_test_config(self.cfg, self.art, self.plan, "experiment")

    def test_test_gate_rejects_manipulated_score(self):
        self.make_candidates()
        summary = self.select()
        summary["scores"][0]["exact_mean"] = "1"
        runner.atomic_json(self.art.run_dir("experiment") / "tuning_summary.json", summary)
        with self.assertRaisesRegex(ValueError, "Invalid beta"):
            tuning.selected_test_config(self.cfg, self.art, self.plan, "experiment")

    def test_wrong_candidate_beta_rejected_at_collection(self):
        tuning.prepare_session(self.cfg, self.art, self.plan, "experiment")
        with patch.object(par.subprocess, "run", side_effect=self.fake_process):
            for worker in range(2):
                par.automatic_action(self.args("dev-retrieve", worker, 0), self.cfg, self.art, [], None, self.plan)
        args = self.args("dev-collect")
        args.run_id = tuning.dev_run_id("experiment", .25)
        args.retrieval_only, args.beta = True, .5
        with self.assertRaisesRegex(ValueError, "different beta"):
            par.collect(args, self.cfg, self.art, [], None, self.plan)


if __name__ == "__main__":
    unittest.main()
