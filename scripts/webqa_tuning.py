"""Dev-only Document-ID Recall@5 selection and immutable test configuration.

No model loading and no test scoring. Test gold labels never enter score_candidates.
"""
from fractions import Fraction
from pathlib import Path

import webqa_run as runner
from memgraphrag_v.config import ExperimentConfig
from memgraphrag_v.schemas import RetrievalResult

GRID = [0.25, 0.5, 0.75]
RETRIEVAL_FILES = ("retrieval.jsonl", "diagnostics.jsonl", "retrieval_details.jsonl", "labels.jsonl")
POLICY = "exclude_dev_questions_in_preparation_split_overlap"


def dev_run_id(run_id, beta):
    return f"{run_id}-dev-beta-{round(beta * 100):03d}"


def immutable_json(path, value):
    if path.exists() and runner.read_json(path) != value:
        raise ValueError(f"Existing decision/session differs: {path}; use a new run ID")
    runner.atomic_json(path, value)


def eligible_dev(artifacts):
    questions = runner.selection(artifacts, "dev", 0)
    excluded = set(runner.preparation(artifacts)["split_overlap"]["exclude_from_dev_tuning"])
    eligible = [q for q in questions if q.qid not in excluded]
    if not eligible:
        raise ValueError("No eligible dev questions remain after the existing overlap exclusion policy")
    if any(not q.gold_doc_ids for q in eligible):
        raise ValueError("Eligible dev questions must have at least one gold document")
    return questions, eligible


def session_value(config, artifacts, plan):
    return {"version": 2, "grid": GRID, "eligibility_policy": POLICY,
            "config_hash": config.config_hash(), "plan_sha256": runner.stamp(plan),
            "questions_sha256": runner.sha(artifacts.questions),
            "preparation_sha256": runner.sha(artifacts.root / "preparation.json"),
            "expected_test_questions": 120}


def prepare_session(config, artifacts, plan, run_id):
    if sorted(config.retrieval.beta_grid) != GRID:
        raise ValueError("Automatic workflow requires beta_grid=[0.25, 0.5, 0.75]")
    # Check test-readiness now; do not wait until expensive dev retrieval finishes.
    candidate = config.model_copy(update={"retrieval": config.retrieval.model_copy(update={"beta": GRID[0]})})
    candidate.require_frozen()
    questions, eligible = eligible_dev(artifacts)
    tests = runner.selection(artifacts, "test", 0)
    if len(tests) != 120:
        raise ValueError(f"Expected 120 test questions, found {len(tests)}")
    all_ids = [q.qid for q in questions + tests]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Question IDs must be unique across dev and test")
    prep = runner.preparation(artifacts)
    if prep["smoke_only"]:
        raise ValueError("Automatic tuning/test requires the full prepared corpus")
    directory = artifacts.run_dir(run_id)
    immutable_json(directory / "tuning_session.json", session_value(config, artifacts, plan))
    print(f"Tuning: {len(eligible)}/{len(questions)} eligible dev questions; test: 120")


def verify_session(config, artifacts, plan, run_id):
    directory = artifacts.run_dir(run_id)
    if runner.read_json(directory / "tuning_session.json") != session_value(config, artifacts, plan):
        raise ValueError("Tuning session inputs/settings changed")
    return directory


def verify_collection(directory):
    contract = runner.read_json(directory / "run.json")
    done = runner.read_json(directory / "retrieval_completed.json")
    if (done["run_sha256"] != runner.stamp(contract)
            or done["questions"] != len(contract["questions"])
            or set(done["files"]) != set(RETRIEVAL_FILES)):
        raise ValueError(f"Incomplete retrieval collection: {directory}")
    for name, digest in done["files"].items():
        if not digest or runner.sha(directory / name) != digest:
            raise ValueError(f"Retrieval file changed: {directory / name}")
    return contract, done


def score_candidates(gold_by_qid, candidates):
    """Macro mean over questions; exact fractions make the tie rule deterministic."""
    if not gold_by_qid or any(not set(gold) for gold in gold_by_qid.values()):
        raise ValueError("Recall requires a nonempty eligible set and nonempty gold IDs")
    if set(candidates) != set(GRID):
        raise ValueError("All three beta candidates are required")
    results, per_question = [], []
    for beta in GRID:
        ranked = candidates[beta]
        if set(ranked) != set(gold_by_qid):
            raise ValueError("Candidates must cover exactly the same eligible questions")
        recalls = []
        for qid, gold in gold_by_qid.items():
            ids = ranked[qid]
            if len(ids) > 5 or len(set(ids)) != len(ids):
                raise ValueError("Ranking must contain at most five distinct document IDs")
            gold = set(gold)
            hits = len(set(ids) & gold)
            recall = Fraction(hits, len(gold))
            recalls.append(recall)
            per_question.append({"beta": beta, "question_id": qid, "recall_at_5": float(recall),
                                 "hits": hits, "exact_recall": str(recall),
                                 "gold_count": len(gold), "top_k_doc_ids": ids})
        mean = sum(recalls, Fraction()) / len(recalls)
        results.append({"beta": beta, "mean_document_id_recall_at_5": float(mean),
                        "exact_mean": str(mean), "questions": len(recalls)})
    selected = min(results, key=lambda row: (-Fraction(row["exact_mean"]), row["beta"]))["beta"]
    return selected, results, per_question


def select_beta(config, artifacts, plan, run_id):
    directory = verify_session(config, artifacts, plan, run_id)
    questions, eligible = eligible_dev(artifacts)
    expected_questions = [[q.qid, q.question] for q in questions]
    candidates, sources, common_identity = {}, {}, None
    for beta in GRID:
        source = artifacts.run_dir(dev_run_id(run_id, beta))
        contract, done = verify_collection(source)
        if (contract["split"] != "dev" or contract["beta"] != beta
                or contract["config_hash"] != config.config_hash()
                or contract["index_manifest_sha256"] != runner.sha(artifacts.index_manifest)
                or contract["parallel_plan"] != runner.stamp(plan)
                or contract["preparation_sha256"] != runner.sha(artifacts.root / "preparation.json")
                or contract["runner_sha256"] != runner.sha(runner.__file__)
                or contract["questions"] != expected_questions or contract["partial_run"]):
            raise ValueError(f"Different/incomplete candidate inputs: beta={beta}")
        identity = {k: v for k, v in contract.items() if k not in ("beta", "shards")}
        if common_identity is not None and identity != common_identity:
            raise ValueError("Candidates used different inputs/settings")
        common_identity = identity
        results = [RetrievalResult.model_validate(row) for row in runner.read_rows(source / "retrieval.jsonl")]
        if [r.question_id for r in results] != [q.qid for q in questions]:
            raise ValueError("Candidate has missing/duplicate/unexpected questions")
        ranked = {r.question_id: [e.doc_id for e in r.evidence] for r in results}
        candidates[beta] = {q.qid: ranked[q.qid] for q in eligible}
        sources[str(beta)] = {"run_id": source.name, "run_sha256": runner.stamp(contract),
                              "completion_sha256": runner.sha(source / "retrieval_completed.json"),
                              "files": done["files"]}
    selected, scores, per_question = score_candidates({q.qid: q.gold_doc_ids for q in eligible}, candidates)
    frozen = config.model_copy(update={"retrieval": config.retrieval.model_copy(update={"beta": selected})})
    frozen.require_frozen()
    frozen_path = directory / "test_config.yaml"
    if frozen_path.exists() and ExperimentConfig.load(frozen_path).model_dump() != frozen.model_dump():
        raise ValueError("Existing frozen test configuration differs; never retune an existing test run")
    frozen.save(frozen_path)
    eligible_ids = [q.qid for q in eligible]
    summary = {"version": 2, "metric": "macro Document-ID Recall@5", "tie_break": "smaller beta",
               "selected_beta": selected, "scores": scores, "eligibility_policy": POLICY,
               "eligible_question_ids": eligible_ids,
               "excluded_question_ids": [q.qid for q in questions if q.qid not in set(eligible_ids)],
               "dev_questions": len(questions), "eligible_questions": len(eligible),
               "test_labels_used_for_scoring": False, "sources": sources,
               "session_sha256": runner.sha(directory / "tuning_session.json"),
               "index_manifest_sha256": runner.sha(artifacts.index_manifest),
               "test_config_sha256": runner.sha(frozen_path), "test_config_hash": frozen.config_hash(),
               "per_question_scores_sha256": runner.stamp(per_question)}
    # Existing decisions are immutable; retries may reproduce the same decision only.
    immutable_json(directory / "tuning_summary.json", summary)
    runner.write_rows(directory / "dev_recall_by_question.jsonl", per_question)
    print(f"Selected beta={selected}; Recall@5 scores: {scores}")


def selected_test_config(config, artifacts, plan, run_id):
    directory = verify_session(config, artifacts, plan, run_id)
    summary = runner.read_json(directory / "tuning_summary.json")
    path = directory / "test_config.yaml"
    if (summary["session_sha256"] != runner.sha(directory / "tuning_session.json")
            or summary["index_manifest_sha256"] != runner.sha(artifacts.index_manifest)
            or summary["test_config_sha256"] != runner.sha(path)):
        raise ValueError("Tuning decision/index/frozen test configuration changed")
    questions, eligible = eligible_dev(artifacts)
    candidates = {}
    for beta in GRID:
        source = artifacts.run_dir(dev_run_id(run_id, beta))
        contract, done = verify_collection(source)
        saved = summary["sources"][str(beta)]
        if (runner.stamp(contract) != saved["run_sha256"] or done["files"] != saved["files"]
                or runner.sha(source / "retrieval_completed.json") != saved["completion_sha256"]):
            raise ValueError("Dev candidate changed after selection")
        rows = [RetrievalResult.model_validate(row) for row in runner.read_rows(source / "retrieval.jsonl")]
        if [r.question_id for r in rows] != [q.qid for q in questions]:
            raise ValueError("Dev candidate coverage changed")
        ranked = {r.question_id: [e.doc_id for e in r.evidence] for r in rows}
        candidates[beta] = {q.qid: ranked[q.qid] for q in eligible}
    winner, scores, per_question = score_candidates({q.qid: q.gold_doc_ids for q in eligible}, candidates)
    if (winner != summary["selected_beta"] or scores != summary["scores"]
            or [q.qid for q in eligible] != summary["eligible_question_ids"]
            or runner.stamp(per_question) != summary["per_question_scores_sha256"]
            or runner.read_rows(directory / "dev_recall_by_question.jsonl") != per_question):
        raise ValueError("Invalid beta selection")
    frozen = ExperimentConfig.load(path)
    expected = config.model_copy(update={"retrieval": config.retrieval.model_copy(update={"beta": winner})})
    if frozen.model_dump() != expected.model_dump() or frozen.config_hash() != summary["test_config_hash"]:
        raise ValueError("Only selected beta may differ in test configuration")
    frozen.require_frozen()
    if len(runner.selection(artifacts, "test", 0)) != 120:
        raise ValueError("Test set must contain exactly 120 questions")
    return path, frozen
