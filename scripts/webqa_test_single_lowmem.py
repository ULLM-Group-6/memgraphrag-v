#!/usr/bin/env python3
"""Copy of webqa_test_single.py whose read step uses webqa_read_lowmem.py (vision-attention OOM fix)."""
import argparse
import os
from pathlib import Path
import subprocess


def commands(args, selected_path, beta, root, scripts, env):
    common = ["--config", str(selected_path), "--artifacts", str(root),
              "--run-id", args.output_run_id, "--split", "test", "--beta", str(beta),
              "--shard-index", "0", "--shard-count", "1"]
    runner = str(scripts / "webqa_run.py")
    return [
        ("queries", [env["MGRV_VISION_PYTHON"], runner, "queries", *common]),
        ("retrieve", [env["MGRV_TEXT_PYTHON"], str(scripts / "webqa_test_single.py"),
                      "retrieve", *common, "--base-config", args.config,
                      "--base-run-id", args.run_id]),
        ("read", [env["MGRV_READER_PYTHON"], str(scripts / "webqa_read_lowmem.py"), "read", *common,
                  "--reader-model", env["MGRV_READER_MODEL"]]),
    ]


def run(args):
    import webqa_parallel as api
    from webqa_sequential import chain_lock
    from webqa_validation_cache import HashCache, timed, log
    from webqa_fast_validation import ValidatedRun

    config, art = api.runner.context(args)
    work = art.index_dir / "parallel-v2"
    scripts = Path(__file__).resolve().parent
    # Same writer locks as the sequential chain; no overlap with tuning or old QA.
    with chain_lock(art.root), api.lock(work / "phase.lock"):
        with HashCache(work / "sequential-v1/validation-v2/hashes.sqlite3") as cache:
            prepared = ValidatedRun(api, config, art, args.run_id, cache).prepare()
            selected_path, selected = api.tuning.selected_test_config(
                config, art, prepared.plan, args.run_id)
        questions = api.runner.selection(art, "test", 0, 0, 1)
        if len(questions) != 120:
            raise ValueError(f"Expected exactly 120 test questions, got {len(questions)}")
        log(f"Single-worker test: 120 questions; selected beta={selected.retrieval.beta}; output={args.output_run_id}")
        for action, command in commands(args, selected_path, selected.retrieval.beta,
                                         art.root, scripts, os.environ):
            with timed(f"single-worker test {action}: all 120 questions"):
                subprocess.run(command, check=True)

        directory = art.run_dir(args.output_run_id)
        done = api.runner.read_json(directory / "completed.json")
        contract = api.runner.read_json(directory / "run.json")
        expected_questions = [[q.qid, q.question] for q in questions]
        if (done["questions"] != 120 or contract["questions"] != expected_questions
                or done["run_sha256"] != api.runner.stamp(contract)
                or done["answers_sha256"] != api.runner.sha(directory / "answers.jsonl")):
            raise ValueError("Single-worker test completion does not match the 120-question run")
        summary = art.run_dir(args.run_id) / "tuning_summary.json"
        api.runner.atomic_json(directory / "tuning_provenance.json", {
            "tuning_summary": str(summary), "tuning_summary_sha256": api.runner.sha(summary),
            "selected_beta": selected.retrieval.beta,
            "test_config_sha256": api.runner.sha(selected_path),
            "single_worker_launcher_sha256": api.runner.sha(__file__),
            "workers": 1, "questions": 120})
        log(f"TEST COMPLETE: 120 questions; results in {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    launch = sub.add_parser("run")
    for name in ("config", "artifacts", "run-id", "output-run-id"):
        launch.add_argument("--" + name, required=True)
    retrieve = sub.add_parser("retrieve")
    for name in ("config", "artifacts", "run-id", "base-config", "base-run-id"):
        retrieve.add_argument("--" + name, required=True)
    retrieve.add_argument("--split", choices=("test",), required=True)
    retrieve.add_argument("--beta", type=float, required=True)
    retrieve.add_argument("--shard-index", type=int, choices=(0,), required=True)
    retrieve.add_argument("--shard-count", type=int, choices=(1,), required=True)
    args = parser.parse_args()
    if args.action == "run":
        if args.output_run_id == args.run_id:
            parser.error("Test output must differ from the tuning run ID")
        run(args)
    else:
        # Call the existing function directly: its CLI restricts shards to two,
        # but the function and the original runner both support shard_count=1.
        from webqa_fast_qa import retrieve as existing_retrieve
        args.limit, args.device = 0, "cuda"
        existing_retrieve(args)


if __name__ == "__main__":
    main()
