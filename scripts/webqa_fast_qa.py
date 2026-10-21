#!/usr/bin/env python3
"""Original retrieval loop with an explicit cached index loader; no runtime patches."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import webqa_parallel as api
import webqa_run as runner
from webqa_run import (context, selection, run_contract, read_json, stamp, sha,
                       CachedQuery, synchronize, SYSTEM, atomic_json, read_rows,
                       write_rows, environment)
from webqa_validation_cache import HashCache, timed, log
from webqa_fast_validation import ValidatedRun
from webqa_fast_stage import validate_stage


def load_index_cached(args, config, artifacts):
    base, _ = runner.context(SimpleNamespace(config=args.base_config, artifacts=args.artifacts))
    cache_path = artifacts.index_dir / "parallel-v2/sequential-v1/validation-v2/hashes.sqlite3"
    with HashCache(cache_path) as cache:
        run = ValidatedRun(api, base, artifacts, args.base_run_id, cache).prepare()
        if args.split == "test":
            _, selected = api.tuning.selected_test_config(base, artifacts, run.plan, args.base_run_id)
            if selected.model_dump(mode="json") != config.model_dump(mode="json") or args.beta != selected.retrieval.beta:
                raise ValueError("Test must use the dev-selected configuration and beta")
        elif config.model_dump(mode="json") != base.model_dump(mode="json") or args.beta not in api.tuning.GRID:
            raise ValueError("Dev retrieval must use the frozen base config and candidate grid")
        if not set(api.p.STAGES).issubset(run.manifest["stages"]):
            raise ValueError("Index incomplete")
        api.p._validate_chain(run.manifest)
        with timed("validate query index using cached hashes"):
            run.validate_inputs(run.items()["image"], "image")
            for stage in api.p.STAGES:
                validate_stage(run, stage)
        assembled = run.manifest["stages"]["assemble"]
        text = api.p.effective_text(config)
        enc = config.visual_encoder
        with timed("load saved image/crop vectors"):
            image_set = api.EmbeddingSet.load(artifacts.embeddings_dir(enc.name, "image"))
            crop_set = api.EmbeddingSet.load(artifacts.embeddings_dir(enc.name, "crop"))
        wanted = api.visual.encoder_meta(enc)
        for embedded in (image_set, crop_set):
            if any(getattr(embedded.meta, key) != value for key, value in wanted.items()):
                raise ValueError("Visual embedding metadata changed")
        with timed("load text backend and retrieval graph"):
            backend = api.p.make_backend(config, artifacts, text)
            api.p.retrieval.prepare_retrieval_objects(backend, artifacts.memory_path)
        provenance = {"index_manifest_sha256": cache.digest(artifacts.index_manifest),
            "graph_sha256": assembled["outputs"]["graph"], "config_hash": config.config_hash(),
            "memgraphrag_commit": api.p.MEMGRAPHRAG_COMMIT,
            "text": {"llm_name": text.llm_name, "embedding_model": text.embedding_model},
            "visual_encoder": wanted,
            "validation_policy": "SHA-256 reused only for unchanged device/inode/size/mtime_ns/ctime_ns"}
        return api.p.retrieval.build_index(backend, config.retrieval, image_set, crop_set,
            api.read_jsonl(artifacts.texts, api.TextRecord), api.read_jsonl(artifacts.chunks, api.ChunkRecord),
            api.read_jsonl(artifacts.indexed_images, api.p.IndexedImage), provenance)


def run_worker(run, args):
    """Existing query/retrieve/read order and contracts, in separate environments."""
    api.tuning.verify_session(run.config, run.artifacts, run.plan, run.run_id)
    if args.action == "dev-retrieve":
        split, beta = "dev", api.tuning.GRID[args.candidate]
        base_id = api.tuning.dev_run_id(run.run_id, beta)
        config_path = args.config
        retrieval_only = True
    else:
        path, selected = api.tuning.selected_test_config(run.config, run.artifacts, run.plan, run.run_id)
        split, beta, base_id = "test", selected.retrieval.beta, f"{run.run_id}-test"
        config_path, retrieval_only = str(path), False
    child = f"{base_id}-part-{args.worker}"
    common = ["--config", config_path, "--artifacts", str(run.artifacts.root),
              "--run-id", child, "--split", split, "--beta", str(beta),
              "--shard-index", str(args.worker), "--shard-count", str(api.WORKERS)]
    run.cache.flush()  # Child retrieval opens the same SQLite cache; no open write transaction.
    commands = [("queries", "MGRV_VISION_PYTHON"), ("retrieve", "MGRV_TEXT_PYTHON")]
    if not retrieval_only:
        commands.append(("read", "MGRV_READER_PYTHON"))
    for action, env in commands:
        if action == "retrieve":
            command = [os.environ[env], str(Path(__file__).resolve()), *common,
                       "--base-config", args.config, "--base-run-id", run.run_id]
        else:
            command = [os.environ[env], str(Path(runner.__file__).resolve()), action, *common]
            if action == "read":
                command += ["--reader-model", os.environ["MGRV_READER_MODEL"]]
        with timed(f"{action}: {split} worker={args.worker}, beta={beta}"):
            subprocess.run(command, check=True)
    directory = run.artifacts.run_dir(child)
    if retrieval_only:
        contract = runner.read_json(directory / "run.json")
        runner.atomic_json(directory / "retrieval_completed.json", {
            "questions": len(contract["questions"]), "run_sha256": runner.stamp(contract),
            "files": {f: api.required_hash(directory / f) for f in api.tuning.RETRIEVAL_FILES}})
    runner.atomic_json(directory / "parallel_complete.json", {
        "plan": runner.stamp(run.plan),
        "files": {f: api.required_hash(directory / f) for f in api.completion_files(retrieval_only)}})


def retrieve(args):
    import numpy as np
    from memgraphrag_v import pipeline
    from memgraphrag_v.retrieval import Question
    from memgraphrag_v.visual_indexing import encoder_meta
    config, artifacts = context(args)
    questions = selection(artifacts, args.split, args.limit, args.shard_index, args.shard_count)
    run_dir, contract = run_contract(args, config, artifacts, questions)
    cache = read_json(run_dir / "query_vectors.json")
    if cache["contract_sha256"] != stamp(contract) or cache["encoder"] != encoder_meta(config.visual_encoder):
        raise ValueError("Query cache belongs to different inputs or encoder settings")
    if cache["vectors_sha256"] != sha(run_dir / "query_vectors.npy"):
        raise ValueError("Query vector file changed")
    vectors = np.load(run_dir / "query_vectors.npy", allow_pickle=False)
    if len(vectors) != len(questions) or len(cache["rows"]) != len(questions):
        raise ValueError("Query cache row count mismatch")
    index = load_index_cached(args, config, artifacts)
    texts = {r["record_id"]: r for r in read_rows(artifacts.texts)}
    images = {r["image_id"]: r for r in read_rows(artifacts.image_manifest)}
    records_dir = run_dir / "retrieved"
    records_dir.mkdir(exist_ok=True)
    cold = True
    for number, (q, saved, vector) in enumerate(zip(questions, cache["rows"], vectors)):
        if [saved["qid"], saved["text"]] != [q.qid, q.question]:
            raise ValueError("Query cache order/text mismatch")
        path = records_dir / f"{number:06d}.json"
        if path.exists():
            previous = read_json(path)
            if previous["question_id"] != q.qid or previous["contract_sha256"] != stamp(contract):
                raise ValueError("Saved retrieval result belongs to a different run")
            continue
        encoder = CachedQuery(q.question, vector, saved["truncated"])
        synchronize()
        start = time.perf_counter()
        result, diagnostics = pipeline.retrieve(index, [Question(q.qid, q.question)], encoder,
                                                beta=contract["beta"])[0]
        synchronize()
        elapsed = time.perf_counter() - start
        raw = []
        for evidence in result.evidence:
            records = [texts[pid] for pid in evidence.passage_ids]
            if any(r["doc_id"] != evidence.doc_id for r in records):
                raise ValueError("Retrieved text belongs to a different document")
            image = images[evidence.image_id] if evidence.image_id else None
            if image and image["doc_id"] != evidence.doc_id:
                raise ValueError("Retrieved image belongs to a different document")
            image_path = artifacts.images_dir / image["path"] if image else None
            raw.append({**evidence.model_dump(mode="json"), "records": records,
                        "image_path": str(image_path) if image else None,
                        "image_sha256": sha(image_path) if image else None})
        result_dict = result.model_dump(mode="json")
        result_dict["system"] = SYSTEM
        atomic_json(path, {"question_id": q.qid, "question": q.question,
                          "contract_sha256": stamp(contract), "result": result_dict,
                          "diagnostics": diagnostics.model_dump(mode="json"),
                          "top_k_doc_ids": [e.doc_id for e in result.evidence],
                          "selected_image_ids": [e.image_id for e in result.evidence if e.image_id],
                          "raw_evidence": raw, "graph_retrieval_seconds": elapsed,
                          "eva_query_seconds": saved["eva_seconds"],
                          "retrieval_compute_seconds": elapsed + saved["eva_seconds"],
                          "cold_first_query": cold, "eva_cold_first_query": number == 0})
        cold = False
    rows = [read_json(records_dir / f"{i:06d}.json") for i in range(len(questions))]
    write_rows(run_dir / "retrieval.jsonl", [r["result"] for r in rows])
    write_rows(run_dir / "diagnostics.jsonl", [r["diagnostics"] for r in rows])
    write_rows(run_dir / "retrieval_details.jsonl", rows)
    atomic_json(run_dir / "retrieval_provenance.json", {
        "index": index.provenance, "environment": environment(),
        "latency_definition": "EVA query encode + graph retrieval compute in separate processes; "
                              "excludes model/index loading, serialization and reader generation",
        "question_encoder_environment": cache["environment"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "artifacts", "run-id", "base-config", "base-run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--shard-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--shard-count", type=int, choices=(2,), required=True)
    args = parser.parse_args()
    args.limit = 0
    args.device = "cuda"
    with timed("retrieval worker"):
        retrieve(args)


if __name__ == "__main__":
    main()

