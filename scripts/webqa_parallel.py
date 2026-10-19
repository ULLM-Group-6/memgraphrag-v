#!/usr/bin/env python3
"""Two-worker WebQA continuation. Workers never write the index manifest.

Uses the existing grounding, embedding and retrieval algorithms unchanged.
Run through submit_webqa_parallel.sh on Snellius, with the original serial job stopped.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import webqa_run as runner
import webqa_tuning as tuning
import numpy as np
from memgraphrag_v import pipeline as p, sam3_processing as sam, visual_indexing as visual
from memgraphrag_v.embeddings import EmbeddingMeta, EmbeddingSet, embed_files, items_from_images
from memgraphrag_v.io import read_jsonl
from memgraphrag_v.paths import external_path, MG2_ENV
from memgraphrag_v.schemas import ImageRecord, TextRecord, ChunkRecord, EntityRecord

WORKERS = 2
FILES = ("meta.json", "rows.jsonl", "vectors.npy")
RESULTS = ("retrieval.jsonl", "diagnostics.jsonl", "retrieval_details.jsonl",
           "reader_inputs.jsonl", "answers.jsonl", "evaluation_inputs.jsonl", "labels.jsonl")


def partition(rows, worker, count=WORKERS):
    if count < 1 or not 0 <= worker < count:
        raise ValueError("Invalid worker index/count")
    return rows[worker::count]


@contextmanager
def lock(path, shared=False):
    # Linux flock releases automatically if Slurm kills the process.
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        flag = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(file, flag | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another parallel job owns {path}; do not overlap chains") from None
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def required_hash(path):
    value = p._file_sha256(path)
    if value is None:
        raise FileNotFoundError(path)
    return value


def snapshot(config, artifacts, images):
    """Freeze source inputs, text results and code, not the evolving index manifest."""
    text = p._read_manifest(artifacts)["stages"].get("text")
    if text is None:
        raise RuntimeError("Finish text indexing before starting parallel continuation")
    p._verify_outputs(artifacts, config, "text", text["outputs"])
    sources = {name: required_hash(getattr(artifacts, name))
               for name in ("text_manifest", "captions", "image_manifest", "questions")}
    wanted = {"text_manifest": sources["text_manifest"], "captions": sources["captions"],
              "image_identities": p._json_sha256([[i.image_id, i.doc_id] for i in images])}
    if text["inputs"] != wanted:
        raise RuntimeError("Text stage inputs are stale; do not continue this index")
    effective = p.effective_text(config)
    for key in ("llm_name", "embedding_model", "chunk_size", "chunk_overlap",
                "ontology_filter_mode", "ontology_min_frequency", "ontology_low_percent"):
        if text["settings"].get(key) != getattr(effective, key):
            raise RuntimeError(f"Text setting {key} differs from the built index; restore its original value")
    checkpoint = sam.default_checkpoint(artifacts)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    root = Path(__file__).resolve().parents[1]
    code = sorted((root / "src" / "memgraphrag_v").glob("*.py"))
    code += [Path(__file__).resolve(), Path(runner.__file__).resolve(), Path(tuning.__file__).resolve()]
    return {"workers": WORKERS, "config": config.model_dump(mode="json"),
            "effective_text": p.effective_text(config).model_dump(mode="json"),
            "text_stage": text, "sources": sources,
            "grounder": sam.grounder_identity(checkpoint),
            "images": [[i.image_id, i.path, required_hash(artifacts.image_path(i))] for i in images],
            "code": {str(f.relative_to(root)): required_hash(f) for f in code}}


def grounding_context(artifacts, images):
    links = p.link_images(images, read_jsonl(artifacts.texts, TextRecord),
                          read_jsonl(artifacts.chunks, ChunkRecord))
    entities = read_jsonl(artifacts.entities, EntityRecord)
    return {link.image_id: link.chunk_ids for link in links}, entities


def validate_grounding(artifacts, config, images, grounder_id):
    """Read-only validation: never quietly re-ground in a finalizer."""
    links, entities = grounding_context(artifacts, images)
    by_chunk = {}
    for entity in entities:
        for chunk_id in entity.chunk_ids:
            by_chunk.setdefault(chunk_id, []).append(entity)
    records = sam.load_grounding(artifacts, [i.image_id for i in images])
    for image in images:
        record = records[image.image_id]
        prompts = sam.image_prompts(links[image.image_id], by_chunk)
        fingerprint = sam._fingerprint(artifacts.image_path(image), prompts, config.sam3, grounder_id)
        if (record.image_id != image.image_id or record.doc_id != image.doc_id or record.status == "failed"
                or record.fingerprint != fingerprint
                or not sam.grounding_artifacts_valid(record, artifacts)):
            raise RuntimeError(f"Grounding incomplete/stale for {image.image_id}: {record.error}")
    return [records[i.image_id] for i in images]


def ground_worker(args, config, artifacts, images, work, plan):
    subset = partition(images, args.worker)
    links, entities = grounding_context(artifacts, subset)
    held = []
    def get_grounder():
        obj = sam.MG2Sam3(external_path(MG2_ENV, "MG2-RAG checkout"),
            sam.default_checkpoint(artifacts), config.sam3.device, config.sam3.threshold,
            config.sam3.prompt_batch_size, work / f"mg2-{args.worker}")
        held.append(obj)
        return obj
    try:
        sam.ground_images(subset, links, entities, artifacts, config.sam3,
                          get_grounder, plan["grounder"])
    finally:
        for obj in held:
            obj.close()
        held.clear()
        sam.release_gpu()
    validate_grounding(artifacts, config, subset, plan["grounder"])


def checked_stage(artifacts, config, name):
    entry = p._read_manifest(artifacts)["stages"].get(name)
    if entry is None:
        raise RuntimeError(f"Finalize {name} before proceeding")
    p._verify_outputs(artifacts, config, name, entry["outputs"])
    return entry


def embedding_items(artifacts, images):
    records = sam.load_grounding(artifacts, [i.image_id for i in images])
    return {"image": items_from_images(images, artifacts.images_dir.relative_to(artifacts.root).as_posix()),
            "crop": visual.crop_items([records[i.image_id] for i in images])}


def saved_set(directory, identity):
    marker = directory / "complete.json"
    if not marker.is_file():
        return None  # Ignore files interrupted before the completion marker.
    saved = runner.read_json(marker)
    if saved["identity"] != identity:
        raise RuntimeError(f"Different inputs in {directory}; do not mix runs")
    actual = {f: required_hash(directory / f) for f in FILES}
    if actual != saved["files"]:
        raise RuntimeError(f"Corrupt embedding shard: {directory}")
    return EmbeddingSet.load(directory)


def save_set(directory, result, identity):
    result.save(directory)
    runner.atomic_json(directory / "complete.json", {
        "identity": identity, "files": {f: required_hash(directory / f) for f in FILES}})


def check_set(result, items, kind, config, artifacts):
    reason = visual.cache_problem(result, items, kind,
                                  visual.encoder_meta(config.visual_encoder), artifacts.root)
    if reason:
        raise RuntimeError(f"Invalid {kind} embeddings: {reason}")
    if not np.isfinite(result.vectors).all():
        raise RuntimeError(f"Nonfinite {kind} vectors")


def embed_worker(args, config, artifacts, images, work, plan):
    ground = checked_stage(artifacts, config, "ground")
    items = embedding_items(artifacts, partition(images, args.worker))
    held = []
    dim = None
    try:
        for kind in ("image", "crop"):
            directory = work / "embeddings" / str(args.worker) / kind
            identity = {"plan": runner.stamp(plan), "ground": ground,
                        "worker": args.worker, "kind": kind}
            result = saved_set(directory, identity)
            if result is None:
                if items[kind]:
                    if not held:
                        held.append(visual.load_encoder(config.visual_encoder, artifacts, "cuda"))
                    result = embed_files(items[kind], kind, held[0], artifacts.root,
                                         visual.encoder_meta(config.visual_encoder))
                else:
                    if dim is None:
                        raise RuntimeError("Empty image shard; need at least two images")
                    meta = EmbeddingMeta(kind=kind, dim=dim, n_rows=0, n_failed=0,
                                         **visual.encoder_meta(config.visual_encoder))
                    result = EmbeddingSet(meta, [], np.empty((0, dim), dtype=np.float32))
                check_set(result, items[kind], kind, config, artifacts)
                save_set(directory, result, identity)
            check_set(result, items[kind], kind, config, artifacts)
            dim = result.meta.dim
            del result
    finally:
        held.clear()
        sam.release_gpu()


def merge_sets(parts, expected_ids):
    """Restore canonical item order and remap vector row indices."""
    if not parts:
        raise ValueError("No embedding shards")
    base = parts[0].meta.model_dump(exclude={"n_rows", "n_failed"})
    indexed = {}
    for part in parts:
        if part.meta.model_dump(exclude={"n_rows", "n_failed"}) != base:
            raise ValueError("Embedding metadata differs between shards")
        for row in part.rows:
            if row.item_id in indexed:
                raise ValueError(f"Duplicate embedding: {row.item_id}")
            indexed[row.item_id] = (row, part)
    if len(set(expected_ids)) != len(expected_ids) or set(indexed) != set(expected_ids):
        raise ValueError("Embedding shards do not exactly cover expected items")
    rows, vectors = [], []
    for item_id in expected_ids:
        row, part = indexed[item_id]
        number = None
        if row.row is not None:
            number = len(vectors)
            vectors.append(part.vectors[row.row])
        rows.append(row.model_copy(update={"row": number}))
    array = np.stack(vectors) if vectors else np.empty((0, base["dim"]), dtype=np.float32)
    meta = EmbeddingMeta(**base, n_rows=len(rows), n_failed=len(rows) - len(vectors))
    return EmbeddingSet(meta, rows, array)


def finalize_embed(args, config, artifacts, images, work, plan):
    if "embed" in p._read_manifest(artifacts)["stages"]:
        # Never overwrite an already finalized cache (or silently repair a stale one).
        p.index_with_memory_v(artifacts, config, stages=["embed"], device="cpu")
        return
    ground = checked_stage(artifacts, config, "ground")
    all_items = embedding_items(artifacts, images)
    for kind in ("image", "crop"):
        parts = []
        for worker in range(WORKERS):
            identity = {"plan": runner.stamp(plan), "ground": ground,
                        "worker": worker, "kind": kind}
            part = saved_set(work / "embeddings" / str(worker) / kind, identity)
            if part is None:
                raise RuntimeError(f"Worker {worker} has not completed {kind} embeddings")
            subset_items = embedding_items(artifacts, partition(images, worker))[kind]
            check_set(part, subset_items, kind, config, artifacts)
            parts.append(part)
        merged = merge_sets(parts, [i.item_id for i in all_items[kind]])
        check_set(merged, all_items[kind], kind, config, artifacts)
        merged.save(artifacts.embeddings_dir(config.visual_encoder.name, kind))
        del parts, part, merged
    # Both canonical caches are complete and validated: the normal stage reuses them.
    p.index_with_memory_v(artifacts, config, stages=["embed"], device="cpu")


def qa_worker(args, config, artifacts, images, work, plan):
    if not config.retrieval.skip_fact_rerank:
        raise ValueError("This launcher requires the existing skip_fact_rerank=true configuration")
    # Separate runs prevent query caches, answer files and run.json from colliding.
    child = f"{args.run_id}-part-{args.worker}"
    common = ["--config", args.config, "--artifacts", str(artifacts.root),
              "--run-id", child, "--split", args.split,
              "--shard-index", str(args.worker), "--shard-count", str(WORKERS)]
    if getattr(args, "beta", None) is not None:
        common += ["--beta", str(args.beta)]
    retrieval_only = getattr(args, "retrieval_only", False)
    commands = [("queries", "MGRV_VISION_PYTHON"), ("retrieve", "MGRV_TEXT_PYTHON")]
    if not retrieval_only:
        commands.append(("read", "MGRV_READER_PYTHON"))
    for command, env in commands:
        call = [os.environ[env], str(Path(runner.__file__).resolve()), command, *common]
        if command == "read":
            call += ["--reader-model", os.environ["MGRV_READER_MODEL"]]
        subprocess.run(call, check=True)
    directory = artifacts.run_dir(child)
    if retrieval_only:
        contract = runner.read_json(directory / "run.json")
        runner.atomic_json(directory / "retrieval_completed.json", {
            "questions": len(contract["questions"]), "run_sha256": runner.stamp(contract),
            "files": {f: required_hash(directory / f) for f in tuning.RETRIEVAL_FILES}})
    runner.atomic_json(directory / "parallel_complete.json", {
        "plan": runner.stamp(plan), "files": {f: required_hash(directory / f)
            for f in completion_files(retrieval_only)}})


def completion_files(retrieval_only):
    if retrieval_only:
        return (*tuning.RETRIEVAL_FILES, "run.json", "retrieval_completed.json", "retrieval_provenance.json")
    return (*RESULTS, "run.json", "completed.json", "reader_settings.json", "retrieval_provenance.json")


def merge_rows(parts, expected_ids, key="question_id"):
    indexed = {}
    for rows in parts:
        for row in rows:
            qid = row[key]
            if qid in indexed:
                raise ValueError(f"Duplicate question: {qid}")
            indexed[qid] = row
    if len(set(expected_ids)) != len(expected_ids) or set(indexed) != set(expected_ids):
        raise ValueError("QA shards do not exactly cover the selected questions")
    return [indexed[qid] for qid in expected_ids]


def collect(args, config, artifacts, images, work, plan):
    retrieval_only = getattr(args, "retrieval_only", False)
    result_files = tuning.RETRIEVAL_FILES if retrieval_only else RESULTS
    questions = runner.selection(artifacts, args.split, 0)
    ids = [q.qid for q in questions]
    children = [artifacts.run_dir(f"{args.run_id}-part-{w}") for w in range(WORKERS)]
    contracts, settings = [], []
    for worker, directory in enumerate(children):
        marker = runner.read_json(directory / "parallel_complete.json")
        if marker["plan"] != runner.stamp(plan):
            raise ValueError("QA shard belongs to a different plan")
        expected_files = set(completion_files(retrieval_only))
        if set(marker["files"]) != expected_files:
            raise ValueError("QA completion marker is incomplete")
        for name, digest in marker["files"].items():
            if required_hash(directory / name) != digest:
                raise ValueError(f"QA file changed: {directory / name}")
        contract = runner.read_json(directory / "run.json")
        if contract["questions"] != [[q.qid, q.question] for q in partition(questions, worker)]:
            raise ValueError("Wrong questions in QA shard")
        if (contract["index_manifest_sha256"] != required_hash(artifacts.index_manifest)
                or contract["config_hash"] != config.config_hash() or contract["split"] != args.split
                or contract["shard"] != {"index": worker, "count": WORKERS}):
            raise ValueError("QA shard settings/index changed")
        if getattr(args, "beta", None) is not None and contract["beta"] != args.beta:
            raise ValueError("Worker used a different beta")
        done = runner.read_json(directory / ("retrieval_completed.json" if retrieval_only else "completed.json"))
        if (done["run_sha256"] != runner.stamp(contract)
                or (not retrieval_only and done["answers_sha256"] != required_hash(directory / "answers.jsonl"))
                or done["questions"] != len(contract["questions"])):
            raise ValueError("QA completion contract mismatch")
        if retrieval_only:
            tuning.verify_collection(directory)
        contracts.append(contract)
        settings.append({} if retrieval_only else {
            k: v for k, v in runner.read_json(directory / "reader_settings.json").items()
            if k != "contract_sha256"})
    common = lambda c: {k: v for k, v in c.items() if k not in ("questions", "shard")}
    if common(contracts[0]) != common(contracts[1]) or settings[0] != settings[1]:
        raise ValueError("QA shards used different settings")
    out = artifacts.run_dir(args.run_id)
    merged_contract = {**common(contracts[0]), "questions": [[q.qid, q.question] for q in questions],
                       "shards": [str(d) for d in children], "parallel_plan": runner.stamp(plan)}
    if (out / "run.json").exists() and runner.read_json(out / "run.json") != merged_contract:
        raise ValueError("Destination run already exists with different inputs")
    runner.atomic_json(out / "run.json", merged_contract)
    # Write completion last. Interrupted collection is safe to repeat.
    for name in result_files:
        rows = merge_rows([runner.read_rows(d / name) for d in children], ids,
                          "qid" if name == "labels.jsonl" else "question_id")
        runner.write_rows(out / name, rows)
    config.save(out / "config.yaml")
    if not retrieval_only:
        runner.atomic_json(out / "reader_settings.json", settings[0])
    runner.atomic_json(out / "retrieval_provenance.json", {
        "workers": [runner.read_json(d / "retrieval_provenance.json") for d in children],
        "note": "Two independent warmups. Keep cold_first_query flags when reporting latency."})
    completion = {"questions": len(ids), "scoring_performed": False,
        "run_sha256": runner.stamp(merged_contract),
        "files": {f: required_hash(out / f) for f in result_files}, "shards": [str(d) for d in children]}
    if not retrieval_only:
        completion["answers_sha256"] = required_hash(out / "answers.jsonl")
    runner.atomic_json(out / ("retrieval_completed.json" if retrieval_only else "completed.json"), completion)
    print(f"Collected {len(ids)} questions in {out}")


def automatic_action(args, config, artifacts, images, work, plan):
    """Only the test subprocesses receive the selected config; the index plan stays fixed."""
    from copy import copy
    base_id = args.run_id
    tuning.verify_session(config, artifacts, plan, base_id)
    if args.action == "select-beta":
        tuning.select_beta(config, artifacts, plan, base_id)
        return
    child_args = copy(args)
    if args.action in ("dev-retrieve", "dev-collect"):
        child_args.split = "dev"
        child_args.beta = tuning.GRID[args.candidate]
        child_args.run_id = tuning.dev_run_id(base_id, child_args.beta)
        child_args.retrieval_only = True
    else:
        path, config = tuning.selected_test_config(config, artifacts, plan, base_id)
        child_args.config = str(path)
        child_args.split = "test"
        child_args.beta = config.retrieval.beta
        child_args.run_id = f"{base_id}-test"
        child_args.retrieval_only = False
    action = collect if args.action in ("dev-collect", "test-collect") else qa_worker
    action(child_args, config, artifacts, images, work, plan)
    if args.action == "test-collect":
        # An audit pointer from final test output back to the dev-only decision.
        runner.atomic_json(artifacts.run_dir(child_args.run_id) / "tuning_provenance.json", {
            "tuning_summary": str(artifacts.run_dir(base_id) / "tuning_summary.json"),
            "tuning_summary_sha256": required_hash(artifacts.run_dir(base_id) / "tuning_summary.json"),
            "selected_beta": config.retrieval.beta})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "ground", "ground-finalize", "embed",
                                           "embed-finalize", "assemble", "qa", "collect", "prepare-auto",
                                           "dev-retrieve", "dev-collect", "select-beta", "test-qa", "test-collect"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--worker", type=int, choices=range(WORKERS), default=0)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--run-id", default="parallel-dev-v1")
    parser.add_argument("--candidate", type=int, choices=range(3), default=0)
    args = parser.parse_args()
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config, artifacts = runner.context(args)
    artifacts.run_dir(args.run_id)  # Validate the name before creating any paths.
    work = artifacts.index_dir / "parallel-v2"
    worker_action = args.action in ("ground", "embed", "qa", "dev-retrieve", "test-qa")
    name = f"{args.action}-{args.worker}" if worker_action else args.action
    started = time.time()
    with lock(work / "phase.lock", shared=worker_action), lock(work / f"{name}.lock"):
        images = read_jsonl(artifacts.image_manifest, ImageRecord)
        if len(images) < WORKERS or len({i.image_id for i in images}) != len(images):
            raise ValueError("Need at least two unique images")
        current = snapshot(config, artifacts, images)
        path = work / "plan.json"
        if args.action in ("prepare", "prepare-auto"):
            runner.preparation(artifacts)
            if not config.retrieval.skip_fact_rerank:
                raise ValueError("Keep skip_fact_rerank=true; this continuation does not start vLLM")
            if config.retrieval.n_docs != 5 or config.retrieval.beta is None:
                raise ValueError("QA requires n_docs=5 and an explicit retrieval.beta")
            if len(runner.selection(artifacts, args.split, 0)) < WORKERS:
                raise ValueError("Need at least two questions for parallel QA")
            if args.split == "test":
                config.require_frozen()
            if path.exists() and runner.read_json(path) != current:
                raise ValueError("Existing parallel plan has different inputs; do not overwrite it")
            runner.atomic_json(path, current)
            if args.action == "prepare-auto":
                tuning.prepare_session(config, artifacts, current, args.run_id)
            print(f"Plan ready: {len(images)} images, {WORKERS} workers; existing grounding will be reused")
        else:
            if runner.read_json(path) != current:
                raise ValueError("Inputs/settings/code changed since plan creation; stop and inspect")
            if args.action == "ground-finalize":
                validate_grounding(artifacts, config, images, current["grounder"])
                p.index_with_memory_v(artifacts, config, stages=["ground"], device="cpu")
            elif args.action == "assemble":
                p.index_with_memory_v(artifacts, config, stages=["assemble"])
            elif args.action in ("dev-retrieve", "dev-collect", "select-beta", "test-qa", "test-collect"):
                automatic_action(args, config, artifacts, images, work, current)
            else:
                actions = {"ground": ground_worker, "embed": embed_worker,
                           "embed-finalize": finalize_embed, "qa": qa_worker, "collect": collect}
                actions[args.action](args, config, artifacts, images, work, current)
        runner.atomic_json(work / "timings" / f"{name}-{os.environ.get('SLURM_JOB_ID', 'local')}.json",
                           {"action": name, "started": started, "finished": time.time(),
                            "job_id": os.environ.get("SLURM_JOB_ID"), "status": "completed"})


if __name__ == "__main__":
    main()
