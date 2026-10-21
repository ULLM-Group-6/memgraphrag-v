#!/usr/bin/env python3
"""Cached preparation/finalization; uses the original graph-building algorithm."""
import argparse

from webqa_validation_cache import HashCache, log, timed
from webqa_fast_validation import ValidatedRun, CachedEmbeddings


def validate_stage(run, name):
    entry = run.manifest["stages"].get(name)
    if entry is None or entry["inputs"] != run.expected_inputs(name):
        raise ValueError(f"Missing/stale {name} stage inputs")
    if name == "embed":
        wanted = {"visual_encoder": run.api.visual.encoder_meta(run.config.visual_encoder)}
    elif name == "assemble":
        text = run.api.p.effective_text(run.config)
        wanted = {"llm_name": text.llm_name, "embedding_model": text.embedding_model}
    else:
        wanted = entry["settings"]  # text/ground settings already checked by preparation.
    if entry["settings"] != wanted:
        raise ValueError(f"Changed {name} settings")
    run.verify_outputs(name)


def stamp_stage(run, name, outputs, settings):
    run.manifest["stages"][name] = {"inputs": run.expected_inputs(name), "settings": settings, "outputs": outputs}
    run.api.p._validate_chain(run.manifest)
    run.manifest["provenance"] = {
        "config_hash": run.config.config_hash(),
        "config": run.config.model_dump(mode="json", exclude={"artifacts_root"}),
        "memgraphrag_commit": run.api.p.MEMGRAPHRAG_COMMIT,
        "mg2_commit": run.api.sam.MG2_COMMIT}
    run.api.p._write_manifest(run.artifacts, run.manifest)
    run.cache.flush()


def finalize(run):
    api, art, config = run.api, run.artifacts, run.config
    run.validate_inputs(run.items()["image"], "image")
    validate_stage(run, "text")
    validate_stage(run, "ground")
    if "embed" in run.manifest["stages"]:
        validate_stage(run, "embed")
        log("Embedding stage already complete and validated")
        return
    cached = CachedEmbeddings(run)
    all_items = run.items()
    with timed("merge existing embedding workers (no encoder loading)"):
        for kind in ("image", "crop"):
            parts = []
            for worker in range(api.WORKERS):
                identity = {"plan": api.runner.stamp(run.plan), "ground": run.ground, "worker": worker, "kind": kind}
                part = cached.saved_set(run.work / "embeddings" / str(worker) / kind, identity)
                if part is None:
                    raise ValueError(f"Worker {worker} has not completed {kind} embeddings")
                cached.check_set(part, run.items(worker)[kind], kind, config, art)
                parts.append(part)
            merged = cached.merge_sets(parts, [i.item_id for i in all_items[kind]])
            cached.check_set(merged, all_items[kind], kind, config, art)
            merged.save(art.embeddings_dir(config.visual_encoder.name, kind))
            del parts, part, merged
        outputs = {f"{kind}_embeddings": run.embedding_digest(art.embeddings_dir(config.visual_encoder.name, kind))
                   for kind in ("image", "crop")}
        stamp_stage(run, "embed", outputs, {"visual_encoder": api.visual.encoder_meta(config.visual_encoder)})


def assemble(run):
    run.validate_inputs(run.items()["image"], "image")
    for stage in ("text", "ground", "embed"):
        validate_stage(run, stage)
    if "assemble" in run.manifest["stages"]:
        validate_stage(run, "assemble")
        log("Assembly already complete and validated")
        return
    text = run.api.p.effective_text(run.config)
    if not text.llm_name:
        raise ValueError("text.llm_name is required")
    with timed("build graph using the original pipeline._stage_assemble"):
        outputs = run.api.p._stage_assemble(run.artifacts, run.config, text)
    stamp_stage(run, "assemble", outputs, {"llm_name": text.llm_name, "embedding_model": text.embedding_model})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-auto", "embed-finalize", "assemble", "dev-retrieve",
                                           "dev-collect", "select-beta", "test-qa", "test-collect"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker", type=int, choices=(0, 1), default=0)
    parser.add_argument("--candidate", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    with timed(f"imports/config for {args.action}"):
        import webqa_parallel as api
        config, artifacts = api.runner.context(args)
        artifacts.run_dir(args.run_id)
    work = artifacts.index_dir / "parallel-v2"
    with api.lock(work / "phase.lock"), HashCache(work / "sequential-v1/validation-v2/hashes.sqlite3") as cache:
        run = ValidatedRun(api, config, artifacts, args.run_id, cache).prepare()
        if args.action == "embed-finalize":
            finalize(run)
        elif args.action == "assemble":
            assemble(run)
        elif args.action in ("dev-retrieve", "test-qa"):
            from webqa_fast_qa import run_worker
            run_worker(run, args)
        elif args.action in ("dev-collect", "select-beta", "test-collect"):
            with timed(args.action):
                api.automatic_action(args, config, artifacts, run.images, work, run.plan)
        else:
            log("Preparation ready; following steps can share this verified receipt")


if __name__ == "__main__":
    main()
