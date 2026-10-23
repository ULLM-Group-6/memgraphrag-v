#!/usr/bin/env python3
"""Serial EVA continuation with small, committed embedding segments.

This adds checkpoints without changing the original two-worker partition,
encoder settings, canonical row order, or the frozen parallel plan.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

SEGMENT_SIZE = 128  # Multiple of embed_files' existing 64-item loading batch.
YIELD = 75


class SliceFinished(Exception):
    """The soft deadline was reached at a checkpoint boundary."""


def resume_kind(api, items, kind, config, artifacts, directory, segments,
                identity, encode, empty, deadline, clock=time.time):
    """Injectable checkpoint loop; no model imports needed to test recovery."""
    started = time.monotonic()
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), f"Checking {kind} checkpoints: {directory}", flush=True)
    result = api.saved_set(directory, identity)
    if result is not None:
        api.check_set(result, items, kind, config, artifacts)
        print(f"Reusing complete {directory}; checked in {time.monotonic() - started:.1f}s", flush=True)
        return result
    parts = []
    for start in range(0, len(items), SEGMENT_SIZE):
        batch = items[start:start + SEGMENT_SIZE]
        path = segments / f"{start:08d}"
        key = {"version": 1, "parent": identity, "segment_size": SEGMENT_SIZE,
               "start": start, "ids": [item.item_id for item in batch]}
        part = api.saved_set(path, key)
        if part is None:
            if clock() >= deadline:
                raise SliceFinished()
            part = encode(batch, kind)
            api.check_set(part, batch, kind, config, artifacts)
            # save_set writes complete.json last. An interrupted uncommitted
            # segment is recomputed; committed segments are never overwritten.
            api.save_set(path, part, key)
            print(f"Saved {kind}: {start + len(batch)}/{len(items)} ({directory})", flush=True)
        else:
            api.check_set(part, batch, kind, config, artifacts)
            if start % (8 * SEGMENT_SIZE) == 0:
                print(time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      f"Reused {kind} through {start + len(batch)}/{len(items)}", flush=True)
        parts.append(part)
    if clock() >= deadline:
        raise SliceFinished()
    result = api.merge_sets(parts, [item.item_id for item in items]) if parts else empty(kind)
    api.check_set(result, items, kind, config, artifacts)
    # Use the ORIGINAL full-worker identity so existing finalizers can consume it.
    api.save_set(directory, result, identity)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    from webqa_validation_cache import HashCache, log, timed
    from webqa_fast_validation import ValidatedRun, CachedEmbeddings
    with timed("embedding imports/config"):
        import webqa_parallel as api
        config, artifacts = api.runner.context(args)
    work = artifacts.index_dir / "parallel-v2"
    held = []
    dim = None

    def encode(items, kind):
        # Authoritative hashes come from the original plan/grounding records,
        # not from whatever bytes happen to be present when encoding starts.
        with timed(f"validate {len(items)} {kind} inputs"):
            run.validate_inputs(items, kind)
        if not held:
            with timed("load EVA once for this job"):
                held.append(api.visual.load_encoder(config.visual_encoder, artifacts, "cuda"))
        with timed(f"encode {len(items)} {kind} inputs"):
            return api.embed_files(items, kind, held[0], artifacts.root,
                                   api.visual.encoder_meta(config.visual_encoder))

    def empty(kind):
        if dim is None:
            raise RuntimeError("Empty image worker: at least two images are required")
        meta = api.EmbeddingMeta(kind=kind, dim=dim, n_rows=0, n_failed=0,
                                 **api.visual.encoder_meta(config.visual_encoder))
        return api.EmbeddingSet(meta, [], api.np.empty((0, dim), dtype=api.np.float32))

    try:
        # Exclusive phase lock: this helper processes both workers sequentially.
        with api.lock(work / "phase.lock"), HashCache(work / "sequential-v1/validation-v2/hashes.sqlite3") as cache:
            run = ValidatedRun(api, config, artifacts, args.run_id, cache).prepare()
            cached = CachedEmbeddings(run)
            if "embed" in run.manifest["stages"]:
                log("Embedding stage already stamped; finalizer will verify it")
                return 0
            for worker in range(api.WORKERS):
                items = run.items(worker)
                dim = None
                for kind in ("image", "crop"):
                    identity = {"plan": api.runner.stamp(run.plan), "ground": run.ground,
                                "worker": worker, "kind": kind}
                    result = resume_kind(cached, items[kind], kind, config, artifacts,
                        work / "embeddings" / str(worker) / kind,
                        work / "sequential-v1" / "segments" / str(worker) / kind,
                        identity, encode, empty, args.deadline)
                    dim = result.meta.dim
                    del result
    except SliceFinished:
        print("Embedding slice finished; committed segments will be reused next job", flush=True)
        return YIELD
    finally:
        held.clear()
        api.sam.release_gpu()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
