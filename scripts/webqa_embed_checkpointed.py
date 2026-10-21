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
    result = api.saved_set(directory, identity)
    if result is not None:
        api.check_set(result, items, kind, config, artifacts)
        print(f"Reusing complete {directory}", flush=True)
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
    args = parser.parse_args()
    import webqa_parallel as api
    config, artifacts = api.runner.context(args)
    work = artifacts.index_dir / "parallel-v2"
    held = []
    dim = None

    def encode(items, kind):
        if not held:
            held.append(api.visual.load_encoder(config.visual_encoder, artifacts, "cuda"))
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
        with api.lock(work / "phase.lock"):
            images = api.read_jsonl(artifacts.image_manifest, api.ImageRecord)
            if len(images) < api.WORKERS or len({i.image_id for i in images}) != len(images):
                raise ValueError("Need at least two unique images")
            plan = api.snapshot(config, artifacts, images)
            if api.runner.read_json(work / "plan.json") != plan:
                raise ValueError("Inputs/settings/code changed since the original plan")
            ground = api.checked_stage(artifacts, config, "ground")
            if "embed" in api.p._read_manifest(artifacts)["stages"]:
                print("Embedding stage already stamped; finalizer will verify it", flush=True)
                return 0
            for worker in range(api.WORKERS):
                items = api.embedding_items(artifacts, api.partition(images, worker))
                dim = None
                for kind in ("image", "crop"):
                    identity = {"plan": api.runner.stamp(plan), "ground": ground,
                                "worker": worker, "kind": kind}
                    result = resume_kind(api, items[kind], kind, config, artifacts,
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
