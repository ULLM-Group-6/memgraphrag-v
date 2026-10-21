"""Explicit cached validation adapter; does not patch upstream or frozen modules.

The original plan/checkpoint identities are retained. Preparation validates small
metadata and text outputs; image/crop bytes are checked when consumed. Full mask
checks are retained at finalization/assembly, with persistent hash reuse.
"""
import json
import hashlib
from pathlib import Path

from webqa_validation_cache import log, timed, receipt_valid


class ValidatedRun:
    def __init__(self, api, config, artifacts, run_id, cache):
        self.api, self.config, self.artifacts = api, config, artifacts
        self.run_id, self.cache = run_id, cache
        self.work = artifacts.index_dir / "parallel-v2"
        self.fast = self.work / "sequential-v1" / "validation-v2"

    def prepare(self):
        api, art, config = self.api, self.artifacts, self.config
        context = {"version": 2, "root": str(art.root), "run_id": self.run_id,
                   "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "config": config.model_dump(mode="json"),
                   "effective_text": api.p.effective_text(config).model_dump(mode="json")}
        path = self.fast / "preparation.json"
        with timed("preparation receipt check"):
            receipt = api.runner.read_json(path) if path.exists() else {}
            reuse = receipt_valid(receipt, context)
        if reuse:
            with timed("reuse verified preparation (no source/mask/crop byte scan)"):
                value = receipt["value"]
                self.plan, self.manifest = value["plan"], value["manifest"]
                self.images = [api.ImageRecord.model_validate(row) for row in value["images"]]
                self.records = {row["image_id"]: api.sam.ImageGrounding.model_validate(row) for row in value["records"]}
                self.ground = self.manifest["stages"]["ground"]
        else:
            with timed("validate preparation metadata/text (image/crop bytes deferred to batches)"):
                self.cache.observed = {}
                self.plan = self.cache.read_json(self.work / "plan.json")
                self.manifest = self.cache.read_json(art.index_manifest)
                self.images = api.read_jsonl(art.image_manifest, api.ImageRecord)
                if len(self.images) < api.WORKERS or len({i.image_id for i in self.images}) != len(self.images):
                    raise ValueError("Need at least two unique images")
                if self.plan["workers"] != api.WORKERS or self.plan["config"] != context["config"]:
                    raise ValueError("Config/worker count changed since the frozen plan")
                if self.plan["effective_text"] != context["effective_text"]:
                    raise ValueError("Effective text settings changed")
                text = self.manifest["stages"].get("text")
                self.ground = self.manifest["stages"].get("ground")
                if text != self.plan["text_stage"] or self.ground is None:
                    raise ValueError("Text stage changed or ground stage is incomplete")
                for name, expected in self.plan["sources"].items():
                    self.cache.require(getattr(art, name), expected)
                original_root = Path(api.__file__).resolve().parents[1]
                source_dir = original_root / "src/memgraphrag_v"
                self.cache.observe(source_dir)  # New/deleted source files invalidate the receipt.
                code = sorted(source_dir.glob("*.py"))
                code += [Path(api.__file__).resolve(), Path(api.runner.__file__).resolve(), Path(api.tuning.__file__).resolve()]
                current_names = {str(f.relative_to(original_root)) for f in code}
                if current_names != set(self.plan["code"]):
                    raise ValueError("Source file set changed since the frozen plan")
                for name, expected in self.plan["code"].items():
                    self.cache.require(original_root / name, expected)
                checkpoint = api.sam.default_checkpoint(art)
                self.cache.observe(checkpoint)  # Preserve existing checkpoint identity policy.
                if api.sam.grounder_identity(checkpoint) != self.plan["grounder"]:
                    raise ValueError("Grounder identity changed")
                if [[i.image_id, i.path] for i in self.images] != [row[:2] for row in self.plan["images"]]:
                    raise ValueError("Image identities/order/paths changed")
                if text["inputs"] != self.expected_inputs("text"):
                    raise ValueError("Text inputs are stale")
                if self.ground["inputs"] != self.expected_inputs("ground"):
                    raise ValueError("Ground inputs are stale")
                if self.ground["settings"] != {"sam3": config.sam3.model_dump(mode="json"), "grounder": self.plan["grounder"]}:
                    raise ValueError("Ground settings changed")
                self.verify_outputs("text")
                self.records = {}
                with timed("read grounding records (no masks or crop bytes)"):
                    for number, image in enumerate(self.images, 1):
                        file = art.grounding_dir / "records" / f"{api.sam._key(image.image_id)}.json"
                        record = api.sam.ImageGrounding.model_validate(self.cache.read_json(file))
                        if record.image_id != image.image_id or record.doc_id != image.doc_id or record.status == "failed":
                            raise ValueError(f"Invalid/incomplete grounding record: {image.image_id}")
                        self.records[image.image_id] = record
                        if number % 500 == 0:
                            log(f"Grounding metadata: {number}/{len(self.images)} images")
                payload = [self.records[i.image_id].model_dump(mode="json") for i in self.images]
                if api.p._json_sha256(payload) != self.ground["outputs"]["grounding"]:
                    raise ValueError("Grounding records changed since the ground stage")
                self.cache.observe(art.root / "preparation.json")
                api.runner.preparation(art)
                api.tuning.prepare_session(config, art, self.plan, self.run_id)
                receipt = {"context": context, "dependencies": dict(self.cache.observed),
                           "value": {"plan": self.plan, "manifest": self.manifest,
                                     "images": [i.model_dump(mode="json") for i in self.images], "records": payload}}
                api.runner.atomic_json(path, receipt)
                self.cache.flush()
        # Always verify the small immutable tuning-session file, even on reuse.
        api.tuning.verify_session(config, art, self.plan, self.run_id)
        if not config.retrieval.skip_fact_rerank or config.retrieval.n_docs != 5:
            raise ValueError("Keep skip_fact_rerank=true and n_docs=5")
        self.image_hashes = {row[0]: row[2] for row in self.plan["images"]}
        self.crop_hashes = {}
        for record in self.records.values():
            for det in record.detections:
                if det.crop_status == "saved":
                    if not det.crop_id or not det.crop_path or not det.crop_hash or det.crop_id in self.crop_hashes:
                        raise ValueError(f"Missing/duplicate saved crop identity: {det.detection_id}")
                    self.crop_hashes[det.crop_id] = (det.crop_path, det.crop_hash)
        return self

    def expected_inputs(self, name):
        p, sources = self.api.p, self.plan["sources"]
        text = self.manifest["stages"]["text"]["outputs"]
        if name == "text":
            return {"text_manifest": sources["text_manifest"], "captions": sources["captions"],
                    "image_identities": p._json_sha256([[i.image_id, i.doc_id] for i in self.images])}
        if name == "ground":
            return {"image_manifest": sources["image_manifest"], "fingerprint": p._json_sha256(self.plan["images"]),
                    **{key: text[key] for key in ("texts", "chunks", "entities")}}
        ground = self.manifest["stages"]["ground"]["outputs"]
        if name == "embed":
            return {"image_manifest": sources["image_manifest"], "fingerprint": p._json_sha256(self.plan["images"]),
                    "grounding": ground["grounding"]}
        embed = self.manifest["stages"]["embed"]["outputs"]
        return {"image_manifest": sources["image_manifest"], **{key: text[key] for key in ("memory", "texts", "chunks")},
                "grounding": ground["grounding"], **{key: embed[key] for key in ("image_embeddings", "crop_embeddings")}}

    def items(self, worker=None):
        images = self.images if worker is None else self.api.partition(self.images, worker)
        return {"image": self.api.items_from_images(images, self.artifacts.images_dir.relative_to(self.artifacts.root).as_posix()),
                "crop": self.api.visual.crop_items([self.records[i.image_id] for i in images])}

    def validate_inputs(self, items, kind):
        for item in items:
            self.cache.require(self.artifacts.root / item.path, self.expected_hash(item, kind))

    def expected_hash(self, item, kind):
        if kind == "image":
            return self.image_hashes[item.item_id]
        path, expected = self.crop_hashes[item.item_id]
        if path != item.path:
            raise ValueError(f"Crop path mismatch: {item.item_id}")
        return expected

    def embedding_digest(self, directory):
        parts = [self.cache.digest(directory / name) for name in self.api.FILES]
        return None if None in parts else self.api.p._json_sha256(parts)

    def verify_outputs(self, stage):
        """Same artifact/hash checks as pipeline._verify_outputs, through the cache."""
        art, api = self.artifacts, self.api
        outputs = self.manifest["stages"][stage]["outputs"]
        with timed(f"verify {stage} outputs using cached hashes"):
            for key, path in {"texts": art.texts, "chunks": art.chunks, "memory": art.memory_path,
                              "entities": art.entities, "images": art.indexed_images}.items():
                if key in outputs:
                    self.cache.require(path, outputs[key])
            if stage in ("text", "assemble"):
                stores = outputs.get("text_embedding_files")
                required = {"chunk"} if stage == "text" else {"chunk", "entity", "fact"}
                if not isinstance(stores, dict) or not required.issubset(stores):
                    raise ValueError(f"Missing text embedding store records for {stage}")
                for name in required:
                    self.cache.require(art.root / stores[name]["path"], stores[name]["sha256"])
            for kind, key in (("image", "image_embeddings"), ("crop", "crop_embeddings")):
                if key in outputs and (not outputs[key] or self.embedding_digest(art.embeddings_dir(self.config.visual_encoder.name, kind)) != outputs[key]):
                    raise ValueError(f"{kind} embedding files changed")
            if "graph" in outputs:
                self.cache.require(art.root / outputs["graph_file"], outputs["graph"])
            if "grounding" in outputs:
                # Metadata hash was verified in prepare, or its unchanged receipt.
                for number, record in enumerate(self.records.values(), 1):
                    if record.mask_file is None:
                        if record.mask_hash is not None or record.detections:
                            raise ValueError(f"Missing mask: {record.image_id}")
                        continue
                    self.cache.require(art.root / record.mask_file, record.mask_hash)
                    for det in record.detections:
                        if det.crop_path is None:
                            if det.crop_status == "saved" or det.crop_hash is not None:
                                raise ValueError(f"Missing crop: {det.detection_id}")
                        else:
                            self.cache.require(art.root / det.crop_path, det.crop_hash)
                    if number % 100 == 0:
                        self.cache.flush()
                        log(f"Ground artifact audit: {number}/{len(self.records)} images")
            self.cache.flush()


class CachedEmbeddings:
    """Checkpoint-compatible API for the existing resume_kind loop."""
    def __init__(self, run):
        self.run, self.api, self.cache = run, run.api, run.cache

    def saved_set(self, directory, identity):
        marker = directory / "complete.json"
        if not marker.is_file():
            return None
        saved = self.api.runner.read_json(marker)
        if saved["identity"] != identity:
            raise ValueError(f"Different inputs in {directory}")
        if set(saved["files"]) != set(self.api.FILES):
            raise ValueError(f"Incomplete checkpoint marker: {directory}")
        for name, expected in saved["files"].items():
            self.cache.require(directory / name, expected)
        return self.api.EmbeddingSet.load(directory)

    def save_set(self, directory, result, identity):
        self.api.save_set(directory, result, identity)
        self.cache.flush()

    def merge_sets(self, parts, expected_ids):
        return self.api.merge_sets(parts, expected_ids)

    def check_set(self, result, items, kind, config, artifacts):
        wanted = self.api.visual.encoder_meta(config.visual_encoder)
        if result.meta.kind != kind or any(getattr(result.meta, k) != v for k, v in wanted.items()):
            raise ValueError("Embedding model metadata changed")
        if [row.item_id for row in result.rows] != [item.item_id for item in items]:
            raise ValueError("Embedding item IDs/order changed")
        self.run.validate_inputs(items, kind)
        for row, item in zip(result.rows, items):
            if (row.path, row.image_id, row.doc_id, row.entity_id) != (item.path, item.image_id, item.doc_id, item.entity_id):
                raise ValueError(f"Embedding row metadata changed: {item.item_id}")
            if row.status == "failed":
                if self.api.visual._readable(artifacts.root / item.path):
                    raise ValueError(f"Failed embedding is now readable: {item.item_id}")
            elif row.sha256 != self.run.expected_hash(item, kind):
                raise ValueError(f"Embedding row has a different source hash: {item.item_id}")
        if not self.api.np.isfinite(result.vectors).all():
            raise ValueError("Nonfinite embedding vectors")
