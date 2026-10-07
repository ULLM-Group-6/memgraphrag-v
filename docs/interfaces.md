# Shared interfaces (task [01], schema version 1)

This is the contract that every pipeline stage reads and writes. The code in [`src/memgraphrag_v/`](../src/memgraphrag_v) enforces it, and this page explains it. If the two disagree, the code is a bug: fix one of them and bump `SCHEMA_VERSION` when a record changes shape.

It implements [method.md](method.md) §2–§5 and §8 and the outputs that [implementation_tasks.md](implementation_tasks.md) asks for. Settings owned by [02] are deliberately left as `null` (see [Configuration](#configuration)).

## Stable IDs — `ids.py`

| ID | Format | Who creates it |
|---|---|---|
| `doc_id`, `question_id` | The benchmark's own string (non-empty, no whitespace) | Benchmark |
| `passage_id`, `image_id` | The benchmark's own ID if it has one; otherwise `"{doc_id}#p0003"` / `"{doc_id}#img012"` (0-based, in source order) | Benchmark / [09] |
| `entity_id` | `"entity-" + md5(name)`, MemGraphRAG's `compute_mdhash_id` | MemGraphRAG only |
| `chunk_id` | `"chunk-" + md5(passage text)`, MemGraphRAG's passage node | MemGraphRAG only |
| image node | `"image-" + md5(image_id)`, the graph vertex of a whole image | [20] |
| `crop_id` | `"crop-" + md5(f"{image_id}\|{entity_id}\|{rank}")`; `rank` is the detection's 0-based position among the pair's kept detections, by confidence descending | [13] |

- We **never create entities**. `upstream_entity_id()` / `upstream_chunk_id()` exist only to check and join against upstream output, and the tests compare them with upstream's function.
- Upstream passage IDs are content hashes, so [11] exports `passage_map.jsonl` to join them to benchmark passage IDs. Two passages with identical text get the same `chunk_id` and therefore share one graph node. We keep upstream's behaviour and use this rule: the node's PPR score is credited to every passage mapped to it, so each of their documents can be ranked by it, and an image linked to one of those passages is linked to the shared node. `validate_manifests` warns about every such group and about images whose shared node spans documents.
- A crop's `rank` is fixed before MG²'s 10 px crop filter, so `crop_id` joins to exactly one detection and the saved crops of a pair can skip ranks.
- **Crops are records, not graph vertices.**

## Files and records — `schemas.py`, `io.py`

All records are JSON Lines, one record per line, written with `write_jsonl`. Each file `x.jsonl` has a sidecar `x.jsonl.meta.json` holding `schema_version`, `record_type`, `count` and `sha256`. By default, `read_jsonl` validates every line and checks the sidecar. Records are frozen and reject unknown fields.

**All file paths inside records are POSIX paths relative to the artifacts root**, for example `corpus/images/d1_0.jpg`. Absolute paths, `..` and backslashes are rejected. The same manifests therefore work on a laptop and on Snellius.

| File (under the artifacts root) | Record | Fields | Written by |
|---|---|---|---|
| `manifests/text.jsonl` | `PassageRecord` | doc_id, passage_id, text, source_location? | [09] |
| `manifests/images.jsonl` | `ImageRecord` | image_id, doc_id, passage_ids[] (source order, no repeats), path, page_or_figure?, caption? (original only) | [09] |
| `manifests/questions.jsonl` | `QuestionRecord` | question_id, text, answers[≥1], gold_doc_ids[≥1], split ∈ {dev, test}, gold_image_ids?, group_id? | [09]/[10] |
| `manifests/crops.jsonl` | `CropRecord` | crop_id, image_id, entity_id, prompt, bbox (pixel x1,y1,x2,y2), confidence ∈ [0,1], path, mask_path | [13]/[18] |
| `grounding/sam3/attempts.jsonl` | `GroundingRecord` | image_id, entity_id, prompt, threshold, detections[] (`Detection`: rank, bbox, confidence, mask_path, crop_id?) | [13]/[18] |
| `descriptions/descriptions.jsonl` | `DescriptionRecord` | image_id, text, model, model_revision, prompt_version, max_tokens, input/output_tokens?, status ∈ {ok, failed, truncated}, note? | [14] |
| `index/text/exports/passage_map.jsonl` | `PassageMapRecord` | passage_id, doc_id, chunk_id | [11] |
| `index/text/exports/entities.jsonl` | `EntityRecord` | entity_id (must equal hash of name), name, type?, passage_ids[] | [11] |
| `index/caption/exports/chunk_map.jsonl` | `CaptionChunkMapRecord` | chunk_id, doc_id, passage_id?, image_id? (at least one; image_id when the chunk holds that image's description) | [22] |
| `embeddings/<encoder>/<kind>/rows.jsonl` | `EmbeddingRow` | row, item_id, key (`EmbeddingKey`), status ∈ {ok, failed, truncated}, note? | [12]/[19] |
| `runs/<run_id>/retrieval.jsonl` | `RetrievalResult` | see below | [16]/[24] |
| `runs/<run_id>/diagnostics.jsonl` | `DiagnosticsRecord` | see below | [16]/[24] |

`GroundingRecord` keeps one line for every (image, entity) prompt, including prompts with zero detections. That makes "SAM3 missed it" distinguishable from "never prompted" in the error analysis, and an image with no detections stays in the corpus. Every kept detection (confidence ≥ threshold) is stored with its box, mask and confidence, in rank order. `crop_id` is null when MG²'s crop code dropped the box for being under 10 px, so an edge can exist without any crop but no detection is lost. `n_kept`, `n_crops` and `max_confidence` are computed from the list. The edge exists when `n_kept > 0`, and its weight is `max_confidence`, never a sum over detections.

`EmbeddingKey` = encoder, model revision, processor revision, kind ∈ {image, crop, query} and the input's sha256. A cached vector is reused only if every field matches. The vectors themselves sit next to `rows.jsonl`, in a format [12] chooses, where row *i* is vector *i*.

`validate_manifests(passages, images, questions, crops, artifacts_root)` runs the checks [09] needs:

- **Errors (block indexing):** duplicate IDs, an image that references an unknown passage or a passage from another document, a crop of an unknown image, and missing files (images, crops, masks). When `grounding` is passed: a pair prompted twice, a crop with no matching detection or whose image, entity, confidence or box differ from it, and a detection whose crop is missing from the crop manifest.
- **Warnings (coverage report):** gold documents or images that are not in the corpus, images with no associated passage, and passages with identical text (see the chunk rule above).

`check_config_consistency(config, grounding, descriptions)` checks cached artifacts against the experiment config: the grounding threshold must equal `sam3.threshold`, and descriptions must come from `reader.model` at `reader.revision`, with `caption.prompt_version` and `caption.description_max_tokens` (§6 uses the chosen reader VLM for descriptions).

### Retrieval output

```json
{"question_id": "q1", "system": "G", "selection": "image", "route": "graph",
 "seeds": {"text_used": true, "image_used": true, "crop_used": true, "beta": 0.5},
 "evidence": [{"rank": 1, "doc_id": "d1", "score": 0.031, "image_id": "d1#img000",
               "passage_id": null, "attached_passage_ids": ["d1#p0000"]}],
 "config_hash": "<sha256 of the frozen config>"}
```

- `route` ∈ {graph, direct, fallback}, so fallbacks are always logged.
- `selection` says how documents were ranked (§5.3): by their best image node or by their best passage. G0 is the only system that runs both ways. G0 with `image` is the ablation row, and G0 with `passage` is the T-vs-G0 graph diagnostic, which uses the same passage-to-document ranking as T.
- `seeds` records which components put restart mass into the seed vector (`*_used`), not which were computed. `beta` is set exactly when text and a visual component were both used.
- Evidence ranks are 1..n with n ≤ 5, document IDs are unique, and scores do not increase with rank. This means one image (or passage) per document, at most five documents.
- Each item has an `image_id` (image selection), a `passage_id` (passage selection or fallback), or both (a native-image fallback attaches the passage's first image), plus the passages whose text goes to the reader.
- `latency_ms` is retrieval time without generation, used for the median and p95 latency report. It is null for G-caption, which copies G's evidence (same IDs, order and text) without retrieving again.
- Reader inputs and answers (`reader.jsonl`) and scores (`scores.jsonl`) belong to [17] and [15]; their paths are already reserved.

`RetrievalResult` rejects combinations that contradict §6:

| System | Selection | Routes | Seeds allowed |
|---|---|---|---|
| T, C | passage | graph, fallback | text |
| D | image | direct, fallback | none (`seeds` is null) |
| G | image | graph, fallback | text, image, crop |
| G-image, G-ground | image | graph, fallback | text, image |
| G0 | image or passage | graph, fallback | text; missing text seeds go to the fallback, never to visual seeds |
| G-caption | image | copied from G | copied from G, or null |

On the graph route at least one component must be used and `seeds` must be present (G-caption excepted). On the fallback route none may be used.

The text-regression check in §7 (T and the extension on a text-only subset) doesn't need its own system ID. It runs as T and G in a separate run directory whose config names the other dataset, so its `config_hash` already tells it apart.

### Diagnostics output

`runs/<run_id>/diagnostics.jsonl` holds one `DiagnosticsRecord` per question and system, written next to `retrieval.jsonl`. It contains what the §7 error analysis needs to separate "visual encoder", "graph propagation" and "SAM3 grounding" failures:

- `image_candidates`: the top `retrieval.diagnostics_top_n` images by SigLIP2 similarity. The first `top_k_candidates` (20) are the ones that seeded the graph, and for D this list is the direct ranking.
- `crop_candidates`: crop_id, image_id, entity_id and similarity, with the same cut-off.
- `seeds`: every node with non-zero restart mass, its channel (text, image or crop) and its weight after normalisation and fusion. A node seeded by two channels appears once per channel.
- `ppr_ranking`: the top image IDs (image selection) or passage IDs (passage selection) after PPR. Empty for D.

Gold labels are not written here, because retrieval code never sees them. The analysis joins with `questions.jsonl` to get the gold image's rank before PPR (`image_candidates`) and after it (`ppr_ranking`), reporting "> N" when it is outside the stored top N.

## Artifact layout — `paths.py`

Everything lives under one **artifacts root outside the repo**. The root comes from `$MGRV_ARTIFACTS`, or from `artifacts_root` in a local, uncommitted config.

```
$MGRV_ARTIFACTS/
├── corpus/images/                     image files referenced by ImageRecord.path
├── manifests/{text,images,questions,crops}.jsonl
├── splits/                            question lists, grouping rule, seed, hashes ([10])
├── index/text/memgraphrag/            upstream save_dir, unaugmented graph ([11])
├── index/text/exports/{passage_map,entities}.jsonl
├── index/caption/                     separate index for baseline C ([22]), exports/chunk_map.jsonl
├── descriptions/descriptions.jsonl    cached image descriptions ([14])
├── grounding/sam3/{attempts.jsonl,crops/,masks/}
├── embeddings/<encoder-slug>/{image,crop,query}/
├── graph/                             image–passage and image–entity augmentation ([20], [21])
└── runs/<run_id>/{config.yaml,retrieval.jsonl,diagnostics.jsonl,reader.jsonl,scores.jsonl}
```

`<encoder-slug>` turns `/` into `--`, for example `google--siglip2-so400m-patch14-384`. Use `layout.resolve(record.path)` to open a file that a manifest refers to.

## Configuration — `config.py`, `configs/default.yaml`

`ExperimentConfig` has these sections: dataset, upstream, text_index, visual_encoder, sam3, retrieval, caption, reader, judge, evaluation and artifacts_root. Unknown keys are rejected.

- Values already fixed by method.md are filled in: SAM3 threshold 0.5, top-20 candidates, visual mix 0.5, β grid {0.25, 0.5, 0.75}, five documents, a 256-token text budget and a 256-token description limit, 2,000 bootstrap resamples, and seed 42.
- Values that are still open are `null` and annotated with the task that will resolve them. These include the reader, judge, text encoder, PPR restart α, model revisions, the MemGraphRAG and MG² commits, the crop preprocessing, prompt versions, and the corpus and split hashes. The visual encoder defaults to SigLIP2, pending the EVA-CLIP decision in [05]/[12].
- The text index pins `chunking` and `upstream_settings`, a copy of the upstream `BaseConfig` used for indexing (extraction, ontology filtering, conflict resolution, `linking_top_k`, `passage_node_weight`, …) without machine paths. The dense-passage fallback is upstream's `dense_passage_retrieval`, so it uses `text_index.text_encoder`; only `retrieval.fallback_top_k` is separate. `retrieval.text_budget_tokenizer` names the tokenizer the 256-token budgets are counted with, and `dataset.distractor_set` records where the fixed distractors come from.
- `unresolved()` lists every null setting. `require_frozen()` raises while any remain, so call it before a test evaluation.
- `config_hash()` is the sha256 of the canonical JSON. It ignores key order and `artifacts_root`, so the same experiment hashes the same on every machine. Each `RetrievalResult` carries this hash, and each run saves its config at `runs/<run_id>/config.yaml`.

## Open points for other tasks

These don't block [01], but the interfaces assume an answer:

- **[03] Text-only questions.** `QuestionRecord` has no query-image field, because method.md §1–2 asks for text questions that make sense without a query image. InfoSeek (proposed on the `dataset` branch) is a VQA benchmark whose questions come with a query image, so it may conflict with this.
- **[03]/[15] Alternative valid evidence.** `gold_doc_ids` is a single set, which matches the Recall@5 formula in method.md §7. If the benchmark lists alternative evidence sets ("any of"), the question record needs another field.
- **[09]/[11] Caption entities.** Under §3.3, SAM3 candidates come from entities in the image's passages *and caption*. MemGraphRAG only extracts entities from indexed passages, so the caption has to be part of a passage's text for its entities to exist.
- **[22]/[17] Descriptions in baseline C.** method.md doesn't say whether C indexes each description as its own chunk or appends it to an associated passage, or which description fills a document's 256-token description slot when C ranks by passage. `chunk_map.jsonl` supports both layouts. [22] picks the layout and [17] picks the rule (for example, the description of the first image attached to the selected passage).
- **[11] Entity names.** `EntityRecord` requires `entity_id == "entity-" + md5(name)`. Upstream hashes the exact fact-layer string (`MemGraphRAG.py:838–841, 892`), so export `name` unchanged, without `text_processing`.

## Running on Snellius

Clone the repo and keep artifacts on project or scratch space, not in `$HOME`:

```bash
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1   # Python 3.11; only needed by GPU stages
python -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -e ".[dev]" && pytest -q
export MGRV_ARTIFACTS=/scratch-shared/$USER/mgrv   # or project space
```

The package itself needs only `pydantic` and `pyyaml`, so it installs into the separate SAM3 and encoder environments too. `.gitattributes` forces LF line endings, and `file_sha256` normalises CRLF, so hashes match between Windows and Linux.
