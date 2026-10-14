# Shared interfaces

This page describes the files that retrieval reads and writes. The code in [`src/memgraphrag_v/`](../src/memgraphrag_v) enforces it: `schemas.py` for the manifests and retrieval output, `embeddings.py` for the EVA-CLIP vectors. If this page and the code disagree, fix one of them.

It covers what retrieval needs, plus the [combined pipeline](#combined-pipeline-implementation-note) that builds the index (entity export, chunk map, SAM3 detections).

## Artifacts root

Everything lives under one directory outside the repo, `$MGRV_ARTIFACTS`. On Snellius that should be project space or `/scratch-shared`, not `$HOME`. `paths.Artifacts` knows the layout:

```
$MGRV_ARTIFACTS/
├── manifests/            output of Misha's build_mmqa_manifest.py, copied as is
│   ├── text_manifest.jsonl, image_manifest.jsonl, questions.jsonl
│   └── captions.jsonl    generated image descriptions (caption systems only)
├── corpus/images/        MMQA final_dataset_images/; ImageRecord.path is relative to this
├── models/               EVA-CLIP-8B/, clip-vit-large-patch14/ (scripts/download_eva_clip.py)
├── embeddings/eva-clip-8b/{image,query,crop}/   vectors.npy, rows.jsonl, meta.json
├── index/                combined MemGraphRAG-V index (see the implementation note below)
└── runs/<run_id>/        retrieval.jsonl, config.yaml (+ diagnostics.jsonl, provenance.json for the combined pipeline)
```

## Manifests (plan task 1): `schemas.py`

The records mirror the builder's output field for field, so the files are read without a conversion step. Records are frozen and reject unknown fields. IDs are non-empty strings and may contain spaces, because MMQA `doc_id`s are Wikipedia titles.

| File | Record | Fields |
|---|---|---|
| `text_manifest.jsonl` | `PassageRecord` | doc_id, passage_id, text, source_url? |
| `image_manifest.jsonl` | `ImageRecord` | image_id, doc_id, path, source_url?, passage_ids, width?, height?, ok |
| `questions.jsonl` | `QuestionRecord` | qid, question, answers, gold_doc_ids, gold_image_ids, gold_passage_ids, distractor_image_ids, distractor_passage_ids, q_type?, modalities, rephrasing_confidence?, split ∈ {dev, test}, group_id |
| `captions.jsonl` | `CaptionRecord` | image_id, text, model, prompt_version (provisional) |

Retrieval reads only `qid` and `question` from a question. The gold and distractor fields are for evaluation. Misha's `checkjson.py` validates the manifests: uniqueness, references, image files, split hygiene and leakage.

## EVA-CLIP embeddings (plan tasks 3 and 4): `embeddings.py`, `eva_clip.py`

The encoder is MG2's setup: `BAAI/EVA-CLIP-8B` with the `openai/clip-vit-large-patch14` image processor. Both revisions are pinned in `configs/default.yaml`. `EvaClip` reproduces MG2's `EvaClipModel` preprocessing (fp16, the model's `encode_image`/`encode_text`, 77-token text limit, L2 normalisation), and `scripts/check_eva_parity.py` checks that both give the same vectors. There is one deliberate difference: MG2 replaces an unreadable image with a black one, but we open every file ourselves and record the failure instead.

Each kind (`image`, `crop`, `query`) has its own directory:

- **`vectors.npy`**: float32 `(n_ok, 1280)`, L2-normalised, so a dot product is the cosine similarity.
- **`rows.jsonl`**: one `EmbeddingRow` per input item, failures included:

  | Field | Meaning |
  |---|---|
  | `row` | index into `vectors.npy`; null iff the item failed |
  | `kind` | image, crop or query |
  | `item_id` | image_id, crop_id or qid |
  | `image_id`, `doc_id` | the image (or the crop's parent image) and its document |
  | `entity_id` | crops only: the MemGraphRAG entity SAM3 grounded |
  | `path`, `sha256` | input file relative to the root, and its hash (detects stale vectors) |
  | `width`, `height` | size of the image as encoded |
  | `status`, `error` | ok, truncated (query over 77 tokens, still embedded) or failed |

- **`meta.json`**: `EmbeddingMeta`: encoder, revision, processor, processor_revision, dim, kind, n_rows, n_failed.

Because rows carry `doc_id` and `entity_id`, direct retrieval (image → document) and crop seeds (crop → entity) need no joins. `EmbeddingSet.load(dir)` gives you `row_for(id)`, `vector(id)` and `top_k(query_vector, k)`, ranked best first with ties broken by item_id.

Whole images and questions are embedded on a GPU node with `scripts/embed_eva_clip.sbatch`. SAM3 crops are embedded from Python with `embed_files(items, "crop", ...)`, using `EmbedItem`s that the SAM3 step builds.

## Whole-image retrieval (plan task 3): `image_retrieval.py`

- **`image_passage_edges(images, image_set)`** gives the graph edges `(image_id, passage_id, 1.0)` from each image to its document's passages. Images without a vector are left out, so PPR can never select an image the reader cannot see.
- **`select_documents(image_scores, images, n_docs=5)`** scores each document by its best image and returns the top documents as `Evidence`, each with that image and its passages. Both D (cosine scores) and G (PPR image-node scores) use it, so they pick documents the same way.
- **`retrieve_direct(...)`** is system D: the question's EVA-CLIP vector against every whole-image vector. `python -m memgraphrag_v.image_retrieval --run-id <id> [--split dev]` writes `runs/<id>/retrieval.jsonl` and `config.yaml`, and prints how often the gold document is ranked first and in the top 5 (a sanity check, not the evaluation). `python scripts/show_retrieval.py --run-id <id>` turns a run into `runs/<id>/report.html`: each question with its retrieved documents' thumbnails, scores and first passage, gold document outlined.

The attached passages are not trimmed here: see `passage_ids` under *Retrieval output* for where the 256-token cut belongs.

**Dummy dataset.** `python -m memgraphrag_v.dummy <root>` writes a synthetic corpus in Misha's format: twelve documents "<Colour> <shape>", each with one picture and the question "Which picture shows a red circle?", plus edge cases (a document with two pictures, a picture without passages, a text-only document and an unreadable image). `tests/test_image_retrieval.py` runs embedding and retrieval on it with a colour-matching stand-in encoder; `scripts/dummy_image_retrieval.sbatch` runs it with the real EVA-CLIP on Snellius.

## Retrieval output (plan tasks 3 and 5)

`runs/<run_id>/retrieval.jsonl` holds one `RetrievalResult` per question and system:

```json
{"question_id": "abc123", "system": "D", "route": "direct",
 "evidence": [{"doc_id": "Barack Obama", "score": 0.31, "image_id": "f0e1…",
               "passage_ids": ["0a1b…"]}]}
```

- `route` ∈ {graph, direct, fallback}, so fallbacks are always logged.
- `evidence` holds at most five documents, each listed once, best first with scores that never increase. `image_id` is the document's single selected image, or null for text-only systems.
- `passage_ids` lists the evidence's associated passages in source order, **untrimmed**. For an image that is every paragraph of its Wikipedia page. The reader input keeps only the first `retrieval.text_budget_tokens` (256) tokens of them. That cut happens once, in the shared reader-input step, for every system, so D and G get the same text budget. It is still to be built (reader/logging work), using the reader's tokenizer.

## Configuration: `config.py`, `configs/default.yaml`

The config holds the dataset, visual encoder, SAM3 threshold and retrieval settings. Settings still to be decided are `null`: `sam3.threshold` (task 4) and `retrieval.beta` (tuned on dev). `unresolved()` lists them, and `require_frozen()` refuses to run while any remain, so call it before a test evaluation. `config_hash()` ignores key order and `artifacts_root`, so the same experiment hashes the same on every machine. Later tasks add their own sections.

## Running on Snellius

```bash
git clone … && cd memgraphrag-v
module load 2023 Python/3.11.3-GCCcore-12.3.0
python -m venv .venv-eva && source .venv-eva/bin/activate
pip install -e ".[eva,dev]" huggingface_hub && pytest -q
export MGRV_ARTIFACTS=/scratch-shared/$USER/mgrv
python scripts/download_eva_clip.py                     # login node, about 30 GB
sbatch scripts/dummy_image_retrieval.sbatch             # end-to-end test on the dummy dataset
sbatch scripts/embed_eva_clip.sbatch --limit 20         # MMQA smoke run, then without --limit
python -m memgraphrag_v.image_retrieval --run-id d-dev --split dev
python scripts/show_retrieval.py --run-id d-dev           # runs/d-dev/report.html
```

To run the parity check against MG2, use a GPU job that runs `python scripts/check_eva_parity.py --mg2 <MG2-RAG checkout>` (MG2 commit `91f0eed`).

## Combined pipeline (implementation note)

`pipeline.py` builds **one** multimodal graph and retrieves from it. It stops at evidence ready for a future reader. This note overrides the older scope in `method.md` where they differ: EVA-CLIP (not SigLIP2); one indexing invocation and one graph (no separate text baseline snapshot, ablations, evaluation, tuning, caption generation or reader); the dataset supplies passages, captions and images; the benchmark and reader VLM stay undecided. Nothing here was executed: see the checklist below.

### Modules

| File | Responsibility |
|---|---|
| `sam3_processing.py` | entity-name prompts per image, detections (≥ `sam3.threshold`), masks, MG2-style crops, per-image resumable records, adapter over MG2's `SAM3Grounding` |
| `visual_indexing.py` | EVA-CLIP embeddings of whole images and crops with explicit save and reuse validation; image vertices, image–passage and image–entity edges |
| `retrieval.py` | adapted `prepare_retrieval_objects`, text and visual seeds, fusion, CPU PPR, `select_documents`, dense fallback, diagnostics |
| `pipeline.py` | owns the upstream text backend (thin subclass), stages, `index_with_memory_v`, `load_index`, `retrieve`, CLI |

`schemas.py`, `config.py` and `paths.py` gained the records, settings and paths below. `embeddings.py`, `eva_clip.py` and `image_retrieval.py` are reused unchanged (`EmbedItem`, `EmbeddingSet`, `items_from_images`, `embed_files`, `image_passage_edges`, `select_documents`, `Evidence`, `RetrievalResult`).

### Indexing stages

```bash
python -m memgraphrag_v.pipeline index --llm-name <served name>                 # all four stages
python -m memgraphrag_v.pipeline index --stages text assemble --llm-name <name>   # MemGraphRAG environment
python -m memgraphrag_v.pipeline index --stages ground embed                      # MG2 / SAM3 / EVA environment
```

1. **text**: passages and supplied captions become `TextRecord`s (passages in manifest order, then captions as `caption:<image_id>:<n>`, in the parent image's document). Each record is chunked on its own (upstream's token-window algorithm; the text is never prefixed with IDs). Chunk IDs hash (record, position, text), so equal text in two records never collapses. Then upstream `triple_extraction`, `build_from_raw_openie_results`, `extract_memory_schema`, `filter_memory_ontology` (default: absolute, minimum frequency 1), `detect_memory_conflicts`, `resolve_memory_conflicts`. The surviving entities and their supporting chunks are exported from the final memory (`entities.jsonl`).
2. **ground**: each image is prompted with the entities of its *explicitly associated* chunks (its `passage_ids` and its captions; a shared document is not an association).
3. **embed**: EVA-CLIP embeds every manifest image (so the set equals what `embeddings.py image` writes; failures stay as failed rows) and every saved crop (`kind="crop"`, with the existing entity ID).
4. **assemble**: upstream `build_memory_graph` and `install_memory_graph` (their save is deferred), then one vertex per usable embedded image, `image_passage` edges (weight 1) to the chunks of its associated records, and `image_entity` edges weighted by the maximum retained SAM3 confidence of the pair. Each new pair is added once; text edges, including parallel ones, are untouched. The finished graph is written once.

Image vertices are named `image-<image_id>`; graph vertex kinds are entity, type, passage and image. Crops and facts are never vertices.

Each stage records its input hashes, settings and output hashes in `index/manifest.json`. A stage whose stamp no longer matches raises `StaleIndexError` (use `--force` to recompute); `load_index` also checks that the stages agree with each other and that the saved files still have the recorded hashes. `embed_cached` reuses saved vectors only if encoder and processor revisions, item IDs, paths and file hashes match. SAM3 results are saved per image, so an interrupted `ground` resumes.

```
index/
├── manifest.json            stage stamps (inputs, settings, output hashes)
├── texts.jsonl, chunks.jsonl, entities.jsonl, images.jsonl   TextRecord, ChunkRecord, EntityRecord, IndexedImage
├── text/                    MemGraphRAG save_dir: OpenIE cache, LLM cache, memory.json,
│   └── <llm>_<embedding>/   chunk/entity/fact stores and graph.graphml (the final multimodal graph)
└── grounding/               records/<md5(image_id)>.json (ImageGrounding), masks/*.npz, crops/<md5>/*.png
```

### Retrieval

`load_index` loads the saved graph and stores without rebuilding or grounding, and `retrieve` reads only question IDs and texts.

- **Text seeds** (upstream seed building without PPR): raw query–fact similarity ≥ `fact_similarity_threshold` (0.4) selects facts (`skip_fact_rerank`, `use_raw_threshold_filter`); min–max normalized fact scores become entity weights (a later fact overwrites an earlier one for the same entity, as upstream), divided by the entity's passage count; `get_top_k_weights` keeps `linking_top_k` entities; then `t_raw = phrase_weights + passage_node_weight × normalized passage scores` (0.05). No selected fact: channel unavailable.
- **Visual seeds**: one EVA question vector against whole images and crops; top 20 of each by raw cosine (ties by ID), negatives clamped to 0. Image scores seed image vertices; crop scores seed existing entity vertices (max per image–entity pair, then max across images). No confidence weighting or normalization other than the final `N(x) = x / sum(x)`.
- **Fusion**: `v = 0.5 i + 0.5 c` (or the available one), `s = (1 − β) t + β v` (or the available channel), neither → dense-passage fallback. β must be given (`--beta` or `retrieval.beta`); no value is claimed optimal and nothing tunes it.
- **PPR**: `igraph personalized_pagerank`, all vertices, undirected, edge `weight`, PRPACK, `damping` from the config (restart probability `1 − damping`). `select_documents` keeps the best image of at most five documents; `route="graph"`.
- **Fallback** (`route="fallback"`): best chunk per document by dense similarity, at most five documents, each with the first usable image explicitly associated with the chunk's record in source order, else `image_id=None`. It also answers when no image receives PPR mass, so arbitrary images are never returned with score 0.
- `Evidence.passage_ids` are text **record** IDs (passage IDs and `caption:<image_id>:<n>`), in source order: passages in manifest order, then captions.
- `runs/<run_id>/` gets `retrieval.jsonl`, `diagnostics.jsonl` (facts, component availability, strongest seeds, β, top image PPR scores, selected IDs, fallback reason), `provenance.json` and `config.yaml`.

### Configuration and environment

`config.py` / `configs/default.yaml` gained `sam3` (threshold 0.5, minimum crop 10 px, prompt batch 1), `text` (`llm_name` is still null; chunk 256/32; ontology filter) and the retrieval settings above. Machine-specific inputs are environment variables, read only when the operation needs them:

| Variable | Needed by | Meaning |
|---|---|---|
| `MGRV_ARTIFACTS` | all | artifacts root (or `artifacts_root`) |
| `MGRV_MEMGRAPHRAG` | text, assemble, retrieve | checkout of [MemGraphRAG @ 49f20cec](https://github.com/XMUDeepLIT/MemGraphRAG/tree/49f20cec41290c95209724d33ec527f72449d300); its `code/` is imported |
| `MGRV_MG2_RAG` | ground | checkout of MG2-RAG @ `91f0eed` (bundled `sam3/`, `assets/`, the wrapper) |
| `MGRV_SAM3_CHECKPOINT` | ground | SAM3 weights (default `<artifacts>/models/sam3/sam3.pt`) |
| `MGRV_TEXT_EMBEDDING_MODEL` | text, assemble, retrieve | overrides `text.embedding_model`, e.g. an offline snapshot path; it names the stores, so keep it the same |
| `OPENAI_API_KEY` | text, assemble, retrieve | for the OpenAI-compatible endpoint (`text.llm_base_url`), e.g. a local vLLM server (any value) |

**Environments.** MemGraphRAG pins `torch==2.5.1`, `transformers==4.45.2`, `pydantic==2.10.4` (`snellius-work-memgraph/req_memgraph.txt`); MG2's requirements and this repo's `eva` extra use `torch==2.7.0` and `transformers==4.49.0` (SAM3 was smoke-tested on torch 2.7.0). A single environment is therefore unverified. Run `text` and `assemble` in the MemGraphRAG environment and `ground` and `embed` in the MG2 environment; `retrieve` needs the upstream backend and EVA-CLIP in one process (see the open issues).

### Adaptations of upstream MemGraphRAG

The text backend is upstream imported from a checkout, with a documented subclass in `pipeline.py` (no runtime patching): chunks are inserted under source-aware IDs (upstream hashes the text); the OpenIE cache is matched by chunk ID and text, in source order (upstream re-hashes it); the entity-to-triples index, which re-hashes chunk IDs, is skipped; the text-only graph is not saved on installation; `get_top_k_weights` keeps its logic without the final assertion that fails for a zero-weight entity. `retrieval.prepare_retrieval_objects` replaces upstream's, whose vertex-count assertion excludes images. The local `snellius-work-memgraph/backups/MemGraphRAG.py` (no `run_ppr`) is not used; `MemGraphRAG.py_bkp` has the byte size of the pinned commit's file (121,890 bytes).

### Open issues

- **Not executed or tested.** No imports, tests or models were run; the code was written against the sources of the pinned commits.
- `retrieve` needs EVA-CLIP (transformers 4.49.0, the smoke-tested pin) and the upstream backend (4.45.2 pin) in one process. Whether one environment serves both is unverified. `retrieval.retrieve_questions` takes any object with `encode_texts`, so precomputed question vectors can be supplied if it does not.
- Whether MG2's modules and upstream's generic `src` package coexist in one process (`index` without `--stages`) is unverified; the staged runs avoid it.
- `MG2Sam3` follows `component-verification/smoke_sam3.py`; masks are assumed boolean `(n, 1, H, W)` and boxes `(n, 4)` in original coordinates, as that script asserts. Crops reproduce MG2's `_process_single_crop` rules; a parity check against `get_save_crop` is not written.
- `skip_fact_rerank=False` needs upstream's DSPy reranker configuration, which this pipeline does not expose.
- The 256-token cut of `Evidence.passage_ids` (and caption position within it) belongs to the future reader input step.

### Snellius verification checklist (not yet run)

1. `pytest -q` (config and pure-logic tests), in the MG2 environment.
2. Environments: install MemGraphRAG's requirements in one and MG2's (+ this repo with `.[eva]`) in the other; try importing the upstream backend and EVA-CLIP in one process.
3. Dummy dataset (`python -m memgraphrag_v.dummy`), then `index --stages text`; inspect `chunks.jsonl`, `entities.jsonl`, that chunk IDs in `memory.json`, the stores and the graph agree, and that equal text in two records gives two chunk vertices.
4. `index --stages ground` on a few images: positive, empty and failing prompts; open `crops/` and `masks/`; check IDs and scores in `records/*.json`; rerun to see per-image reuse.
5. `index --stages embed assemble`: compare `images.jsonl`, `stats` in the manifest (vertices, both edge counts), no crop vertices, no duplicate image pairs, image vectors identical to a direct `embeddings.py image` run.
6. Tamper checks: change a manifest line, a setting or an image and confirm `StaleIndexError`; rerun with `--force`.
7. `retrieve --run-id <id> --beta <β> --limit 5`: check `diagnostics.jsonl` for seeds, components, fallback reasons; try a question that selects no fact (visual only) and one with neither channel (fallback).
8. Time and memory: stage by stage on the real corpus before the full run.
