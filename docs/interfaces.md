# Shared interfaces

This page describes the files that retrieval reads and writes. The code in [`src/memgraphrag_v/`](../src/memgraphrag_v) enforces it: `schemas.py` for the manifests and retrieval output, `embeddings.py` for the EVA-CLIP vectors. If this page and the code disagree, fix one of them.

It covers only what retrieval needs. Records that only graph construction uses (the MemGraphRAG entity export, the passage-to-chunk map, SAM3 detections) are added with plan tasks 2 and 4.

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
└── runs/<run_id>/retrieval.jsonl
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

## Retrieval output (plan tasks 3 and 5)

`runs/<run_id>/retrieval.jsonl` holds one `RetrievalResult` per question and system:

```json
{"question_id": "abc123", "system": "D", "route": "direct",
 "evidence": [{"doc_id": "Barack Obama", "score": 0.31, "image_id": "f0e1…",
               "passage_ids": ["0a1b…"]}]}
```

- `route` ∈ {graph, direct, fallback}, so fallbacks are always logged.
- `evidence` holds at most five documents, each listed once, best first with scores that never increase. `image_id` is the document's single selected image, or null for text-only systems.
- `passage_ids` lists the passages whose text goes to the reader, in source order.

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
sbatch scripts/embed_eva_clip.sbatch --limit 20         # smoke run, then without --limit
```

To run the parity check against MG2, use a GPU job that runs `python scripts/check_eva_parity.py --mg2 <MG2-RAG checkout>` (MG2 commit `91f0eed`).
