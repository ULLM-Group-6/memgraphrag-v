# Shared interfaces

This page describes the files that retrieval reads and writes. The code in [`src/memgraphrag_v/`](../src/memgraphrag_v) enforces it: `schemas.py` for the manifests and retrieval output, `embeddings.py` for the EVA-CLIP vectors. If this page and the code disagree, fix one of them.

It covers only what retrieval needs. Records that only graph construction uses (the MemGraphRAG entity export, the passage-to-chunk map, SAM3 detections) are added with plan tasks 2 and 4.

## Artifacts root

Everything lives under one directory outside the repo, `$MGRV_ARTIFACTS`. On Snellius that should be project space or `/scratch-shared`, not `$HOME`. `paths.Artifacts` knows the layout:

```
$MGRV_ARTIFACTS/
├── webqa/                Misha's WebQA subset (dataset branch), copied as is:
│                         text_manifest.jsonl, image_manifest.jsonl, caption_manifest.jsonl,
│                         build_qa.json, test_qa.json
├── manifests/            written by `python -m memgraphrag_v.webqa manifests`
│   ├── text_manifest.jsonl, image_manifest.jsonl, questions.jsonl
│   └── captions.jsonl    generated image descriptions (caption systems only)
├── corpus/images/        <image_id>.jpg, from `webqa images`; ImageRecord.path is relative to this
├── models/               EVA-CLIP-8B/, clip-vit-large-patch14/ (scripts/download_eva_clip.py)
├── embeddings/eva-clip-8b/{image,query,crop}/   vectors.npy, rows.jsonl, meta.json
└── runs/<run_id>/        retrieval.jsonl, config.yaml
```

## Manifests (plan task 1): `schemas.py`

The dataset is Misha's WebQA subset (`subsetcheck4.py` on the `dataset` branch): 4,221 images, 12,080 passages (one caption `cap_<image_id>` per image plus `txt_<snippet_id>` text snippets), 480 graph-building questions (360 image, 120 text) and 120 held-out image questions. Misha also generated a description of every image with Qwen2.5-VL-7B (`createCaption.py`, `caption_manifest.jsonl`). `webqa.py` turns it into the manifests below:

- `images --tsv <file>` unpacks the images from the TSV Misha cut out of WebQA's own `imgs.tsv` (`/scratch-shared/scur0462/WebQA/shortlisted_imgs.tsv`, one base64 image per line) into `corpus/images/<image_id>.jpg`. These are the files WebQA was built with, and Misha captioned all 4,221 of them, so none are missing. We used to download them from Wikimedia instead, which took about 7 hours and lost 77 images that had since been deleted.
- `manifests` copies the passages, adds each image's local `path`, `width`, `height` and `ok`, and normalises the questions: `build_qa.json` becomes split `dev` (used to tune β), `test_qa.json` becomes split `test`, snippet IDs get their passages' `txt_` prefix, `gold_doc_ids` are the gold sources' titles, and `group_id` joins questions that share gold evidence. Misha's descriptions become `CaptionRecord`s with the model and prompt that made them. It fails on repeated IDs, on captions or gold sources missing from the corpus and on a description of an unknown image, and prints how many gold images do not open and how many images have no description.

Records are frozen and reject unknown fields. IDs are non-empty strings and may contain spaces, because `doc_id`s are page titles. A title is **not** unique to one image: images from the same Wikipedia article share it (up to 14), and 264 titles also have text snippets.

| File | Record | Fields |
|---|---|---|
| `text_manifest.jsonl` | `PassageRecord` | doc_id, passage_id, text, source_url? |
| `image_manifest.jsonl` | `ImageRecord` | image_id, doc_id, path, source_url?, commons_url?, passage_ids, width?, height?, ok |
| `questions.jsonl` | `QuestionRecord` | qid, question, answers, gold_doc_ids, gold_image_ids, gold_passage_ids, distractor_image_ids, distractor_passage_ids, q_type? (WebQA Qcate), topic?, modalities ⊆ {image, text}, split ∈ {dev, test}, group_id |
| `captions.jsonl` | `CaptionRecord` | image_id, text, model (`Qwen/Qwen2.5-VL-7B-Instruct`), prompt_version (`createCaption.py@2c94363`) |

Two kinds of caption: an image's WebQA caption is a passage (`cap_<image_id>`, in `text_manifest.jsonl`) and is what every system's reader gets with the image. The generated description in `captions.jsonl` is only for the caption baselines (C, G-caption).

Retrieval reads only `qid` and `question` from a question. The gold and distractor fields are for evaluation. Test questions' gold images are part of the corpus the dev questions built. The 120 test questions use only 49 distinct gold images, 33 of which are also gold for a dev question (104 test questions touch one), so most groups span both splits and β tuned on dev has seen the test images as gold.

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

- **`image_passage_edges(images, image_set)`** gives the graph edges `(image_id, passage_id, 1.0)` from each image to its passages (in WebQA, its one caption). Images without a vector are left out, so PPR can never select an image the reader cannot see.
- **`select_documents(image_scores, images, n_docs=5)`** scores each document by its best image and returns the top documents as `Evidence`, each with that image and its passages. Both D (cosine scores) and G (PPR image-node scores) use it, so they pick documents the same way.
- **`retrieve_direct(...)`** is system D: the question's EVA-CLIP vector against every whole-image vector. `python -m memgraphrag_v.image_retrieval --run-id <id> [--split dev]` writes `runs/<id>/retrieval.jsonl` and `config.yaml`, and prints, over the questions that have a gold image, how often a gold document is first and in the top 5, and the mean doc and image Recall@5 (a sanity check, not the evaluation). Image recall is the stricter and more meaningful one here, because a retrieved title can match a gold title through a different image of the same article. `python scripts/show_retrieval.py --run-id <id>` turns a run into `runs/<id>/report.html`: each question with its retrieved documents' thumbnails, scores and first passage, gold images outlined.

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
- `passage_ids` lists the evidence's associated passages in source order, **untrimmed**. For a WebQA image that is its caption. The reader input keeps only the first `retrieval.text_budget_tokens` (256) tokens of them. That cut happens once, in the shared reader-input step, for every system, so D and G get the same text budget. It is still to be built (reader/logging work), using the reader's tokenizer.

## Configuration: `config.py`, `configs/default.yaml`

The config holds the dataset name, visual encoder, SAM3 threshold and retrieval settings. Settings still to be decided are `null`: `sam3.threshold` (task 4) and `retrieval.beta` (tuned on dev). `unresolved()` lists them, and `require_frozen()` refuses to run while any remain, so call it before a test evaluation. `config_hash()` ignores key order and `artifacts_root`, so the same experiment hashes the same on every machine. Later tasks add their own sections.

## Running on Snellius

```bash
git clone … && cd memgraphrag-v
module load 2023 Python/3.11.3-GCCcore-12.3.0
python -m venv .venv-eva && source .venv-eva/bin/activate
pip install -e ".[eva,dev]" huggingface_hub && pytest -q
export MGRV_ARTIFACTS=/scratch-shared/$USER/mgrv
python scripts/download_eva_clip.py                     # login node, about 30 GB
sbatch scripts/dummy_image_retrieval.sbatch             # end-to-end test on the dummy dataset
mkdir -p $MGRV_ARTIFACTS/webqa && git fetch origin dataset
for f in text_manifest.jsonl image_manifest.jsonl caption_manifest.jsonl build_qa.json test_qa.json; do
  git show origin/dataset:$f > $MGRV_ARTIFACTS/webqa/$f; done
python -m memgraphrag_v.webqa images --tsv /scratch-shared/scur0462/WebQA/shortlisted_imgs.tsv   # login node
python -m memgraphrag_v.webqa manifests
sbatch scripts/webqa_image_retrieval.sbatch             # embed, then D on dev and test
# results: log lines "image_recall_at_5"; reports in runs/d-dev/ and runs/d-test/report.html
```

To run the parity check against MG2, use a GPU job that runs `python scripts/check_eva_parity.py --mg2 <MG2-RAG checkout>` (MG2 commit `91f0eed`).
