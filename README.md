# MemGraphRAG-V

MemGraphRAG-V extends [MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG) with native image evidence for graph-based retrieval and question answering. It combines textual memory, SAM3 entity grounding, EVA-CLIP visual embeddings and Personalized PageRank (PPR), followed by a Qwen2.5-VL reader.

Developed for **2AMM20 Research Topics in Data Mining, TU/e**. The implemented pipeline and completed WebQA experiments are on [`combined-memgraphrag-v`](https://github.com/ULLM-Group-6/memgraphrag-v/tree/combined-memgraphrag-v).

## How it works

1. **Text:** index WebQA snippets, original image captions and generated image descriptions with MemGraphRAG; extract entities, facts and memory schemas.
2. **Ground:** prompt SAM3 with entity names associated with each image. Save detections, masks and crops; retain detections with confidence at least 0.5.
3. **Embed:** use EVA-CLIP-8B to embed whole images and grounded crops. Crop matches seed existing entity nodes; crops are not graph nodes.
4. **Assemble:** extend the textual graph with image nodes, image–passage edges and confidence-weighted image–entity edges.
5. **Retrieve and answer:** combine normalized text and visual seeds as `s = (1 - beta) * s_text + beta * s_visual`, then run PPR. Image and crop-entity components are separately normalized and equally weighted when both are available. The current selector ranks documents by their highest passage or image PPR score, returning up to five documents. Dense passage retrieval is the fallback when graph retrieval yields no usable evidence.

The reader receives original images and original WebQA text, with a 256-token text budget per document. Generated descriptions are used for indexing, not supplied to the multimodal reader.

## Data and models

The WebQA subset contains **4,221 images**, **12,080 base text records** (including original captions), and **4,221 additional generated descriptions**. `dataset/` contains manifests and question files; image files and model weights must be obtained separately.

There are 480 build/development questions and 120 test questions. Beta selection uses 425 eligible development questions, excluding 55 that share gold evidence with test questions. All 120 test questions require image evidence.

Text processing uses **Qwen2.5-7B-Instruct** through local vLLM and **BAAI/bge-large-en-v1.5** embeddings. Vision uses **SAM3** through [MG2-RAG](https://github.com/Daboolu/MG2-RAG), **EVA-CLIP-8B** with the `openai/clip-vit-large-patch14` processor, and **Qwen2.5-VL-7B-Instruct** for descriptions and reading.

## Setup

Run from the repository root. Keep separate text (PyTorch 2.5.1 / Transformers 4.45.2), vision (PyTorch 2.7.0 / Transformers 4.49.0), and vLLM environments. Install upstream dependencies as well as this package:

```bash
# Text environment, after installing MemGraphRAG dependencies:
python -m pip install -e '.[dev]'

# Vision environment, alongside the required SAM3/model dependencies:
python -m pip install -e '.[eva,dev]'
```

Use MemGraphRAG commit `49f20cec41290c95209724d33ec527f72449d300` and MG2-RAG commit `91f0eed`. Apply the MemGraphRAG patches in `patches/` in this order: `ner-json`, `openie-retries`, `openie-triple-repair`, `triple-prompt`, `memory-json`, `conflict-context` (all prefixed `memgraphrag-`). They adapt extraction parsing, retries and memory processing for the local model.

Set these variables to your own absolute paths before submitting jobs:

| Variables | Point to |
|---|---|
| `MGRV_ARTIFACTS` | Persistent output/checkpoint directory |
| `MGRV_RUN_CONFIG` | `configs/webqa_run.yaml` |
| `MGRV_TEXT_PYTHON`, `MGRV_VISION_PYTHON` | Text and vision Python executables |
| `MGRV_GROUND_PYTHON`, `MGRV_READER_PYTHON` | Grounding and reader Python executables; may use the vision environment |
| `MGRV_MEMGRAPHRAG`, `MGRV_MG2_RAG` | Upstream checkouts |
| `MGRV_SAM3_CHECKPOINT` | SAM3 checkpoint file |
| `MGRV_TEXT_EMBEDDING_MODEL`, `MGRV_TEXT_LLM_MODEL` | Local BGE and Qwen2.5-7B model directories |
| `MGRV_READER_MODEL` | Local Qwen2.5-VL-7B model directory |
| `MGRV_EVA_MODELS` | Directory containing `EVA-CLIP-8B/` and `clip-vit-large-patch14/` |
| `MGRV_VLLM` | vLLM executable |

Use `OPENAI_API_KEY=dummy` for the local text server. Set all paths explicitly; older scripts contain shared-user defaults.

## Running the pipeline

### Prepare and index

Unpack WebQA images using `dataset/unpack_tsv.py`, adjusting its input/output paths. Then prepare the artifacts:

```bash
"$MGRV_TEXT_PYTHON" scripts/webqa_run.py prepare \
  --dataset dataset --images /absolute/path/to/images \
  --artifacts "$MGRV_ARTIFACTS"
```

Inside an allocated GPU job, start the local text vLLM server and run the text stage, then stop that server before running grounding:

```bash
"$MGRV_TEXT_PYTHON" scripts/webqa_run.py index \
  --config "$MGRV_RUN_CONFIG" --artifacts "$MGRV_ARTIFACTS" \
  --stages text --llm-base-url http://127.0.0.1:8000/v1

"$MGRV_GROUND_PYTHON" scripts/webqa_run.py index \
  --config "$MGRV_RUN_CONFIG" --artifacts "$MGRV_ARTIFACTS" --stages ground
```

`scripts/webqa_combined.sbatch` contains the vLLM startup and staged execution commands. GPU work belongs in a compute allocation, not on a login node.

### Continue embedding, assemble and tune

After grounding is complete, the checkpointed continuation launcher submits embedding, assembly, development tuning and testing:

```bash
bash scripts/submit_webqa_sequential.sh webqa-tuned-v1 index
```

Limits are **4 hours embedding + optional 4-hour backup**, **8 hours assembly**, and **3 hours each for tuning/testing** (older comments say 5/6 hours). Only one GPU job runs at a time. Assembly requires completed embeddings. Preserve artifact paths and run IDs to resume checkpoints; assembly is not checkpointed midway.

Tuning compares beta values `0.25`, `0.50`, `0.75` by eligible-development Recall@5, breaking ties toward smaller beta. It saves `tuning_summary.json` and a frozen test configuration, overriding the provisional base-YAML beta.

### Final single-worker test

The released run used this launcher after tuning, with block-wise vision attention to reduce memory use. It handles all 120 questions. Use it instead of the chain's standard test job; do not schedule both against the same run.

```bash
sbatch scripts/webqa_test_single_lowmem.sbatch \
  webqa-tuned-v1 webqa-tuned-v1-test-single
```

To queue before tuning finishes, add `--dependency=afterok:TUNING_JOB_ID --kill-on-invalid-dep=yes` to `sbatch`.

## Results and offline evaluation

The final test run selected **beta = 0.25** and achieved **67.50% Document-ID Recall@5** and **63.75% supplied-image Recall@5**. Its paired recall difference against the caption-augmented dense baseline has a confidence interval containing zero. The build run also scored 0% on text questions, so the image-only test result should not be generalized to mixed workloads. Lexical answer matching is a heuristic, not semantic answer accuracy.

Saved outputs include document rankings (`retrieval.jsonl`), references (`labels.jsonl`), answers and reader evidence (`answers.jsonl`), evaluation inputs, settings and completion/provenance records. Evaluate the committed test results locally without a GPU:

```bash
python evaluation/evaluate_webqa.py results/webqa-tuned-v1-test-single \
  --output output/webqa-evaluation
```

This writes recall scores, grouped-bootstrap 95% intervals, per-question results and examples for manual review. See [results/README.md](results/README.md) for baseline comparisons and additional metrics. Large test diagnostics are available in the [WebQA results release](https://github.com/ULLM-Group-6/memgraphrag-v/releases/tag/webqa-test-results-v1); the evaluator does not require that archive.

## Repository map and checks

| Directory | Contents |
|---|---|
| `src/memgraphrag_v/` | Schemas, indexing, grounding, embeddings and retrieval |
| `scripts/` | Runners, checkpoint helpers, Slurm launchers and optional diagnostics |
| `configs/`, `dataset/`, `patches/` | Configuration, dataset manifests and upstream adaptations |
| `evaluation/`, `results/`, `evaluation_results/` | Evaluation tools, multimodal outputs and baseline results |
| `tests/` | Unit, regression and submission checks |
| `docs/` | Methodology and development notes; some earlier planning notes are historical |

Run Python checks in the configured environment with `python -m pytest -q`. Shell submission checks in `tests/` use mocked scheduling; they do not run the GPU pipeline. Optional `scripts/check_*.py`, `smoke_eva_clip.py` and `show_retrieval.py` support diagnostics rather than normal execution.

Model weights and images are external dependencies. Keep credentials and machine-specific paths outside Git, and follow the respective upstream code, model and dataset licences.
