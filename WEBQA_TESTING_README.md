# WebQA End-to-End Evaluation & Testing Guide

This directory contains the pipeline, code structure, issues encountered, and benchmark results for evaluating MemGraphRAG retrieval performance (Recall@5) on the WebQA dataset.

---

## 1. Directory Structure

- web_qa_testing/: Direct CPU-based dense vector retrieval scripts (retrieval_dataset_test.py).
- web_qa_scripts/: Evaluation utilities (eval_retrieval.py, webqa_common.py) and execution runner (run_webqa_eval.sh).
- evaluation_results/: Output candidate predictions (qa_results_*.json) and computed metrics (recall_*.json) across two system configurations (webqa_run_T and webqa_run_C).

---

## 2. Issues Encountered & Technical Fixes

### Issue 1: GPU Out-Of-Memory (OOM) & Resource Allocation Timeouts
- Symptom: Full engine evaluation required holding LLM weights and vector indexes in GPU memory, causing node timeouts and OOM crashes during batch processing.
- Fix: Decoupled evaluation from GPU runtimes. Implemented web_qa_testing/retrieval_dataset_test.py to perform dense retrieval on CPU by loading passage embeddings directly from the binary Parquet artifact (vdb_chunk.parquet) via pandas and numpy.

### Issue 2: 0.0 Recall@5 Score (Index Parsing Mismatch)
- Symptom: Initial evaluation returned 0.0 Recall across all queries, throwing the error: passage without 'idx:' prefix: the one-chunk-per-passage driver was not used.
- Root Cause: webqa_common.py parses returned document strings using regex ^(\d+): to extract the zero-based integer line position corresponding to passages.jsonl. Outputting raw chunk IDs or plain un-indexed text caused regex lookup failures.
- Fix: Added text normalization (clean_text()) in retrieval_dataset_test.py to map vector DB content strings directly back to their corpus line index, formatting document strings as {line_index}:{normalized_text}.

---

## 3. End-to-End Execution Workflow

Step 1: Environment Setup
source ~/venvs/memgraph/bin/activate
export CODE_DIR="$(pwd)"
export W="/home/scur0466/work/webqa"
export D="/home/scur0466/dataset"
export CUDA_VISIBLE_DEVICES=""

Step 2: Run Retrieval Generation
SAVE_DIR_T=$(find $W/webqa_run_T/outputs/ -name "graph.graphml" -exec dirname {} \; | head -n 1)
SAVE_DIR_C=$(find $W/webqa_run_C/outputs/ -name "graph.graphml" -exec dirname {} \; | head -n 1)

python web_qa_testing/retrieval_dataset_test.py --questions $D/test_qa.json --save-dir "$SAVE_DIR_T" --passages $W/webqa_run_T/corpus/passages.jsonl --output evaluation_results/webqa_run_T/qa_results_test.json --retrieval-top-k 50
python web_qa_testing/retrieval_dataset_test.py --questions $D/build_qa.json --save-dir "$SAVE_DIR_T" --passages $W/webqa_run_T/corpus/passages.jsonl --output evaluation_results/webqa_run_T/qa_results_build.json --retrieval-top-k 50

python web_qa_testing/retrieval_dataset_test.py --questions $D/test_qa.json --save-dir "$SAVE_DIR_C" --passages $W/webqa_run_C/corpus/passages.jsonl --output evaluation_results/webqa_run_C/qa_results_test.json --retrieval-top-k 50
python web_qa_testing/retrieval_dataset_test.py --questions $D/build_qa.json --save-dir "$SAVE_DIR_C" --passages $W/webqa_run_C/corpus/passages.jsonl --output evaluation_results/webqa_run_C/qa_results_build.json --retrieval-top-k 50

Step 3: Compute Recall@5 Metrics
cd web_qa_scripts/
for SYS in webqa_run_T webqa_run_C; do 
  for SET in test build; do 
    python eval_retrieval.py \
      --qa-results ../evaluation_results/$SYS/qa_results_$SET.json \
      --questions $D/${SET}_qa.json \
      --passages $W/$SYS/corpus/passages.jsonl \
      --data-dir $D \
      --out ../evaluation_results/$SYS/recall_$SET.json
  done
done

---

## 4. Final Verified Results (Recall@5)

| System Run | Split | Sample Size (N) | Document-ID Recall@5 | 95% Grouped Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: |
| webqa_run_T | Test (test_qa.json) | 120 | 0.4125 | [0.294, 0.578] |
| webqa_run_T | Build (build_qa.json) | 480 | 0.6538 | [0.605, 0.704] |
| webqa_run_C | Test (test_qa.json) | 120 | 0.5917 | [0.442, 0.668] |
| webqa_run_C | Build (build_qa.json) | 480 | 0.6653 | [0.611, 0.717] |
