#!/bin/bash
# Submit from the repository root; required environment is in SNELLIUS.md.
# Usage: sbatch scripts/webqa_combined.sbatch all|index|qa dev|test run-id [limit]
# `index` builds all stages; `qa` reuses the index and runs query/retrieve/reader.
# The text vLLM server is terminated BEFORE SAM3/EVA/reader use the same GPU.
#SBATCH --job-name=webqa-g
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=webqa-g-%j.log

set -euo pipefail
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "$SLURM_SUBMIT_DIR"
: "${MGRV_ARTIFACTS:?Set MGRV_ARTIFACTS}"
: "${MGRV_TEXT_PYTHON:?Path to the tested MemGraphRAG environment's python}"
: "${MGRV_VISION_PYTHON:?Path to the tested EVA environment's python}"
: "${MGRV_TEXT_EMBEDDING_MODEL:?Local BGE snapshot directory}"
: "${MGRV_MEMGRAPHRAG:?MemGraphRAG checkout}"
: "${MGRV_MG2_RAG:?MG2-RAG checkout}"
: "${MGRV_SAM3_CHECKPOINT:?SAM3 weights}"
MGRV_READER_PYTHON=${MGRV_READER_PYTHON:-$MGRV_VISION_PYTHON}
MGRV_GROUND_PYTHON=${MGRV_GROUND_PYTHON:-$MGRV_VISION_PYTHON}
MGRV_READER_MODEL=${MGRV_READER_MODEL:-/scratch-shared/scur0462/hf_cache}
CONFIG=${MGRV_RUN_CONFIG:-configs/webqa_run.yaml}
MODE=${1:-all}
SPLIT=${2:-dev}
RUN_ID=${3:-smoke-dev}
LIMIT=${4:-0}
[[ "$MODE" =~ ^(all|index|qa)$ ]] || { echo 'Mode must be all, index or qa'; exit 2; }
[[ "$SPLIT" =~ ^(dev|test)$ ]] || { echo 'Split must be dev or test'; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi
COMMON=(--config "$CONFIG" --artifacts "$MGRV_ARTIFACTS")

if [[ "$MODE" != qa ]]; then
    : "${MGRV_VLLM:?Path to existing vllm executable}"
    : "${MGRV_TEXT_LLM_MODEL:?Local Qwen2.5-7B-Instruct snapshot}"
    PORT=$("$MGRV_TEXT_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
    LOG="$MGRV_ARTIFACTS/vllm-${SLURM_JOB_ID}.log"
    "$MGRV_VLLM" serve "$MGRV_TEXT_LLM_MODEL" \
        --served-model-name qwen2.5-7b-instruct --host 127.0.0.1 --port "$PORT" \
        --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.70 \
        --enforce-eager --override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":-1}' \
        > "$LOG" 2>&1 &
    SERVER_PID=$!
    trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
    READY=0
    for _ in $(seq 1 120); do
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -50 "$LOG"; exit 1; }
        if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            READY=1
            break
        fi
        sleep 5
    done
    [[ "$READY" == 1 ]] || { tail -50 "$LOG"; exit 1; }
    "$MGRV_TEXT_PYTHON" -u scripts/webqa_run.py index "${COMMON[@]}" \
        --stages text --llm-base-url "http://127.0.0.1:$PORT/v1"
    kill "$SERVER_PID"
    wait "$SERVER_PID" || true
    trap - EXIT
    "$MGRV_GROUND_PYTHON" -u scripts/webqa_run.py index "${COMMON[@]}" --stages ground
    "$MGRV_VISION_PYTHON" -u scripts/webqa_run.py index "${COMMON[@]}" --stages embed
    "$MGRV_TEXT_PYTHON" -u scripts/webqa_run.py index "${COMMON[@]}" --stages assemble
fi

if [[ "$MODE" != index ]]; then
    QUERY=("${COMMON[@]}" --split "$SPLIT" --run-id "$RUN_ID" --limit "$LIMIT")
    "$MGRV_VISION_PYTHON" -u scripts/webqa_run.py queries "${QUERY[@]}"
    "$MGRV_TEXT_PYTHON" -u scripts/webqa_run.py retrieve "${QUERY[@]}"
    "$MGRV_READER_PYTHON" -u scripts/webqa_run.py read "${QUERY[@]}" --reader-model "$MGRV_READER_MODEL"
fi
