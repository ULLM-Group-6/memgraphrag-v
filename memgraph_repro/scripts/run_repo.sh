#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --job-name=repro
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
# inputs (env): TAG  CORPUS  QUESTIONS  [SAMPLE_NUM=0]  [WORKERS=8]  [EXTRA_ARGS]
set -u
TAG=${TAG:?}; CORPUS=${CORPUS:?}; QUESTIONS=${QUESTIONS:?}
SAMPLE_NUM=${SAMPLE_NUM:-0}; WORKERS=${WORKERS:-8}; EXTRA_ARGS=${EXTRA_ARGS:-}
W=$HOME/work/memgraph_repro
REPO=$HOME/work/MemGraphRAG
OUT=$W/$TAG/outputs; RES=$W/$TAG/qa_results.json
mkdir -p $OUT

export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 VLLM_USE_FLASHINFER_SAMPLER=0
module purge; module load 2023; module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/memgraph/bin/activate
cd $REPO
echo "== TAG=$TAG host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION"
echo "== corpus=$CORPUS ($(stat -c%s $CORPUS) bytes)"

# a node is shared by several jobs: use a private free port, loopback only
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
echo "vLLM port: $PORT"

TOTAL=$(python -c "import torch; f,t=torch.cuda.mem_get_info(); print(round(t/2**30,1))")
UTIL=$(python -c "t=$TOTAL; print(round(min(0.85,(t-3)/t),2))")
echo "GPU GiB: $TOTAL  vLLM util: $UTIL"
QWEN=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
EMBEDDING_MODEL=$(ls -d $HF_HOME/hub/models--BAAI--bge-large-en-v1.5/snapshots/* | head -1)
LOG=$W/logs/vllm_${TAG}_$SLURM_JOB_ID.log

$HOME/venvs/vllm/bin/vllm serve "$QWEN" --served-model-name qwen2.5-7b-instruct \
  --host 127.0.0.1 --port $PORT --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization $UTIL --enforce-eager \
  --override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":-1,"repetition_penalty":1.0}' > $LOG 2>&1 &
VLLM_PID=$!
trap 'echo "vLLM chat requests served: $(grep -c "POST /v1/chat/completions" $LOG)"; kill $VLLM_PID 2>/dev/null' EXIT

UP=0
for i in $(seq 1 120); do
  kill -0 $VLLM_PID 2>/dev/null || { echo "VLLM DIED"; tail -60 $LOG; exit 1; }
  if grep -q "Application startup complete" $LOG && curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
    UP=1; echo "vLLM up after ~$((i*5))s"; break
  fi
  sleep 5
done
[ $UP -eq 1 ] || { echo "VLLM NOT UP"; tail -60 $LOG; exit 1; }

# prove that OUR server (not another job's) answers real requests
RESP=$(curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-instruct","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}')
echo "test reply: $(echo "$RESP" | cut -c1-250)"
sleep 2
grep -q "POST /v1/chat/completions" $LOG || { echo "ABORT: test request not seen in our own vLLM log (wrong server answered)"; exit 1; }
echo "OK: our vLLM served the test request"

export OPENAI_API_KEY=dummy
LLM_NAME=qwen2.5-7b-instruct
LLM_BASE_URL=http://127.0.0.1:$PORT/v1

echo "== indexing start: $(date)"
python code/index.py \
  --corpus $CORPUS \
  --save-dir $OUT \
  --llm-name "$LLM_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --tokenizer "$EMBEDDING_MODEL" \
  --chunk-size 256 \
  --chunk-overlap 32 \
  --artifact-mode default \
  --memory-max-workers $WORKERS $EXTRA_ARGS || { echo "INDEXING FAILED"; tail -40 $LOG; exit 1; }
echo "== indexing end: $(date)"

python - << PY
import sqlite3, glob
for f in glob.glob("$OUT/llm_cache/*.sqlite"):
    c = sqlite3.connect(f)
    for (t,) in c.execute("select name from sqlite_master where type='table'"):
        print("LLM cache rows:", t, c.execute(f'select count(*) from "{t}"').fetchone()[0])
PY

echo "== retrieval + QA start: $(date)"
python code/retrieval_dataset_test.py \
  --questions $QUESTIONS \
  --save-dir $OUT \
  --output $RES \
  --llm-name "$LLM_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --question-type all \
  --sample-num $SAMPLE_NUM \
  --skip-fact-rerank true \
  --fact-similarity-threshold 0.4 \
  --use-raw-threshold-filter true || echo "RETRIEVAL FAILED"
echo "== retrieval + QA end: $(date)"
echo "== done"
