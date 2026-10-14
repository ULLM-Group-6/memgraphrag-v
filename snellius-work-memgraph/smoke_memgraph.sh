#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:50:00
#SBATCH --job-name=smoke
#SBATCH --output=slurm_%x_%A.out
#SBATCH --error=slurm_%x_%A.err

W=$HOME/snellius-work
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
module purge; module load 2023; module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/memgraph/bin/activate
cd $W/MemGraphRAG

echo "== import test (from code/)"
(cd code && python -c "import src.MemGraphRAG; print('import OK')") || { echo "IMPORT FAILED"; exit 1; }

echo "== GPU memory visible to this job"
TOTAL=$(python -c "import torch; f,t=torch.cuda.mem_get_info(); print(round(t/2**30,1))")
echo "total GiB: $TOTAL"
# leave ~3 GiB for the embedding model, cap at 0.85
UTIL=$(python -c "t=$TOTAL; print(round(min(0.85,(t-3)/t),2))")
echo "vLLM gpu-memory-utilization: $UTIL"

QWEN=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
BGE=$(ls -d $HF_HOME/hub/models--BAAI--bge-large-en-v1.5/snapshots/* | head -1)
LOG=$W/vllm_server_$SLURM_JOB_ID.log

echo "== starting vLLM"
$HOME/venvs/vllm/bin/vllm serve "$QWEN" --served-model-name qwen2.5-7b-instruct \
  --port 8000 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization $UTIL --enforce-eager --override-generation-config '{"temperature":0.0,"top_p":1.0,"top_k":-1,"repetition_penalty":1.0}' > $LOG 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

UP=0
for i in $(seq 1 120); do
  if curl -s localhost:8000/v1/models > /dev/null 2>&1; then UP=1; echo "server up after ~$((i*5))s"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "VLLM DIED"; tail -60 $LOG; exit 1; fi
  sleep 5
done
[ $UP -eq 1 ] || { echo "VLLM NOT UP IN TIME"; tail -60 $LOG; exit 1; }

echo "== test request"
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-instruct","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}'
echo

export OPENAI_API_KEY=dummy
LLM=qwen2.5-7b-instruct
URL=http://localhost:8000/v1
OUT=$W/tiny/outputs2

echo "== indexing"; date
python code/index.py --corpus $W/tiny/corpus.txt --save-dir $OUT \
  --llm-name $LLM --llm-base-url $URL \
  --embedding-model $BGE --tokenizer $BGE \
  --chunk-size 256 --chunk-overlap 32 --artifact-mode default \
  --memory-max-workers 8 \
  --ontology-filter-mode absolute --ontology-min-frequency 1 || { echo "INDEXING FAILED"; tail -40 $LOG; exit 1; }
date

echo "== retrieval + QA"
python code/retrieval_dataset_test.py --questions $W/tiny/questions.json \
  --save-dir $OUT --output $W/tiny/qa_results.json \
  --llm-name $LLM --llm-base-url $URL --embedding-model $BGE \
  --question-type all --sample-num 0 \
  --skip-fact-rerank true --fact-similarity-threshold 0.4 --use-raw-threshold-filter true || echo "RETRIEVAL FAILED"
date

echo "== outputs"
find $OUT -maxdepth 2 | head -40
python - << 'PY'
import json
d = json.load(open("/home/scur0466/snellius-work/tiny/qa_results.json"))
print("keys:", list(d.keys()))
q = json.load(open("/home/scur0466/snellius-work/tiny/questions.json"))
for k in ("answers","solutions"):
    if k in d: print(k, json.dumps(d[k], indent=1)[:2500])
print("GOLD:", [x["answer"] for x in q])
PY
echo "== done"
