#!/bin/bash
#SBATCH --account=gputue013
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

cd /home/scur0466/work/MemGraphRAG/code
export CODE_DIR="/gpfs/home6/scur0466/work/MemGraphRAG/code"
export W="/home/scur0466/work/webqa"
export D="/home/scur0466/dataset"
export OPENAI_API_KEY="dummy-local-key"

python3 -c "import torch; torch.cuda.empty_cache()"

for SYS in webqa_run_T webqa_run_C; do
  SAVE_DIR=$(find $W/$SYS/outputs/ -name "graph.graphml" -exec dirname {} \; | head -n 1)
  echo "Using save-dir for $SYS: $SAVE_DIR"

  for SET in test build; do
    echo "=== Running local retrieval & QA for $SYS on $SET ==="
    python3 $CODE_DIR/retrieval_dataset_test.py \
      --questions $D/${SET}_qa.json \
      --save-dir "$SAVE_DIR" \
      --output $W/$SYS/qa_results_${SET}.json \
      --llm-name Qwen/Qwen2.5-7B-Instruct \
      --embedding-model BAAI/bge-large-en-v1.5 \
      --retrieval-top-k 50
  done
done
